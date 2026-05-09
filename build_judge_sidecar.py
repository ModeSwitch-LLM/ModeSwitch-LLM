from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd


# ============================================================
# OpenAI client setup
# ============================================================

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError(
        "Missing openai package. Install with: pip install openai"
    ) from exc


# ============================================================
# Basic helpers
# ============================================================

ANSWER_COLUMN_CANDIDATES = [
    "generated_text",
    "output_text",
    "response_text",
    "completion",
    "model_output",
    "generated_output",
    "answer",
    "text",
]

PROMPT_COLUMN_CANDIDATES = [
    "prompt",
    "input_prompt",
    "user_prompt",
    "workload_prompt",
]


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def find_first_existing_column(df: pd.DataFrame, candidates) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def extract_item_id(workload_name: str) -> Optional[str]:
    """
    Example:
    mt_bench_eval__mt_bench_0001 -> mt_bench_0001
    alpacaeval2_lc_eval__alpacaeval2_0001 -> alpacaeval2_0001
    """
    if "__" not in str(workload_name):
        return None
    return str(workload_name).split("__", 1)[1]


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_prompt_lookup(benchmark_data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
    {
        "mt_bench_0001": {...},
        "alpacaeval2_0001": {...}
    }
    """
    lookup = {}

    for filename in ["mt_bench_eval.jsonl", "alpacaeval2_lc_eval.jsonl"]:
        path = benchmark_data_dir / filename

        for row in load_jsonl(path):
            lookup[row["id"]] = row

    return lookup


def latest_results_csv(raw_results_dir: Path) -> Path:
    patterns = [
        "dense_final_results_*.csv",
        "dense_final_prejudge_results_*.csv",
        "dense_final_partial_*.csv",
        "benchmark_results_*.csv",
    ]

    paths = []
    for pattern in patterns:
        paths.extend(raw_results_dir.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No dense final / benchmark CSV files found in {raw_results_dir}"
        )

    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


# ============================================================
# Judge schemas
# ============================================================

MT_BENCH_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "minimum": 1,
            "maximum": 10,
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {
            "type": "string",
            "enum": ["candidate", "baseline", "tie"],
        },
        "candidate_score": {
            "type": "number",
            "minimum": 1,
            "maximum": 10,
        },
        "baseline_score": {
            "type": "number",
            "minimum": 1,
            "maximum": 10,
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["winner", "candidate_score", "baseline_score", "reason"],
    "additionalProperties": False,
}


# ============================================================
# OpenAI structured judging call
# ============================================================

def call_structured_judge(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: Dict[str, Any],
    max_retries: int = 3,
    sleep_s: float = 2.0,
) -> Dict[str, Any]:
    """
    Uses Responses API structured output.
    Falls back to Chat Completions structured output if needed.
    """

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
                temperature=0,
            )

            return json.loads(response.output_text)

        except TypeError:
            # Older SDK fallback.
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                },
                temperature=0,
            )

            return json.loads(response.choices[0].message.content)

        except Exception as exc:
            if attempt == max_retries:
                raise

            print(f"Judge call failed on attempt {attempt}/{max_retries}: {exc}")
            time.sleep(sleep_s * attempt)

    raise RuntimeError("Judge failed unexpectedly.")


# ============================================================
# MT-Bench-style single-answer judge
# ============================================================

def judge_mt_bench(
    client: OpenAI,
    model: str,
    prompt: str,
    answer: str,
) -> Dict[str, Any]:

    system_prompt = (
        "You are a strict but fair evaluator of assistant responses. "
        "Score the answer from 1 to 10 based on helpfulness, correctness, "
        "instruction following, clarity, and completeness. "
        "Do not reward verbosity by itself. Return only the structured JSON."
    )

    user_prompt = f"""
Evaluate the assistant answer.

TASK PROMPT:
{prompt}

ASSISTANT ANSWER:
{answer}

Scoring guide:
1-2: mostly wrong, irrelevant, or unsafe
3-4: weak answer with major issues
5-6: partially useful but incomplete
7-8: good answer with minor issues
9-10: excellent, correct, clear, and complete
""".strip()

    return call_structured_judge(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="mt_bench_judgment",
        schema=MT_BENCH_SCHEMA,
    )


# ============================================================
# AlpacaEval-style pairwise judge
# ============================================================

def judge_pairwise_vs_baseline(
    client: OpenAI,
    model: str,
    prompt: str,
    baseline_answer: str,
    candidate_answer: str,
) -> Dict[str, Any]:

    system_prompt = (
        "You are a strict pairwise evaluator of instruction-following responses. "
        "Compare the candidate answer against the baseline answer. "
        "Judge correctness, helpfulness, instruction following, clarity, and conciseness. "
        "Do not prefer an answer only because it is longer. Return only structured JSON."
    )

    user_prompt = f"""
Instruction:
{prompt}

Baseline answer:
{baseline_answer}

Candidate answer:
{candidate_answer}

Decide which answer is better:
- candidate
- baseline
- tie
""".strip()

    return call_structured_judge(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="pairwise_judgment",
        schema=PAIRWISE_SCHEMA,
    )


# ============================================================
# Main sidecar builder
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/scratch/as18181/ModeSwitch-LLM/ModeSwitch-LLM"),
    )

    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Optional explicit dense_final_results_*.csv path.",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.environ.get("JUDGE_MODEL", "gpt-4o-mini"),
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for testing the judge script.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing judge sidecar instead of appending/skipping existing rows.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output sidecar path.",
    )

    args = parser.parse_args()

    project_root = args.project_root
    benchmark_data_dir = project_root / "benchmark_data"
    raw_results_dir = project_root / "results" / "raw"

    output_path = args.output_path or (benchmark_data_dir / "judge_scores_sidecar.jsonl")

    results_csv = args.results_csv or latest_results_csv(raw_results_dir)

    print("Project root:", project_root)
    print("Results CSV:", results_csv)
    print("Benchmark data dir:", benchmark_data_dir)
    print("Output judge sidecar:", output_path)
    print("Judge model:", args.judge_model)

    client = OpenAI()

    results_df = pd.read_csv(results_csv)

    answer_col = find_first_existing_column(results_df, ANSWER_COLUMN_CANDIDATES)
    prompt_col = find_first_existing_column(results_df, PROMPT_COLUMN_CANDIDATES)

    if answer_col is None:
        print("Available columns:")
        print(results_df.columns.tolist())
        raise ValueError(
            "Could not find generated answer column. "
            "Update ANSWER_COLUMN_CANDIDATES with your actual output column name."
        )

    print("Using answer column:", answer_col)
    print("Using prompt column:", prompt_col)

    prompt_lookup = load_prompt_lookup(benchmark_data_dir)

    judge_df = results_df[
        results_df["workload_name"].astype(str).str.startswith(
            ("mt_bench_eval__", "alpacaeval2_lc_eval__")
        )
    ].copy()

    if "success" in judge_df.columns:
        judge_df = judge_df[judge_df["success"].astype(str).str.lower().isin(["true", "1", "yes"])]

    if args.max_rows is not None:
        judge_df = judge_df.head(args.max_rows).copy()

    print("Rows to judge:", len(judge_df))

    if len(judge_df) == 0:
        print("Nothing to judge. Did you run mt_bench_eval / alpacaeval2_lc_eval yet?")
        return

    # Existing sidecar rows, so script can resume.
    existing_keys = set()

    if output_path.exists() and not args.overwrite:
        for row in load_jsonl(output_path):
            key = (
                str(row.get("workload_name")),
                str(row.get("mode_name")),
                int(row.get("trial_index", 0)),
                str(row.get("benchmark_suite")),
            )
            existing_keys.add(key)

    mode_col = "mode_name"
    trial_col = "trial_index"

    # Build baseline answers for AlpacaEval pairwise judging.
    fp16_rows = judge_df[judge_df[mode_col] == "fp16_baseline"].copy()

    baseline_by_workload_trial: Dict[Tuple[str, int], str] = {}
    baseline_by_workload: Dict[str, str] = {}

    for _, row in fp16_rows.iterrows():
        workload_name = str(row["workload_name"])
        trial_index = int(row[trial_col]) if trial_col in row and pd.notna(row[trial_col]) else 0
        answer = str(row[answer_col])

        baseline_by_workload_trial[(workload_name, trial_index)] = answer

        if workload_name not in baseline_by_workload:
            baseline_by_workload[workload_name] = answer

    mode = "w" if args.overwrite else "a"

    num_written = 0
    num_skipped = 0

    with open(output_path, mode, encoding="utf-8") as f:
        for _, row in judge_df.iterrows():
            workload_name = str(row["workload_name"])
            mode_name = str(row[mode_col])
            trial_index = int(row[trial_col]) if trial_col in row and pd.notna(row[trial_col]) else 0
            candidate_answer = str(row[answer_col])

            if workload_name.startswith("mt_bench_eval__"):
                benchmark_suite = "mt_bench"
            elif workload_name.startswith("alpacaeval2_lc_eval__"):
                benchmark_suite = "alpacaeval2_lc"
            else:
                continue

            key = (workload_name, mode_name, trial_index, benchmark_suite)

            if key in existing_keys:
                num_skipped += 1
                continue

            item_id = extract_item_id(workload_name)

            if prompt_col is not None and pd.notna(row[prompt_col]):
                prompt = str(row[prompt_col])
            elif item_id in prompt_lookup:
                prompt = str(prompt_lookup[item_id]["prompt"])
            else:
                print(f"Skipping {workload_name}: could not find prompt.")
                num_skipped += 1
                continue

            print(f"Judging {benchmark_suite} | {mode_name} | {workload_name} | trial={trial_index}")

            try:
                if benchmark_suite == "mt_bench":
                    judgment = judge_mt_bench(
                        client=client,
                        model=args.judge_model,
                        prompt=prompt,
                        answer=candidate_answer,
                    )

                    mt_score = float(judgment["score"])

                    sidecar_row = {
                        "workload_name": workload_name,
                        "mode_name": mode_name,
                        "trial_index": trial_index,
                        "benchmark_suite": "mt_bench",
                        "mt_bench_score": mt_score,
                        "benchmark_primary_metric_value": mt_score,
                        "judge_model": args.judge_model,
                        "judge_reason": judgment.get("reason", ""),
                    }

                else:
                    # Baseline against itself gets tie.
                    if mode_name == "fp16_baseline":
                        win_rate = 0.5
                        judgment = {
                            "winner": "tie",
                            "candidate_score": 0.0,
                            "baseline_score": 0.0,
                            "reason": "Baseline compared against itself.",
                        }
                    else:
                        baseline_answer = baseline_by_workload_trial.get(
                            (workload_name, trial_index)
                        )

                        if baseline_answer is None:
                            baseline_answer = baseline_by_workload.get(workload_name)

                        if baseline_answer is None:
                            print(f"Skipping {workload_name}: missing fp16_baseline answer.")
                            num_skipped += 1
                            continue

                        judgment = judge_pairwise_vs_baseline(
                            client=client,
                            model=args.judge_model,
                            prompt=prompt,
                            baseline_answer=baseline_answer,
                            candidate_answer=candidate_answer,
                        )

                        winner = judgment["winner"]

                        if winner == "candidate":
                            win_rate = 1.0
                        elif winner == "baseline":
                            win_rate = 0.0
                        else:
                            win_rate = 0.5

                    sidecar_row = {
                        "workload_name": workload_name,
                        "mode_name": mode_name,
                        "trial_index": trial_index,
                        "benchmark_suite": "alpacaeval2_lc",
                        "alpacaeval2_lc_win_rate": win_rate,
                        "benchmark_primary_metric_value": win_rate,
                        "judge_model": args.judge_model,
                        "judge_winner": judgment.get("winner", ""),
                        "judge_candidate_score": judgment.get("candidate_score", None),
                        "judge_baseline_score": judgment.get("baseline_score", None),
                        "judge_reason": judgment.get("reason", ""),
                    }

                f.write(json.dumps(sidecar_row, ensure_ascii=False) + "\n")
                f.flush()

                existing_keys.add(key)
                num_written += 1

            except Exception as exc:
                print(f"Failed judging {workload_name} | {mode_name} | trial={trial_index}: {exc}")
                num_skipped += 1

    print("Done.")
    print("Rows written:", num_written)
    print("Rows skipped:", num_skipped)
    print("Judge sidecar:", output_path)


if __name__ == "__main__":
    main()