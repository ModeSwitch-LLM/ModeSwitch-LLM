"""
==============================================================================
 reporter.py
==============================================================================

Purpose:
- load raw benchmark results produced by ModeSwitch-LLM
- compute grouped aggregate statistics across repeated trials
- compare each mode against the fp16_baseline on the same workload
- quantify whether a workload is prefill-dominated or decode-dominated
- generate a markdown summary plus optional plots for paper/report use

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from config import CONFIG
import re

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    np = None
    _MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib/numpy not installed. Plot generation is disabled.")


def _collapse_report_workload_name(name):
    name = str(name)
    if "__" in name:
        name = name.split("__", 1)[0]
    name = re.sub(r"_v\d+$", "", name)
    return name


def _collapse_report_df_for_plotting(df):
    if df is None or len(df) == 0 or "workload_name" not in df.columns:
        return df

    out = df.copy()
    out["workload_name"] = out["workload_name"].apply(_collapse_report_workload_name)

    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
    agg_map = {col: "mean" for col in numeric_cols}

    for col in ["n", "failure_count"]:
        if col in agg_map:
            agg_map[col] = "sum"

    non_numeric_cols = [
        col for col in out.columns
        if col not in numeric_cols and col not in ["mode_name", "workload_name"]
    ]
    for col in non_numeric_cols:
        agg_map[col] = "first"

    return (
        out
        .groupby(["mode_name", "workload_name"], as_index=False)
        .agg(agg_map)
    )

# =============================================================================
# Result-schema fields used by this adapted reporter
# =============================================================================

FLOAT_METRICS = [
    "ttft_ms",
    "avg_tbt_ms",
    "tbt_median_ms",
    "tbt_p95_ms",
    "tbt_p99_ms",
    "tbt_std_ms",
    "prefill_latency_ms",
    "decode_latency_ms",
    "total_latency_ms",
    "tokens_per_second",
    "batched_tokens_per_second",
    "prefill_throughput_tps",
    "decode_throughput_tps",
    "decode_prefill_ratio",
    "peak_gpu_memory_mb",
    "reserved_gpu_memory_mb",
    "cpu_ram_peak_mb",
    "cpu_ram_delta_mb",
    "gpu_peak_delta_mb",
    "gpu_reserved_delta_mb",
    "kv_cache_estimate_mb",
    "avg_power_w",
    "energy_joules",
    "energy_per_token_j",
    "reference_rouge_l_f1",
    "reference_token_f1",
    "baseline_similarity_rouge_l_f1",
    "quality_degradation_vs_baseline",
    "benchmark_primary_metric_value",
    "mmlu_pro_accuracy",
    "gsm8k_exact_match_accuracy",
    "truthfulqa_accuracy",
    "gpqa_accuracy",
    "mlu_accuracy",
    "tam_accuracy",
    "mt_bench_score",
    "alpacaeval2_lc_win_rate",
]

DEFAULT_DELTA_METRICS = [
    "ttft_ms",
    "avg_tbt_ms",
    "tbt_p95_ms",
    "decode_throughput_tps",
    "energy_per_token_j",
    "peak_gpu_memory_mb",
    "reference_rouge_l_f1",
    "baseline_similarity_rouge_l_f1",
    "quality_degradation_vs_baseline",
]

HIGHER_IS_BETTER_METRICS = {
    "tokens_per_second",
    "batched_tokens_per_second",
    "prefill_throughput_tps",
    "decode_throughput_tps",
    "reference_exact_match",
    "reference_rouge_l_f1",
    "reference_token_f1",
    "baseline_similarity_rouge_l_f1",
    "benchmark_primary_metric_value",
    "mmlu_pro_accuracy",
    "gsm8k_exact_match_accuracy",
    "truthfulqa_accuracy",
    "gpqa_accuracy",
    "mlu_accuracy",
    "tam_accuracy",
    "mt_bench_score",
    "alpacaeval2_lc_win_rate",
}

BENCHMARK_DISPLAY_METRICS = [
    ("MMLU-Pro", "mmlu_pro_accuracy"),
    ("GSM8K EM", "gsm8k_exact_match_accuracy"),
    ("TruthfulQA", "truthfulqa_accuracy"),
    ("GPQA", "gpqa_accuracy"),
    ("MLU", "mlu_accuracy"),
    ("TAM", "tam_accuracy"),
    ("MT-Bench", "mt_bench_score"),
    ("AlpacaEval2 LC", "alpacaeval2_lc_win_rate"),
]

QUALITY_METRIC_DISPLAY_NAMES = {
    "reference_rouge_l_f1": "Qual",
    "benchmark_primary_metric_value": "BenchmarkScore",
    **{metric_name: display_name for display_name, metric_name in BENCHMARK_DISPLAY_METRICS},
}

def _get_baseline_mode_name() -> str:
    """Resolve the configured baseline mode name with a safe fallback."""
    return getattr(getattr(CONFIG, "system", None), "baseline_reference_mode_name", "fp16_baseline")

# =============================================================================
# Low-level numeric helpers
# =============================================================================

def _is_valid_number(value) -> bool:
    """Return True only for finite ints/floats."""
    return isinstance(value, (int, float)) and not math.isnan(value) and math.isfinite(value)


def _valid_numbers(values: Iterable) -> List[float]:
    """Filter an arbitrary iterable down to valid numeric values."""
    return [float(v) for v in values if _is_valid_number(v)]


def _safe_mean(values: Iterable) -> Optional[float]:
    """Mean that gracefully returns None for empty/invalid inputs."""
    valid = _valid_numbers(values)
    return (sum(valid) / len(valid)) if valid else None


def _safe_std(values: Iterable) -> Optional[float]:
    """Population standard deviation that returns 0.0 for one valid value."""
    valid = _valid_numbers(values)
    if not valid:
        return None
    if len(valid) == 1:
        return 0.0
    mean = sum(valid) / len(valid)
    return math.sqrt(sum((x - mean) ** 2 for x in valid) / len(valid))


def _safe_percentile(values: Iterable, percentile: float) -> Optional[float]:
    """
    Compute a percentile using linear interpolation.
    """
    valid = sorted(_valid_numbers(values))
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    position = (percentile / 100.0) * (len(valid) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return valid[lower]

    weight = position - lower
    return valid[lower] * (1.0 - weight) + valid[upper] * weight


def _safe_min(values: Iterable) -> Optional[float]:
    """Minimum over valid numeric inputs, else None."""
    valid = _valid_numbers(values)
    return min(valid) if valid else None


def _safe_max(values: Iterable) -> Optional[float]:
    """Maximum over valid numeric inputs, else None."""
    valid = _valid_numbers(values)
    return max(valid) if valid else None


def _metric_stats(values: Iterable) -> Dict[str, Optional[float]]:
    """Return the standard aggregate statistics block for one metric."""
    return {
        "mean": _safe_mean(values),
        "std": _safe_std(values),
        "median": _safe_percentile(values, 50),
        "p95": _safe_percentile(values, 95),
        "min": _safe_min(values),
        "max": _safe_max(values),
    }


def _fmt_number(value, digits: int = 2) -> str:
    """Human-readable formatter for markdown tables and console logs."""
    if not _is_valid_number(value):
        return "—"
    return f"{float(value):.{digits}f}"

def _fmt_ratio(value, digits: int = 2) -> str:
    """Pretty formatter for x-ratios such as speedup/throughput/energy ratios."""
    if not _is_valid_number(value):
        return "—"
    return f"{float(value):.{digits}f}x"

def _fmt_delta(value) -> str:
    """Pretty formatter for delta percentages."""
    if not _is_valid_number(value):
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{float(value):.1f}%"


# =============================================================================
# Result loading
# =============================================================================

def _load_results_from_json_file(path: Path) -> List[dict]:
    """
    Load a benchmark JSON file.

    Expected format: a top-level list of result dictionaries, exactly matching
    the JSON written by benchmark_modes.py.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    raise ValueError(f"Unsupported JSON structure in {path}. Expected a list of dicts.")

def _coerce_csv_value(value: str):
    """
    Best-effort CSV value coercion.

    This makes CSV input usable for numeric aggregation instead of leaving every
    field as a string.
    """
    if value is None:
        return None

    text = value.strip()
    if text == "":
        return None

    lowered = text.lower()
    if lowered in {"none", "null", "nan"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except Exception:
            pass

    try:
        if any(ch in lowered for ch in [".", "e"]):
            return float(text)
        return int(text)
    except Exception:
        pass

    return value


def _load_results_from_csv_file(path: Path) -> List[dict]:
    """
    Load a benchmark CSV file.

    CSV support is included because the project writes both JSON and CSV, but
    JSON is still the preferred source because list-valued fields are preserved.
    """
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({key: _coerce_csv_value(value) for key, value in dict(row).items()})
    return rows


def load_results(input_path: str | Path, merge_all_jsons: bool = False) -> List[dict]:
    """
    Load benchmark results from either:
    - a specific benchmark_results_*.json file,
    - a specific benchmark_results_*.csv file, or
    - a directory containing one or more benchmark_results_*.json files.

    Directory behavior:
    - by default, the newest benchmark_results_*.json file is loaded,
    - if merge_all_jsons=True, all matching JSON files are concatenated.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() == ".json":
            results = _load_results_from_json_file(input_path)
        elif input_path.suffix.lower() == ".csv":
            results = _load_results_from_csv_file(input_path)
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")

        logger.info("Loaded %d result rows from %s", len(results), input_path)
        return results

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    json_candidates = sorted(input_path.glob("benchmark_results_*.json"))
    if not json_candidates:
        raise FileNotFoundError(
            f"No benchmark_results_*.json files were found in directory: {input_path}"
        )

    if merge_all_jsons:
        merged: List[dict] = []
        for json_file in json_candidates:
            merged.extend(_load_results_from_json_file(json_file))
        logger.info(
            "Loaded %d merged result rows from %d JSON files in %s",
            len(merged),
            len(json_candidates),
            input_path,
        )
        return merged

    latest = max(json_candidates, key=lambda p: p.stat().st_mtime)
    results = _load_results_from_json_file(latest)
    logger.info("Loaded %d result rows from latest JSON file %s", len(results), latest)
    return results


# =============================================================================
# Schema adaptation helpers
# =============================================================================

def _infer_workload_cell(row: dict) -> str:
    """
    Infer a workload-cell label from the current project's metadata.

    Rules:
    - repeated-prefix workloads get the special cell "PREFIX"
    - memory-pressure workloads get the special cell "MEM"
    - otherwise we infer S/L from prompt_tokens_target and max_new_tokens
    """
    if row.get("repeated_prefix"):
        return "PREFIX"
    if row.get("memory_pressure"):
        return "MEM"

    prompt_tokens = row.get("prompt_tokens_target")
    max_new_tokens = row.get("max_new_tokens")

    prompt_side = "S" if _is_valid_number(prompt_tokens) and float(prompt_tokens) <= 256 else "L"
    output_side = "S" if _is_valid_number(max_new_tokens) and float(max_new_tokens) <= 64 else "L"
    return prompt_side + output_side


def _infer_system_condition(row: dict) -> str:
    """
    Infer a light-weight system-condition label from current fields.
    """
    if row.get("memory_pressure"):
        return "memory_pressure"

    batch_size = row.get("num_requests_in_batch")
    if _is_valid_number(batch_size) and int(float(batch_size)) > 1:
        return f"batch_{int(float(batch_size))}"

    return "baseline"

def _group_workload_name_from_row(row: dict) -> str:
    """
    Collapse sidecar-expanded benchmark example names back to their parent workload.

    Example:
        mmlu_pro_eval__q0001 -> mmlu_pro_eval
    """
    workload_name = row.get("workload_name")
    if (
        isinstance(workload_name, str)
        and "__" in workload_name
        and (
            row.get("benchmark_example_id") is not None
            or row.get("benchmark_suite") is not None
            or row.get("benchmark_primary_metric_name") is not None
            or row.get("benchmark_primary_metric_value") is not None
        )
    ):
        return workload_name.split("__", 1)[0]
    return workload_name

def _enrich_result_row(row: dict) -> dict:
    """
    Add derived metadata so current results can be reported  without forcing upstream schema changes.
    """
    enriched = dict(row)

    # Derive a workload cell label if it is not already present.
    enriched.setdefault("workload_cell", _infer_workload_cell(row))

    # Derive a simple system-condition label if it is not already present.
    enriched.setdefault("system_condition", _infer_system_condition(row))

    # Derive a workload-group label so benchmark example rows aggregate back
    # to the parent benchmark workload.
    enriched["workload_group_name"] = _group_workload_name_from_row(enriched)

    # Add GB-converted aliases because those are easier to read in reports.
    peak_mb = enriched.get("peak_gpu_memory_mb")
    kv_mb = enriched.get("kv_cache_estimate_mb")
    reserved_mb = enriched.get("reserved_gpu_memory_mb")

    enriched["peak_vram_gb"] = (float(peak_mb) / 1024.0) if _is_valid_number(peak_mb) else None
    enriched["kv_cache_estimated_gb"] = (float(kv_mb) / 1024.0) if _is_valid_number(kv_mb) else None
    enriched["reserved_vram_gb"] = (float(reserved_mb) / 1024.0) if _is_valid_number(reserved_mb) else None

    # Add Ali-style aliases so the report code reads naturally.
    enriched["tbt_mean_ms"] = enriched.get("avg_tbt_ms")
    enriched["total_decode_ms"] = enriched.get("decode_latency_ms")
    enriched["total_inference_ms"] = enriched.get("total_latency_ms")
    enriched["total_energy_j"] = enriched.get("energy_joules")
    enriched["mean_power_w"] = enriched.get("avg_power_w")
    enriched["rougeL_f"] = enriched.get("reference_rouge_l_f1")

    # Normalize success/status semantics.
    success_flag = enriched.get("success")
    if success_flag is None:
        success_flag = enriched.get("error") in (None, "")
    enriched["status"] = "ok" if success_flag else "error"

    return enriched


def prepare_results(results: Sequence[dict]) -> List[dict]:
    """Apply schema enrichment to every raw result row."""
    return [_enrich_result_row(row) for row in results]


# =============================================================================
# Statistical aggregation
# =============================================================================

def aggregate_results(
    results: List[dict],
    group_by: Tuple[str, ...] = ("mode_name", "workload_group_name", "system_condition"),
) -> Dict[tuple, dict]:
    """
    Group raw sample results by the specified keys and compute summary statistics.

    Default grouping is by exact workload name rather than only by workload cell,
    because the current project already uses workload_name as the main benchmark
    key. We still preserve the derived workload_cell as additional metadata.
    """
    prepared = prepare_results(results)

    grouped_rows: Dict[tuple, List[dict]] = defaultdict(list)
    for row in prepared:
        key = tuple(row.get(k, "unknown") for k in group_by)
        grouped_rows[key].append(row)

    aggregated: Dict[tuple, dict] = {}
    for key, group_rows in grouped_rows.items():
        successful_rows = [row for row in group_rows if row.get("status") == "ok"]
        if not successful_rows:
            continue

        first = successful_rows[0]
        failure_count = len(group_rows) - len(successful_rows)
        failure_rate = (failure_count / len(group_rows)) if group_rows else None
        benchmark_primary_metric_name = next(
            (
                row.get("benchmark_primary_metric_name")
                for row in successful_rows
                if row.get("benchmark_primary_metric_name")
            ),
            next(
                (
                    row.get("benchmark_primary_metric_name")
                    for row in group_rows
                    if row.get("benchmark_primary_metric_name")
                ),
                None,
            ),
        )

        aggregate_row = {
            "n": len(successful_rows),
            "num_runs": len(group_rows),
            "group_keys": dict(zip(group_by, key)),
            "mode_name": first.get("mode_name"),
            "workload_name": first.get("workload_group_name") or first.get("workload_name"),
            "workload_cell": first.get("workload_cell"),
            "system_condition": first.get("system_condition"),
            "backend": first.get("backend"),
            "benchmark_suite": first.get("benchmark_suite"),
            "benchmark_subset": first.get("benchmark_subset"),
            "benchmark_language": first.get("benchmark_language"),
            "evaluation_mode": first.get("evaluation_mode"),
            "benchmark_primary_metric_name": benchmark_primary_metric_name,
            "baseline_mode_name": _get_baseline_mode_name(),
            "num_requests_in_batch_mean": _safe_mean(
                row.get("num_requests_in_batch") for row in successful_rows
            ),
            "repeated_prefix": any(bool(row.get("repeated_prefix")) for row in successful_rows),
            "memory_pressure": any(bool(row.get("memory_pressure")) for row in successful_rows),
            "prompt_tokens_target_mean": _safe_mean(
                row.get("prompt_tokens_target") for row in successful_rows
            ),
            "max_new_tokens_mean": _safe_mean(
                row.get("max_new_tokens") for row in successful_rows
            ),
            "failure_count": failure_count,
            "failure_rate": failure_rate,
        }

        for metric in FLOAT_METRICS:
            aggregate_row[metric] = _metric_stats(row.get(metric) for row in successful_rows)

        aggregated[key] = aggregate_row

    return aggregated


# =============================================================================
# Failure summary
# =============================================================================

def build_failure_summary(
    results: List[dict],
    group_by: Tuple[str, ...] = ("mode_name", "workload_group_name", "system_condition"),
) -> Dict[tuple, dict]:
    """
    Summarize failures separately so we do not lose reliability information.
    """
    prepared = prepare_results(results)

    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in prepared:
        key = tuple(row.get(k, "unknown") for k in group_by)
        groups[key].append(row)

    summary: Dict[tuple, dict] = {}
    for key, group_rows in groups.items():
        failures = [row for row in group_rows if row.get("status") != "ok"]
        error_counter = defaultdict(int)
        for row in failures:
            error_type = row.get("error_type") or row.get("error") or "unknown"
            error_counter[str(error_type)] += 1

        most_common_error = None
        most_common_count = 0
        for error_type, count in error_counter.items():
            if count > most_common_count:
                most_common_error = error_type
                most_common_count = count

        first = group_rows[0]
        summary[key] = {
            "mode_name": first.get("mode_name"),
            "workload_name": first.get("workload_group_name") or first.get("workload_name"),
            "workload_cell": first.get("workload_cell"),
            "system_condition": first.get("system_condition"),
            "num_runs": len(group_rows),
            "num_failures": len(failures),
            "failure_rate": (len(failures) / len(group_rows)) if group_rows else None,
            "most_common_error": most_common_error,
        }

    return summary


# =============================================================================
# Delta vs baseline
# =============================================================================

def compute_delta_table(
    aggregated: Dict[tuple, dict],
    baseline_mode: Optional[str] = None,
    metrics_to_compare: Optional[List[str]] = None,
) -> Dict[tuple, dict]:
    """
    Compute relative change (%) of each mode vs the fp16_baseline for the same
    workload_name + system_condition pair.

    Positive delta means the metric numerically increased. Interpretation depends
    on the metric: for latency/energy/memory that is usually worse, but for
    throughput it is usually better.
    """
    baseline_mode = baseline_mode or _get_baseline_mode_name()

    if metrics_to_compare is None:
        metrics_to_compare = list(DEFAULT_DELTA_METRICS)

    # Build a lookup of baseline aggregates keyed by workload + condition.
    baseline_lookup = {}
    for key, agg in aggregated.items():
        mode = agg.get("mode_name")
        workload_name = agg.get("workload_name")
        condition = agg.get("system_condition")
        if mode == baseline_mode:
            baseline_lookup[(workload_name, condition)] = agg

    deltas: Dict[tuple, dict] = {}
    for key, agg in aggregated.items():
        mode = agg.get("mode_name")
        workload_name = agg.get("workload_name")
        condition = agg.get("system_condition")
        if mode == baseline_mode:
            continue

        baseline = baseline_lookup.get((workload_name, condition))
        if baseline is None:
            continue

        delta_row = {
            "mode_name": mode,
            "workload_name": workload_name,
            "workload_cell": agg.get("workload_cell"),
            "system_condition": condition,
            "latency_speedup_vs_baseline": None,
            "throughput_ratio_vs_baseline": None,
            "energy_ratio_vs_baseline": None,
        }

        for metric in metrics_to_compare:
            mode_value = (agg.get(metric) or {}).get("mean")
            baseline_value = (baseline.get(metric) or {}).get("mean")

            if _is_valid_number(mode_value) and _is_valid_number(baseline_value) and float(baseline_value) != 0.0:
                delta_row[metric] = round((float(mode_value) - float(baseline_value)) / abs(float(baseline_value)) * 100.0, 2)
            else:
                delta_row[metric] = None

        baseline_latency = (baseline.get("total_latency_ms") or {}).get("mean")
        mode_latency = (agg.get("total_latency_ms") or {}).get("mean")
        if _is_valid_number(baseline_latency) and _is_valid_number(mode_latency) and float(mode_latency) != 0.0:
            delta_row["latency_speedup_vs_baseline"] = round(float(baseline_latency) / float(mode_latency), 4)

        baseline_tps = (baseline.get("tokens_per_second") or {}).get("mean")
        mode_tps = (agg.get("tokens_per_second") or {}).get("mean")
        if _is_valid_number(baseline_tps) and _is_valid_number(mode_tps) and float(baseline_tps) != 0.0:
            delta_row["throughput_ratio_vs_baseline"] = round(float(mode_tps) / float(baseline_tps), 4)

        baseline_energy = (baseline.get("energy_per_token_j") or {}).get("mean")
        mode_energy = (agg.get("energy_per_token_j") or {}).get("mean")
        if _is_valid_number(baseline_energy) and _is_valid_number(mode_energy) and float(baseline_energy) != 0.0:
            delta_row["energy_ratio_vs_baseline"] = round(float(mode_energy) / float(baseline_energy), 4)

        deltas[key] = delta_row

    return deltas


# =============================================================================
# Phase dominance analysis
# =============================================================================

def compute_phase_dominance(aggregated: Dict[tuple, dict]) -> Dict[tuple, dict]:
    """
    For each group, compute how much of total latency is spent in prefill vs
    decode.
    """
    dominance: Dict[tuple, dict] = {}
    for key, agg in aggregated.items():
        prefill_ms = (agg.get("prefill_latency_ms") or {}).get("mean")
        decode_ms = (agg.get("decode_latency_ms") or {}).get("mean")

        prefill_ms = float(prefill_ms) if _is_valid_number(prefill_ms) else 0.0
        decode_ms = float(decode_ms) if _is_valid_number(decode_ms) else 0.0
        total_ms = prefill_ms + decode_ms
        if total_ms <= 0:
            continue

        prefill_pct = (prefill_ms / total_ms) * 100.0
        decode_pct = (decode_ms / total_ms) * 100.0

        if prefill_pct > 60.0:
            dominated_by = "prefill"
        elif decode_pct > 60.0:
            dominated_by = "decode"
        else:
            dominated_by = "balanced"

        dominance[key] = {
            "mode_name": agg.get("mode_name"),
            "workload_name": agg.get("workload_name"),
            "workload_cell": agg.get("workload_cell"),
            "system_condition": agg.get("system_condition"),
            "prefill_ms": round(prefill_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "total_ms": round(total_ms, 3),
            "prefill_pct": round(prefill_pct, 1),
            "decode_pct": round(decode_pct, 1),
            "dominated_by": dominated_by,
        }

    return dominance


# =============================================================================
# Pareto frontier
# =============================================================================

def _resolve_quality_metric_name(
    aggregated: Dict[tuple, dict],
    preferred: str = "auto",
) -> str:
    """
    Choose a usable quality metric.

    Priority:
    - explicit user choice if it exists
    - reference_rouge_l_f1 if present anywhere
    - baseline_similarity_rouge_l_f1 if present anywhere
    - quality_degradation_vs_baseline if present anywhere (inverted logic later)
    """
    if preferred != "auto":
        return preferred

    candidates = [
        "benchmark_primary_metric_value",
        "mmlu_pro_accuracy",
        "gsm8k_exact_match_accuracy",
        "truthfulqa_accuracy",
        "gpqa_accuracy",
        "mlu_accuracy",
        "tam_accuracy",
        "mt_bench_score",
        "alpacaeval2_lc_win_rate",
        "reference_rouge_l_f1",
        "baseline_similarity_rouge_l_f1",
        "quality_degradation_vs_baseline",
    ]

    for metric in candidates:
        for agg in aggregated.values():
            if _is_valid_number((agg.get(metric) or {}).get("mean")):
                return metric

    return "reference_rouge_l_f1"


def find_pareto_frontier(
    aggregated: Dict[tuple, dict],
    obj1_metric: str = "energy_per_token_j",
    obj2_metric: str = "reference_rouge_l_f1",
    obj1_minimize: bool = True,
    obj2_minimize: bool = False,
    workload_filter: Optional[str] = None,
) -> List[tuple]:
    """
    Identify Pareto-optimal group keys on two objectives.

    A point is dominated if another point is at least as good on both objectives
    and strictly better on at least one.
    """
    points: List[Tuple[tuple, float, float]] = []

    for key, agg in aggregated.items():
        workload_name = agg.get("workload_name")
        if workload_filter and workload_name != workload_filter:
            continue

        v1 = (agg.get(obj1_metric) or {}).get("mean")
        v2 = (agg.get(obj2_metric) or {}).get("mean")
        if not _is_valid_number(v1) or not _is_valid_number(v2):
            continue

        # Normalize to a minimization problem for the dominance test.
        x = float(v1) if obj1_minimize else -float(v1)
        y = float(v2) if obj2_minimize else -float(v2)
        points.append((key, x, y))

    frontier: List[tuple] = []
    for i, (key_i, x_i, y_i) in enumerate(points):
        dominated = False
        for j, (key_j, x_j, y_j) in enumerate(points):
            if i == j:
                continue
            if x_j <= x_i and y_j <= y_i and (x_j < x_i or y_j < y_i):
                dominated = True
                break
        if not dominated:
            frontier.append(key_i)

    return frontier


# =============================================================================
# Optional CSV / JSON writers
# =============================================================================

def _write_json(data, output_path: Path) -> None:
    """Write JSON with parent directory creation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _write_csv_rows(rows: List[dict], output_path: Path) -> None:
    """Write a list of row dictionaries to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_aggregated_results(aggregated: Dict[tuple, dict]) -> List[dict]:
    """Convert nested aggregate dictionaries into flat CSV-friendly rows."""
    rows: List[dict] = []
    for _, agg in aggregated.items():
        row = {
            "mode_name": agg.get("mode_name"),
            "workload_name": agg.get("workload_name"),
            "workload_cell": agg.get("workload_cell"),
            "system_condition": agg.get("system_condition"),
            "backend": agg.get("backend"),
            "benchmark_suite": agg.get("benchmark_suite"),
            "benchmark_subset": agg.get("benchmark_subset"),
            "benchmark_language": agg.get("benchmark_language"),
            "evaluation_mode": agg.get("evaluation_mode"),
            "benchmark_primary_metric_name": agg.get("benchmark_primary_metric_name"),
            "n": agg.get("n"),
            "failure_count": agg.get("failure_count"),
            "failure_rate": agg.get("failure_rate"),
            "num_requests_in_batch_mean": agg.get("num_requests_in_batch_mean"),
            "prompt_tokens_target_mean": agg.get("prompt_tokens_target_mean"),
            "max_new_tokens_mean": agg.get("max_new_tokens_mean"),
            "repeated_prefix": agg.get("repeated_prefix"),
            "memory_pressure": agg.get("memory_pressure"),
        }

        for metric in FLOAT_METRICS:
            stats = agg.get(metric, {})
            row[f"{metric}_mean"] = stats.get("mean")
            row[f"{metric}_std"] = stats.get("std")
            row[f"{metric}_median"] = stats.get("median")
            row[f"{metric}_p95"] = stats.get("p95")
            row[f"{metric}_min"] = stats.get("min")
            row[f"{metric}_max"] = stats.get("max")

        rows.append(row)

    rows.sort(key=lambda r: (r["workload_name"], r["mode_name"], r["system_condition"]))
    return rows


def flatten_delta_table(deltas: Dict[tuple, dict]) -> List[dict]:
    """Convert the delta table dictionary to a sorted list of rows."""
    rows = list(deltas.values())
    rows.sort(key=lambda r: (r.get("workload_name"), r.get("mode_name"), r.get("system_condition")))
    return rows


def flatten_phase_dominance(phase_dominance: Dict[tuple, dict]) -> List[dict]:
    """Convert the phase-dominance dictionary to a sorted list of rows."""
    rows = list(phase_dominance.values())
    rows.sort(key=lambda r: (r.get("workload_name"), r.get("mode_name"), r.get("system_condition")))
    return rows


def flatten_failure_summary(failure_summary: Dict[tuple, dict]) -> List[dict]:
    """Convert the failure summary dictionary to a sorted list of rows."""
    rows = list(failure_summary.values())
    rows.sort(key=lambda r: (r.get("workload_name"), r.get("mode_name"), r.get("system_condition")))
    return rows


def build_winner_rows(
    aggregated: Dict[tuple, dict],
    quality_metric_name: str,
) -> List[dict]:
    """
    Build a compact "winner table" so the best mode per workload/condition can
    be skimmed quickly
    """
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for agg in aggregated.values():
        grouped[(agg.get("workload_name"), agg.get("system_condition"))].append(agg)

    quality_minimize = quality_metric_name not in HIGHER_IS_BETTER_METRICS
    rows: List[dict] = []

    def _pick(entries: List[dict], metric_name: str, minimize: bool):
        valid_entries = [
            entry for entry in entries
            if _is_valid_number((entry.get(metric_name) or {}).get("mean"))
        ]
        if not valid_entries:
            return None
        selector = min if minimize else max
        return selector(valid_entries, key=lambda entry: (entry.get(metric_name) or {}).get("mean"))

    for (workload_name, system_condition), entries in grouped.items():
        best_latency = _pick(entries, "total_latency_ms", True)
        best_energy = _pick(entries, "energy_per_token_j", True)
        best_memory = _pick(entries, "peak_gpu_memory_mb", True)
        best_quality = _pick(entries, quality_metric_name, quality_minimize)

        rows.append({
            "workload_name": workload_name,
            "workload_cell": entries[0].get("workload_cell"),
            "system_condition": system_condition,
            "best_latency_mode": best_latency.get("mode_name") if best_latency else None,
            "best_latency_value": (best_latency.get("total_latency_ms") or {}).get("mean") if best_latency else None,
            "best_energy_mode": best_energy.get("mode_name") if best_energy else None,
            "best_energy_value": (best_energy.get("energy_per_token_j") or {}).get("mean") if best_energy else None,
            "best_memory_mode": best_memory.get("mode_name") if best_memory else None,
            "best_memory_value": (best_memory.get("peak_gpu_memory_mb") or {}).get("mean") if best_memory else None,
            "best_quality_mode": best_quality.get("mode_name") if best_quality else None,
            "best_quality_value": (best_quality.get(quality_metric_name) or {}).get("mean") if best_quality else None,
        })

    rows.sort(key=lambda row: (row.get("workload_name"), row.get("system_condition")))
    return rows


# =============================================================================
# Markdown report generation
# =============================================================================

def generate_markdown_report(
    aggregated: Dict[tuple, dict],
    delta_table: Dict[tuple, dict],
    phase_dominance: Dict[tuple, dict],
    failure_summary: Dict[tuple, dict],
    run_id: str,
    output_path: str | Path,
    baseline_mode: Optional[str] = None,
    quality_metric_name: str = "reference_rouge_l_f1",
) -> None:
    """
    Write a markdown report adapted to the current benchmark schema.
    """
    baseline_mode = baseline_mode or _get_baseline_mode_name()
    winner_rows = build_winner_rows(aggregated, quality_metric_name)
    lines: List[str] = []
    add_line = lines.append

    quality_metric_display_name = QUALITY_METRIC_DISPLAY_NAMES.get(
        quality_metric_name,
        quality_metric_name,
    )

    add_line("# ModeSwitch-LLM Benchmark Report")
    add_line("")
    add_line(f"**Run ID:** `{run_id}`")
    add_line("---")
    add_line("")

    # ---------------------------------------------------------------------
    # Section 1: aggregate metrics
    # ---------------------------------------------------------------------
    add_line("## 1. Aggregate Metrics by Mode and Workload")
    add_line("")
    add_line(
        "Latency is in milliseconds, throughput in tokens/second, GPU memory in MB, "
        "and energy in joules/token. `Cell` is inferred from the current workload metadata."
    )
    add_line("")


    headers = [
        "Mode",
        "Workload",
        "Cell",
        "Cond",
        "n",
        "TTFT",
        "TBT_mean",
        "TBT_p95",
        "Prefill_tps",
        "Decode_tps",
        "Total_tps",
        "GPU_MB",
        "E/tok",
        quality_metric_display_name,
    ]
    add_line("| " + " | ".join(headers) + " |")
    add_line("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, agg in sorted(aggregated.items(), key=lambda item: (
        item[1].get("workload_name"), item[1].get("mode_name"), item[1].get("system_condition")
    )):
        qual_value = (agg.get(quality_metric_name) or {}).get("mean")
        row = [
            str(agg.get("mode_name")),
            str(agg.get("workload_name")),
            str(agg.get("workload_cell")),
            str(agg.get("system_condition")),
            str(agg.get("n")),
            _fmt_number((agg.get("ttft_ms") or {}).get("mean"), 1),
            _fmt_number((agg.get("avg_tbt_ms") or {}).get("mean"), 1),
            _fmt_number((agg.get("tbt_p95_ms") or {}).get("mean"), 1),
            _fmt_number((agg.get("prefill_throughput_tps") or {}).get("mean"), 1),
            _fmt_number((agg.get("decode_throughput_tps") or {}).get("mean"), 1),
            _fmt_number((agg.get("tokens_per_second") or {}).get("mean"), 1),
            _fmt_number((agg.get("peak_gpu_memory_mb") or {}).get("mean"), 1),
            _fmt_number((agg.get("energy_per_token_j") or {}).get("mean"), 4),
            _fmt_number(qual_value, 4),
        ]
        add_line("| " + " | ".join(row) + " |")

    add_line("")

    # -------------------------------------------------------------------------
    # Benchmark accuracy / chat quality section
    # -------------------------------------------------------------------------
    benchmark_rows = []
    for (mode_name, workload_name, system_condition), row in sorted(aggregated.items()):
        rendered_values = []
        any_present = False
        for _, metric_name in BENCHMARK_DISPLAY_METRICS:
            metric_stats = row.get(metric_name, {})
            metric_mean = metric_stats.get("mean") if isinstance(metric_stats, dict) else None
            if metric_mean is None:
                rendered_values.append("—")
            else:
                any_present = True
                rendered_values.append(f"{metric_mean:.4f}")

        if any_present:
            benchmark_rows.append(
                (
                    mode_name,
                    workload_name,
                    row.get("workload_cell", "—"),
                    system_condition,
                    rendered_values,
                )
            )

    if benchmark_rows:
        lines.append("## 1b. Benchmark Accuracy / Chat Quality")
        lines.append("")
        header = "| Mode | Workload | Cell | Cond | " + " | ".join(name for name, _ in BENCHMARK_DISPLAY_METRICS) + " |"
        sep = "| --- | --- | --- | --- | " + " | ".join(["---"] * len(BENCHMARK_DISPLAY_METRICS)) + " |"
        lines.append(header)
        lines.append(sep)
        for mode_name, workload_name, workload_cell, system_condition, rendered_values in benchmark_rows:
            lines.append(
                f"| {mode_name} | {workload_name} | {workload_cell} | {system_condition} | "
                + " | ".join(rendered_values)
                + " |"
            )
        lines.append("")

    # ---------------------------------------------------------------------
    # Section 2: deltas vs baseline
    # ---------------------------------------------------------------------
    add_line(f"## 2. Relative Change vs `{baseline_mode}`")
    add_line("")
    add_line(
        f"Each delta compares a mode against `{baseline_mode}` on the **same workload** and "
        "**same inferred system condition**. Positive values mean the metric numerically "
        "increased. For latency/energy/memory this is usually worse; for throughput it is usually better."
    )
    add_line("")

    delta_headers = [
        "Mode",
        "Workload",
        "Cell",
        "Cond",
        "ΔTTFT",
        "ΔTBT_mean",
        "ΔTBT_p95",
        "ΔDecode_tps",
        "Lat_x",
        "TPS_x",
        "Energy_x",
        "ΔE/tok",
        "ΔGPU_MB",
        f"Δ{quality_metric_display_name}",
    ]
    add_line("| " + " | ".join(delta_headers) + " |")
    add_line("| " + " | ".join(["---"] * len(delta_headers)) + " |")

    for _, row in sorted(delta_table.items(), key=lambda item: (
        item[1].get("workload_name"), item[1].get("mode_name"), item[1].get("system_condition")
    )):
        values = [
            str(row.get("mode_name")),
            str(row.get("workload_name")),
            str(row.get("workload_cell")),
            str(row.get("system_condition")),
            _fmt_delta(row.get("ttft_ms")),
            _fmt_delta(row.get("avg_tbt_ms")),
            _fmt_delta(row.get("tbt_p95_ms")),
            _fmt_delta(row.get("decode_throughput_tps")),
            _fmt_ratio(row.get("latency_speedup_vs_baseline")),
            _fmt_ratio(row.get("throughput_ratio_vs_baseline")),
            _fmt_ratio(row.get("energy_ratio_vs_baseline")),
            _fmt_delta(row.get("energy_per_token_j")),
            _fmt_delta(row.get("peak_gpu_memory_mb")),
            _fmt_delta(row.get(quality_metric_name)),
        ]
        add_line("| " + " | ".join(values) + " |")

    add_line("")

    # ---------------------------------------------------------------------
    # Section 3: phase dominance
    # ---------------------------------------------------------------------
    add_line("## 3. Phase Dominance Analysis")
    add_line("")
    add_line("")

    phase_headers = [
        "Mode",
        "Workload",
        "Cell",
        "Cond",
        "Prefill_ms",
        "Decode_ms",
        "Pre%",
        "Dec%",
        "Dominated_by",
    ]
    add_line("| " + " | ".join(phase_headers) + " |")
    add_line("| " + " | ".join(["---"] * len(phase_headers)) + " |")

    for _, row in sorted(phase_dominance.items(), key=lambda item: (
        item[1].get("workload_name"), item[1].get("mode_name"), item[1].get("system_condition")
    )):
        values = [
            str(row.get("mode_name")),
            str(row.get("workload_name")),
            str(row.get("workload_cell")),
            str(row.get("system_condition")),
            _fmt_number(row.get("prefill_ms"), 1),
            _fmt_number(row.get("decode_ms"), 1),
            _fmt_number(row.get("prefill_pct"), 1) + "%" if _is_valid_number(row.get("prefill_pct")) else "—",
            _fmt_number(row.get("decode_pct"), 1) + "%" if _is_valid_number(row.get("decode_pct")) else "—",
            f"**{str(row.get('dominated_by', 'unknown')).upper()}**",
        ]
        add_line("| " + " | ".join(values) + " |")

    add_line("")

    # ---------------------------------------------------------------------
    # Section 4: failures / reliability
    # ---------------------------------------------------------------------
    add_line("## 4. Failure / Reliability Summary")
    add_line("")
    add_line(
        "Aggregate statistics only use successful runs, but the table below keeps failed runs visible so "
        "reliability trade-offs do not disappear from the final analysis."
    )
    add_line("")

    fail_headers = ["Mode", "Workload", "Cell", "Cond", "Runs", "Failures", "Failure_rate", "Most_common_error"]
    add_line("| " + " | ".join(fail_headers) + " |")
    add_line("| " + " | ".join(["---"] * len(fail_headers)) + " |")

    for _, row in sorted(failure_summary.items(), key=lambda item: (
        item[1].get("workload_name"), item[1].get("mode_name"), item[1].get("system_condition")
    )):
        values = [
            str(row.get("mode_name")),
            str(row.get("workload_name")),
            str(row.get("workload_cell")),
            str(row.get("system_condition")),
            str(row.get("num_runs")),
            str(row.get("num_failures")),
            _fmt_number(row.get("failure_rate"), 3),
            str(row.get("most_common_error") or "—"),
        ]
        add_line("| " + " | ".join(values) + " |")

    add_line("")

    # ---------------------------------------------------------------------
    # Section 5: quick winners
    # ---------------------------------------------------------------------
    add_line("## 5. Quick Winners by Workload")
    add_line("")
    add_line(
        "This compact table highlights the easiest takeaways: which mode wins on "
        "latency, energy, memory, and quality for each workload/condition."
    )
    add_line("")

    winner_headers = [
        "Workload",
        "Cell",
        "Cond",
        "Best_latency",
        "Best_energy",
        "Best_memory",
        "Best_quality",
    ]
    add_line("| " + " | ".join(winner_headers) + " |")
    add_line("| " + " | ".join(["---"] * len(winner_headers)) + " |")

    for row in winner_rows:
        values = [
            str(row.get("workload_name")),
            str(row.get("workload_cell")),
            str(row.get("system_condition")),
            f"{row.get('best_latency_mode')} ({_fmt_number(row.get('best_latency_value'), 1)} ms)" if row.get("best_latency_mode") else "—",
            f"{row.get('best_energy_mode')} ({_fmt_number(row.get('best_energy_value'), 4)} J/tok)" if row.get("best_energy_mode") else "—",
            f"{row.get('best_memory_mode')} ({_fmt_number(row.get('best_memory_value'), 1)} MB)" if row.get("best_memory_mode") else "—",
            f"{row.get('best_quality_mode')} ({_fmt_number(row.get('best_quality_value'), 4)})" if row.get("best_quality_mode") else "—",
        ]
        add_line("| " + " | ".join(values) + " |")

    add_line("")

    # ---------------------------------------------------------------------
    # Section 6: observations
    # ---------------------------------------------------------------------
    add_line("## 6. Key Observations")
    add_line("")
    _generate_observations(
        lines,
        aggregated,
        phase_dominance,
        baseline_mode,
        quality_metric_name,
    )
    add_line("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Markdown report saved to %s", output_path)


def _generate_observations(
    lines: List[str],
    aggregated: Dict[tuple, dict],
    phase_dominance: Dict[tuple, dict],
    baseline_mode: str,
    quality_metric_name: str,
) -> None:
    """Auto-generate plain-English observations from aggregate results."""
    baseline_mode = baseline_mode or _get_baseline_mode_name()
    observations: List[str] = []

    # -------------------------------------------------------------
    # Best-energy mode per workload in baseline condition.
    # -------------------------------------------------------------
    by_workload = defaultdict(list)
    for agg in aggregated.values():
        if agg.get("system_condition") != "baseline":
            continue
        energy = (agg.get("energy_per_token_j") or {}).get("mean")
        if _is_valid_number(energy):
            by_workload[agg.get("workload_name")].append(agg)

    for workload_name, entries in by_workload.items():
        best_entry = min(entries, key=lambda a: (a.get("energy_per_token_j") or {}).get("mean"))
        baseline_entry = next((a for a in entries if a.get("mode_name") == baseline_mode), None)
        best_energy = (best_entry.get("energy_per_token_j") or {}).get("mean")
        baseline_energy = (baseline_entry.get("energy_per_token_j") or {}).get("mean") if baseline_entry else None
        if _is_valid_number(best_energy) and _is_valid_number(baseline_energy) and float(baseline_energy) > 0:
            reduction_pct = (float(baseline_energy) - float(best_energy)) / float(baseline_energy) * 100.0
            observations.append(
                f"- **{workload_name}**: the most energy-efficient mode is `{best_entry.get('mode_name')}` "
                f"({_fmt_number(best_energy, 4)} J/token), which is a {_fmt_number(reduction_pct, 1)}% "
                f"reduction versus `{baseline_mode}`."
            )

    # -------------------------------------------------------------
    # Baseline phase dominance per workload.
    # -------------------------------------------------------------
    baseline_phase_rows = [
        row for row in phase_dominance.values()
        if row.get("mode_name") == baseline_mode and row.get("system_condition") == "baseline"
    ]
    decode_dominated = [row.get("workload_name") for row in baseline_phase_rows if row.get("dominated_by") == "decode"]
    prefill_dominated = [row.get("workload_name") for row in baseline_phase_rows if row.get("dominated_by") == "prefill"]

    if decode_dominated:
        observations.append(
            "- Decode-dominated baseline workloads: **" + ", ".join(sorted(set(decode_dominated))) +
            "**. These are the workloads where decode-focused ideas such as quantization, speculative decoding, "
            "and KV-cache optimizations should matter most."
        )

    if prefill_dominated:
        observations.append(
            "- Prefill-dominated baseline workloads: **" + ", ".join(sorted(set(prefill_dominated))) +
            "**. These are the workloads where prefill-side strategies such as prefix reuse, chunked prefill, "
            "and graph/runtime setup choices matter most."
        )

    # -------------------------------------------------------------
    # Best-quality mode per workload, if quality exists.
    # -------------------------------------------------------------
    qual_by_workload = defaultdict(list)
    for agg in aggregated.values():
        quality = (agg.get(quality_metric_name) or {}).get("mean")
        if _is_valid_number(quality):
            qual_by_workload[agg.get("workload_name")].append(agg)

    for workload_name, entries in qual_by_workload.items():
        quality_minimize = quality_metric_name not in HIGHER_IS_BETTER_METRICS
        if quality_minimize:
            best_entry = min(entries, key=lambda a: (a.get(quality_metric_name) or {}).get("mean"))
            observations.append(
                f"- **{workload_name}**: the lowest `{quality_metric_name}` is achieved by `{best_entry.get('mode_name')}` "
                f"({_fmt_number((best_entry.get(quality_metric_name) or {}).get('mean'), 4)})."
            )
        else:
            best_entry = max(entries, key=lambda a: (a.get(quality_metric_name) or {}).get("mean"))
            observations.append(
                f"- **{workload_name}**: the strongest `{quality_metric_name}` is achieved by `{best_entry.get('mode_name')}` "
                f"({_fmt_number((best_entry.get(quality_metric_name) or {}).get('mean'), 4)})."
            )

    if not observations:
        observations.append("- Insufficient data was available to generate automatic observations.")

    lines.extend(observations)


# =============================================================================
# Plot generation
# =============================================================================

def plot_ttft_vs_tbt_scatter(
    aggregated: Dict[tuple, dict],
    output_path: str | Path,
    condition_filter: str = "baseline",
) -> None:
    """Scatter plot of TTFT vs mean TBT for baseline condition rows."""
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib unavailable; skipping TTFT/TBT scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    used_labels = set()

    for agg in aggregated.values():
        if agg.get("system_condition") != condition_filter:
            continue

        ttft = (agg.get("ttft_ms") or {}).get("mean")
        tbt = (agg.get("avg_tbt_ms") or {}).get("mean")
        if not _is_valid_number(ttft) or not _is_valid_number(tbt):
            continue

        label = agg.get("mode_name")
        shown_label = label if label not in used_labels else "_nolegend_"
        used_labels.add(label)

        ax.scatter(float(ttft), float(tbt), s=90, label=shown_label, alpha=0.85)
        ax.annotate(
            str(agg.get("workload_name")),
            (float(ttft), float(tbt)),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
        )

    ax.set_xlabel("TTFT (ms)")
    ax.set_ylabel("Mean TBT (ms)")
    ax.set_title(f"TTFT vs Mean TBT ({condition_filter})")
    ax.grid(True, alpha=0.3)
    if used_labels:
        ax.legend(title="Mode", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved TTFT/TBT scatter plot to %s", output_path)


def plot_energy_per_token_bar(
    aggregated: Dict[tuple, dict],
    output_path: str | Path,
    condition_filter: str = "baseline",
) -> None:
    """Bar chart of energy/token by workload, grouped by mode."""
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib unavailable; skipping energy/token bar chart.")
        return

    workload_names = sorted({agg.get("workload_name") for agg in aggregated.values() if agg.get("system_condition") == condition_filter})
    mode_names = sorted({agg.get("mode_name") for agg in aggregated.values() if agg.get("system_condition") == condition_filter})

    if not workload_names or not mode_names:
        logger.info("No data for energy/token plot.")
        return

    x = np.arange(len(workload_names))
    width = 0.8 / max(len(mode_names), 1)

    num_workloads = len(workload_names)
    fig_width = min(24, max(10, num_workloads * 0.9))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for i, mode_name in enumerate(mode_names):
        values = []
        for workload_name in workload_names:
            matching = [
                agg for agg in aggregated.values()
                if agg.get("mode_name") == mode_name
                and agg.get("workload_name") == workload_name
                and agg.get("system_condition") == condition_filter
            ]
            if matching:
                value = (matching[0].get("energy_per_token_j") or {}).get("mean")
            else:
                value = None
            values.append(float(value) if _is_valid_number(value) else 0.0)

        offset = (i - len(mode_names) / 2 + 0.5) * width
        ax.bar(x + offset, values, width * 0.92, label=mode_name, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(workload_names, rotation=20, ha="right")
    ax.set_xlabel("Workload")
    ax.set_ylabel("Energy per token (J/token)")
    ax.set_title(f"Energy per Token by Mode ({condition_filter})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Mode", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved energy/token bar chart to %s", output_path)


def plot_memory_heatmap(
    aggregated: Dict[tuple, dict],
    output_path: str | Path,
    condition_filter: str = "baseline",
) -> None:
    """Heatmap of mean peak GPU memory (MB): modes × workloads."""
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib unavailable; skipping memory heatmap.")
        return

    workload_names = sorted({agg.get("workload_name") for agg in aggregated.values() if agg.get("system_condition") == condition_filter})
    mode_names = sorted({agg.get("mode_name") for agg in aggregated.values() if agg.get("system_condition") == condition_filter})

    if not workload_names or not mode_names:
        logger.info("No data for memory heatmap.")
        return

    matrix = []
    for mode_name in mode_names:
        row_values = []
        for workload_name in workload_names:
            matching = [
                agg for agg in aggregated.values()
                if agg.get("mode_name") == mode_name
                and agg.get("workload_name") == workload_name
                and agg.get("system_condition") == condition_filter
            ]
            if matching:
                value = (matching[0].get("peak_gpu_memory_mb") or {}).get("mean")
                row_values.append(float(value) if _is_valid_number(value) else float("nan"))
            else:
                row_values.append(float("nan"))
        matrix.append(row_values)

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, len(workload_names) * 1.3), max(5, len(mode_names) * 0.8)))
    image = ax.imshow(arr, aspect="auto")
    plt.colorbar(image, ax=ax, label="Peak GPU memory (MB)")

    ax.set_xticks(range(len(workload_names)))
    ax.set_xticklabels(workload_names, rotation=20, ha="right")
    ax.set_yticks(range(len(mode_names)))
    ax.set_yticklabels(mode_names)
    ax.set_title(f"Peak GPU Memory Heatmap ({condition_filter})")

    for row_index in range(len(mode_names)):
        for col_index in range(len(workload_names)):
            value = arr[row_index, col_index]
            if not math.isnan(value):
                ax.text(col_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved memory heatmap to %s", output_path)


def plot_phase_dominance_bar(
    phase_dominance: Dict[tuple, dict],
    output_path: str | Path,
    condition_filter: str = "baseline",
) -> None:
    """
    Stacked bar chart showing prefill% vs decode% for each mode/workload row.
    """
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib unavailable; skipping phase-dominance plot.")
        return

    rows = [
        row for row in phase_dominance.values()
        if row.get("system_condition") == condition_filter
    ]
    rows.sort(key=lambda r: (r.get("workload_name"), r.get("mode_name")))

    if not rows:
        logger.info("No rows for phase-dominance plot.")
        return

    labels = [f"{row.get('workload_name')}\n{row.get('mode_name')}" for row in rows]
    prefill = [float(row.get("prefill_pct")) if _is_valid_number(row.get("prefill_pct")) else 0.0 for row in rows]
    decode = [float(row.get("decode_pct")) if _is_valid_number(row.get("decode_pct")) else 0.0 for row in rows]

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.9), 6))
    ax.bar(x, prefill, label="Prefill %", alpha=0.9)
    ax.bar(x, decode, bottom=prefill, label="Decode %", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Share of total latency (%)")
    ax.set_title(f"Phase Dominance by Mode and Workload ({condition_filter})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved phase-dominance bar chart to %s", output_path)


# =============================================================================
# End-to-end report generation
# =============================================================================

def generate_full_report(
    input_path: str | Path,
    output_dir: Optional[str | Path] = None,
    merge_all_jsons: bool = False,
    baseline_mode: Optional[str] = None,
    quality_metric: str = "auto",
) -> Path:
    """
    Load results, compute all analyses, and write a full report bundle.

    Output files include:
    - prepared_results.json
    - aggregated.json / aggregated.csv
    - deltas.json / deltas.csv
    - phase_dominance.json / phase_dominance.csv
    - failure_summary.json / failure_summary.csv
    - pareto.json
    - report.md
    - optional plots if matplotlib is available
    """
    input_path = Path(input_path)
    baseline_mode = baseline_mode or _get_baseline_mode_name()
    raw_results = load_results(input_path, merge_all_jsons=merge_all_jsons)
    prepared_results = prepare_results(raw_results)

    aggregated = aggregate_results(prepared_results)
    failure_summary = build_failure_summary(prepared_results)
    quality_metric_name = _resolve_quality_metric_name(aggregated, preferred=quality_metric)
    delta_metrics = list(DEFAULT_DELTA_METRICS)
    if quality_metric_name not in delta_metrics:
        delta_metrics.append(quality_metric_name)
    delta_table = compute_delta_table(
        aggregated,
        baseline_mode=baseline_mode,
        metrics_to_compare=delta_metrics,
    )
    phase_dominance = compute_phase_dominance(aggregated)

    pareto = find_pareto_frontier(
        aggregated,
        obj1_metric="energy_per_token_j",
        obj2_metric=quality_metric_name,
        obj1_minimize=True,
        obj2_minimize=(quality_metric_name not in HIGHER_IS_BETTER_METRICS),
    )

    if output_dir is None:
        if input_path.is_file():
            report_dir = input_path.with_name(f"{input_path.stem}_report")
        else:
            report_dir = input_path / "report"
    else:
        report_dir = Path(output_dir)

    report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save machine-readable outputs.
    # ------------------------------------------------------------------
    _write_json(prepared_results, report_dir / "prepared_results.json")
    _write_json({str(k): v for k, v in aggregated.items()}, report_dir / "aggregated.json")
    _write_json({str(k): v for k, v in delta_table.items()}, report_dir / "deltas.json")
    _write_json({str(k): v for k, v in phase_dominance.items()}, report_dir / "phase_dominance.json")
    _write_json({str(k): v for k, v in failure_summary.items()}, report_dir / "failure_summary.json")
    _write_json([str(k) for k in pareto], report_dir / "pareto.json")

    _write_csv_rows(flatten_aggregated_results(aggregated), report_dir / "aggregated.csv")
    _write_csv_rows(flatten_delta_table(delta_table), report_dir / "deltas.csv")
    _write_csv_rows(flatten_phase_dominance(phase_dominance), report_dir / "phase_dominance.csv")
    _write_csv_rows(flatten_failure_summary(failure_summary), report_dir / "failure_summary.csv")

    # ------------------------------------------------------------------
    # Save the human-readable markdown report.
    # ------------------------------------------------------------------
    run_id = input_path.stem if input_path.is_file() else input_path.name
    generate_markdown_report(
        aggregated=aggregated,
        delta_table=delta_table,
        phase_dominance=phase_dominance,
        failure_summary=failure_summary,
        run_id=run_id,
        output_path=report_dir / "report.md",
        baseline_mode=baseline_mode,
        quality_metric_name=quality_metric_name,
    )

    # ------------------------------------------------------------------
    # Save plots if matplotlib is available.
    # ------------------------------------------------------------------
    if _MATPLOTLIB_AVAILABLE:
        plot_ttft_vs_tbt_scatter(aggregated, report_dir / "ttft_vs_tbt_scatter.png")
        plot_energy_per_token_bar(aggregated, report_dir / "energy_per_token.png")
        plot_memory_heatmap(aggregated, report_dir / "memory_heatmap.png")
        plot_phase_dominance_bar(phase_dominance, report_dir / "phase_dominance.png")

    logger.info("Full report written to %s", report_dir)
    return report_dir


# =============================================================================
# CLI entry point
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate an adapted benchmark report for ModeSwitch-LLM."
    )
    parser.add_argument(
        "input_path",
        help=(
            "Path to benchmark_results_*.json, benchmark_results_*.csv, or a directory "
            "containing benchmark_results_*.json files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for the report bundle.",
    )
    parser.add_argument(
        "--merge-all-jsons",
        action="store_true",
        help="When input_path is a directory, merge all benchmark_results_*.json files instead of only loading the newest.",
    )
    parser.add_argument(
        "--quality-metric",
        default="auto",
        choices=[
            "auto",
            "reference_rouge_l_f1",
            "baseline_similarity_rouge_l_f1",
            "quality_degradation_vs_baseline",
            "benchmark_primary_metric_value",
            "mmlu_pro_accuracy",
            "gsm8k_exact_match_accuracy",
            "truthfulqa_accuracy",
            "gpqa_accuracy",
            "mlu_accuracy",
            "tam_accuracy",
            "mt_bench_score",
            "alpacaeval2_lc_win_rate",
        ],
        help="Quality metric to use for reporting and Pareto analysis.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    report_dir = generate_full_report(
        input_path=args.input_path,
        output_dir=args.output_dir,
        merge_all_jsons=args.merge_all_jsons,
        quality_metric=args.quality_metric,
    )
    print(f"\n[reporter] Full report written to: {report_dir}")


if __name__ == "__main__":
    main()
