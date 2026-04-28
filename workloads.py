"""
workloads.py

Workload construction utilities for ModeSwitch-LLM.

Purpose:
- convert abstract workload definitions from config.py into runnable prompts
- create prompt text for short/long/repeated-prefix workloads
- provide a clean workload object for runner.py

Design principle:
config.py    = what workload buckets exist
workloads.py = how those workload buckets become actual benchmark inputs
"""

from enum import Enum
import csv
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from config import (
    CONFIG,
    BENCHMARK_DATA_DIR,
    WorkloadConfig,
    get_workload_by_name,
)

# =============================================================================
# System-condition metadata
# =============================================================================

class SystemCondition(Enum):
    BASELINE = "baseline"
    MEM_PRESSURE_50 = "mem_pressure_50"
    MEM_PRESSURE_75 = "mem_pressure_75"
    BATCH_2 = "batch_2"
    BATCH_4 = "batch_4"
    BATCH_8 = "batch_8"


@dataclass
class SystemConditionConfig:
    """
    Runtime system-condition metadata.

    For now this is descriptive/bookkeeping only unless the benchmark runner
    explicitly chooses to act on it.
    """
    condition: SystemCondition
    vram_fraction: float
    batch_size: int
    description: str


SYSTEM_CONDITIONS: Dict[SystemCondition, SystemConditionConfig] = {
    SystemCondition.BASELINE: SystemConditionConfig(
        condition=SystemCondition.BASELINE,
        vram_fraction=0.0,
        batch_size=1,
        description="No memory pressure; single request",
    ),
    SystemCondition.MEM_PRESSURE_50: SystemConditionConfig(
        condition=SystemCondition.MEM_PRESSURE_50,
        vram_fraction=0.50,
        batch_size=1,
        description="~50% memory pressure",
    ),
    SystemCondition.MEM_PRESSURE_75: SystemConditionConfig(
        condition=SystemCondition.MEM_PRESSURE_75,
        vram_fraction=0.75,
        batch_size=1,
        description="~75% memory pressure",
    ),
    SystemCondition.BATCH_2: SystemConditionConfig(
        condition=SystemCondition.BATCH_2,
        vram_fraction=0.0,
        batch_size=2,
        description="Batch size 2",
    ),
    SystemCondition.BATCH_4: SystemConditionConfig(
        condition=SystemCondition.BATCH_4,
        vram_fraction=0.0,
        batch_size=4,
        description="Batch size 4",
    ),
    SystemCondition.BATCH_8: SystemConditionConfig(
        condition=SystemCondition.BATCH_8,
        vram_fraction=0.0,
        batch_size=8,
        description="Batch size 8",
    ),
}

# =============================================================================
# Runtime workload representation
# =============================================================================

@dataclass
class RuntimeWorkload:
    """
    Concrete runnable workload instance.

    This is what runner.py will actually consume.
    """

    # Canonical workload name
    name: str

    # Actual input prompt text
    prompt: str

    # Max new tokens to generate
    max_new_tokens: int

    # Optional follow-up prompt used for repeated-prefix experiments.
    # When present, the benchmark runner can issue `prompt` first to prime the
    # engine and then time `followup_prompt` on the same engine instance.
    followup_prompt: Optional[str] = None

    # Optional reference answer for the primary prompt.
    reference_answer: Optional[str] = None

    # Optional reference answer for the follow-up prompt when repeated-prefix
    # timing uses the follow-up prompt rather than the initial one.
    followup_reference_answer: Optional[str] = None

    # Optional benchmark-evaluation metadata
    benchmark_suite: Optional[str] = None
    benchmark_subset: Optional[str] = None
    benchmark_language: Optional[str] = None
    evaluation_mode: Optional[str] = None
    benchmark_example_id: Optional[str] = None

    # Workload metadata borrowed from the broader benchmark design.
    task_type: Optional[str] = None
    workload_cell: Optional[str] = None
    system_condition_name: str = "baseline"

    # Human-readable description
    description: str = ""

    # Approximate prompt token target from config
    prompt_tokens_target: int = 0

    # Whether this workload is designed for repeated-prefix testing
    repeated_prefix: bool = False

    # Whether this workload is designed for memory-pressure testing
    memory_pressure: bool = False

    # Extra metadata for later logging/filtering/grouping
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the workload object into a plain dictionary.
        """
        output = asdict(self)
        if output["metadata"] is None:
            output["metadata"] = {}
        return output

# =============================================================================
# Metadata helpers
# =============================================================================

def infer_workload_cell(prompt_tokens: int, max_new_tokens: int) -> str:
    """
    Infer a coarse SS / SL / LS / LL label from token budgets.

    Short prompt  = <= 256 tokens
    Long prompt   = > 256 tokens
    Short output  = <= 64 tokens
    Long output   = > 64 tokens
    """
    prompt_side = "S" if prompt_tokens <= 256 else "L"
    output_side = "S" if max_new_tokens <= 64 else "L"
    return prompt_side + output_side

# =============================================================================
# Prompt templates
# =============================================================================

# Short reusable instruction-style seed
BASE_SHORT_PROMPT = (
    "You are a helpful assistant. Answer the following question clearly and concisely:\n\n"
    "Explain the importance of efficient inference for large language models."
)

# Longer reusable instruction-style seed
BASE_LONG_PROMPT = (
    "You are a helpful assistant. Carefully read the following detailed context and answer "
    "the question that follows.\n\n"
    "Large language model inference often consists of two major phases: prefill and decode. "
    "The prefill stage processes the input prompt and builds the key-value cache, while the "
    "decode stage generates output tokens one at a time. These two phases can stress hardware "
    "differently. Prefill is often more compute-intensive because it processes many prompt tokens "
    "together, while decode can become more memory-bandwidth-bound because the model repeatedly "
    "reads and updates cached activations over many steps. Practical efficiency techniques may "
    "therefore help one phase more than the other. Quantization can reduce memory footprint, "
    "speculative decoding can reduce per-token latency, prefix caching can avoid recomputing "
    "shared prompt segments, and chunked prefill can improve scheduling behavior when prompts "
    "are long. Because of this, a fixed inference mode may not always be optimal across all "
    "requests, models, or hardware settings.\n\n"
    "Question: Explain why separating prefill and decode metrics is useful when benchmarking "
    "LLM inference modes on a single GPU."
)

# Shared prefix used for prefix caching experiments
SHARED_PREFIX = (
    "System: You are an expert assistant helping with efficient AI benchmarking.\n"
    "Context: The user is evaluating inference-time optimizations on a single GPU.\n"
    "Instructions: Answer clearly, use concise reasoning, and focus on practical tradeoffs.\n\n"
)


# =============================================================================
# Prompt generation helpers
# =============================================================================

def _expand_text_to_target_length(base_text: str, target_tokens: int) -> str:
    """
    Naively expand a base text until it is roughly large enough for the desired
    prompt length bucket.

    Important:
    This is an approximation. Exact token counts depend on the tokenizer.
    For early development, this is usually sufficient. Later, you can replace
    this with tokenizer-based prompt calibration if needed.
    """
    # Very rough approximation:
    # assume around 0.75 words per token or simply treat repeated text length
    # as a coarse way to separate "short" vs "long" workloads.
    text = base_text.strip()

    if target_tokens <= 0:
        return text

    while len(text.split()) < target_tokens:
        text += "\n\n" + base_text.strip()

    return text


def _build_standard_prompt(prompt_tokens: int) -> str:
    """
    Build a standard prompt for short/long workload buckets.
    """
    if prompt_tokens <= 256:
        return _expand_text_to_target_length(BASE_SHORT_PROMPT, prompt_tokens)

    return _expand_text_to_target_length(BASE_LONG_PROMPT, prompt_tokens)


def _build_repeated_prefix_prompt(prompt_tokens: int, variant_id: int = 0) -> str:
    """
    Build a repeated-prefix workload prompt.

    The initial shared section stays the same across variants, while the tail
    changes slightly. This makes it useful for prefix caching experiments.
    """
    suffixes = [
        "Task: Summarize how prefix caching can reduce repeated prefill work.",
        "Task: Explain when prefix caching helps and when it may not help.",
        "Task: Compare prefix caching with chunked prefill in simple language.",
        "Task: Describe practical workloads where repeated prompt prefixes occur.",
    ]

    suffix = suffixes[variant_id % len(suffixes)]
    base_text = SHARED_PREFIX + suffix

    return _expand_text_to_target_length(base_text, prompt_tokens)

def _first_present_value(row: Dict[str, Any], keys: List[str]):
    """
    Return the first non-empty value from a row across candidate keys.
    """
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _build_prompt_from_sidecar_row(row: Dict[str, Any]) -> Optional[str]:
    """
    Build a prompt from a benchmark sidecar row.

    Supported shapes:
    - {"prompt": "..."}
    - {"input": "..."}
    - {"question": "...", "choices": [...]}
    - {"question": "...", "options": {...}}
    """
    direct_prompt = _first_present_value(
        row,
        ["prompt", "input", "full_prompt", "question_prompt", "user_prompt"],
    )
    if direct_prompt is not None:
        return _append_benchmark_answer_instruction(str(direct_prompt), row)

    question = _first_present_value(
        row,
        ["question", "query", "problem", "instruction"],
    )
    if question is None:
        return None

    choices = row.get("choices")
    if choices is None:
        choices = row.get("options")

    choice_lines: List[str] = []
    if isinstance(choices, dict):
        for label, text in choices.items():
            choice_lines.append(f"{label}. {text}")
    elif isinstance(choices, list):
        for idx, item in enumerate(choices):
            default_label = chr(ord("A") + idx)
            if isinstance(item, dict):
                label = str(item.get("label", default_label))
                text = item.get("text")
                if text is None:
                    text = item.get("option")
                if text is None:
                    text = item.get("value")
            else:
                label = default_label
                text = item
            choice_lines.append(f"{label}. {text}")

    if choice_lines:
        prompt = (
            f"{question}\n\n"
            f"Options:\n" + "\n".join(choice_lines)
        )

        return _append_benchmark_answer_instruction(prompt, row)

    return _append_benchmark_answer_instruction(str(question), row)


def _extract_reference_from_sidecar_row(row: Dict[str, Any]) -> Optional[str]:
    """
    Extract the reference / gold answer from a benchmark sidecar row.
    """
    reference = _first_present_value(
        row,
        [
            "reference",
            "answer",
            "target",
            "label",
            "gold",
            "gold_answer",
            "correct_answer",
            "correct_option",
            "expected_answer",
        ],
    )
    if reference is None:
        return None
    return str(reference)


def _infer_valid_labels_from_sidecar_row(row: Dict[str, Any]) -> Optional[List[str]]:
    """
    Infer valid multiple-choice labels from a sidecar row when possible.
    """
    explicit = row.get("valid_labels")
    if explicit:
        if isinstance(explicit, list):
            return [str(x) for x in explicit]
        return [str(explicit)]

    choices = row.get("choices")
    if choices is None:
        choices = row.get("options")

    if isinstance(choices, dict):
        return [str(key) for key in choices.keys()]

    if isinstance(choices, list) and choices:
        labels = []
        for idx, item in enumerate(choices):
            default_label = chr(ord("A") + idx)
            if isinstance(item, dict) and item.get("label") not in (None, ""):
                labels.append(str(item["label"]))
            else:
                labels.append(default_label)
        return labels

    return None


def _append_benchmark_answer_instruction(prompt: str, row: Dict[str, Any]) -> str:
    """
    Add stricter output-format instructions for benchmark grading.

    This reduces parser ambiguity for multiple-choice and final-answer tasks.
    """
    prompt = str(prompt).rstrip()
    lowered_prompt = prompt.lower()
    evaluation_mode = str(row.get("evaluation_mode") or "").strip().lower()

    if evaluation_mode in {"multiple_choice", "multiple_choice_accuracy"}:
        if "entire response must be exactly one uppercase letter" in lowered_prompt:
            return prompt

        prompt = re.sub(
            r"\n?\s*Answer with the correct letter only\.?\s*$",
            "",
            prompt,
            flags=re.IGNORECASE,
        ).rstrip()

        valid_labels = _infer_valid_labels_from_sidecar_row(row)
        if valid_labels:
            label_text = ", ".join(str(label) for label in valid_labels)
            return (
                prompt
                + f"\n\nReturn exactly one uppercase option label from {{{label_text}}} on a single line.\n"
                  "Do not repeat the question.\n"
                  "Do not repeat the options.\n"
                  "Do not explain your answer.\n"
                  "Your entire response must be exactly one uppercase letter."
            )

        return prompt + "\n\nRespond with exactly one option label. Do not explain your answer."

    if evaluation_mode == "final_answer_exact_match":
        if "final answer:" in lowered_prompt and "do not include units" in lowered_prompt:
            return prompt

        return (
            prompt
            + "\n\nSolve the problem carefully.\n"
              "Your last line must be exactly:\n"
              "Final answer: <number>\n"
              "Do not include units, commas, or any text after that final line."
        )

    return prompt


def _build_standard_reference_answer(prompt_tokens: int) -> str:
    """
    Reference answer for the standard prompt families.
    """
    if prompt_tokens <= 256:
        return (
            "Efficient inference is important because it reduces latency, lowers serving cost, "
            "improves throughput, and makes large language models practical for real-time use. "
            "It also helps deploy larger models within limited compute and memory budgets."
        )

    return (
        "Separating prefill and decode metrics is useful because they stress the GPU in different ways. "
        "Prefill processes many prompt tokens at once and is more compute-heavy, while decode generates "
        "tokens step by step and is often more memory-bandwidth-bound. Measuring them separately shows "
        "which optimization helps which phase instead of hiding the effect inside one combined latency."
    )


def _build_repeated_prefix_reference_answer(variant_id: int) -> str:
    """
    Reference answers for repeated-prefix benchmark tasks.
    """
    answers = [
        (
            "Prefix caching helps by reusing the KV-cache for the shared beginning of repeated prompts. "
            "Instead of recomputing the same prefix during prefill, the system can start from the cached "
            "state and only process the new suffix, reducing repeated prefill work and latency."
        ),
        (
            "Prefix caching helps when many requests share the same long prefix, such as a system prompt, "
            "retrieved context, or conversation history. It helps less when prompts are mostly unique, "
            "the shared prefix is short, or cache reuse is limited by memory pressure or eviction."
        ),
        (
            "Prefix caching and chunked prefill solve different problems. Prefix caching avoids recomputing "
            "shared prompt prefixes across related requests, while chunked prefill improves how long prompts "
            "are scheduled and processed. Prefix caching is about reuse across requests; chunked prefill is "
            "about better handling of large prompt computation."
        ),
        (
            "Repeated prompt prefixes commonly appear in chat systems with fixed system prompts, enterprise "
            "assistants that attach the same policy or tool instructions, retrieval pipelines that reuse a "
            "shared context block, and multi-turn conversations where most of the history is unchanged."
        ),
    ]
    return answers[variant_id % len(answers)]


def _build_memory_pressure_reference_answer() -> str:
    """
    Reference answer for the memory-pressure workload.
    """
    return (
        "When GPU memory is limited, long prompts and large KV caches increase memory pressure and can reduce "
        "throughput or cause OOM failures. Quantization can shrink model or cache memory needs, while runtime "
        "scheduling strategies such as chunked prefill or batching trade off memory and latency. The overall "
        "behavior depends on how model size, prompt length, generated length, and cache growth interact."
    )

def _build_memory_pressure_prompt(prompt_tokens: int) -> str:
    """
    Build a large prompt intended for memory-pressure scenarios.
    """
    base_text = (
        "You are analyzing GPU memory behavior during large language model inference. "
        "Discuss how long prompts, large KV caches, quantization, and runtime scheduling "
        "interact when GPU memory is limited."
    )
    return _expand_text_to_target_length(base_text, prompt_tokens)

def _resolve_benchmark_sidecar_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BENCHMARK_DATA_DIR / path


def _load_benchmark_sidecar_rows(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark sidecar file not found: {path}. "
            "Create it before expanding benchmark workloads."
        )

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of dicts in JSON sidecar: {path}")
        return data

    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    raise ValueError(
        f"Unsupported benchmark sidecar type: {path.suffix}. "
        "Use JSONL, JSON, or CSV."
    )


def _build_runtime_workloads_from_benchmark_sidecar(
    workload: WorkloadConfig,
    system_condition_name: Optional[str] = None,
) -> List[RuntimeWorkload]:
    if not workload.benchmark_source_path:
        return []

    resolved_system_condition_name = system_condition_name or workload.system_condition or "baseline"
    sidecar_path = _resolve_benchmark_sidecar_path(workload.benchmark_source_path)
    sidecar_rows = _load_benchmark_sidecar_rows(sidecar_path)

    runtime_rows: List[RuntimeWorkload] = []
    for idx, row in enumerate(sidecar_rows):
        if not isinstance(row, dict):
            continue

        row_for_prompt = dict(workload.metadata)
        row_for_prompt.update(row)

        prompt = _build_prompt_from_sidecar_row(row_for_prompt)
        if not prompt:
            raise ValueError(
                f"Benchmark sidecar row {idx} in {sidecar_path} is missing a usable prompt field."
            )

        example_id = str(
            row.get("id")
            or row.get("benchmark_example_id")
            or f"{workload.name}_{idx:04d}"
        )
        runtime_name = f"{workload.name}__{example_id}"

        row_metadata = dict(row.get("metadata") or {})
        metadata = dict(workload.metadata)
        metadata.update(row_metadata)

        inferred_valid_labels = _infer_valid_labels_from_sidecar_row(row_for_prompt)
        if inferred_valid_labels is not None and metadata.get("valid_labels") is None:
            metadata["valid_labels"] = inferred_valid_labels

        for passthrough_key in [
            "valid_labels",
            "question_id",
            "judge_prompt",
            "judge_reference",
            "benchmark_primary_metric_name",
            "benchmark_primary_metric_value",
            "mmlu_pro_accuracy",
            "gsm8k_exact_match_accuracy",
            "truthfulqa_accuracy",
            "gpqa_accuracy",
            "mlu_accuracy",
            "tam_accuracy",
            "mt_bench_score",
            "alpacaeval2_lc_win_rate",
        ]:
            if row.get(passthrough_key) is not None and passthrough_key not in metadata:
                metadata[passthrough_key] = row.get(passthrough_key)

        prompt_tokens_target = int(row.get("prompt_tokens_target", workload.prompt_tokens))
        max_new_tokens = int(row.get("max_new_tokens", workload.max_new_tokens))

        runtime_rows.append(
            RuntimeWorkload(
                name=runtime_name,
                prompt=str(prompt),
                max_new_tokens=max_new_tokens,
                followup_prompt=None,
                reference_answer=_extract_reference_from_sidecar_row(row),
                followup_reference_answer=None,
                benchmark_suite=row.get("benchmark_suite") or workload.benchmark_suite,
                benchmark_subset=row.get("benchmark_subset") or workload.benchmark_subset,
                benchmark_language=row.get("benchmark_language") or workload.benchmark_language,
                evaluation_mode=row.get("evaluation_mode") or workload.evaluation_mode,
                benchmark_example_id=example_id,
                task_type=workload.task_type,
                workload_cell=row.get("workload_cell") or workload.workload_cell or infer_workload_cell(
                    prompt_tokens_target,
                    max_new_tokens,
                ),
                system_condition_name=resolved_system_condition_name,
                description=row.get("description") or workload.description,
                prompt_tokens_target=prompt_tokens_target,
                repeated_prefix=False,
                memory_pressure=workload.memory_pressure,
                metadata=metadata,
            )
        )

    return runtime_rows

# =============================================================================
# Workload builders
# =============================================================================

def build_runtime_workload(
    workload: WorkloadConfig,
    repeated_prefix_variant: int = 0,
    system_condition_name: Optional[str] = None,
) -> RuntimeWorkload:
    """
    Convert an abstract WorkloadConfig into a concrete RuntimeWorkload.
    """
    runtime_name = workload.name
    followup_prompt = None
    reference_answer = None
    followup_reference_answer = None
    resolved_system_condition_name = system_condition_name or workload.system_condition or "baseline"

    if workload.benchmark_source_path:
        expanded = _build_runtime_workloads_from_benchmark_sidecar(
            workload,
            system_condition_name=system_condition_name,
        )
        if len(expanded) != 1:
            raise ValueError(
                f"Workload '{workload.name}' expands to {len(expanded)} benchmark examples. "
                "Use build_runtime_workloads_for_name(...) or an expanded runtime name "
                "like '<workload_name>__<example_id>'."
            )
        return expanded[0]

    if workload.repeated_prefix:
        runtime_name = f"{workload.name}_v{repeated_prefix_variant}"
        prompt = _build_repeated_prefix_prompt(
            prompt_tokens=workload.prompt_tokens,
            variant_id=repeated_prefix_variant,
        )
        followup_prompt = _build_repeated_prefix_prompt(
            prompt_tokens=workload.prompt_tokens,
            variant_id=repeated_prefix_variant + 1,
        )
        reference_answer = _build_repeated_prefix_reference_answer(repeated_prefix_variant)
        followup_reference_answer = _build_repeated_prefix_reference_answer(repeated_prefix_variant + 1)
    elif workload.memory_pressure:
        prompt = _build_memory_pressure_prompt(
            prompt_tokens=workload.prompt_tokens,
        )
        reference_answer = _build_memory_pressure_reference_answer()
    else:
        prompt = _build_standard_prompt(
            prompt_tokens=workload.prompt_tokens,
        )
        reference_answer = _build_standard_reference_answer(workload.prompt_tokens)

    # Allow config-level override if a future workload wants an explicit
    # reference answer instead of the synthetic default.
    if workload.reference_output is not None:
        reference_answer = workload.reference_output

    metadata = dict(workload.metadata)
    if workload.benchmark_suite is not None:
        metadata.setdefault("benchmark_suite", workload.benchmark_suite)
    if workload.benchmark_subset is not None:
        metadata.setdefault("benchmark_subset", workload.benchmark_subset)
    if workload.benchmark_language is not None:
        metadata.setdefault("benchmark_language", workload.benchmark_language)
    if workload.evaluation_mode is not None:
        metadata.setdefault("evaluation_mode", workload.evaluation_mode)
    metadata["base_workload_name"] = workload.name
    metadata["prompt_tokens_target"] = workload.prompt_tokens
    metadata["max_new_tokens"] = workload.max_new_tokens
    metadata["task_type"] = workload.task_type
    metadata["system_condition"] = resolved_system_condition_name
    metadata["workload_cell"] = workload.workload_cell or infer_workload_cell(
        workload.prompt_tokens,
        workload.max_new_tokens,
    )

    if workload.repeated_prefix:
        metadata["repeated_prefix_variant"] = repeated_prefix_variant
        metadata["followup_repeated_prefix_variant"] = repeated_prefix_variant + 1

    return RuntimeWorkload(
        name=runtime_name,
        prompt=prompt,
        followup_prompt=followup_prompt,
        reference_answer=reference_answer,
        followup_reference_answer=followup_reference_answer,
        benchmark_suite=workload.benchmark_suite or metadata.get("benchmark_suite"),
        benchmark_subset=workload.benchmark_subset or metadata.get("benchmark_subset"),
        benchmark_language=workload.benchmark_language or metadata.get("benchmark_language"),
        evaluation_mode=workload.evaluation_mode or metadata.get("evaluation_mode"),
        benchmark_example_id=metadata.get("benchmark_example_id"),
        task_type=workload.task_type,
        workload_cell=workload.workload_cell or infer_workload_cell(
            workload.prompt_tokens,
            workload.max_new_tokens,
        ),
        system_condition_name=resolved_system_condition_name,
        max_new_tokens=workload.max_new_tokens,
        description=workload.description,
        prompt_tokens_target=workload.prompt_tokens,
        repeated_prefix=workload.repeated_prefix,
        memory_pressure=workload.memory_pressure,
        metadata=metadata,
    )

def build_runtime_workloads_for_name(
    workload_name: str,
    repeated_prefix_variants: int = 2,
    system_condition_name: Optional[str] = None,
) -> List[RuntimeWorkload]:
    workload = get_workload_by_name(workload_name)

    if workload.benchmark_source_path:
        return _build_runtime_workloads_from_benchmark_sidecar(
            workload,
            system_condition_name=system_condition_name,
        )

    if workload.repeated_prefix:
        return [
            build_runtime_workload(
                workload=workload,
                repeated_prefix_variant=variant_id,
                system_condition_name=system_condition_name,
            )
            for variant_id in range(repeated_prefix_variants)
        ]

    return [
        build_runtime_workload(
            workload=workload,
            repeated_prefix_variant=0,
            system_condition_name=system_condition_name,
        )
    ]



def build_runtime_workload_by_name(
    workload_name: str,
    repeated_prefix_variant: int = 0,
    system_condition_name: Optional[str] = None,
) -> RuntimeWorkload:
    """
    Fetch a workload config by name and build its runtime form.
    """
    if "__" in workload_name:
        base_name, _, example_id = workload_name.partition("__")
        workload = get_workload_by_name(base_name)
        if workload.benchmark_source_path:
            expanded = _build_runtime_workloads_from_benchmark_sidecar(
                workload,
                system_condition_name=system_condition_name,
            )
            for runtime_workload in expanded:
                if runtime_workload.name == workload_name:
                    return runtime_workload
            raise ValueError(
                f"Unknown benchmark runtime workload name: {workload_name}. "
                f"No example with id '{example_id}' was found in '{base_name}'."
            )

    workload = get_workload_by_name(workload_name)
    return build_runtime_workload(
        workload=workload,
        repeated_prefix_variant=repeated_prefix_variant,
        system_condition_name=system_condition_name,
    )


def get_all_runtime_workloads(
    repeated_prefix_variants: int = 2,
) -> List[RuntimeWorkload]:
    """
    Build all configured workloads into runtime-ready workload objects.

    For repeated-prefix workloads, create multiple variants so you can test
    shared-prefix behavior more realistically.
    """
    runtime_workloads: List[RuntimeWorkload] = []

    for workload in CONFIG.workloads:
        if workload.benchmark_source_path:
            runtime_workloads.extend(
                _build_runtime_workloads_from_benchmark_sidecar(workload)
            )
        elif workload.repeated_prefix:
            for variant_id in range(repeated_prefix_variants):
                runtime_workloads.append(
                    build_runtime_workload(
                        workload=workload,
                        repeated_prefix_variant=variant_id,
                    )
                )
        else:
            runtime_workloads.append(
                build_runtime_workload(workload=workload)
            )

    return runtime_workloads


# =============================================================================
# Optional utility helpers
# =============================================================================

def summarize_workload(runtime_workload: RuntimeWorkload) -> str:
    """
    Return a short human-readable summary of a workload.
    """
    summary = (
        f"{runtime_workload.name} | "
        f"target_prompt_tokens={runtime_workload.prompt_tokens_target} | "
        f"max_new_tokens={runtime_workload.max_new_tokens}"
    )

    if runtime_workload.repeated_prefix:
        summary += " | repeated_prefix=True"
    if runtime_workload.memory_pressure:
        summary += " | memory_pressure=True"
    if runtime_workload.benchmark_suite:
        summary += f" | benchmark_suite={runtime_workload.benchmark_suite}"
    if runtime_workload.benchmark_example_id:
        summary += f" | benchmark_example_id={runtime_workload.benchmark_example_id}"
    return summary


def get_prompt_preview(runtime_workload: RuntimeWorkload, max_chars: int = 200) -> str:
    """
    Return a short preview of the workload prompt for debugging/logging.
    """
    prompt = runtime_workload.prompt.strip().replace("\n", " ")
    if len(prompt) <= max_chars:
        return prompt
    return prompt[:max_chars] + "..."