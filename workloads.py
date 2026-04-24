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
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from config import CONFIG, WorkloadConfig, get_workload_by_name

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


def build_runtime_workload_by_name(
    workload_name: str,
    repeated_prefix_variant: int = 0,
    system_condition_name: Optional[str] = None,
) -> RuntimeWorkload:
    """
    Fetch a workload config by name and build its runtime form.
    """
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
        if workload.repeated_prefix:
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

    return summary


def get_prompt_preview(runtime_workload: RuntimeWorkload, max_chars: int = 200) -> str:
    """
    Return a short preview of the workload prompt for debugging/logging.
    """
    prompt = runtime_workload.prompt.strip().replace("\n", " ")
    if len(prompt) <= max_chars:
        return prompt
    return prompt[:max_chars] + "..."