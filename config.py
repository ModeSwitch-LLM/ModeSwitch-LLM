"""
config.py

Central configuration file for the ModeSwitch-LLM benchmarking pipeline.

This file stores:
- global experiment settings
- model/runtime configuration
- candidate fixed modes
- workload groups
- output/logging paths
- benchmark defaults
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Project paths
# =============================================================================

# Root directory of the benchmarking project.
PROJECT_ROOT = Path(__file__).resolve().parent

# Folder for all output artifacts.
RESULTS_DIR = PROJECT_ROOT / "results"

# Raw benchmark outputs, typically one file per run / mode / trial.
RAW_RESULTS_DIR = RESULTS_DIR / "raw"

# Processed summaries, merged CSVs, cleaned result tables, etc.
PROCESSED_RESULTS_DIR = RESULTS_DIR / "processed"

# Plots and visualizations.
PLOTS_DIR = RESULTS_DIR / "plots"

# Logs from benchmark runs.
LOGS_DIR = PROJECT_ROOT / "logs"

# Temporary files if needed.
TMP_DIR = PROJECT_ROOT / "tmp"

# Benchmark-sidecar files (JSONL / JSON / CSV) for benchmark-style workloads.
BENCHMARK_DATA_DIR = PROJECT_ROOT / "benchmark_data"

# =============================================================================
# Basic runtime / model config
# =============================================================================

@dataclass
class ModelConfig:
    """
    Configuration related to the base LLM and tokenizer.
    """

    # Hugging Face model name or local checkpoint path.
    model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Separate tokenizer path. Usually same as model path.
    tokenizer_name_or_path: Optional[str] = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Primary runtime/backend to use.
    # Examples: "transformers", "vllm", "tgi"
    backend: str = "vllm"

    # Primary device for inference.
    device: str = "cuda"

    # Dtype for the baseline full-precision-ish configuration.
    # Usually "float16" for modern single-GPU inference.
    baseline_dtype: str = "float16"

    # Whether to trust remote code when loading models/tokenizers.
    trust_remote_code: bool = True

    # Maximum context length to assume for runs if needed.
    max_model_len: int = 4096

    # Tensor parallel replicas for vLLM.
    tensor_parallel_size: int = 1

    # Fraction of GPU memory that vLLM is allowed to use.
    # Keep this below 1.0 so the engine has room for runtime buffers / KV cache
    # without immediately OOMing on a 40 GB class GPU.
    gpu_memory_utilization: float = 0.82

    # CPU swap space exposed to vLLM (GiB per GPU).
    swap_space_gb: int = 4

    # Scheduling / batching defaults.
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 8

    # CUDA-graph capture window used when graph mode is enabled.
    max_seq_len_to_capture: int = 2048

    # Baseline policy:
    # make the FP16 baseline eager so the CUDA-graphs mode is a real ablation.
    enforce_eager_baseline: bool = True

    # Optional alternate checkpoints for quantized / draft-model runs.
    # These are real model IDs rather than fake "turn the base checkpoint into
    # AWQ/GPTQ/INT8 with one flag" placeholders.
    int8_model_name_or_path: Optional[str] = "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
    awq_model_name_or_path: Optional[str] = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    gptq_model_name_or_path: Optional[str] = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    speculative_model_name_or_path: Optional[str] = "meta-llama/Llama-3.2-1B-Instruct"

    # Environment variable to use for gated / authenticated HF downloads.
    hf_token_env_var: str = "HF_TOKEN"

@dataclass
class GenerationConfig:
    """
    Default text generation parameters.
    These should stay fixed unless workload-specific logic overrides them.
    """

    # Decoding strategy. For clean benchmarking, greedy is usually safest.
    do_sample: bool = False

    # Temperature should be 0 or ignored if using greedy decoding.
    temperature: float = 0.0

    # Top-p / nucleus sampling. Mostly irrelevant if do_sample=False.
    top_p: float = 1.0

    # Top-k token filtering.
    top_k: int = 0

    # Repetition penalty, if used.
    repetition_penalty: float = 1.0

    # Stop strings can be added later if needed.
    stop_sequences: List[str] = field(default_factory=list)

    # Number of sequences to generate per prompt.
    num_return_sequences: int = 1


@dataclass
class SystemConfig:
    """
    System-level and reproducibility settings.
    """

    # Random seed for reproducibility.
    seed: int = 42

    # Number of warmup runs before actual timing.
    warmup_runs: int = 1

    # Number of measured trials per (mode, workload) pair.
    num_trials: int = 2

    # Whether to clear CUDA cache between runs.
    clear_cuda_cache_between_runs: bool = True

    # Whether to force synchronization around timing calls.
    sync_cuda_for_timing: bool = True

    # Whether to save generated text outputs for later quality checks.
    save_generated_outputs: bool = True

    # Whether to collect GPU memory stats.
    collect_memory_stats: bool = True

    # Whether to collect power / energy stats if supported.
    collect_energy_stats: bool = True

    # Whether to collect CPU RAM stats.
    collect_cpu_memory_stats: bool = True

    # Whether to compute quality metrics when a reference is available.
    collect_quality_metrics: bool = True

    # Interval in seconds for optional power polling.
    power_poll_interval_s: float = 0.05

    # Optional timeout per run in seconds.
    run_timeout_s: Optional[float] = None

    # For the "continuous batching" benchmark mode, measure a small multi-request
    # batch on one engine instance.
    continuous_batching_batch_size: int = 4

    # Baseline mode used for baseline-quality comparisons in postprocessing.
    baseline_reference_mode_name: str = "fp16_baseline"

    # If True, synthetic non-benchmark workloads receive a trial-specific
    # header before timing. This prevents prefix-caching modes from getting
    # accidental cache reuse just because the same synthetic prompt is repeated
    # across trials on one loaded engine.
    unique_synthetic_prompt_per_trial: bool = True

# =============================================================================
# Mode definitions
# =============================================================================

@dataclass
class ModeConfig:
    """
    Defines one fixed inference mode.

    A "fixed mode" means one predefined inference configuration that remains
    constant for a run, such as FP16, INT8, AWQ, speculative decoding, etc.
    """

    # Unique mode name used everywhere in the pipeline.
    name: str

    # Short human-readable description.
    description: str

    # High-level category for grouping later.
    # Examples: "baseline", "quantization", "decoding", "cache", "scheduler"
    category: str

    # Backend/runtime used by this mode.
    backend: str = "vllm"

    # Primary dtype or precision label if relevant.
    dtype: Optional[str] = None

    # Quantization method name, if applicable.
    # Examples: "bitsandbytes_int8", "awq", "gptq"
    quantization: Optional[str] = None

    # Whether speculative decoding is enabled.
    speculative_decoding: bool = False

    # Whether KV-cache compression / quantization is enabled.
    kv_cache_compression: bool = False

    # Whether prefix caching is enabled.
    prefix_caching: bool = False

    # Whether chunked prefill is enabled.
    chunked_prefill: bool = False

    # Whether continuous batching is enabled.
    continuous_batching: bool = False

    # Whether CUDA graphs are enabled.
    cuda_graphs: bool = False

    # Arbitrary backend-specific flags.
    extra_args: Dict[str, Any] = field(default_factory=dict)

    # Optional note on which phase this mode is expected to help most.
    # Examples: "prefill", "decode", "both"
    primary_phase: str = "both"

    # Whether this mode is currently enabled for the benchmark sweep.
    enabled: bool = True


# =============================================================================
# Workload definitions
# =============================================================================

@dataclass
class WorkloadConfig:
    """
    Defines one workload bucket for benchmarking.
    """

    # Unique workload name.
    name: str

    # Prompt token length target / category.
    prompt_tokens: int

    # Max new tokens to generate.
    max_new_tokens: int

    # Optional human-readable description.
    description: str = ""

    # Task label used in reporting / grouping.
    task_type: Optional[str] = None

    # Explicit workload cell label (SS / SL / LS / LL).
    workload_cell: Optional[str] = None

    # Named system condition associated with this workload.
    system_condition: Optional[str] = None

    # Reference output for lightweight quality checks.
    reference_output: Optional[str] = None

    # Whether this workload is meant to test shared-prefix behavior.
    repeated_prefix: bool = False

    # Whether this workload is meant to run under memory pressure.
    memory_pressure: bool = False

    # Optional benchmark-suite metadata.
    benchmark_suite: Optional[str] = None
    benchmark_subset: Optional[str] = None
    benchmark_language: Optional[str] = None
    evaluation_mode: Optional[str] = None

    # Optional path to a sidecar file containing concrete benchmark examples.
    # Supports JSONL, JSON(list[dict]), or CSV.
    benchmark_source_path: Optional[str] = None

    # Whether this workload requires an external judge / sidecar scorer after generation.
    benchmark_judge_required: bool = False

    # Optional custom metadata for later grouping/filtering.
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Default candidate modes
# =============================================================================

DEFAULT_MODES: List[ModeConfig] = [
    ModeConfig(
        name="fp16_baseline",
        description="FP16 baseline inference mode",
        category="baseline",
        backend="vllm",
        dtype="float16",
        primary_phase="both",
        extra_args={
            # Make the baseline a real eager baseline so CUDA-graphs mode is
            # a meaningful ablation instead of already being implicitly enabled.
            "enforce_eager": True,
        },
        enabled=True,
    ),
    ModeConfig(
        name="int8_quant",
        description="INT8 W8A8 quantized checkpoint mode",
        category="quantization",
        backend="vllm",
        # Red Hat / Neural Magic W8A8 checkpoints are loaded as real quantized
        # checkpoints. Let vLLM infer from the model config when possible.
        quantization="compressed-tensors",
        primary_phase="decode",
        extra_args={
            "model_name_or_path": "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
            "dtype": "auto",
        },
        enabled=True,
    ),
    ModeConfig(
        name="awq_4bit",
        description="4-bit AWQ quantized inference mode",
        category="quantization",
        backend="vllm",
        quantization="awq",
        primary_phase="decode",
        extra_args={
            "model_name_or_path": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            "dtype": "float16",
        },
        enabled=True,
    ),
    ModeConfig(
        name="gptq_4bit",
        description="4-bit GPTQ quantized inference mode",
        category="quantization",
        backend="vllm",
        quantization="gptq",
        primary_phase="decode",
        extra_args={
            "model_name_or_path": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
            "dtype": "float16",
        },
        enabled=True,
    ),
    ModeConfig(
        name="speculative_decoding",
        description="Speculative decoding mode",
        category="decoding",
        backend="vllm",
        speculative_decoding=True,
        primary_phase="decode",
        extra_args={
            "speculative_model": "meta-llama/Llama-3.2-1B-Instruct",
            "num_speculative_tokens": 5,
        },
        enabled=True,
    ),
    ModeConfig(
        name="kv_cache_compression",
        description="Quantized KV-cache mode",
        category="cache",
        backend="vllm",
        kv_cache_compression=True,
        primary_phase="decode",
        extra_args={
            "kv_cache_dtype": "fp8_e4m3",
        },
        enabled=True,
    ),
    ModeConfig(
        name="prefix_caching",
        description="Automatic prefix caching mode",
        category="cache",
        backend="vllm",
        prefix_caching=True,
        primary_phase="prefill",
        enabled=True,
    ),
    ModeConfig(
        name="chunked_prefill",
        description="Chunked prefill scheduling mode",
        category="scheduler",
        backend="vllm",
        chunked_prefill=True,
        primary_phase="prefill",
        extra_args={
            # Must stay >= max_model_len, otherwise vLLM rejects the engine config.
            "max_num_batched_tokens": 4096,
        },
        enabled=True,
    ),
    ModeConfig(
        name="continuous_batching",
        description="Continuous batching mode",
        category="scheduler",
        backend="vllm",
        continuous_batching=True,
        primary_phase="decode",
        extra_args={
            "max_num_seqs": 4,
            "max_num_batched_tokens": 4096,
        },
        enabled=True,
    ),
    ModeConfig(
        name="cuda_graphs",
        description="CUDA graph capture / replay mode",
        category="runtime",
        backend="vllm",
        cuda_graphs=True,
        primary_phase="both",
        extra_args={
            "enforce_eager": False,
            "max_seq_len_to_capture": 2048,
        },
        enabled=True,
    ),
]


# =============================================================================
# Default workloads
# =============================================================================

DEFAULT_WORKLOADS: List[WorkloadConfig] = [
    WorkloadConfig(
        name="short_prompt_short_output",
        prompt_tokens=128,
        max_new_tokens=32,
        description="Short prompt, short output",
        task_type="qa",
        workload_cell="SS",
    ),
    WorkloadConfig(
        name="short_prompt_long_output",
        prompt_tokens=128,
        max_new_tokens=128,
        description="Short prompt, long output",
        task_type="qa",
        workload_cell="SL",
    ),
    WorkloadConfig(
        name="long_prompt_short_output",
        prompt_tokens=1024,
        max_new_tokens=32,
        description="Long prompt, short output",
        task_type="analysis",
        workload_cell="LS",
    ),
    WorkloadConfig(
        name="long_prompt_long_output",
        prompt_tokens=1024,
        max_new_tokens=128,
        description="Long prompt, long output",
        task_type="analysis",
        workload_cell="LL",
    ),
    WorkloadConfig(
        name="shared_prefix_chat",
        prompt_tokens=1024,
        max_new_tokens=64,
        description="Shared-prefix workload for prefix caching experiments",
        repeated_prefix=True,
        task_type="chat",
        workload_cell="LS",
        metadata={
            "workload_family": "shared_prefix",
            "repeated_prefix_variants": 10,
        },
    ),
    WorkloadConfig(
        name="memory_pressure_long_context",
        prompt_tokens=2048,
        max_new_tokens=128,
        description="Long-context workload under artificial memory pressure",
        memory_pressure=True,
        task_type="analysis",
        workload_cell="LS",
        system_condition="mem_pressure_50",
        metadata={
            "workload_family": "memory_pressure",
            "memory_pressure_variants": 10,
        },
    ),
    WorkloadConfig(
        name="mmlu_pro_eval",
        prompt_tokens=512,
        max_new_tokens=8,
        description="Sidecar-backed MMLU-Pro evaluation workload",
        task_type="benchmark",
        benchmark_suite="mmlu_pro",
        benchmark_language="en",
        evaluation_mode="multiple_choice_accuracy",
        benchmark_source_path="mmlu_pro_eval.jsonl",
        metadata={"benchmark_family": "knowledge_reasoning"},
    ),
    WorkloadConfig(
        name="gsm8k_eval",
        prompt_tokens=512,
        max_new_tokens=384,
        description="Sidecar-backed GSM8K evaluation workload",
        task_type="benchmark",
        benchmark_suite="gsm8k",
        benchmark_language="en",
        evaluation_mode="final_answer_exact_match",
        benchmark_source_path="gsm8k_eval.jsonl",
        metadata={"benchmark_family": "math_reasoning"},
    ),
    WorkloadConfig(
        name="truthfulqa_eval",
        prompt_tokens=512,
        max_new_tokens=8,
        description="Sidecar-backed TruthfulQA evaluation workload",
        task_type="benchmark",
        benchmark_suite="truthfulqa",
        benchmark_language="en",
        evaluation_mode="multiple_choice_accuracy",
        benchmark_source_path="truthfulqa_eval.jsonl",
        metadata={"benchmark_family": "truthfulness"},
    ),
    WorkloadConfig(
        name="gpqa_eval",
        prompt_tokens=512,
        max_new_tokens=8,
        description="Sidecar-backed GPQA evaluation workload",
        task_type="benchmark",
        benchmark_suite="gpqa",
        benchmark_language="en",
        evaluation_mode="multiple_choice_accuracy",
        benchmark_source_path="gpqa_eval.jsonl",
        metadata={"benchmark_family": "science_reasoning"},
    ),
    WorkloadConfig(
        name="mlu_eval",
        prompt_tokens=512,
        max_new_tokens=8,
        description="Sidecar-backed MLU evaluation workload",
        task_type="benchmark",
        benchmark_suite="mlu",
        benchmark_language="multilingual",
        evaluation_mode="multiple_choice_accuracy",
        benchmark_source_path="mlu_eval.jsonl",
        metadata={"benchmark_family": "multilingual_understanding"},
    ),
    WorkloadConfig(
        name="mt_bench_eval",
        prompt_tokens=512,
        max_new_tokens=1024,
        description="Sidecar-backed MT-Bench evaluation workload",
        task_type="benchmark",
        benchmark_suite="mt_bench",
        benchmark_language="en",
        evaluation_mode="external_judge",
        benchmark_source_path="mt_bench_eval.jsonl",
        benchmark_judge_required=True,
        metadata={"benchmark_family": "chat_quality"},
    ),
    WorkloadConfig(
        name="alpacaeval2_lc_eval",
        prompt_tokens=512,
        max_new_tokens=512,
        description="Sidecar-backed AlpacaEval 2 LC evaluation workload",
        task_type="benchmark",
        benchmark_suite="alpacaeval2_lc",
        benchmark_language="en",
        evaluation_mode="external_judge",
        benchmark_source_path="alpacaeval2_lc_eval.jsonl",
        benchmark_judge_required=True,
        metadata={"benchmark_family": "chat_quality"},
    ),
]


# =============================================================================
# Top-level experiment config
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Master experiment configuration object.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    modes: List[ModeConfig] = field(default_factory=lambda: DEFAULT_MODES.copy())
    workloads: List[WorkloadConfig] = field(default_factory=lambda: DEFAULT_WORKLOADS.copy())


# Global singleton-style config object
CONFIG = ExperimentConfig()


# =============================================================================
# Helper functions
# =============================================================================

def ensure_directories() -> None:
    """
    Create all standard output directories if they do not already exist.
    """
    for directory in [
        RESULTS_DIR,
        RAW_RESULTS_DIR,
        PROCESSED_RESULTS_DIR,
        PLOTS_DIR,
        LOGS_DIR,
        TMP_DIR,
        BENCHMARK_DATA_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def get_enabled_modes() -> List[ModeConfig]:
    """
    Return only the modes currently enabled for benchmarking.
    """
    return [mode for mode in CONFIG.modes if mode.enabled]


def get_workload_by_name(name: str) -> WorkloadConfig:
    """
    Fetch a workload config by name.
    """
    for workload in CONFIG.workloads:
        if workload.name == name:
            return workload
    raise ValueError(f"Unknown workload name: {name}")


def get_mode_by_name(name: str) -> ModeConfig:
    """
    Fetch a mode config by name.
    """
    for mode in CONFIG.modes:
        if mode.name == name:
            return mode
    raise ValueError(f"Unknown mode name: {name}")


# Create directories automatically when this file is imported.
ensure_directories()