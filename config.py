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


# =============================================================================
# Basic runtime / model config
# =============================================================================

@dataclass
class ModelConfig:
    """
    Configuration related to the base LLM and tokenizer.
    """

    # Hugging Face model name or local checkpoint path.
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Separate tokenizer path. Usually same as model path.
    tokenizer_name_or_path: Optional[str] = None

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
    max_model_len: int = 8192


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
    num_trials: int = 3

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

    # Interval in seconds for optional power polling.
    power_poll_interval_s: float = 0.05

    # Optional timeout per run in seconds.
    run_timeout_s: Optional[float] = None


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

    # Whether this workload is meant to test shared-prefix behavior.
    repeated_prefix: bool = False

    # Whether this workload is meant to run under memory pressure.
    memory_pressure: bool = False

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
        dtype="float16",
        primary_phase="both",
    ),
    ModeConfig(
        name="int8_quant",
        description="INT8 quantized inference mode",
        category="quantization",
        quantization="int8",
        primary_phase="decode",
    ),
    ModeConfig(
        name="awq_4bit",
        description="4-bit AWQ quantized inference mode",
        category="quantization",
        quantization="awq",
        primary_phase="decode",
    ),
    ModeConfig(
        name="gptq_4bit",
        description="4-bit GPTQ quantized inference mode",
        category="quantization",
        quantization="gptq",
        primary_phase="decode",
        enabled=True, 
    ),
    ModeConfig(
        name="speculative_decoding",
        description="Speculative decoding mode",
        category="decoding",
        speculative_decoding=True,
        primary_phase="decode",
    ),
    ModeConfig(
        name="kv_cache_compression",
        description="KV-cache compression / quantization mode",
        category="cache",
        kv_cache_compression=True,
        primary_phase="decode",
    ),
    ModeConfig(
        name="prefix_caching",
        description="Automatic prefix caching mode",
        category="cache",
        prefix_caching=True,
        primary_phase="prefill",
    ),
    ModeConfig(
        name="chunked_prefill",
        description="Chunked prefill scheduling mode",
        category="scheduler",
        chunked_prefill=True,
        primary_phase="prefill",
    ),
    ModeConfig(
        name="continuous_batching",
        description="Continuous batching mode",
        category="scheduler",
        continuous_batching=True,
        primary_phase="decode",
    ),
    ModeConfig(
        name="cuda_graphs",
        description="CUDA graph capture / replay mode",
        category="runtime",
        cuda_graphs=True,
        primary_phase="both",
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
    ),
    WorkloadConfig(
        name="short_prompt_long_output",
        prompt_tokens=128,
        max_new_tokens=256,
        description="Short prompt, long output",
    ),
    WorkloadConfig(
        name="long_prompt_short_output",
        prompt_tokens=2048,
        max_new_tokens=32,
        description="Long prompt, short output",
    ),
    WorkloadConfig(
        name="long_prompt_long_output",
        prompt_tokens=2048,
        max_new_tokens=256,
        description="Long prompt, long output",
    ),
    WorkloadConfig(
        name="shared_prefix_chat",
        prompt_tokens=1024,
        max_new_tokens=128,
        description="Shared-prefix workload for prefix caching experiments",
        repeated_prefix=True,
    ),
    WorkloadConfig(
        name="memory_pressure_long_context",
        prompt_tokens=4096,
        max_new_tokens=128,
        description="Long-context workload under artificial memory pressure",
        memory_pressure=True,
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