"""
modes.py

Backend-facing mode utilities for ModeSwitch-LLM.

Purpose:
- interpret abstract fixed inference modes from config.py
- convert them into backend/runtime-specific argument dictionaries

Design principle:
config.py = what modes exist
modes.py  = how those modes map to runnable settings
"""

from dataclasses import dataclass, asdict
from dataclasses import dataclass, asdict, field, replace
from typing import Any, Dict, List, Optional

from config import CONFIG, ModeConfig, get_mode_by_name


# =============================================================================
# Backend-facing runtime mode representation
# =============================================================================

@dataclass
class RuntimeMode:
    """
    Concrete runtime-ready representation of a mode.

    This is the object returned by the mode builder and later consumed by
    model_loader.py / runner.py.
    """

    # Canonical mode name
    name: str

    # Short description
    description: str

    # Backend/runtime, e.g. "vllm", "transformers"
    backend: str

    # Precision / dtype if relevant
    dtype: Optional[str] = None

    # Quantization label if relevant
    quantization: Optional[str] = None

    # High-level phase this mode primarily targets
    primary_phase: str = "both"

    # Human notes for logging / debugging
    notes: str = ""

    # Generic feature toggles
    speculative_decoding: bool = False
    kv_cache_compression: bool = False
    prefix_caching: bool = False
    chunked_prefill: bool = False
    continuous_batching: bool = False
    cuda_graphs: bool = False

    # Backend-specific runtime kwargs
    runtime_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Runner-side kwargs. These are NOT passed into the backend constructor.
    # They control how the benchmark driver exercises the engine.
    runner_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the dataclass into a plain dictionary.
        """
        return asdict(self)


# =============================================================================
# Validation helpers
# =============================================================================

def validate_mode_config(mode: ModeConfig) -> None:
    """
    Basic validation checks for a mode before converting it to runtime form.
    """
    if mode.backend not in {"vllm", "transformers", "tgi"}:
        raise ValueError(
            f"Unsupported backend '{mode.backend}' for mode '{mode.name}'."
        )

    if mode.quantization is not None and mode.dtype is not None:
        # This is not always illegal, but it can often indicate confusion.
        # Keep it strict for now to reduce ambiguity.
        pass

    if mode.primary_phase not in {"prefill", "decode", "both"}:
        raise ValueError(
            f"Mode '{mode.name}' has invalid primary_phase='{mode.primary_phase}'."
        )


# =============================================================================
# Backend mapping logic
# =============================================================================

def _build_vllm_runtime_kwargs(mode: ModeConfig) -> Dict[str, Any]:
    """
    Convert a ModeConfig into a vLLM-oriented kwargs dictionary.

    NOTE:
    These keys are deliberately generic for now. Later, once your exact
    runtime/API entrypoint is finalized, you can adapt the names to match
    your actual LLM(...) constructor or serve command.
    """
    kwargs: Dict[str, Any] = {
        "dtype": mode.dtype or CONFIG.model.baseline_dtype,
        "tensor_parallel_size": CONFIG.model.tensor_parallel_size,
        "gpu_memory_utilization": CONFIG.model.gpu_memory_utilization,
        "swap_space": CONFIG.model.swap_space_gb,
        "max_num_batched_tokens": CONFIG.model.max_num_batched_tokens,
        "max_num_seqs": CONFIG.model.max_num_seqs,
        "disable_log_stats": True,
    }

    # Baseline dtype
    if mode.dtype is not None:
        kwargs["dtype"] = mode.dtype

    # Quantization modes
    if mode.quantization is not None:
        kwargs["quantization"] = mode.quantization

    # Speculative decoding (vLLM 0.6.x style args)
    if mode.speculative_decoding:
        kwargs.setdefault(
            "speculative_model",
            CONFIG.model.speculative_model_name_or_path,
        )
        kwargs.setdefault("num_speculative_tokens", 5)

    # KV cache quantization
    if mode.kv_cache_compression:
        kwargs["kv_cache_dtype"] = "fp8_e4m3"
        kwargs["calculate_kv_scales"] = True

    if mode.prefix_caching:
        kwargs["enable_prefix_caching"] = True

    if mode.chunked_prefill:
        kwargs["enable_chunked_prefill"] = True

    if mode.continuous_batching:
        kwargs["max_num_seqs"] = max(
            CONFIG.model.max_num_seqs,
            CONFIG.system.continuous_batching_batch_size,
        )

    if mode.cuda_graphs:
        kwargs["enforce_eager"] = False
        kwargs["max_seq_len_to_capture"] = CONFIG.model.max_seq_len_to_capture
    elif mode.name == "fp16_baseline":
        kwargs["enforce_eager"] = CONFIG.model.enforce_eager_baseline

    # Merge any custom extras last so they can override defaults if needed
    kwargs.update(mode.extra_args)

    return kwargs


def _build_runner_kwargs(mode: ModeConfig) -> Dict[str, Any]:
    """
    Build runner-only benchmark controls.

    Some "modes" are not meaningful as one fake engine flag:
    - continuous batching should be exercised by issuing multiple requests to
      one engine instance
    - prefix caching should be exercised by priming a shared prefix and timing
      the follow-up request on the same engine instance
    """
    runner_kwargs: Dict[str, Any] = {}

    if mode.continuous_batching:
        runner_kwargs["request_batch_size"] = CONFIG.system.continuous_batching_batch_size

    return runner_kwargs

def _build_transformers_runtime_kwargs(mode: ModeConfig) -> Dict[str, Any]:
    """
    Convert a ModeConfig into a Transformers-oriented kwargs dictionary.

    These settings are placeholders / adapter-level fields that you can later
    translate into actual AutoModel / quantization / generate() settings.
    """
    kwargs: Dict[str, Any] = {}

    if mode.dtype is not None:
        kwargs["torch_dtype"] = mode.dtype

    if mode.quantization is not None:
        kwargs["quantization"] = mode.quantization

    if mode.speculative_decoding:
        kwargs["speculative_decoding"] = True

    if mode.kv_cache_compression:
        kwargs["kv_cache_compression"] = True

    if mode.prefix_caching:
        kwargs["prefix_caching"] = True

    if mode.chunked_prefill:
        kwargs["chunked_prefill"] = True

    if mode.continuous_batching:
        kwargs["continuous_batching"] = True

    if mode.cuda_graphs:
        kwargs["cuda_graphs"] = True

    kwargs.update(mode.extra_args)

    return kwargs


def _build_tgi_runtime_kwargs(mode: ModeConfig) -> Dict[str, Any]:
    """
    Convert a ModeConfig into a TGI-oriented kwargs dictionary.
    """
    kwargs: Dict[str, Any] = {}

    if mode.dtype is not None:
        kwargs["dtype"] = mode.dtype

    if mode.quantization is not None:
        kwargs["quantization"] = mode.quantization

    if mode.speculative_decoding:
        kwargs["speculative_decoding"] = True

    if mode.kv_cache_compression:
        kwargs["kv_cache_compression"] = True

    if mode.prefix_caching:
        kwargs["prefix_caching"] = True

    if mode.chunked_prefill:
        kwargs["chunked_prefill"] = True

    if mode.continuous_batching:
        kwargs["continuous_batching"] = True

    if mode.cuda_graphs:
        kwargs["cuda_graphs"] = True

    kwargs.update(mode.extra_args)

    return kwargs


# =============================================================================
# Public builders
# =============================================================================

def build_runtime_mode(mode: ModeConfig) -> RuntimeMode:
    """
    Convert an abstract ModeConfig into a concrete RuntimeMode object.
    """
    validate_mode_config(mode)

    if mode.backend == "vllm":
        runtime_kwargs = _build_vllm_runtime_kwargs(mode)
    elif mode.backend == "transformers":
        runtime_kwargs = _build_transformers_runtime_kwargs(mode)
    elif mode.backend == "tgi":
        runtime_kwargs = _build_tgi_runtime_kwargs(mode)
    else:
        raise ValueError(f"Unsupported backend '{mode.backend}'.")

    runner_kwargs = _build_runner_kwargs(mode)

    notes_parts: List[str] = []
    if mode.quantization:
        notes_parts.append(f"quantization={mode.quantization}")
    if mode.speculative_decoding:
        notes_parts.append("speculative decoding enabled")
    if mode.kv_cache_compression:
        notes_parts.append("KV-cache compression enabled")
    if mode.prefix_caching:
        notes_parts.append("prefix caching enabled")
    if mode.chunked_prefill:
        notes_parts.append("chunked prefill enabled")
    if mode.continuous_batching:
        notes_parts.append("continuous batching benchmark uses a multi-request batch")
    if mode.cuda_graphs:
        notes_parts.append("CUDA graphs enabled")

    notes = "; ".join(notes_parts) if notes_parts else "standard mode"

    return RuntimeMode(
        name=mode.name,
        description=mode.description,
        backend=mode.backend,
        dtype=mode.dtype,
        quantization=mode.quantization,
        primary_phase=mode.primary_phase,
        notes=notes,
        speculative_decoding=mode.speculative_decoding,
        kv_cache_compression=mode.kv_cache_compression,
        prefix_caching=mode.prefix_caching,
        chunked_prefill=mode.chunked_prefill,
        continuous_batching=mode.continuous_batching,
        cuda_graphs=mode.cuda_graphs,
        runtime_kwargs=runtime_kwargs,
        runner_kwargs=runner_kwargs,
    )


def build_runtime_mode_by_name(mode_name: str) -> RuntimeMode:
    """
    Fetch a mode from config.py by name and build its concrete runtime form.
    """
    mode = get_mode_by_name(mode_name)
    return build_runtime_mode(mode)


def get_all_runtime_modes(enabled_only: bool = True) -> List[RuntimeMode]:
    """
    Build all configured modes into runtime-ready objects.
    """
    modes = CONFIG.modes
    if enabled_only:
        modes = [mode for mode in modes if mode.enabled]

    return [build_runtime_mode(mode) for mode in modes]


# =============================================================================
# Hybrid modes
# =============================================================================

def build_hybrid_mode(
    name: str,
    base_mode_name: str,
    extra_flags: Dict[str, Any],
    description: Optional[str] = None,
    primary_phase: Optional[str] = None,
) -> RuntimeMode:
    """
    Build a hybrid mode by starting from an existing configured mode and
    overlaying additional runtime flags.

    Example use:
        build_hybrid_mode(
            name="awq_plus_kv_cache",
            base_mode_name="awq_4bit",
            extra_flags={"kv_cache_compression": True},
            description="4-bit AWQ with KV-cache compression",
            primary_phase="decode",
        )
    """
    base_mode = get_mode_by_name(base_mode_name)
    hybrid_cfg = replace(base_mode, extra_args=dict(base_mode.extra_args))

    hybrid_cfg.name = name
    hybrid_cfg.description = description or f"Hybrid mode based on {base_mode_name}"
    if primary_phase is not None:
        hybrid_cfg.primary_phase = primary_phase

    # Treat known keys as high-level mode toggles; everything else becomes an
    # engine override in extra_args.
    high_level_keys = {
        "speculative_decoding",
        "kv_cache_compression",
        "prefix_caching",
        "chunked_prefill",
        "continuous_batching",
        "cuda_graphs",
        "dtype",
        "quantization",
        "backend",
    }
    for key, value in extra_flags.items():
        if key in high_level_keys:
            setattr(hybrid_cfg, key, value)
        else:
            hybrid_cfg.extra_args[key] = value

    return build_runtime_mode(hybrid_cfg)


# =============================================================================
# Convenience hybrid registry
# =============================================================================

def get_default_hybrid_modes() -> List[RuntimeMode]:
    """
    Return a small set of optional hybrid modes that may be useful later.
    Keep these out of the main config until you actually want to benchmark them.
    """
    hybrids: List[RuntimeMode] = []

    hybrids.append(
        build_hybrid_mode(
            name="awq_plus_chunked_prefill",
            base_mode_name="awq_4bit",
            extra_flags={"chunked_prefill": True, "max_num_batched_tokens": 1024},
            description="4-bit AWQ with chunked prefill",
            primary_phase="both",
        )
    )

    return hybrids