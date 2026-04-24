"""
model_loader.py

Model/backend loading utilities for ModeSwitch-LLM.

Purpose:
- load the tokenizer and inference backend for a given RuntimeMode
- keep backend-specific loading logic isolated from benchmarking code
- provide a standard loaded object that runner.py can consume

Current design:
- starts with vLLM as the primary supported backend
- keeps room for Transformers / TGI later
"""

from dataclasses import dataclass
from typing import Any, Optional

import gc
import os

from config import CONFIG
from modes import RuntimeMode

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoConfig = None

AsyncEngineArgs = None
AsyncVLLMEngine = None

try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# Standard loaded bundle
# =============================================================================

@dataclass
class LoadedModelBundle:
    """
    Standard container returned by the loader.

    This lets runner.py interact with a unified object instead of worrying
    about backend-specific return types.
    """

    mode_name: str
    backend: str
    tokenizer: Any
    model: Any
    runtime_mode: RuntimeMode
    hf_model_config: Any = None


# =============================================================================
# Tokenizer loading
# =============================================================================
def _resolve_hf_token() -> Optional[str]:
    """
    Resolve a Hugging Face token from the configured environment variable.
    """
    env_var = CONFIG.model.hf_token_env_var
    if not env_var:
        return None
    token = os.environ.get(env_var)
    return token or None


def _get_vllm_llm_class():
    """
    Lazily import the vLLM async engine classes.
    """
    global AsyncEngineArgs, AsyncVLLMEngine

    if AsyncEngineArgs is None or AsyncVLLMEngine is None:
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs as _AsyncEngineArgs
        except ImportError as exc:
            raise ImportError(
                "vllm async engine args are unavailable. Please check the installed vLLM version."
            ) from exc

        try:
            from vllm.engine.async_llm_engine import AsyncLLMEngine as _AsyncVLLMEngine
        except ImportError as exc:
            raise ImportError(
                "AsyncLLMEngine is unavailable in this vLLM installation."
            ) from exc

        AsyncEngineArgs = _AsyncEngineArgs
        AsyncVLLMEngine = _AsyncVLLMEngine

    return AsyncEngineArgs, AsyncVLLMEngine


def _resolve_model_name_or_path(runtime_mode: RuntimeMode) -> str:
    """
    Resolve the actual model path for a runtime mode.
    """
    override = runtime_mode.runtime_kwargs.get("model_name_or_path")
    return override or CONFIG.model.model_name_or_path


def _resolve_tokenizer_name_or_path(runtime_mode: RuntimeMode, model_name_or_path: str) -> str:
    """
    Resolve the tokenizer path.

    Keep tokenizer selection stable across quantized checkpoints unless the mode
    explicitly overrides it.
    """
    override = runtime_mode.runtime_kwargs.get("tokenizer_name_or_path")
    if override:
        return override
    if CONFIG.model.tokenizer_name_or_path is not None:
        return CONFIG.model.tokenizer_name_or_path
    return model_name_or_path



def load_tokenizer(tokenizer_path: Optional[str] = None):
    """
    Load the tokenizer specified in the global config.

    Returns:
        A tokenizer object compatible with the selected model.
    """
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is not installed, so AutoTokenizer cannot be loaded."
        )

    if tokenizer_path is None:
        tokenizer_path = (
            CONFIG.model.tokenizer_name_or_path
            if CONFIG.model.tokenizer_name_or_path is not None
            else CONFIG.model.model_name_or_path
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=CONFIG.model.trust_remote_code,
        token=_resolve_hf_token(),
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# =============================================================================
# Backend-specific loaders
# =============================================================================

def _load_vllm_model(runtime_mode: RuntimeMode):
    """
    Load a vLLM async engine for the provided runtime mode.

    Notes:
    - We use AsyncLLMEngine rather than offline LLM so the benchmark runner
      can measure streamed TTFT/TBT from real incremental outputs.
    """
    AsyncEngineArgsClass, AsyncVLLMEngineClass = _get_vllm_llm_class()

    model_name_or_path = _resolve_model_name_or_path(runtime_mode)
    tokenizer_name_or_path = _resolve_tokenizer_name_or_path(
        runtime_mode,
        model_name_or_path=model_name_or_path,
    )

    engine_kwargs = {
        "model": model_name_or_path,
        "tokenizer": tokenizer_name_or_path,
        "trust_remote_code": CONFIG.model.trust_remote_code,
        "max_model_len": CONFIG.model.max_model_len,
        "tensor_parallel_size": CONFIG.model.tensor_parallel_size,
        "gpu_memory_utilization": CONFIG.model.gpu_memory_utilization,
        "swap_space": CONFIG.model.swap_space_gb,
        "max_num_batched_tokens": CONFIG.model.max_num_batched_tokens,
        "max_num_seqs": CONFIG.model.max_num_seqs,
        "disable_log_stats": True,
    }

    # Merge in mode-specific runtime kwargs
    engine_kwargs.update(
        {
            key: value
            for key, value in runtime_mode.runtime_kwargs.items()
            if key not in {"model_name_or_path", "tokenizer_name_or_path"}
        }
    )

    engine_args = AsyncEngineArgsClass(**engine_kwargs)
    model = AsyncVLLMEngineClass.from_engine_args(engine_args)
    return model


def _load_transformers_model(runtime_mode: RuntimeMode):
    """
    Load a Hugging Face Transformers causal LM for the provided runtime mode.
    """
    if AutoModelForCausalLM is None:
        raise ImportError(
            "transformers is not installed, so the Transformers backend cannot be used."
        )

    model_kwargs = {
        "trust_remote_code": CONFIG.model.trust_remote_code,
    }

    # Map dtype string from config/runtime mode to torch dtype
    dtype_str = runtime_mode.dtype or CONFIG.model.baseline_dtype
    if torch is not None:
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_str in dtype_map:
            model_kwargs["torch_dtype"] = dtype_map[dtype_str]

    # Do not fake quantization in the legacy Transformers path.
    if runtime_mode.quantization is not None:
        raise NotImplementedError(
            f"Transformers quantization path for '{runtime_mode.quantization}' "
            "is intentionally disabled for the main benchmark path. Use backend='vllm'."
        )

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.model.model_name_or_path,
        **model_kwargs,
    )

    if torch is not None and torch.cuda.is_available() and CONFIG.model.device.startswith("cuda"):
        model = model.to(CONFIG.model.device)

    model.eval()
    return model


def _load_tgi_model(runtime_mode: RuntimeMode):
    """
    Placeholder for TGI backend support.

    TGI is usually served as a separate process/service, so this may later
    become a client connector rather than a direct model loader.
    """
    raise NotImplementedError(
        "TGI backend loader is not implemented yet."
    )


# =============================================================================
# Public loading API
# =============================================================================

def load_model_for_mode(runtime_mode: RuntimeMode) -> LoadedModelBundle:
    """
    Load tokenizer + backend model for the given runtime mode.

    Args:
        runtime_mode: A concrete runtime-ready mode object from modes.py

    Returns:
        LoadedModelBundle containing tokenizer, model, and metadata
    """
    hf_model_config = None

    if runtime_mode.backend == "vllm":
        model_name_or_path = _resolve_model_name_or_path(runtime_mode)
        tokenizer_name_or_path = _resolve_tokenizer_name_or_path(
            runtime_mode,
            model_name_or_path=model_name_or_path,
        )
        tokenizer = load_tokenizer(tokenizer_name_or_path)
        if AutoConfig is not None:
            try:
                hf_model_config = AutoConfig.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=CONFIG.model.trust_remote_code,
                    token=_resolve_hf_token(),
                )
            except Exception:
                hf_model_config = None
        model = _load_vllm_model(runtime_mode)
    elif runtime_mode.backend == "transformers":
        tokenizer = load_tokenizer()
        if AutoConfig is not None:
            try:
                hf_model_config = AutoConfig.from_pretrained(
                    CONFIG.model.model_name_or_path,
                    trust_remote_code=CONFIG.model.trust_remote_code,
                    token=_resolve_hf_token(),
                )
            except Exception:
                hf_model_config = None
        model = _load_transformers_model(runtime_mode)
    elif runtime_mode.backend == "tgi":
        tokenizer = load_tokenizer()
        model = _load_tgi_model(runtime_mode)
    else:
        raise ValueError(f"Unsupported backend: {runtime_mode.backend}")

    return LoadedModelBundle(
        mode_name=runtime_mode.name,
        backend=runtime_mode.backend,
        tokenizer=tokenizer,
        model=model,
        runtime_mode=runtime_mode,
        hf_model_config=hf_model_config,
    )


# =============================================================================
# Cleanup utilities
# =============================================================================

def unload_model(bundle: Optional[LoadedModelBundle]) -> None:
    """
    Best-effort cleanup for a loaded model bundle.

    This is especially useful when sweeping many modes in one script.
    """
    if bundle is None:
        return

    try:
        shutdown_background_loop = getattr(bundle.model, "shutdown_background_loop", None)
        if callable(shutdown_background_loop):
            shutdown_background_loop()
    except Exception:
        pass

    try:
        shutdown = getattr(bundle.model, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass

    try:
        del bundle.model
    except Exception:
        pass

    try:
        del bundle.tokenizer
    except Exception:
        pass

    gc.collect()

    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass