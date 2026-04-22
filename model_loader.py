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

from config import CONFIG
from modes import RuntimeMode

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    from vllm import LLM
except ImportError:
    LLM = None

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


# =============================================================================
# Tokenizer loading
# =============================================================================

def load_tokenizer():
    """
    Load the tokenizer specified in the global config.

    Returns:
        A tokenizer object compatible with the selected model.
    """
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is not installed, so AutoTokenizer cannot be loaded."
        )

    tokenizer_path = (
        CONFIG.model.tokenizer_name_or_path
        if CONFIG.model.tokenizer_name_or_path is not None
        else CONFIG.model.model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=CONFIG.model.trust_remote_code,
    )

    return tokenizer


# =============================================================================
# Backend-specific loaders
# =============================================================================

def _load_vllm_model(runtime_mode: RuntimeMode):
    """
    Load a vLLM model for the provided runtime mode.

    Notes:
    - The exact kwargs here may need adjustment depending on your installed
      vLLM version and which features are supported in Python API vs CLI.
    - For now, we pass through the runtime kwargs and keep the loader modular.
    """
    if LLM is None:
        raise ImportError("vllm is not installed, so the vLLM backend cannot be used.")

    llm_kwargs = {
        "model": CONFIG.model.model_name_or_path,
        "trust_remote_code": CONFIG.model.trust_remote_code,
        "max_model_len": CONFIG.model.max_model_len,
    }

    # Merge in mode-specific runtime kwargs
    if runtime_mode.runtime_kwargs is not None:
        llm_kwargs.update(runtime_mode.runtime_kwargs)

    model = LLM(**llm_kwargs)
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

    # Very basic quantization placeholders for future extension
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
    tokenizer = load_tokenizer()

    if runtime_mode.backend == "vllm":
        model = _load_vllm_model(runtime_mode)
    elif runtime_mode.backend == "transformers":
        model = _load_transformers_model(runtime_mode)
    elif runtime_mode.backend == "tgi":
        model = _load_tgi_model(runtime_mode)
    else:
        raise ValueError(f"Unsupported backend: {runtime_mode.backend}")

    return LoadedModelBundle(
        mode_name=runtime_mode.name,
        backend=runtime_mode.backend,
        tokenizer=tokenizer,
        model=model,
        runtime_mode=runtime_mode,
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