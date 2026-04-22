"""
runner.py

Execution utilities for ModeSwitch-LLM.

Purpose:
- run one benchmark trial for one (mode, workload) pair
- coordinate model loading, prompt construction, generation, and metric capture
- return a finalized BenchmarkResult object

Design principle:
runner.py = execution orchestration
metrics.py = metric computation
"""

from typing import Optional
from threading import Thread
from config import CONFIG
from modes import RuntimeMode, build_runtime_mode_by_name
from model_loader import LoadedModelBundle, load_model_for_mode, unload_model
from workloads import RuntimeWorkload, build_runtime_workload_by_name
from metrics import (
    BenchmarkResult,
    finalize_benchmark_result,
    now_s,
    reset_gpu_peak_memory_stats,
)

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import TextIteratorStreamer
except ImportError:
    TextIteratorStreamer = None

# =============================================================================
# Helper functions
# =============================================================================

def _count_output_tokens(tokenizer, output_text: str) -> Optional[int]:
    """
    Count output tokens using the tokenizer, if available.
    """
    if tokenizer is None or output_text is None:
        return None

    try:
        token_ids = tokenizer.encode(output_text, add_special_tokens=False)
        return len(token_ids)
    except Exception:
        return None


def _run_vllm_generate(bundle: LoadedModelBundle, workload: RuntimeWorkload):
    """
    Run generation using the vLLM backend.

    Returns:
        output_text, first_token_time_s, token_timestamps_s, end_time_s

    Notes:
    - This first version uses a simple non-streaming path.
    - Exact TTFT/TBT may need a streaming-compatible backend path later.
    - For now, first_token_time_s is approximated conservatively.
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=workload.max_new_tokens,
        temperature=CONFIG.generation.temperature,
        top_p=CONFIG.generation.top_p,
        repetition_penalty=CONFIG.generation.repetition_penalty,
    )

    # Start generation
    start_generation_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)

    outputs = bundle.model.generate(
        [workload.prompt],
        sampling_params=sampling_params,
    )

    end_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)

    # vLLM returns a list of request outputs
    output_text = ""
    if outputs and len(outputs) > 0:
        if outputs[0].outputs and len(outputs[0].outputs) > 0:
            output_text = outputs[0].outputs[0].text

    output_text = output_text.replace("</s>", "").strip()

    # Since this path is non-streaming, we do not have exact first-token timestamps yet.
    # Keep TTFT/TBT as approximate placeholders only.
    first_token_time_s = end_time_s

    # No per-token timestamps yet in this basic implementation
    token_timestamps_s = []

    return output_text, first_token_time_s, token_timestamps_s, end_time_s

def _run_transformers_generate(bundle: LoadedModelBundle, workload: RuntimeWorkload):
    """
    Run generation using the Hugging Face Transformers backend.

    Notes:
    - This uses a streamer so TTFT and TBT are based on actual
      streamed output arrival times
    """
    if TextIteratorStreamer is None:
        raise ImportError(
            "transformers TextIteratorStreamer is unavailable. Please update transformers."
        )

    tokenizer = bundle.tokenizer
    model = bundle.model

    encoded = tokenizer(
        workload.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CONFIG.model.max_model_len,
    )

    if torch is not None and torch.cuda.is_available() and CONFIG.model.device.startswith("cuda"):
        encoded = {k: v.to(CONFIG.model.device) for k, v in encoded.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generate_kwargs = {
        **encoded,
        "max_new_tokens": workload.max_new_tokens,
        "do_sample": CONFIG.generation.do_sample,
        "repetition_penalty": CONFIG.generation.repetition_penalty,
        "num_return_sequences": CONFIG.generation.num_return_sequences,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    # Only include sampling params when sampling is enabled.
    if CONFIG.generation.do_sample:
        generate_kwargs["temperature"] = CONFIG.generation.temperature
        generate_kwargs["top_p"] = CONFIG.generation.top_p
        if CONFIG.generation.top_k > 0:
            generate_kwargs["top_k"] = CONFIG.generation.top_k

    start_generation_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)

    generation_thread = Thread(
        target=model.generate,
        kwargs=generate_kwargs,
        daemon=True,
    )
    generation_thread.start()

    output_chunks = []
    token_timestamps_s = []
    first_token_time_s = None

    for new_text in streamer:
        current_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)
        output_chunks.append(new_text)

        # Record first arrival time once.
        if first_token_time_s is None and new_text.strip() != "":
            first_token_time_s = current_time_s

        # Record timestamp for each non-empty streamed chunk.
        if new_text.strip() != "":
            token_timestamps_s.append(current_time_s)

    generation_thread.join()
    end_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)

    output_text = "".join(output_chunks)

    # Safety cleanup: remove any leftover special end-of-sequence markers
    # that may still appear in streamed text on some models/setups.
    output_text = output_text.replace("</s>", "").strip()

    # Fallback in case streamer yields nothing useful.
    if first_token_time_s is None:
        first_token_time_s = end_time_s

    return output_text, first_token_time_s, token_timestamps_s, end_time_s

def _run_generation(bundle: LoadedModelBundle, workload: RuntimeWorkload):
    """
    Dispatch generation to the correct backend-specific implementation.
    """
    if bundle.backend == "vllm":
        return _run_vllm_generate(bundle, workload)
    if bundle.backend == "transformers":
        return _run_transformers_generate(bundle, workload)
    raise NotImplementedError(
        f"Generation runner for backend '{bundle.backend}' is not implemented yet."
    )


# =============================================================================
# Public benchmark runner
# =============================================================================

def run_single_benchmark(
    runtime_mode: RuntimeMode,
    workload: RuntimeWorkload,
    trial_index: int = 0,
) -> BenchmarkResult:
    """
    Run one benchmark trial for one runtime mode and one runtime workload.

    Args:
        runtime_mode: Concrete mode object from modes.py
        workload: Concrete workload object from workloads.py
        trial_index: Which trial number this is

    Returns:
        Finalized BenchmarkResult
    """
    bundle: Optional[LoadedModelBundle] = None

    result = BenchmarkResult(
        mode_name=runtime_mode.name,
        workload_name=workload.name,
        backend=runtime_mode.backend,
        trial_index=trial_index,
        prompt_tokens_target=workload.prompt_tokens_target,
        max_new_tokens=workload.max_new_tokens,
        repeated_prefix=workload.repeated_prefix,
        memory_pressure=workload.memory_pressure,
    )

    try:
        # Reset peak GPU memory tracking before the run
        reset_gpu_peak_memory_stats()

        # Load model/tokenizer for this mode
        bundle = load_model_for_mode(runtime_mode)

        # Optional warmup runs
        for _ in range(CONFIG.system.warmup_runs):
            _run_generation(bundle, workload)

        # Actual timed run
        result.start_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)

        output_text, first_token_time_s, token_timestamps_s, end_time_s = _run_generation(
            bundle=bundle,
            workload=workload,
        )

        result.first_token_time_s = first_token_time_s
        result.end_time_s = end_time_s
        result.token_timestamps_s = token_timestamps_s
        result.output_text = output_text
        result.output_tokens_generated = _count_output_tokens(bundle.tokenizer, output_text)

        if runtime_mode.backend == "vllm":
            result.notes = (
                "Primary trusted metrics for this run: total latency, memory, output tokens, "
                "and rough throughput. TTFT/TBT are approximate in the current offline vLLM path."
            )
        elif runtime_mode.backend == "transformers":
            result.notes = (
                "Transformers streamer path uses chunk-level arrival timing; "
                "TTFT/TBT are approximate."
            )

        # Finalize derived metrics
        result = finalize_benchmark_result(result)

    except Exception as exc:
        result.error = str(exc)
        result.notes = "Benchmark run failed."

    finally:
        unload_model(bundle)

    return result


# =============================================================================
# Convenience wrappers
# =============================================================================

def run_single_benchmark_by_name(
    mode_name: str,
    workload_name: str,
    trial_index: int = 0,
):
    """
    Convenience wrapper to run one benchmark using names instead of objects.
    """
    runtime_mode = build_runtime_mode_by_name(mode_name)
    runtime_workload = build_runtime_workload_by_name(workload_name)

    return run_single_benchmark(
        runtime_mode=runtime_mode,
        workload=runtime_workload,
        trial_index=trial_index,
    )