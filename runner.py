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

from typing import List, Optional
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


def _count_output_tokens_for_texts(tokenizer, output_texts: List[str]) -> Optional[int]:
    """
    Count generated tokens across one or more outputs.
    """
    counts = []
    for text in output_texts:
        count = _count_output_tokens(tokenizer, text)
        if count is None:
            return None
        counts.append(count)
    return sum(counts)


def _sanitize_output_text(text: str) -> str:
    """
    Remove simple special-token artifacts that can still appear in text output.
    """
    return (text or "").replace("</s>", "").strip()


def _build_warmup_prompts(batch_size: int = 1) -> List[str]:
    """
    Build generic warmup prompts so the warmup does not pollute prefix-cache
    experiments with the real benchmark prompt.
    """
    prompts = []
    for i in range(batch_size):
        prompts.append(
            f"Warmup request {i}. Reply with the single word OK."
        )
    return prompts


def _build_batched_prompts(base_prompt: str, batch_size: int) -> List[str]:
    """
    Create a small batch of near-identical prompts for multi-request vLLM runs.
    """
    prompts = []
    for i in range(batch_size):
        prompts.append(
            base_prompt
            + f"\n\nBenchmark instance id: {i}\n"
              "Answer normally, but keep the content specific to this instance id."
        )
    return prompts

def _run_vllm_generate(
    bundle: LoadedModelBundle,
    workload: RuntimeWorkload,
    prompts: Optional[List[str]] = None,
    max_new_tokens_override: Optional[int] = None,
):
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

    if prompts is None:
        prompts = [workload.prompt]

    max_new_tokens = (
        max_new_tokens_override
        if max_new_tokens_override is not None
        else workload.max_new_tokens
    )

    sampling_kwargs = {
        "max_tokens": max_new_tokens,
        "temperature": CONFIG.generation.temperature if CONFIG.generation.do_sample else 0.0,
        "top_p": CONFIG.generation.top_p,
        "repetition_penalty": CONFIG.generation.repetition_penalty,
    }
    if CONFIG.generation.top_k > 0:
        sampling_kwargs["top_k"] = CONFIG.generation.top_k
    if CONFIG.generation.stop_sequences:
        sampling_kwargs["stop"] = CONFIG.generation.stop_sequences

    sampling_params = SamplingParams(**sampling_kwargs)

    outputs = bundle.model.generate(prompts, sampling_params=sampling_params)

    end_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)

    output_texts: List[str] = []
    for request_output in outputs:
        text = ""
        if request_output.outputs and len(request_output.outputs) > 0:
            text = request_output.outputs[0].text
        output_texts.append(_sanitize_output_text(text))

    # Since this path is non-streaming, we do not have exact first-token timestamps yet.
    # Keep TTFT/TBT as approximate placeholders only.
    first_token_time_s = end_time_s

    # No per-token timestamps yet in this basic implementation
    token_timestamps_s = []

    return output_texts, first_token_time_s, token_timestamps_s, end_time_s

def _run_transformers_generate(
    bundle: LoadedModelBundle,
    workload: RuntimeWorkload,
    prompts: Optional[List[str]] = None,
    max_new_tokens_override: Optional[int] = None,
):
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

    if prompts is None:
        prompts = [workload.prompt]
    if len(prompts) != 1:
        raise NotImplementedError(
            "The legacy Transformers streamer path only supports one prompt at a time."
        )

    prompt = prompts[0]
    max_new_tokens = (
        max_new_tokens_override
        if max_new_tokens_override is not None
        else workload.max_new_tokens
    )

    tokenizer = bundle.tokenizer
    model = bundle.model

    encoded = tokenizer(
        prompt,
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
        "max_new_tokens": max_new_tokens,
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
    output_text = _sanitize_output_text(output_text)

    # Fallback in case streamer yields nothing useful.
    if first_token_time_s is None:
        first_token_time_s = end_time_s

    return [output_text], first_token_time_s, token_timestamps_s, end_time_s

def _run_generation(
    bundle: LoadedModelBundle,
    workload: RuntimeWorkload,
    prompts: Optional[List[str]] = None,
    max_new_tokens_override: Optional[int] = None,
):
    """
    Dispatch generation to the correct backend-specific implementation.
    """
    if bundle.backend == "vllm":
        return _run_vllm_generate(
            bundle,
            workload,
            prompts=prompts,
            max_new_tokens_override=max_new_tokens_override,
        )
    if bundle.backend == "transformers":
        return _run_transformers_generate(
            bundle,
            workload,
            prompts=prompts,
            max_new_tokens_override=max_new_tokens_override,
        )
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
    preloaded_bundle: Optional[LoadedModelBundle] = None,
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
    bundle: Optional[LoadedModelBundle] = preloaded_bundle
    owns_bundle = preloaded_bundle is None

    result = BenchmarkResult(
        mode_name=runtime_mode.name,
        workload_name=workload.name,
        backend=runtime_mode.backend,
        trial_index=trial_index,
        prompt_tokens_target=workload.prompt_tokens_target,
        max_new_tokens=workload.max_new_tokens,
        repeated_prefix=workload.repeated_prefix,
        memory_pressure=workload.memory_pressure,
        num_requests_in_batch=1,
    )

    try:
        # Reset peak GPU memory tracking before the run
        reset_gpu_peak_memory_stats()

        # Load model/tokenizer for this mode
        if bundle is None:
            bundle = load_model_for_mode(runtime_mode)

        request_batch_size = int(runtime_mode.runner_kwargs.get("request_batch_size", 1))

        # Optional warmup runs with unrelated prompts so we do not pollute
        # repeated-prefix timing with real cached prefixes.
        for _ in range(CONFIG.system.warmup_runs):
            warmup_prompts = _build_warmup_prompts(
                batch_size=request_batch_size if runtime_mode.continuous_batching else 1
            )
            _run_generation(
                bundle,
                workload,
                prompts=warmup_prompts,
                max_new_tokens_override=min(8, workload.max_new_tokens),
            )

        priming_prompts = None
        timed_prompts = [workload.prompt]

        # Prefix-caching experiments only make sense when the follow-up request
        # is timed on the same engine instance after priming a shared prefix.
        if workload.repeated_prefix and workload.followup_prompt is not None:
            priming_prompts = [workload.prompt]
            timed_prompts = [workload.followup_prompt]

        # "Continuous batching" in vLLM is exercised by batching multiple
        # requests through one engine invocation, not by a fake constructor flag.
        if runtime_mode.continuous_batching:
            if priming_prompts is not None:
                priming_prompts = _build_batched_prompts(priming_prompts[0], request_batch_size)
            timed_prompts = _build_batched_prompts(timed_prompts[0], request_batch_size)

        result.num_requests_in_batch = len(timed_prompts)

        if priming_prompts is not None:
            _run_generation(
                bundle,
                workload,
                prompts=priming_prompts,
                max_new_tokens_override=1,
            )
            result.notes += (
                "Timed follow-up request after priming a shared-prefix request on the same engine instance. "
            )

        # Actual timed run
        result.start_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)

        output_texts, first_token_time_s, token_timestamps_s, end_time_s = _run_generation(
            bundle=bundle,
            workload=workload,
            prompts=timed_prompts,
        )

        result.first_token_time_s = first_token_time_s
        result.end_time_s = end_time_s
        result.token_timestamps_s = token_timestamps_s
        result.output_text = "\n\n===== REQUEST SPLIT =====\n\n".join(output_texts)
        result.output_tokens_generated = _count_output_tokens_for_texts(
            bundle.tokenizer,
            output_texts,
        )

        if runtime_mode.backend == "vllm":
            result.notes += (
                "Primary trusted metrics for this run: total latency, memory, output tokens, "
                "and rough throughput. TTFT/TBT are approximate in the current offline vLLM path. "
            )
        elif runtime_mode.backend == "transformers":
            result.notes += (
                "Transformers streamer path uses chunk-level arrival timing; "
                "TTFT/TBT are approximate. "
            )

        if runtime_mode.continuous_batching:
            result.notes += (
                f"Measured a {len(timed_prompts)}-request batched generate() call on one vLLM engine instance. "
            )

        # Finalize derived metrics
        result = finalize_benchmark_result(result)

    except Exception as exc:
        result.error = str(exc)
        result.notes = "Benchmark run failed."

    finally:
        if owns_bundle:
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