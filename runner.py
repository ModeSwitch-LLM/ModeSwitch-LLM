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
import asyncio
import logging
import re
import uuid
import time
import warnings
from typing import List, Optional
from threading import Thread, Event
from config import CONFIG
from controller import route_runtime_workload
from modes import RuntimeMode, build_runtime_mode_by_name
from model_loader import LoadedModelBundle, load_model_for_mode, unload_model
from workloads import RuntimeWorkload, build_runtime_workload_by_name
from metrics import (
    BenchmarkResult,
    compute_benchmark_suite_metrics,
    compute_exact_match,
    compute_rouge_l_f1,
    compute_token_f1,
    finalize_benchmark_result,
    get_current_gpu_allocated_mb,
    get_process_ram_mb,
    get_reserved_gpu_memory_mb,
    now_s,
    reset_gpu_peak_memory_stats,
)

try:
    import torch
except ImportError:
    torch = None

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    from transformers import TextIteratorStreamer
except ImportError:
    TextIteratorStreamer = None

# =============================================================================
# Helper functions
# =============================================================================
class EnergyPoller:
    """
    Lightweight background GPU power poller.

    This is adapted from Ali's phase_monitor.py, but simplified to fit the
    current runner architecture. It is used only for total timed-run energy /
    average power collection.
    """

    def __init__(self, poll_interval_s: float, device_index: int = 0):
        self._poll_interval_s = poll_interval_s
        self._device_index = device_index
        self._available = False
        self._samples_w = []
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._nvml_initialized = False
        self._handle = None

        if (
            pynvml is None
            or torch is None
            or not torch.cuda.is_available()
            or not CONFIG.system.collect_energy_stats
        ):
            return

        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
            else:
                current_device = device_index
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(current_device)
            self._available = True
        except Exception:
            self._handle = None
            self._available = False

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._samples_w.append((time.perf_counter(), power_mw / 1000.0))
            except Exception:
                pass
            self._stop_event.wait(self._poll_interval_s)

    def __enter__(self) -> "EnergyPoller":
        if not self._available:
            return self
        self._samples_w.clear()
        self._stop_event.clear()
        self._thread = Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._available:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def get_total_energy_joules(self) -> Optional[float]:
        """
        Integrate sampled power over time using trapezoids.
        """
        if not self._available or len(self._samples_w) < 2:
            return None

        total_j = 0.0
        for i in range(1, len(self._samples_w)):
            t0, w0 = self._samples_w[i - 1]
            t1, w1 = self._samples_w[i]
            dt = t1 - t0
            total_j += 0.5 * (w0 + w1) * dt
        return total_j

    def get_mean_power_watts(self) -> Optional[float]:
        if not self._available or not self._samples_w:
            return None
        return sum(w for _, w in self._samples_w) / len(self._samples_w)


class MemoryPressureContext:
    """
    Allocate a dummy CUDA tensor to consume a fraction of currently free VRAM.

    Adapted from Ali's phase_monitor.py and simplified for the current runner.
    """

    def __init__(self, vram_fraction: float = 0.0, device: str = "cuda"):
        self._fraction = max(0.0, min(float(vram_fraction), 0.9))
        self._device = device
        self._pressure_tensor = None
        self._stats = {}

    def __enter__(self) -> "MemoryPressureContext":
        if self._fraction <= 0.0 or torch is None or not torch.cuda.is_available():
            return self

        try:
            total_bytes = torch.cuda.get_device_properties(self._device).total_memory
            allocated_bytes = torch.cuda.memory_allocated(self._device)
            free_bytes = max(0, total_bytes - allocated_bytes)
            target_bytes = int(free_bytes * self._fraction)

            self._stats["free_before_gb"] = free_bytes / (1024 ** 3)
            self._stats["target_allocation_gb"] = target_bytes / (1024 ** 3)

            if target_bytes <= 0:
                return self

            # float16 -> 2 bytes per element
            n_elements = max(1, target_bytes // 2)
            self._pressure_tensor = torch.empty(
                n_elements,
                dtype=torch.float16,
                device=self._device,
            )
            self._pressure_tensor.fill_(0.0)

            actual_allocated_bytes = (
                self._pressure_tensor.element_size() * self._pressure_tensor.numel()
            )
            free_after_bytes = max(
                0,
                total_bytes - torch.cuda.memory_allocated(self._device),
            )

            self._stats["actual_allocated_gb"] = actual_allocated_bytes / (1024 ** 3)
            self._stats["free_after_gb"] = free_after_bytes / (1024 ** 3)
        except torch.cuda.OutOfMemoryError:
            self._pressure_tensor = None
            self._stats["allocation_error"] = "OutOfMemoryError"
        except Exception as exc:
            self._pressure_tensor = None
            self._stats["allocation_error"] = str(exc)

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._pressure_tensor is not None:
            try:
                del self._pressure_tensor
            except Exception:
                pass
            self._pressure_tensor = None

        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def get_stats(self) -> dict:
        return dict(self._stats)


def _resolve_memory_pressure_fraction(workload: RuntimeWorkload) -> float:
    """
    Infer the desired artificial VRAM pressure fraction from workload metadata.

    Priority:
    1. explicit workload.memory_pressure_fraction attribute if present
    2. workload.system_condition_name naming convention
    3. fallback for boolean workload.memory_pressure
    """
    explicit_fraction = getattr(workload, "memory_pressure_fraction", None)
    if explicit_fraction is not None:
        try:
            return max(0.0, min(float(explicit_fraction), 0.9))
        except Exception:
            pass

    system_condition_name = str(getattr(workload, "system_condition_name", "") or "").lower()
    if "75" in system_condition_name:
        return 0.75
    if "50" in system_condition_name:
        return 0.50

    if getattr(workload, "memory_pressure", False):
        return 0.50

    return 0.0

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

def _mean_optional(values: List[Optional[float]]) -> Optional[float]:
    """
    Mean over non-None numeric values.
    """
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _apply_mean_benchmark_metric_dicts(
    result: BenchmarkResult,
    metric_dicts: List[dict],
) -> None:
    """
    Aggregate per-output benchmark metric dicts back onto one BenchmarkResult.
    """
    if not metric_dicts:
        return

    primary_metric_names = [
        d.get("benchmark_primary_metric_name")
        for d in metric_dicts
        if d.get("benchmark_primary_metric_name")
    ]
    if primary_metric_names:
        result.benchmark_primary_metric_name = primary_metric_names[0]

    numeric_metric_fields = [
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

    for field_name in numeric_metric_fields:
        setattr(
            result,
            field_name,
            _mean_optional([metric_dict.get(field_name) for metric_dict in metric_dicts]),
        )

def _sanitize_output_text(text: str) -> str:
    """
    Remove simple special-token artifacts that can still appear in text output.
    """
    cleaned = text or ""

    # Remove common full special-token remnants first.
    cleaned = cleaned.replace("</s>", "").replace("<s>", "")

    # Sometimes streamed / decoded text leaves a truncated leading fragment
    # like "s>" after stripping a broken "</s>" boundary.
    cleaned = re.sub(r"^(?:\s*(?:s>|/s>|</s>|<s>))+", "", cleaned)

    return cleaned.strip()


def _configure_quieter_runtime_logs() -> None:
    """
    Reduce noisy benchmark-time logging from upstream libraries.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*torch_dtype.*deprecated.*dtype instead.*",
    )

    # Hide very verbose vLLM startup/info logs while still letting actual
    # exceptions propagate back through the benchmark result.
    logging.getLogger("vllm").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)


def _filter_reporting_token_ids(tokenizer, token_ids: List[int]) -> List[int]:
    """
    Filter token ids to the subset we want to count/report.

    We exclude tokenizer special tokens (EOS/BOS/PAD/etc.) so the streamed
    token count lines up with the human-visible decoded output.
    """
    if tokenizer is None:
        return list(token_ids)

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    return [token_id for token_id in token_ids if token_id not in special_ids]


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

def _should_add_trial_unique_header(workload: RuntimeWorkload) -> bool:
    """
    Decide whether to make this timed prompt unique for this trial.

    We only do this for synthetic non-benchmark, non-shared-prefix workloads.
    This avoids accidental cross-trial prefix-cache reuse while preserving the
    intended shared-prefix benchmark behavior.
    """
    if not getattr(CONFIG.system, "unique_synthetic_prompt_per_trial", False):
        return False

    if getattr(workload, "repeated_prefix", False):
        return False

    if getattr(workload, "benchmark_suite", None):
        return False

    return True


def _add_trial_unique_header(
    prompts: List[str],
    workload: RuntimeWorkload,
    trial_index: int,
) -> List[str]:
    """
    Add a unique header at the beginning of synthetic prompts.

    Important: the header goes at the beginning, not the end, so prefix-caching
    cannot reuse almost the entire repeated synthetic prompt across trials.
    """
    header = (
        f"Unique benchmark trial id: {workload.name} / trial {trial_index}\n"
        "This identifier is part of the benchmark prompt and should not be discussed.\n\n"
    )
    return [header + prompt for prompt in prompts]


def _resolve_system_prompt_for_workload(workload: RuntimeWorkload) -> Optional[str]:
    """
    Provide a benchmark-aware system prompt for instruct/chat models.
    """
    evaluation_mode = str(getattr(workload, "evaluation_mode", "") or "").strip().lower()

    if evaluation_mode in {"multiple_choice", "multiple_choice_accuracy"}:
        return (
            "You are taking a multiple-choice evaluation. "
            "Choose the single best answer and follow the required output format exactly. "
            "Do not add explanations."
        )

    if evaluation_mode == "final_answer_exact_match":
        return (
            "You are solving a math evaluation problem. "
            "Follow the required final answer format exactly."
        )

    return None

def _format_prompt_for_instruct_model(tokenizer, prompt: str, system_prompt: Optional[str] = None):
    """
    Apply the tokenizer chat template when available.

    This is important for instruct models like Llama-3.1-Instruct.
    Without it, the model may behave more like raw text completion.

    Returns:
        (formatted_prompt, used_chat_template)
    """
    if tokenizer is None:
        return prompt, False

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return prompt, False

    # Avoid double-wrapping prompts that already look templated.
    if any(marker in prompt for marker in (
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "[INST]",
        "<s>[INST]",
    )):
        return prompt, False

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted, True
    except Exception:
        return prompt, False

def _estimate_bytes_per_element_from_runtime_mode(runtime_mode: RuntimeMode) -> int:
    """
    Estimate bytes per KV-cache element from the configured KV-cache dtype.
    """
    kv_cache_dtype = runtime_mode.runtime_kwargs.get("kv_cache_dtype")
    if kv_cache_dtype is None or kv_cache_dtype == "auto":
        dtype_label = runtime_mode.runtime_kwargs.get("dtype", runtime_mode.dtype or CONFIG.model.baseline_dtype)
    else:
        dtype_label = kv_cache_dtype

    dtype_label = str(dtype_label).lower()
    if "fp8" in dtype_label:
        return 1
    if "float16" in dtype_label or "fp16" in dtype_label or "bfloat16" in dtype_label or "bf16" in dtype_label:
        return 2
    if "float32" in dtype_label or "fp32" in dtype_label:
        return 4
    return 2


def _estimate_kv_cache_mb(
    bundle: LoadedModelBundle,
    runtime_mode: RuntimeMode,
    prompts: List[str],
    output_tokens_generated: Optional[int],
) -> Optional[float]:
    """
    Estimate KV-cache footprint using a simple transformer-architecture proxy.
    """
    if bundle.tokenizer is None or bundle.hf_model_config is None:
        return None
    if output_tokens_generated is None:
        return None

    cfg = bundle.hf_model_config
    num_hidden_layers = getattr(cfg, "num_hidden_layers", None)
    num_attention_heads = getattr(cfg, "num_attention_heads", None)
    num_key_value_heads = getattr(cfg, "num_key_value_heads", num_attention_heads)
    hidden_size = getattr(cfg, "hidden_size", None)
    head_dim = getattr(cfg, "head_dim", None)

    if num_hidden_layers is None or num_key_value_heads is None:
        return None
    if head_dim is None:
        if hidden_size is None or num_attention_heads in (None, 0):
            return None
        head_dim = hidden_size // num_attention_heads

    try:
        prompt_token_total = 0
        for prompt in prompts:
            formatted_prompt, used_chat_template = _format_prompt_for_instruct_model(
                bundle.tokenizer,
                prompt,
            )
            prompt_token_total += len(
                bundle.tokenizer.encode(
                    formatted_prompt,
                    add_special_tokens=not used_chat_template,
                )
            )
    except Exception:
        return None

    total_cached_tokens = prompt_token_total + output_tokens_generated
    bytes_per_element = _estimate_bytes_per_element_from_runtime_mode(runtime_mode)

    kv_bytes = (
        2
        * num_hidden_layers
        * total_cached_tokens
        * num_key_value_heads
        * head_dim
        * bytes_per_element
    )
    return kv_bytes / (1024 ** 2)


def _monitor_runtime_stats(stop_event: Event, stats: dict) -> None:
    """
    Poll process RAM during the timed section.
    """
    process_ram_peak_mb = None

    try:
        while not stop_event.is_set():
            if CONFIG.system.collect_cpu_memory_stats:
                current_ram_mb = get_process_ram_mb()
                if current_ram_mb is not None:
                    if process_ram_peak_mb is None or current_ram_mb > process_ram_peak_mb:
                        process_ram_peak_mb = current_ram_mb

            stop_event.wait(CONFIG.system.power_poll_interval_s)
    finally:
        stats["cpu_ram_peak_mb"] = process_ram_peak_mb

def _run_asyncio_coroutine_in_thread(coro):
    """
    Run an async coroutine in a dedicated worker thread.

    This avoids notebook / IPython event-loop conflicts with asyncio.run().
    """
    result_box = {}
    error_box = {}

    def _target():
        try:
            result_box["value"] = asyncio.run(coro)
        except Exception as exc:
            error_box["error"] = exc

    worker = Thread(target=_target, daemon=True)
    worker.start()
    worker.join()

    if "error" in error_box:
        raise error_box["error"]

    return result_box["value"]


def _build_vllm_sampling_params(max_new_tokens: int):
    """
    Build vLLM sampling params for a streamed benchmark request.
    """
    from vllm import SamplingParams

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

    # DELTA mode is ideal because each streamed update contains only the newly
    # generated text/token ids. If unavailable in the installed vLLM version,
    # we gracefully fall back and handle cumulative outputs in the stream parser.
    try:
        from vllm.sampling_params import RequestOutputKind
        sampling_kwargs["output_kind"] = RequestOutputKind.DELTA
    except Exception:
        pass

    return SamplingParams(**sampling_kwargs)


async def _stream_single_vllm_request(
    engine,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
):
    """
    Stream one vLLM request and capture real client-observed timing.

    Returns:
        output_text, first_token_time_s, stream_event_timestamps_s, end_time_s, output_token_count
    """
    sampling_params = _build_vllm_sampling_params(max_new_tokens=max_new_tokens)
    request_id = f"benchmark-{uuid.uuid4().hex}"
    request_start_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)

    accumulated_text = ""
    accumulated_token_ids: List[int] = []
    accumulated_token_count = 0
    token_timestamps_s: List[float] = []
    first_token_time_s = None
    end_time_s = None

    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        for completion in output.outputs:
            raw_text = completion.text or ""
            raw_token_ids = list(getattr(completion, "token_ids", []) or [])

            # Distinguish cumulative-vs-delta style text updates.
            is_cumulative_text = bool(accumulated_text) and raw_text.startswith(accumulated_text)

            # Robust handling across DELTA mode and older cumulative-stream behavior.
            if is_cumulative_text:
                new_text = raw_text[len(accumulated_text):]
            else:
                new_text = raw_text

            # Distinguish cumulative-vs-delta token-id updates.
            if is_cumulative_text and len(raw_token_ids) >= accumulated_token_count:
                new_token_ids = raw_token_ids[accumulated_token_count:]
            else:
                new_token_ids = raw_token_ids

            new_token_count = len(new_token_ids)

            if new_text or new_token_count > 0:
                current_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)
                if first_token_time_s is None:
                    first_token_time_s = current_time_s

                # Record one arrival timestamp per streamed update, not one
                # repeated timestamp per token. Repeating the same timestamp for
                # every token in a multi-token chunk creates artificial zero-gap
                # TBT values and understates decode latency dynamics.
                token_timestamps_s.append(current_time_s)

            if new_text:
                accumulated_text += new_text

            if new_token_count > 0:
                accumulated_token_ids.extend(new_token_ids)
                accumulated_token_count += new_token_count

        if output.finished:
            end_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)
            break

    if end_time_s is None:
        end_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)
    if first_token_time_s is None:
        first_token_time_s = end_time_s

    reporting_token_ids = _filter_reporting_token_ids(tokenizer, accumulated_token_ids)
    decoded_text = (
        tokenizer.decode(reporting_token_ids, skip_special_tokens=True)
        if tokenizer is not None and len(reporting_token_ids) > 0
        else accumulated_text
    )

    return (
        _sanitize_output_text(decoded_text),
        first_token_time_s,
        token_timestamps_s,
        end_time_s,
        len(reporting_token_ids) if len(reporting_token_ids) > 0 else None,
        request_start_time_s,
    )


async def _stream_many_vllm_requests(
    engine,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
):
    """
    Stream one or more vLLM requests concurrently on the same engine.

    This is important for the "continuous batching" benchmark: concurrent
    requests on one async engine allow the scheduler to batch work naturally.
    """
    tasks = [
        asyncio.create_task(
            _stream_single_vllm_request(
                engine=engine,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        )
        for prompt in prompts
    ]

    results = await asyncio.gather(*tasks)

    output_texts = [item[0] for item in results]
    first_token_candidates = [item[1] for item in results if item[1] is not None]
    token_timestamps_s = sorted(
        ts for item in results for ts in item[2]
    )
    end_time_s = max(item[3] for item in results)
    token_counts = [item[4] for item in results]
    output_token_count = (
        sum(token_count for token_count in token_counts if token_count is not None)
        if all(token_count is not None for token_count in token_counts)
        else None
    )

    per_request_stats = []
    for item in results:
        request_first_token_time_s = item[1]
        request_end_time_s = item[3]
        request_output_tokens = item[4]
        request_start_time_s = item[5]

        request_ttft_ms = None
        request_total_latency_ms = None
        if request_first_token_time_s is not None:
            request_ttft_ms = (request_first_token_time_s - request_start_time_s) * 1000.0
        if request_end_time_s is not None:
            request_total_latency_ms = (request_end_time_s - request_start_time_s) * 1000.0

        per_request_stats.append({
            "ttft_ms": request_ttft_ms,
            "total_latency_ms": request_total_latency_ms,
            "output_tokens_generated": request_output_tokens,
        })

    first_token_time_s = min(first_token_candidates) if first_token_candidates else end_time_s

    return output_texts, first_token_time_s, token_timestamps_s, end_time_s, output_token_count, per_request_stats

def _run_vllm_generate(
    bundle: LoadedModelBundle,
    workload: RuntimeWorkload,
    prompts: Optional[List[str]] = None,
    max_new_tokens_override: Optional[int] = None,
):
    """
    Run generation using the vLLM backend.

    Returns:
        output_texts, first_token_time_s, token_timestamps_s, end_time_s, output_token_count

    Notes:
    - This path uses the async vLLM engine and real streamed outputs.
    - TTFT is the first observed streamed delta arrival time.
    - TBT is later derived from streamed decode timing and token count.
    """
    if prompts is None:
        prompts = [workload.prompt]

    system_prompt = _resolve_system_prompt_for_workload(workload)

    formatted_prompts = []
    for prompt in prompts:
        formatted_prompt, _ = _format_prompt_for_instruct_model(
            bundle.tokenizer,
            prompt,
            system_prompt=system_prompt,
        )
        formatted_prompts.append(formatted_prompt)
    prompts = formatted_prompts

    max_new_tokens = (
        max_new_tokens_override
        if max_new_tokens_override is not None
        else workload.max_new_tokens
    )

    return _run_asyncio_coroutine_in_thread(
        _stream_many_vllm_requests(
            engine=bundle.model,
            tokenizer=bundle.tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
        )
    )

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
    system_prompt = _resolve_system_prompt_for_workload(workload)
    prompt, used_chat_template = _format_prompt_for_instruct_model(
        tokenizer=tokenizer,
        prompt=prompt,
        system_prompt=system_prompt,
    )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CONFIG.model.max_model_len,
        add_special_tokens=not used_chat_template,
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

    single_total_latency_ms = None
    if end_time_s is not None:
        single_total_latency_ms = (end_time_s - start_generation_time_s) * 1000.0

    per_request_stats = [{
        "ttft_ms": (first_token_time_s - start_generation_time_s) * 1000.0 if first_token_time_s is not None else None,
        "total_latency_ms": single_total_latency_ms,
        "output_tokens_generated": None,
    }]

    return [output_text], first_token_time_s, token_timestamps_s, end_time_s, None, per_request_stats

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
    controller_name = (
        runtime_mode.runner_kwargs.get("controller_name")
        or runtime_mode.runtime_kwargs.get("controller_name")
    )
    if controller_name:
        route_start_s = now_s(sync_cuda=False)
        decision = route_runtime_workload(
            workload,
            num_requests_in_batch=runtime_mode.runner_kwargs.get("request_batch_size", 1),
        )
        route_end_s = now_s(sync_cuda=False)
        delegated_mode = build_runtime_mode_by_name(decision.selected_mode_name)
        delegated_result = run_single_benchmark(
            runtime_mode=delegated_mode,
            workload=workload,
            trial_index=trial_index,
            preloaded_bundle=preloaded_bundle,
        )
        delegated_result.mode_name = controller_name
        delegated_result.controller_selected_mode_name = decision.selected_mode_name
        delegated_result.controller_phase_label = decision.classification_label
        delegated_result.controller_estimated_prefill_share_pct = decision.estimated_prefill_share_pct
        delegated_result.controller_route_reason = decision.reason
        delegated_result.controller_routing_overhead_ms = (route_end_s - route_start_s) * 1000.0
        delegated_result.controller_decision_source = "online_before_execution"
        delegated_result.evaluation_scope = "online_request_boundary_controller"
        delegated_result.notes = (
            f"Controller `{controller_name}` routed this request to `{decision.selected_mode_name}`. "
            f"{decision.reason} "
            + (delegated_result.notes or "")
        )
        return delegated_result

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
        workload_cell=workload.workload_cell,
        task_type=workload.task_type,
        system_condition=workload.system_condition_name,
        benchmark_suite=workload.benchmark_suite,
        benchmark_subset=workload.benchmark_subset,
        benchmark_language=workload.benchmark_language,
        evaluation_mode=workload.evaluation_mode,
        benchmark_example_id=workload.benchmark_example_id,
    )

    try:
        _configure_quieter_runtime_logs()

        # Reset peak GPU memory tracking before the run
        reset_gpu_peak_memory_stats()

        # Load model/tokenizer for this mode
        if bundle is None:
            bundle = load_model_for_mode(runtime_mode)

        result.gpu_allocated_before_mb = get_current_gpu_allocated_mb()
        result.gpu_reserved_before_mb = get_reserved_gpu_memory_mb()
        result.cpu_ram_before_mb = get_process_ram_mb()

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

        if _should_add_trial_unique_header(workload):
            timed_prompts = _add_trial_unique_header(
                timed_prompts,
                workload=workload,
                trial_index=trial_index,
            )
            result.notes += (
                "Synthetic non-benchmark timed prompt received a trial-specific leading header "
                "to avoid accidental cross-trial prefix-cache reuse. "
            )

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
        memory_pressure_fraction = _resolve_memory_pressure_fraction(workload)
        energy_poller = EnergyPoller(
            poll_interval_s=CONFIG.system.power_poll_interval_s,
            device_index=0,
        )
        monitor_stop_event = Event()
        monitor_stats = {}
        monitor_thread = Thread(
            target=_monitor_runtime_stats,
            args=(monitor_stop_event, monitor_stats),
            daemon=True,
        )
        monitor_thread.start()
        pressure_stats = {}
        try:
            with MemoryPressureContext(
                vram_fraction=memory_pressure_fraction,
                device=CONFIG.model.device,
            ) as pressure_ctx:
                pressure_stats = pressure_ctx.get_stats()
                with energy_poller:
                    result.start_time_s = now_s(sync_cuda=CONFIG.system.sync_cuda_for_timing)
                    (
                        output_texts,
                        first_token_time_s,
                        token_timestamps_s,
                        end_time_s,
                        streamed_output_tokens,
                        per_request_stats,
                    ) = _run_generation(
                        bundle=bundle,
                        workload=workload,
                        prompts=timed_prompts,
                    )
        finally:
            monitor_stop_event.set()
            monitor_thread.join()

        result.first_token_time_s = first_token_time_s
        result.end_time_s = end_time_s
        result.token_timestamps_s = token_timestamps_s
        result.output_text = "\n\n===== REQUEST SPLIT =====\n\n".join(output_texts)
        tokenized_output_tokens = _count_output_tokens_for_texts(
            bundle.tokenizer,
            output_texts,
        )

        # The vLLM streaming path decodes from filtered streamed token ids,
        # so streamed_output_tokens should already line up with the final text.
        # Keep a safe fallback anyway.
        result.output_tokens_generated = (
            streamed_output_tokens
            if streamed_output_tokens is not None
            else tokenized_output_tokens
        )

        result.per_request_ttft_ms = [
            value["ttft_ms"] for value in per_request_stats if value.get("ttft_ms") is not None
        ]
        result.per_request_total_latency_ms = [
            value["total_latency_ms"] for value in per_request_stats if value.get("total_latency_ms") is not None
        ]
        result.per_request_output_tokens_generated = [
            value["output_tokens_generated"] for value in per_request_stats if value.get("output_tokens_generated") is not None
        ]

        result.avg_power_w = energy_poller.get_mean_power_watts()
        result.cpu_ram_peak_mb = monitor_stats.get("cpu_ram_peak_mb")
        result.cpu_ram_after_mb = get_process_ram_mb()
        result.gpu_allocated_after_mb = get_current_gpu_allocated_mb()
        result.gpu_reserved_after_mb = get_reserved_gpu_memory_mb()
        result.kv_cache_estimate_mb = _estimate_kv_cache_mb(
            bundle=bundle,
            runtime_mode=runtime_mode,
            prompts=timed_prompts,
            output_tokens_generated=result.output_tokens_generated,
        )

        expected_reference = None
        if workload.repeated_prefix and workload.followup_prompt is not None:
            expected_reference = workload.followup_reference_answer
        else:
            expected_reference = workload.reference_answer

        if CONFIG.system.collect_quality_metrics and expected_reference:
            if len(output_texts) == 1:
                quality_prediction_text = output_texts[0]
                result.reference_exact_match = compute_exact_match(
                    quality_prediction_text,
                    expected_reference,
                )
                result.reference_rouge_l_f1 = compute_rouge_l_f1(
                    quality_prediction_text,
                    expected_reference,
                )
                result.reference_token_f1 = compute_token_f1(
                    quality_prediction_text,
                    expected_reference,
                )
            else:
                exact_matches = [
                    compute_exact_match(prediction_text, expected_reference)
                    for prediction_text in output_texts
                ]
                rouge_scores = [
                    compute_rouge_l_f1(prediction_text, expected_reference)
                    for prediction_text in output_texts
                ]
                token_f1_scores = [
                    compute_token_f1(prediction_text, expected_reference)
                    for prediction_text in output_texts
                ]

                valid_exact = [value for value in exact_matches if value is not None]
                if valid_exact:
                    result.reference_exact_match = all(valid_exact)
                result.reference_rouge_l_f1 = _mean_optional(rouge_scores)
                result.reference_token_f1 = _mean_optional(token_f1_scores)

        if CONFIG.system.collect_quality_metrics:
            if len(output_texts) == 1:
                benchmark_metric_values = compute_benchmark_suite_metrics(
                    prediction=output_texts[0],
                    reference=expected_reference,
                    benchmark_suite=workload.benchmark_suite,
                    evaluation_mode=workload.evaluation_mode,
                    metadata=workload.metadata or {},
                )
                for key, value in benchmark_metric_values.items():
                    setattr(result, key, value)
            else:
                per_output_metric_dicts = [
                    compute_benchmark_suite_metrics(
                        prediction=prediction_text,
                        reference=expected_reference,
                        benchmark_suite=workload.benchmark_suite,
                        evaluation_mode=workload.evaluation_mode,
                        metadata=workload.metadata or {},
                    )
                    for prediction_text in output_texts
                ]
                _apply_mean_benchmark_metric_dicts(result, per_output_metric_dicts)

            if workload.benchmark_suite in {"mt_bench", "alpacaeval_2_lc", "alpacaeval2_lc"}:
                if result.benchmark_primary_metric_value is None:
                    result.notes += (
                        "This benchmark suite typically requires an external judge / sidecar scorer; "
                        "schema fields are present, but no judge score was supplied for this run. "
                    )

        measured_energy_j = energy_poller.get_total_energy_joules()
        if measured_energy_j is not None:
            result.energy_joules = measured_energy_j

        if runtime_mode.backend == "vllm":
            result.notes += (
                "Primary trusted metrics for this run: total latency, memory, output tokens, "
                "rough throughput, and real streamed TTFT/TBT from the async vLLM engine. "
            )
        elif runtime_mode.backend == "transformers":
            result.notes += (
                "Transformers streamer path uses chunk-level arrival timing; "
                "TTFT/TBT are approximate. "
            )

        if runtime_mode.continuous_batching:
            result.notes += (
                f"Measured {len(timed_prompts)} concurrent requests on one async vLLM engine instance. "
            )

        if memory_pressure_fraction > 0.0:
            result.notes += (
                f"Artificial VRAM pressure applied before timed run "
                f"(target_fraction={memory_pressure_fraction:.2f}). "
            )
            if pressure_stats:
                target_gb = pressure_stats.get("target_allocation_gb")
                actual_gb = pressure_stats.get("actual_allocated_gb")
                if target_gb is not None or actual_gb is not None:
                    result.notes += (
                        f"pressure_target_gb={target_gb}; pressure_actual_gb={actual_gb}. "
                    )

        # Finalize derived metrics
        result = finalize_benchmark_result(result)

    except Exception as exc:
        result.error = str(exc)
        result.error_type = type(exc).__name__
        result.success = False
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