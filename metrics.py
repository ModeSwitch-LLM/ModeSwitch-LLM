"""
metrics.py

Metric computation utilities for ModeSwitch-LLM.

Purpose:
- define the standard benchmark result format
- compute latency and throughput metrics from raw timing data
- capture GPU memory statistics
- provide a clean result object that runner.py can save/export

Design principle:
runner.py should focus on execution
metrics.py should focus on measurement + metric calculation
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import time
import math
import os
import re
from collections import Counter

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# Result schema
# =============================================================================

@dataclass
class BenchmarkResult:
    """
    Standard result object for one benchmark run.

    This should be the main unit saved to CSV/JSON later.
    """

    # Core identifiers
    mode_name: str
    workload_name: str
    backend: str
    trial_index: int

    # Workload metadata
    prompt_tokens_target: int
    max_new_tokens: int
    repeated_prefix: bool = False
    memory_pressure: bool = False
    num_requests_in_batch: int = 1
    workload_cell: Optional[str] = None
    task_type: Optional[str] = None
    system_condition: str = "baseline"

    # Raw timing fields
    start_time_s: Optional[float] = None
    first_token_time_s: Optional[float] = None
    end_time_s: Optional[float] = None

    # Raw decode token timestamps
    token_timestamps_s: List[float] = field(default_factory=list)

    # Output metadata
    output_text: Optional[str] = None
    output_tokens_generated: Optional[int] = None

    # Computed metrics
    ttft_ms: Optional[float] = None
    avg_tbt_ms: Optional[float] = None
    tbt_median_ms: Optional[float] = None
    tbt_p95_ms: Optional[float] = None
    tbt_p99_ms: Optional[float] = None
    tbt_std_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    decode_latency_ms: Optional[float] = None
    prefill_latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    batched_tokens_per_second: Optional[float] = None
    prefill_throughput_tps: Optional[float] = None
    decode_throughput_tps: Optional[float] = None
    decode_prefill_ratio: Optional[float] = None

    # Memory stats
    peak_gpu_memory_mb: Optional[float] = None
    reserved_gpu_memory_mb: Optional[float] = None
    gpu_allocated_before_mb: Optional[float] = None
    gpu_allocated_after_mb: Optional[float] = None
    gpu_allocated_delta_mb: Optional[float] = None
    gpu_peak_delta_mb: Optional[float] = None
    gpu_reserved_before_mb: Optional[float] = None
    gpu_reserved_after_mb: Optional[float] = None
    gpu_reserved_delta_mb: Optional[float] = None
    cpu_ram_before_mb: Optional[float] = None
    cpu_ram_after_mb: Optional[float] = None
    cpu_ram_peak_mb: Optional[float] = None
    cpu_ram_delta_mb: Optional[float] = None
    kv_cache_estimate_mb: Optional[float] = None

    # Optional energy / power stats
    avg_power_w: Optional[float] = None
    energy_joules: Optional[float] = None
    energy_per_token_j: Optional[float] = None

    # Per-request batch metrics
    per_request_ttft_ms: List[float] = field(default_factory=list)
    per_request_total_latency_ms: List[float] = field(default_factory=list)
    per_request_output_tokens_generated: List[int] = field(default_factory=list)

    # Quality metrics
    reference_exact_match: Optional[bool] = None
    reference_rouge_l_f1: Optional[float] = None
    reference_token_f1: Optional[float] = None
    baseline_similarity_rouge_l_f1: Optional[float] = None
    quality_degradation_vs_baseline: Optional[float] = None

    # Free-form notes / errors
    notes: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    success: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result object to a plain dictionary for export.
        """
        return asdict(self)


# =============================================================================
# Timing helpers
# =============================================================================

def now_s(sync_cuda: bool = False) -> float:
    """
    Return current wall-clock time in seconds.

    Args:
        sync_cuda: If True and CUDA is available, synchronize before timing.
                   Useful for more accurate GPU timing boundaries.
    """
    if sync_cuda and torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def compute_ttft_ms(start_time_s: Optional[float], first_token_time_s: Optional[float]) -> Optional[float]:
    """
    Compute TTFT (Time To First Token) in milliseconds.
    """
    if start_time_s is None or first_token_time_s is None:
        return None
    return (first_token_time_s - start_time_s) * 1000.0


def compute_total_latency_ms(start_time_s: Optional[float], end_time_s: Optional[float]) -> Optional[float]:
    """
    Compute total end-to-end latency in milliseconds.
    """
    if start_time_s is None or end_time_s is None:
        return None
    return (end_time_s - start_time_s) * 1000.0


def compute_decode_latency_ms(first_token_time_s: Optional[float], end_time_s: Optional[float]) -> Optional[float]:
    """
    Compute decode-only latency in milliseconds.

    This is the time from first generated token to final completion.
    """
    if first_token_time_s is None or end_time_s is None:
        return None
    return (end_time_s - first_token_time_s) * 1000.0


def compute_avg_tbt_ms(
    token_timestamps_s: List[float],
    output_tokens_generated: Optional[int] = None,
    first_token_time_s: Optional[float] = None,
    end_time_s: Optional[float] = None,
) -> Optional[float]:
    """
    Compute average TBT (Time Between Tokens) in milliseconds.

    Preferred behavior:
    - If we truly have one timestamp per generated token, compute exact
      inter-token average from those timestamps.
    - Otherwise, fall back to decode_latency / (generated_tokens - 1),
      which is a standard aggregate TBT approximation for streamed runs
      when the transport groups multiple tokens in one update.
    """
    if (
        output_tokens_generated is not None
        and output_tokens_generated > 1
        and len(token_timestamps_s) == output_tokens_generated
    ):
        deltas = []
        for i in range(1, len(token_timestamps_s)):
            deltas.append(token_timestamps_s[i] - token_timestamps_s[i - 1])

        if not deltas:
            return None

        return (sum(deltas) / len(deltas)) * 1000.0

    if (
        output_tokens_generated is not None
        and output_tokens_generated > 1
        and first_token_time_s is not None
        and end_time_s is not None
        and end_time_s >= first_token_time_s
    ):
        decode_latency_s = end_time_s - first_token_time_s
        return (decode_latency_s / (output_tokens_generated - 1)) * 1000.0

    if len(token_timestamps_s) < 2:
        return None

    deltas = []
    for i in range(1, len(token_timestamps_s)):
        deltas.append(token_timestamps_s[i] - token_timestamps_s[i - 1])

    if not deltas:
        return None

    return (sum(deltas) / len(deltas)) * 1000.0


def compute_tokens_per_second(output_tokens_generated: Optional[int], total_latency_ms: Optional[float]) -> Optional[float]:
    """
    Compute throughput in generated tokens per second.
    """
    if output_tokens_generated is None or total_latency_ms is None:
        return None
    if total_latency_ms <= 0:
        return None

    total_latency_s = total_latency_ms / 1000.0
    return output_tokens_generated / total_latency_s

def compute_batched_tokens_per_second(
    output_tokens_generated: Optional[int],
    total_latency_ms: Optional[float],
    num_requests_in_batch: int,
) -> Optional[float]:
    """
    Aggregate throughput for batched runs.
    """
    if num_requests_in_batch <= 1:
        return None
    return compute_tokens_per_second(output_tokens_generated, total_latency_ms)

# =============================================================================
# GPU memory helpers
# =============================================================================

def reset_gpu_peak_memory_stats() -> None:
    """
    Reset CUDA peak memory tracking if available.
    """
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def get_peak_gpu_memory_mb() -> Optional[float]:
    """
    Return peak allocated GPU memory in MB, if available.
    """
    if torch is None or not torch.cuda.is_available():
        return None

    try:
        peak_bytes = torch.cuda.max_memory_allocated()
        return peak_bytes / (1024 ** 2)
    except Exception:
        return None

def get_current_gpu_allocated_mb() -> Optional[float]:
    """
    Return current allocated GPU memory in MB, if available.
    """
    if torch is None or not torch.cuda.is_available():
        return None

    try:
        allocated_bytes = torch.cuda.memory_allocated()
        return allocated_bytes / (1024 ** 2)
    except Exception:
        return None

def get_reserved_gpu_memory_mb() -> Optional[float]:
    """
    Return currently reserved GPU memory in MB, if available.
    """
    if torch is None or not torch.cuda.is_available():
        return None

    try:
        reserved_bytes = torch.cuda.memory_reserved()
        return reserved_bytes / (1024 ** 2)
    except Exception:
        return None

def get_process_ram_mb() -> Optional[float]:
    """
    Return the current Python process RSS in MB.
    """
    if psutil is None:
        return None
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
    except Exception:
        return None

# =============================================================================
# Energy helpers
# =============================================================================

def compute_energy_joules(
    avg_power_w: Optional[float],
    total_latency_ms: Optional[float],
) -> Optional[float]:
    """
    Compute energy in joules using average power and total runtime.

    Energy = Power * Time
    """
    if avg_power_w is None or total_latency_ms is None:
        return None

    total_latency_s = total_latency_ms / 1000.0
    return avg_power_w * total_latency_s


def compute_energy_per_token_j(
    energy_joules: Optional[float],
    output_tokens_generated: Optional[int],
) -> Optional[float]:
    """
    Compute energy per generated token in joules/token.
    """
    if energy_joules is None or output_tokens_generated is None:
        return None
    if output_tokens_generated <= 0:
        return None

    return energy_joules / output_tokens_generated


def compute_tbt_stats_ms(
    token_timestamps_s: List[float],
    output_tokens_generated: Optional[int] = None,
    first_token_time_s: Optional[float] = None,
    end_time_s: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute mean / median / p95 / p99 / std of TBT in milliseconds.

    Falls back to a single average-based approximation when exact per-token
    timestamps are not available.
    """
    tbts_ms: List[float] = []

    if (
        output_tokens_generated is not None
        and output_tokens_generated > 1
        and len(token_timestamps_s) == output_tokens_generated
    ):
        tbts_ms = [
            (token_timestamps_s[i] - token_timestamps_s[i - 1]) * 1000.0
            for i in range(1, len(token_timestamps_s))
        ]
    elif (
        output_tokens_generated is not None
        and output_tokens_generated > 1
        and first_token_time_s is not None
        and end_time_s is not None
        and end_time_s >= first_token_time_s
    ):
        approx = ((end_time_s - first_token_time_s) / (output_tokens_generated - 1)) * 1000.0
        tbts_ms = [approx] * (output_tokens_generated - 1)

    if not tbts_ms:
        return {
            "avg_tbt_ms": None,
            "tbt_median_ms": None,
            "tbt_p95_ms": None,
            "tbt_p99_ms": None,
            "tbt_std_ms": None,
        }

    tbts_ms = sorted(tbts_ms)
    n = len(tbts_ms)

    def _pct(p: float) -> float:
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return tbts_ms[lo] * (1 - frac) + tbts_ms[hi] * frac

    mean = sum(tbts_ms) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in tbts_ms) / n) if n > 1 else 0.0

    return {
        "avg_tbt_ms": mean,
        "tbt_median_ms": _pct(50),
        "tbt_p95_ms": _pct(95),
        "tbt_p99_ms": _pct(99),
        "tbt_std_ms": std,
    }

    
def _normalize_text(text: Optional[str]) -> str:
    """
    Normalize text for simple quality metrics.
    """
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize_for_quality(text: Optional[str]) -> List[str]:
    """
    Lightweight tokenization for exact-match / token-F1 / ROUGE-L.
    """
    normalized = _normalize_text(text)
    return re.findall(r"\w+|[^\w\s]", normalized)


def compute_exact_match(prediction: Optional[str], reference: Optional[str]) -> Optional[bool]:
    """
    Compute normalized exact match.
    """
    if prediction is None or reference is None:
        return None
    return _normalize_text(prediction) == _normalize_text(reference)


def compute_token_f1(prediction: Optional[str], reference: Optional[str]) -> Optional[float]:
    """
    Compute token-level F1 using a simple normalized tokenization.
    """
    if prediction is None or reference is None:
        return None

    pred_tokens = _tokenize_for_quality(prediction)
    ref_tokens = _tokenize_for_quality(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = pred_counter & ref_counter
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """
    Compute LCS length for ROUGE-L.
    """
    if not a or not b:
        return 0

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def compute_rouge_l_f1(prediction: Optional[str], reference: Optional[str]) -> Optional[float]:
    """
    Compute a lightweight ROUGE-L F1 score.
    """
    if prediction is None or reference is None:
        return None

    pred_tokens = _tokenize_for_quality(prediction)
    ref_tokens = _tokenize_for_quality(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# Result finalization
# =============================================================================

def finalize_benchmark_result(result: BenchmarkResult) -> BenchmarkResult:
    """
    Fill in all computed metrics for a partially populated BenchmarkResult.
    """
    result.ttft_ms = compute_ttft_ms(result.start_time_s, result.first_token_time_s)
    result.total_latency_ms = compute_total_latency_ms(result.start_time_s, result.end_time_s)
    result.decode_latency_ms = compute_decode_latency_ms(result.first_token_time_s, result.end_time_s)
    result.prefill_latency_ms = result.ttft_ms
    tbt_stats = compute_tbt_stats_ms(
        result.token_timestamps_s,
        output_tokens_generated=result.output_tokens_generated,
        first_token_time_s=result.first_token_time_s,
        end_time_s=result.end_time_s,
    )
    result.avg_tbt_ms = tbt_stats["avg_tbt_ms"]
    result.tbt_median_ms = tbt_stats["tbt_median_ms"]
    result.tbt_p95_ms = tbt_stats["tbt_p95_ms"]
    result.tbt_p99_ms = tbt_stats["tbt_p99_ms"]
    result.tbt_std_ms = tbt_stats["tbt_std_ms"]
    result.tokens_per_second = compute_tokens_per_second(result.output_tokens_generated, result.total_latency_ms)
    if result.prefill_latency_ms is not None and result.prefill_latency_ms > 0:
        result.prefill_throughput_tps = result.prompt_tokens_target / (result.prefill_latency_ms / 1000.0)
    if result.decode_latency_ms is not None and result.decode_latency_ms > 0 and result.output_tokens_generated:
        result.decode_throughput_tps = result.output_tokens_generated / (result.decode_latency_ms / 1000.0)
    if result.prefill_latency_ms is not None and result.prefill_latency_ms > 0 and result.decode_latency_ms is not None:
        result.decode_prefill_ratio = result.decode_latency_ms / result.prefill_latency_ms
    result.batched_tokens_per_second = compute_batched_tokens_per_second(
        result.output_tokens_generated,
        result.total_latency_ms,
        result.num_requests_in_batch,
    )

    # Memory
    if result.peak_gpu_memory_mb is None:
        result.peak_gpu_memory_mb = get_peak_gpu_memory_mb()
    if result.reserved_gpu_memory_mb is None:
        result.reserved_gpu_memory_mb = get_reserved_gpu_memory_mb()
    if result.gpu_allocated_before_mb is not None and result.gpu_allocated_after_mb is not None:
        result.gpu_allocated_delta_mb = result.gpu_allocated_after_mb - result.gpu_allocated_before_mb
    if result.gpu_allocated_before_mb is not None and result.peak_gpu_memory_mb is not None:
        result.gpu_peak_delta_mb = result.peak_gpu_memory_mb - result.gpu_allocated_before_mb
    if result.gpu_reserved_before_mb is not None and result.gpu_reserved_after_mb is not None:
        result.gpu_reserved_delta_mb = result.gpu_reserved_after_mb - result.gpu_reserved_before_mb
    if result.cpu_ram_before_mb is not None and result.cpu_ram_after_mb is not None:
        result.cpu_ram_delta_mb = result.cpu_ram_after_mb - result.cpu_ram_before_mb
 
    # Energy
    if result.energy_joules is None:
        result.energy_joules = compute_energy_joules(result.avg_power_w, result.total_latency_ms)
    if result.energy_per_token_j is None:
        result.energy_per_token_j = compute_energy_per_token_j(
            result.energy_joules,
            result.output_tokens_generated,
        )

    if result.success is None:
        result.success = result.error is None

    return result