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
    total_latency_ms: Optional[float] = None
    decode_latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None

    # Memory stats
    peak_gpu_memory_mb: Optional[float] = None
    reserved_gpu_memory_mb: Optional[float] = None

    # Optional energy / power stats
    avg_power_w: Optional[float] = None
    energy_joules: Optional[float] = None
    energy_per_token_j: Optional[float] = None

    # Free-form notes / errors
    notes: str = ""
    error: Optional[str] = None

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


def compute_avg_tbt_ms(token_timestamps_s: List[float]) -> Optional[float]:
    """
    Compute average TBT (Time Between Tokens) in milliseconds.

    TBT is only meaningful if you have at least 2 token timestamps.
    """
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
    result.avg_tbt_ms = compute_avg_tbt_ms(result.token_timestamps_s)
    result.tokens_per_second = compute_tokens_per_second(result.output_tokens_generated, result.total_latency_ms)

    # Memory
    if result.peak_gpu_memory_mb is None:
        result.peak_gpu_memory_mb = get_peak_gpu_memory_mb()
    if result.reserved_gpu_memory_mb is None:
        result.reserved_gpu_memory_mb = get_reserved_gpu_memory_mb()

    # Energy
    if result.energy_joules is None:
        result.energy_joules = compute_energy_joules(result.avg_power_w, result.total_latency_ms)
    if result.energy_per_token_j is None:
        result.energy_per_token_j = compute_energy_per_token_j(
            result.energy_joules,
            result.output_tokens_generated,
        )

    return result