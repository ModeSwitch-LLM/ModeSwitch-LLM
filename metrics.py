"""
# ============================================================================
# ModeSwitch-LLM Benchmark Metrics and Result Schema
# ============================================================================
# Shared metric utilities and result schema for ModeSwitch-LLM benchmark runs.
#
# Main tasks:
# - Defines the BenchmarkResult dataclass used to store one benchmark run.
# - Computes timing metrics such as TTFT, total latency, decode latency, and TBT.
# - Computes throughput metrics for single-request and batched runs.
# - Collects GPU memory and CPU RAM usage when available.
# - Computes energy and energy-per-token from power measurements.
# - Provides lightweight quality metrics such as exact match, token F1, ROUGE-L,
#   multiple-choice accuracy, and final-answer exact match.
# - Computes benchmark-specific metrics for MMLU-Pro, GSM8K, TruthfulQA, GPQA,
#   MLU, TAM, MT-Bench, and AlpacaEval-style workloads.
# - Finalizes partially populated BenchmarkResult objects before export.
#
# Usage:
# from metrics import BenchmarkResult, finalize_benchmark_result
# ============================================================================
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence
import time
import math
import os
import re
from decimal import Decimal, InvalidOperation
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

    # Benchmark-evaluation metadata
    benchmark_suite: Optional[str] = None
    benchmark_subset: Optional[str] = None
    benchmark_language: Optional[str] = None
    evaluation_mode: Optional[str] = None
    benchmark_example_id: Optional[str] = None
    benchmark_primary_metric_name: Optional[str] = None
    benchmark_primary_metric_value: Optional[float] = None

    # Controller metadata.
    # These are populated for online controller runs and for reconstructed
    # offline controller analyses.
    controller_selected_mode_name: Optional[str] = None
    controller_phase_label: Optional[str] = None
    controller_estimated_prefill_share_pct: Optional[float] = None
    controller_route_reason: Optional[str] = None
    controller_routing_overhead_ms: Optional[float] = None
    controller_decision_source: Optional[str] = None
    evaluation_scope: Optional[str] = None

    # Benchmark accuracy / quality metrics
    mmlu_pro_accuracy: Optional[float] = None
    gsm8k_exact_match_accuracy: Optional[float] = None
    truthfulqa_accuracy: Optional[float] = None
    gpqa_accuracy: Optional[float] = None
    mlu_accuracy: Optional[float] = None
    tam_accuracy: Optional[float] = None
    mt_bench_score: Optional[float] = None
    alpacaeval2_lc_win_rate: Optional[float] = None

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

def _safe_float(value: Any) -> Optional[float]:
    """
    Best-effort float conversion.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_to_accuracy(value: Optional[bool]) -> Optional[float]:
    """
    Convert a correctness boolean into a 0/1 accuracy value.
    """
    if value is None:
        return None
    return 1.0 if value else 0.0

def _tokenize_for_quality(text: Optional[str]) -> List[str]:
    """
    Lightweight tokenization for exact-match / token-F1 / ROUGE-L.
    """
    normalized = _normalize_text(text)
    return re.findall(r"\w+|[^\w\s]", normalized)

def _normalize_suite_name(benchmark_suite: Optional[str]) -> str:
    """
    Normalize benchmark-suite naming across dashes / spaces / casing.
    """
    if benchmark_suite is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", benchmark_suite.strip().lower()).strip("_")


def _extract_choice_label(
    text: Optional[str],
    valid_labels: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    Extract a multiple-choice label such as A/B/C/D from a prediction/reference.
    """
    if text is None:
        return None

    labels = [str(x).strip().upper() for x in (valid_labels or list("ABCDEFGHIJ"))]
    labels = [label for label in labels if label]
    if not labels:
        return None

    label_pattern = "|".join(re.escape(label) for label in labels)
    raw_text = str(text).strip()
    upper_text = raw_text.upper()

    exact_candidate = raw_text.strip().strip("()[]{}.:;- ").upper()
    if exact_candidate in labels:
        return exact_candidate

    marker_patterns = [
        rf"(?:FINAL\s+ANSWER|FINAL|ANSWER|THE\s+ANSWER\s+IS|ANSWER\s+IS|OPTION|CHOICE|CORRECT\s+ANSWER)\s*[:\-]?\s*\(?\s*({label_pattern})\s*\)?",
        rf"(?:I\s+CHOOSE|I\s+SELECT|I\s+PICK|MY\s+ANSWER\s+IS|THEREFORE|THUS|HENCE|SO)\s*[:\-]?\s*\(?\s*({label_pattern})\s*\)?",
        rf"(?:الجواب|الإجابة|الإجابة\s+الصحيحة)\s*[:\-]?\s*\(?\s*({label_pattern})\s*\)?",
        rf"(?:RESPUESTA|LA\s+RESPUESTA|RESPOSTA)\s*[:\-]?\s*\(?\s*({label_pattern})\s*\)?",
     ]

    matches = []
    for pattern in marker_patterns:
        matches.extend(list(re.finditer(pattern, upper_text)))

    if matches:
        return matches[-1].group(1).upper()

    line_patterns = [
        rf"^\(?\s*({label_pattern})\s*\)?\s*$",
        rf"^\(?\s*({label_pattern})\s*\)?\s*[\.\):\-]\s*$",
    ]

    raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    candidate_lines = raw_lines[:3] + raw_lines[-3:]
    seen_lines = set()

    for line in candidate_lines:
        upper_line = line.upper()
        if upper_line in seen_lines:
            continue
        seen_lines.add(upper_line)

        for pattern in line_patterns:
            match = re.search(pattern, upper_line)
            if match:
                return match.group(1).upper()
    return None


def _extract_last_numeric_answer(text: Optional[str]) -> Optional[str]:
    """
    Extract a numeric final answer.

    Prefer explicit final-answer markers, then fall back to the last number.
    """
    if text is None:
        return None

    text = str(text)

    marker_patterns = [
        r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"final\s+answer\s*[:\-]?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"answer\s*[:\-]?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"the\s+answer\s+is\s*[:\-]?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    ]

    for pattern in marker_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", "").strip()

    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None

    return matches[-1].replace(",", "").strip()

def _numeric_strings_equal(a: Optional[str], b: Optional[str]) -> bool:
    """
    Compare numeric strings robustly.

    This treats answers like 109 and 109.0 as equal.
    """
    if a is None or b is None:
        return False

    try:
        da = Decimal(str(a).replace(",", "").strip())
        db = Decimal(str(b).replace(",", "").strip())
        return da == db
    except (InvalidOperation, ValueError):
        return str(a).strip() == str(b).strip()

def compute_exact_match(prediction: Optional[str], reference: Optional[str]) -> Optional[bool]:
    """
    Compute normalized exact match.
    """
    if prediction is None or reference is None:
        return None
    return _normalize_text(prediction) == _normalize_text(reference)

def compute_multiple_choice_accuracy(
    prediction: Optional[str],
    reference: Optional[str],
    valid_labels: Optional[Sequence[str]] = None,
) -> Optional[float]:
    """
    Compute 0/1 multiple-choice accuracy.
    """
    if prediction is None or reference is None:
        return None

    pred_label = _extract_choice_label(prediction, valid_labels=valid_labels)
    ref_label = _extract_choice_label(reference, valid_labels=valid_labels)

    if pred_label is not None and ref_label is not None:
        return 1.0 if pred_label == ref_label else 0.0

    # Fallback to normalized exact match if labels are not extractable.
    return _bool_to_accuracy(compute_exact_match(prediction, reference))


def compute_final_answer_exact_match(
    prediction: Optional[str],
    reference: Optional[str],
) -> Optional[float]:
    """
    Compute GSM8K-style final-answer exact match.
    """
    if prediction is None or reference is None:
        return None

    pred_num = _extract_last_numeric_answer(prediction)
    ref_num = _extract_last_numeric_answer(reference)

    if pred_num is not None and ref_num is not None:
        return 1.0 if _numeric_strings_equal(pred_num, ref_num) else 0.0

    return _bool_to_accuracy(compute_exact_match(prediction, reference))

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

def compute_benchmark_suite_metrics(
    prediction: Optional[str],
    reference: Optional[str],
    benchmark_suite: Optional[str] = None,
    evaluation_mode: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute benchmark-specific evaluation metrics.

    Notes:
    - MMLU-Pro / GPQA / many TruthfulQA / MLU setups are usually multiple-choice.
    - GSM8K is usually final-answer exact match.
    - TAM is left intentionally generic and uses the provided evaluation_mode.
    - MT-Bench and AlpacaEval 2 LC typically require an external judge. This
      function supports them by accepting precomputed values in metadata.
    """
    metadata = metadata or {}
    suite = _normalize_suite_name(benchmark_suite)
    mode = _normalize_suite_name(evaluation_mode)

    results: Dict[str, Optional[float]] = {
        "mmlu_pro_accuracy": None,
        "gsm8k_exact_match_accuracy": None,
        "truthfulqa_accuracy": None,
        "gpqa_accuracy": None,
        "mlu_accuracy": None,
        "tam_accuracy": None,
        "mt_bench_score": None,
        "alpacaeval2_lc_win_rate": None,
        "benchmark_primary_metric_value": None,
    }

    primary_metric_name: Optional[str] = None
    primary_metric_name_override = metadata.get("benchmark_primary_metric_name")
    primary_metric_value_override = _safe_float(metadata.get("benchmark_primary_metric_value"))

    valid_labels = metadata.get("valid_labels")

    if primary_metric_name_override and primary_metric_value_override is not None:
        if primary_metric_name_override in results:
            results[primary_metric_name_override] = primary_metric_value_override
        results["benchmark_primary_metric_name"] = primary_metric_name_override
        results["benchmark_primary_metric_value"] = primary_metric_value_override
        return results

    # Allow external scorers / sidecar evaluators to inject scores directly.
    for field_name in (
        "mmlu_pro_accuracy",
        "gsm8k_exact_match_accuracy",
        "truthfulqa_accuracy",
        "gpqa_accuracy",
        "mlu_accuracy",
        "tam_accuracy",
        "mt_bench_score",
        "alpacaeval2_lc_win_rate",
    ):
        override = _safe_float(metadata.get(field_name))
        if override is not None:
            results[field_name] = override

    if suite in {"mmlu_pro", "mmlu"} and results["mmlu_pro_accuracy"] is None and suite == "mmlu_pro":
        results["mmlu_pro_accuracy"] = compute_multiple_choice_accuracy(
            prediction,
            reference,
            valid_labels=valid_labels,
        )
        primary_metric_name = "mmlu_pro_accuracy"

    if suite == "gsm8k" and results["gsm8k_exact_match_accuracy"] is None:
        results["gsm8k_exact_match_accuracy"] = compute_final_answer_exact_match(
            prediction,
            reference,
        )
        primary_metric_name = "gsm8k_exact_match_accuracy"

    if suite == "truthfulqa" and results["truthfulqa_accuracy"] is None:
        if mode in {"multiple_choice", "multiple_choice_accuracy", ""}:
            results["truthfulqa_accuracy"] = compute_multiple_choice_accuracy(
                prediction,
                reference,
                valid_labels=valid_labels or ["A", "B"],
            )
        elif mode in {"exact_match", "binary_accuracy"}:
            results["truthfulqa_accuracy"] = _bool_to_accuracy(
                compute_exact_match(prediction, reference)
            )
        primary_metric_name = "truthfulqa_accuracy"

    if suite == "gpqa" and results["gpqa_accuracy"] is None:
        results["gpqa_accuracy"] = compute_multiple_choice_accuracy(
            prediction,
            reference,
            valid_labels=valid_labels,
        )
        primary_metric_name = "gpqa_accuracy"

    if suite == "mlu" and results["mlu_accuracy"] is None:
        if mode in {"multiple_choice", "multiple_choice_accuracy", ""}:
            results["mlu_accuracy"] = compute_multiple_choice_accuracy(
                prediction,
                reference,
                valid_labels=valid_labels,
            )
        elif mode == "final_answer_exact_match":
            results["mlu_accuracy"] = compute_final_answer_exact_match(prediction, reference)
        else:
            results["mlu_accuracy"] = _bool_to_accuracy(
                compute_exact_match(prediction, reference)
            )
        primary_metric_name = "mlu_accuracy"

    if suite == "tam" and results["tam_accuracy"] is None:
        if mode == "multiple_choice_accuracy":
            results["tam_accuracy"] = compute_multiple_choice_accuracy(
                prediction,
                reference,
                valid_labels=valid_labels,
            )
        elif mode == "final_answer_exact_match":
            results["tam_accuracy"] = compute_final_answer_exact_match(prediction, reference)
        else:
            results["tam_accuracy"] = _bool_to_accuracy(
                compute_exact_match(prediction, reference)
            )
        primary_metric_name = "tam_accuracy"

    if suite == "mt_bench" and results["mt_bench_score"] is not None:
        primary_metric_name = "mt_bench_score"

    if suite in {"alpacaeval_2_lc", "alpacaeval2_lc"} and results["alpacaeval2_lc_win_rate"] is not None:
        primary_metric_name = "alpacaeval2_lc_win_rate"

    # Backfill primary metric name if it arrived via metadata override.
    if primary_metric_name is None:
        for candidate in (
            "mmlu_pro_accuracy",
            "gsm8k_exact_match_accuracy",
            "truthfulqa_accuracy",
            "gpqa_accuracy",
            "mlu_accuracy",
            "tam_accuracy",
            "mt_bench_score",
            "alpacaeval2_lc_win_rate",
        ):
            if results[candidate] is not None:
                primary_metric_name = candidate
                break

    results["benchmark_primary_metric_value"] = (
        results.get(primary_metric_name) if primary_metric_name is not None else None
    )
    results["benchmark_primary_metric_name"] = primary_metric_name
    return results

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