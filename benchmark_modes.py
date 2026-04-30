"""
benchmark_modes.py

Full benchmark sweep script for ModeSwitch-LLM.

Purpose:
- run all enabled modes across all configured workloads and trials
- collect benchmark results
- save outputs to CSV and JSON for later analysis

Design principle:
This file is the top-level experiment driver.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
from dataclasses import dataclass
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import (
    CONFIG,
    LOGS_DIR,
    RAW_RESULTS_DIR,
)
from modes import (
    get_all_runtime_modes,
    get_default_hybrid_modes,
)
from workloads import (
    build_runtime_workload_by_name,
)
from runner import run_single_benchmark
from metrics import BenchmarkResult, compute_rouge_l_f1

EXTERNAL_BENCHMARK_SCORE_FIELDS = {
    "benchmark_primary_metric_value",
    "mmlu_pro_accuracy",
    "gsm8k_exact_match_accuracy",
    "truthfulqa_accuracy",
    "gpqa_accuracy",
    "mlu_accuracy",
    "tam_accuracy",
    "mt_bench_score",
    "alpacaeval2_lc_win_rate",
}

def _aggregate_workload_name(result: BenchmarkResult) -> str:
    """
    Group sidecar-expanded benchmark examples back to their parent workload.

    Example:
        mmlu_pro_eval__q0001 -> mmlu_pro_eval
    """
    if (
        result.benchmark_example_id is not None
        and isinstance(result.workload_name, str)
        and "__" in result.workload_name
    ):
        return result.workload_name.split("__", 1)[0]
    return result.workload_name

# =============================================================================
# Test plan helpers
# =============================================================================
@dataclass(frozen=True)
class TestCase:
    """
    One concrete test case in the curated benchmark plan.
    """
    mode_name: str
    workload_name: str
    repeated_prefix_variant: int = 0


def _safe_name(text: str) -> str:
    """
    Convert arbitrary text into a filesystem-safe token.
    """
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _fmt_metric(value, digits: int = 2) -> str:
    """
    Format a metric compactly for console / markdown tables.
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _test_log_path(timestamp: str, run_index: int, mode_name: str, workload_name: str, trial_index: int) -> Path:
    """
    Build the per-run log path used to capture verbose backend output.
    """
    filename = (
        f"test_{timestamp}_"
        f"{run_index:03d}_"
        f"{_safe_name(mode_name)}_"
        f"{_safe_name(workload_name)}_"
        f"trial{trial_index}.log"
    )
    return LOGS_DIR / filename


def _add_test_case(
    cases: List[TestCase],
    seen: set,
    available_mode_names: set,
    mode_name: str,
    workload_name: str,
    repeated_prefix_variant: int = 0,
) -> None:
    """
    Add one curated test case if the mode is enabled/available and the case
    has not already been added.
    """
    if mode_name not in available_mode_names:
        return

    case = TestCase(
        mode_name=mode_name,
        workload_name=workload_name,
        repeated_prefix_variant=repeated_prefix_variant,
    )

    if case in seen:
        return

    seen.add(case)
    cases.append(case)


def build_test_plan(
    runtime_modes,
    include_hybrids: bool = False,
    repeated_prefix_variants: int = 2,
    test_profile: str = "controller",
) -> List[TestCase]:
    """
    Build a curated large test plan instead of blindly sweeping every mode
    against every workload.

    Profiles:
    - initial: smaller set for a quick but meaningful meeting-ready run
    - controller: larger set for comparing candidates and selecting a final mode
    - all: the controller set plus the extra repeated-prefix variants
    """
    available_mode_names = {mode.name for mode in runtime_modes}
    cases: List[TestCase] = []
    seen = set()

    repeated_prefix_variants = max(1, repeated_prefix_variants)
    repeated_variant_ids = [0]
    if test_profile in {"controller", "all"} and repeated_prefix_variants > 1:
        repeated_variant_ids = list(range(min(repeated_prefix_variants, 2)))
    if test_profile == "all":
        repeated_variant_ids = list(range(repeated_prefix_variants))

    baseline_workloads = [
        "short_prompt_short_output",
        "short_prompt_long_output",
        "long_prompt_short_output",
        "long_prompt_long_output",
        "memory_pressure_long_context",
    ]
    decode_heavy_workloads = [
        "short_prompt_long_output",
        "long_prompt_long_output",
    ]
    latency_workloads = [
        "short_prompt_short_output",
        "short_prompt_long_output",
        "long_prompt_short_output",
    ]
    long_prefill_workloads = [
        "long_prompt_short_output",
        "long_prompt_long_output",
        "memory_pressure_long_context",
    ]
    batching_workloads = [
        "short_prompt_short_output",
        "short_prompt_long_output",
        "long_prompt_short_output",
    ]
    benchmark_workloads = [
        "mmlu_pro_eval",
        "gsm8k_eval",
        "truthfulqa_eval",
        "gpqa_eval",
        "mlu_eval",
        "mt_bench_eval",
        "alpacaeval2_lc_eval",
    ]

    if test_profile == "initial":
        baseline_workloads = [
            "short_prompt_short_output",
            "short_prompt_long_output",
            "long_prompt_short_output",
            "memory_pressure_long_context",
        ]
        latency_workloads = [
            "short_prompt_short_output",
            "short_prompt_long_output",
        ]
        decode_heavy_workloads = [
            "short_prompt_long_output",
        ]
        long_prefill_workloads = [
            "long_prompt_short_output",
            "memory_pressure_long_context",
        ]
        batching_workloads = [
            "short_prompt_short_output",
        ]

    # Baseline should span the representative workload families because almost
    # every comparison row uses it as the reference.
    for workload_name in baseline_workloads:
        _add_test_case(cases, seen, available_mode_names, "fp16_baseline", workload_name)
    for variant_id in repeated_variant_ids:
        _add_test_case(
            cases,
            seen,
            available_mode_names,
            "fp16_baseline",
            "shared_prefix_chat",
            repeated_prefix_variant=variant_id,
        )

    for mode_name in ["int8_quant", "gptq_4bit"]:
        for workload_name in [
            "short_prompt_short_output",
            "short_prompt_long_output",
            "long_prompt_short_output",
            "long_prompt_long_output",
        ]:
            _add_test_case(cases, seen, available_mode_names, mode_name, workload_name)
        if test_profile in {"controller", "all"}:
            _add_test_case(cases, seen, available_mode_names, mode_name, "memory_pressure_long_context")

    for workload_name in latency_workloads:
        _add_test_case(cases, seen, available_mode_names, "speculative_decoding", workload_name)

    for variant_id in repeated_variant_ids:
        _add_test_case(
            cases,
            seen,
            available_mode_names,
            "gptq_plus_prefix_caching",
            "shared_prefix_chat",
            repeated_prefix_variant=variant_id,
        )

    for workload_name in batching_workloads:
        _add_test_case(
            cases,
            seen,
            available_mode_names,
            "int8_plus_continuous_batching",
            workload_name,
        )

    # The updated controller project direction needs benchmark workloads in the
    # normal "controller" profile too, because accuracy retention against FP16
    # is now a first-class requirement.
    if test_profile in {"controller", "all"}:
        benchmark_candidate_modes = [
            "fp16_baseline",
            "gptq_4bit",
            "int8_quant",
            "speculative_decoding",
            "prefix_caching",
            "controller_v1",
        ]
        if test_profile == "all":
            benchmark_candidate_modes.extend(
                [
                    "gptq_plus_prefix_caching",
                    "int8_plus_continuous_batching",
                    "continuous_batching",
                ]
            )
        for mode_name in benchmark_candidate_modes:
            for workload_name in benchmark_workloads:
                _add_test_case(
                    cases,
                    seen,
                    available_mode_names,
                    mode_name,
                    workload_name,
                )

    if test_profile in {"controller", "all"}:
        controller_workloads = list(dict.fromkeys(baseline_workloads + benchmark_workloads))
        for workload_name in controller_workloads:
            _add_test_case(cases, seen, available_mode_names, "controller_v1", workload_name)
        for variant_id in repeated_variant_ids:
            _add_test_case(
                cases,
                seen,
                available_mode_names,
                "controller_v1",
                "shared_prefix_chat",
                repeated_prefix_variant=variant_id,
            )

    return cases


def _run_test_case_quietly(
    timestamp: str,
    run_index: int,
    total_runs: int,
    runtime_mode,
    runtime_workload,
    trial_index: int,
) -> BenchmarkResult:
    """
    Run one benchmark case while redirecting the verbose backend logs into a
    per-run file.
    """
    log_path = _test_log_path(
        timestamp=timestamp,
        run_index=run_index,
        mode_name=runtime_mode.name,
        workload_name=runtime_workload.name,
        trial_index=trial_index,
    )

    with open(log_path, "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            result = run_single_benchmark(
                runtime_mode=runtime_mode,
                workload=runtime_workload,
                trial_index=trial_index,
            )

    if result.error:
        print(
            f"[{run_index}/{total_runs}] "
            f"{runtime_mode.name:<24} | {runtime_workload.name:<28} | "
            f"trial={trial_index} | FAILED | log={log_path.name}"
        )
    else:
        print(
            f"[{run_index}/{total_runs}] "
            f"{runtime_mode.name:<24} | {runtime_workload.name:<28} | "
            f"trial={trial_index} | "
            f"ttft={_fmt_metric(result.ttft_ms)} ms | "
            f"lat={_fmt_metric(result.total_latency_ms)} ms | "
            f"tps={_fmt_metric(result.tokens_per_second)} | "
            f"J/tok={_fmt_metric(result.energy_per_token_j, 3)} | "
            f"gpu={_fmt_metric(result.peak_gpu_memory_mb)} MB"
        )

    result.notes += f"log_file={log_path.name}. "
    return result


# =============================================================================
# Result saving helpers
# =============================================================================
def _timestamp_str() -> str:
    """
    Return a compact timestamp string for filenames.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _results_to_dicts(results: List[BenchmarkResult]) -> List[dict]:
    """
    Convert BenchmarkResult objects into plain dictionaries.
    """
    return [result.to_dict() for result in results]


def save_results_json(results: List[BenchmarkResult], output_path: Path) -> None:
    """
    Save benchmark results to a JSON file.
    """
    result_dicts = _results_to_dicts(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dicts, f, indent=2, ensure_ascii=False)


def save_results_csv(results: List[BenchmarkResult], output_path: Path) -> None:
    """
    Save benchmark results to a CSV file.

    Handles the union of all keys across result dictionaries.
    """
    result_dicts = _results_to_dicts(results)

    if not result_dicts:
        # Create an empty file with no rows if needed.
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    # Collect all possible field names across rows
    fieldnames = []
    seen = set()
    for row in result_dicts:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_dicts)

def save_summary_csv(results: List[BenchmarkResult], output_path: Path) -> None:
    """
    Save a compact summary focused on the metrics we trust for tomorrow:
    total latency, output length, rough throughput, and memory.
    """
    rows = []
    for r in results:
        rows.append({
            "mode_name": r.mode_name,
            "workload_name": r.workload_name,
            "workload_group_name": _aggregate_workload_name(r),
            "benchmark_example_id": r.benchmark_example_id,
            "workload_cell": r.workload_cell,
            "task_type": r.task_type,
            "system_condition": r.system_condition,
            "backend": r.backend,
            "trial_index": r.trial_index,
            "num_requests_in_batch": r.num_requests_in_batch,
            "benchmark_suite": r.benchmark_suite,
            "benchmark_subset": r.benchmark_subset,
            "benchmark_language": r.benchmark_language,
            "evaluation_mode": r.evaluation_mode,
            "benchmark_primary_metric_name": r.benchmark_primary_metric_name,
            "benchmark_primary_metric_value": r.benchmark_primary_metric_value,
            "success": r.success,
            "error_type": r.error_type,
            "total_latency_ms": r.total_latency_ms,
            "prefill_latency_ms": r.prefill_latency_ms,
            "decode_latency_ms": r.decode_latency_ms,
            "ttft_ms": r.ttft_ms,
            "avg_tbt_ms": r.avg_tbt_ms,
            "tokens_per_second": r.tokens_per_second,
            "batched_tokens_per_second": r.batched_tokens_per_second,
            "output_tokens_generated": r.output_tokens_generated,
            "peak_gpu_memory_mb": r.peak_gpu_memory_mb,
            "reserved_gpu_memory_mb": r.reserved_gpu_memory_mb,
            "cpu_ram_peak_mb": r.cpu_ram_peak_mb,
            "cpu_ram_delta_mb": r.cpu_ram_delta_mb,
            "gpu_peak_delta_mb": r.gpu_peak_delta_mb,
            "gpu_reserved_delta_mb": r.gpu_reserved_delta_mb,
            "kv_cache_estimate_mb": r.kv_cache_estimate_mb,
            "avg_power_w": r.avg_power_w,
            "energy_joules": r.energy_joules,
            "energy_per_token_j": r.energy_per_token_j,
            "reference_rouge_l_f1": r.reference_rouge_l_f1,
            "reference_token_f1": r.reference_token_f1,
            "baseline_similarity_rouge_l_f1": r.baseline_similarity_rouge_l_f1,
            "quality_degradation_vs_baseline": r.quality_degradation_vs_baseline,
            "mmlu_pro_accuracy": r.mmlu_pro_accuracy,
            "gsm8k_exact_match_accuracy": r.gsm8k_exact_match_accuracy,
            "truthfulqa_accuracy": r.truthfulqa_accuracy,
            "gpqa_accuracy": r.gpqa_accuracy,
            "mlu_accuracy": r.mlu_accuracy,
            "tam_accuracy": r.tam_accuracy,
            "mt_bench_score": r.mt_bench_score,
            "alpacaeval2_lc_win_rate": r.alpacaeval2_lc_win_rate,
            "error": r.error,
            "notes": r.notes,
        })

    if not rows:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _percentile(values: List[float], q: float):
    """
    Compute a percentile with linear interpolation.
    """
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _mean(values: List[float]):
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: List[float]):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mu = _mean(values)
    variance = sum((value - mu) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _metric_summary(values: List[float], prefix: str):
    return {
        f"{prefix}_mean": _mean(values),
        f"{prefix}_std": _std(values),
        f"{prefix}_p50": _percentile(values, 0.50),
        f"{prefix}_p95": _percentile(values, 0.95),
        f"{prefix}_p99": _percentile(values, 0.99),
    }


def annotate_results_with_baseline_similarity(results: List[BenchmarkResult]) -> None:
    """
    Compare each result output to the baseline output on the same workload.
    """
    baseline_by_workload = {}

    for result in results:
        if (
            result.mode_name == CONFIG.system.baseline_reference_mode_name
            and result.error is None
            and result.output_text is not None
            and (result.workload_name, result.system_condition) not in baseline_by_workload
        ):
            baseline_by_workload[(result.workload_name, result.system_condition)] = result.output_text

    for result in results:
        baseline_output = baseline_by_workload.get((result.workload_name, result.system_condition))
        if baseline_output is None or result.output_text is None:
            continue
        similarity = compute_rouge_l_f1(result.output_text, baseline_output)
        result.baseline_similarity_rouge_l_f1 = similarity
        if similarity is not None:
            result.quality_degradation_vs_baseline = 1.0 - similarity


def build_aggregate_rows(results: List[BenchmarkResult]) -> List[dict]:
    """
    Build grouped aggregate rows across repeated trials.
    """
    grouped = defaultdict(list)
    for result in results:
        key = (
            result.mode_name,
            _aggregate_workload_name(result),
            result.system_condition,
            result.backend,
        )
        grouped[key].append(result)

    rows = []
    for (mode_name, workload_name, system_condition, backend), group in grouped.items():
        successful = [result for result in group if result.error is None]
        failures = [result for result in group if result.error is not None]
        first = group[0]
        benchmark_primary_metric_name = next(
            (r.benchmark_primary_metric_name for r in successful if r.benchmark_primary_metric_name),
            next((r.benchmark_primary_metric_name for r in group if r.benchmark_primary_metric_name), None),
        )
        selected_controller_modes = [
            r.controller_selected_mode_name
            for r in successful
            if getattr(r, "controller_selected_mode_name", None)
        ]
        selected_controller_phases = [
            r.controller_phase_label
            for r in successful
            if getattr(r, "controller_phase_label", None)
        ]
        selected_controller_reasons = [
            r.controller_route_reason
            for r in successful
            if getattr(r, "controller_route_reason", None)
        ]

        ttft_values = [result.ttft_ms for result in successful if result.ttft_ms is not None]
        latency_values = [result.total_latency_ms for result in successful if result.total_latency_ms is not None]
        tps_values = [result.tokens_per_second for result in successful if result.tokens_per_second is not None]

        error_types = [result.error_type for result in failures if result.error_type]
        most_common_error_type = None
        if error_types:
            most_common_error_type = Counter(error_types).most_common(1)[0][0]

        row = {
            "mode_name": mode_name,
            "workload_name": workload_name,
            "workload_cell": first.workload_cell,
            "task_type": first.task_type,
            "system_condition": system_condition,
            "backend": backend,
            "benchmark_suite": first.benchmark_suite,
            "benchmark_subset": first.benchmark_subset,
            "benchmark_language": first.benchmark_language,
            "evaluation_mode": first.evaluation_mode,
            "benchmark_primary_metric_name": benchmark_primary_metric_name,
            "controller_selected_mode_name": (
                Counter(selected_controller_modes).most_common(1)[0][0]
                if selected_controller_modes
                else None
            ),
            "controller_phase_label": (
                Counter(selected_controller_phases).most_common(1)[0][0]
                if selected_controller_phases
                else None
            ),
            "controller_route_reason": selected_controller_reasons[0] if selected_controller_reasons else None,
            "num_runs": len(group),
            "completed_runs": len(successful),
            "failed_runs": len(failures),
            "failure_rate": (len(failures) / len(group)) if group else None,
            "most_common_error_type": most_common_error_type,
            "reference_rouge_l_f1_mean": _mean([r.reference_rouge_l_f1 for r in successful if r.reference_rouge_l_f1 is not None]),
            "baseline_similarity_rouge_l_f1_mean": _mean([r.baseline_similarity_rouge_l_f1 for r in successful if r.baseline_similarity_rouge_l_f1 is not None]),
            "quality_degradation_vs_baseline_mean": _mean([r.quality_degradation_vs_baseline for r in successful if r.quality_degradation_vs_baseline is not None]),
            "benchmark_primary_metric_value_mean": _mean([r.benchmark_primary_metric_value for r in successful if r.benchmark_primary_metric_value is not None]),
            "mmlu_pro_accuracy_mean": _mean([r.mmlu_pro_accuracy for r in successful if r.mmlu_pro_accuracy is not None]),
            "gsm8k_exact_match_accuracy_mean": _mean([r.gsm8k_exact_match_accuracy for r in successful if r.gsm8k_exact_match_accuracy is not None]),
            "truthfulqa_accuracy_mean": _mean([r.truthfulqa_accuracy for r in successful if r.truthfulqa_accuracy is not None]),
            "gpqa_accuracy_mean": _mean([r.gpqa_accuracy for r in successful if r.gpqa_accuracy is not None]),
            "mlu_accuracy_mean": _mean([r.mlu_accuracy for r in successful if r.mlu_accuracy is not None]),
            "tam_accuracy_mean": _mean([r.tam_accuracy for r in successful if r.tam_accuracy is not None]),
            "mt_bench_score_mean": _mean([r.mt_bench_score for r in successful if r.mt_bench_score is not None]),
            "alpacaeval2_lc_win_rate_mean": _mean([r.alpacaeval2_lc_win_rate for r in successful if r.alpacaeval2_lc_win_rate is not None]),
            "avg_power_w_mean": _mean([r.avg_power_w for r in successful if r.avg_power_w is not None]),
            "energy_joules_mean": _mean([r.energy_joules for r in successful if r.energy_joules is not None]),
            "energy_per_token_j_mean": _mean([r.energy_per_token_j for r in successful if r.energy_per_token_j is not None]),
            "peak_gpu_memory_mb_mean": _mean([r.peak_gpu_memory_mb for r in successful if r.peak_gpu_memory_mb is not None]),
            "cpu_ram_peak_mb_mean": _mean([r.cpu_ram_peak_mb for r in successful if r.cpu_ram_peak_mb is not None]),
            "kv_cache_estimate_mb_mean": _mean([r.kv_cache_estimate_mb for r in successful if r.kv_cache_estimate_mb is not None]),
        }

        row.update(_metric_summary(ttft_values, "ttft_ms"))
        row.update(_metric_summary(latency_values, "total_latency_ms"))
        row.update(_metric_summary(tps_values, "tokens_per_second"))

        rows.append(row)

    return rows


def save_aggregate_csv(rows: List[dict], output_path: Path) -> None:
    """
    Save grouped aggregate rows to CSV.
    """
    if not rows:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def build_comparison_rows(results: List[BenchmarkResult]) -> List[dict]:
    """
    Build aggregate comparison rows against the FP16 baseline on the same workload.
    """
    aggregate_rows = build_aggregate_rows(results)
    baseline_rows = {
        (row["workload_name"], row["system_condition"]): row
        for row in aggregate_rows
        if row["mode_name"] == CONFIG.system.baseline_reference_mode_name
    }

    comparison_rows = []
    for row in aggregate_rows:
        if row["mode_name"] == CONFIG.system.baseline_reference_mode_name:
            continue
        baseline = baseline_rows.get((row["workload_name"], row["system_condition"]))
        if baseline is None:
            continue

        comparison_row = dict(row)
        comparison_row["latency_speedup_vs_baseline"] = None
        comparison_row["throughput_ratio_vs_baseline"] = None
        comparison_row["energy_per_token_ratio_vs_baseline"] = None
        comparison_row["peak_gpu_memory_delta_vs_baseline_mb"] = None
        comparison_row["benchmark_primary_metric_delta_vs_baseline"] = None

        baseline_latency = baseline.get("total_latency_ms_mean")
        row_latency = row.get("total_latency_ms_mean")
        if baseline_latency not in (None, 0) and row_latency not in (None, 0):
            comparison_row["latency_speedup_vs_baseline"] = baseline_latency / row_latency

        baseline_tps = baseline.get("tokens_per_second_mean")
        row_tps = row.get("tokens_per_second_mean")
        if baseline_tps not in (None, 0) and row_tps is not None:
            comparison_row["throughput_ratio_vs_baseline"] = row_tps / baseline_tps

        baseline_energy = baseline.get("energy_per_token_j_mean")
        row_energy = row.get("energy_per_token_j_mean")
        if baseline_energy not in (None, 0) and row_energy is not None:
            comparison_row["energy_per_token_ratio_vs_baseline"] = row_energy / baseline_energy

        baseline_mem = baseline.get("peak_gpu_memory_mb_mean")
        row_mem = row.get("peak_gpu_memory_mb_mean")
        if baseline_mem is not None and row_mem is not None:
            comparison_row["peak_gpu_memory_delta_vs_baseline_mb"] = row_mem - baseline_mem

        baseline_benchmark = baseline.get("benchmark_primary_metric_value_mean")
        row_benchmark = row.get("benchmark_primary_metric_value_mean")
        if baseline_benchmark is not None and row_benchmark is not None:
            comparison_row["benchmark_primary_metric_delta_vs_baseline"] = row_benchmark - baseline_benchmark
  
        comparison_rows.append(comparison_row)

    return comparison_rows


def _load_sidecar_score_rows(sidecar_path: Path) -> List[dict]:
    suffix = sidecar_path.suffix.lower()

    if suffix == ".jsonl":
        rows = []
        with open(sidecar_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix == ".json":
        with open(sidecar_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of dicts in sidecar JSON: {sidecar_path}")
        return data

    if suffix == ".csv":
        with open(sidecar_path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    raise ValueError(
        f"Unsupported sidecar score file type: {sidecar_path.suffix}. "
        "Use JSONL, JSON, or CSV."
    )


def apply_external_score_sidecar(
    results: List[BenchmarkResult],
    sidecar_path: Path,
) -> int:
    rows = _load_sidecar_score_rows(sidecar_path)

    by_workload_key = {}
    by_example_key = {}
    for result in results:
        by_workload_key[(result.mode_name, result.workload_name, result.trial_index)] = result
        if result.benchmark_example_id is not None:
            by_example_key[(result.mode_name, result.benchmark_example_id, result.trial_index)] = result

    applied = 0
    for row in rows:
        mode_name = row.get("mode_name")
        workload_name = row.get("workload_name")
        benchmark_example_id = row.get("benchmark_example_id")
        trial_index = row.get("trial_index")
        trial_index = int(trial_index) if trial_index not in (None, "") else None

        target = None
        if mode_name is not None and workload_name is not None and trial_index is not None:
            target = by_workload_key.get((mode_name, workload_name, trial_index))
        if target is None and mode_name is not None and benchmark_example_id is not None and trial_index is not None:
            target = by_example_key.get((mode_name, benchmark_example_id, trial_index))
        if target is None:
            continue

        if row.get("benchmark_primary_metric_name"):
            target.benchmark_primary_metric_name = str(row["benchmark_primary_metric_name"])

        if row.get("benchmark_primary_metric_value") not in (None, ""):
            target.benchmark_primary_metric_value = float(row["benchmark_primary_metric_value"])

        for field_name in EXTERNAL_BENCHMARK_SCORE_FIELDS:
            value = row.get(field_name)
            if value not in (None, ""):
                setattr(target, field_name, float(value))

        applied += 1

    return applied

def save_test_table_md(rows: List[dict], output_path: Path, title: str, columns: List[str]) -> None:
    """
    Save a compact markdown table.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")

        if not rows:
            f.write("_No rows available._\n")
            return

        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")

        for row in rows:
            values = []
            for column in columns:
                value = row.get(column)
                if isinstance(value, float):
                    values.append(_fmt_metric(value))
                elif value is None:
                    values.append("-")
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def build_test_table_rows(results: List[BenchmarkResult]) -> List[dict]:
    """
    Build a compact, readable table from aggregate rows.
    """
    rows = build_aggregate_rows(results)
    rows.sort(key=lambda row: (row["workload_name"], row["mode_name"]))
    return rows

# =============================================================================
# Benchmark sweep logic
# =============================================================================

def run_full_benchmark(
    include_hybrids: bool = False,
    repeated_prefix_variants: int = 2,
    test_profile: str = "controller",
) -> List[BenchmarkResult]:
    """
    Run the curated large test plan across modes, workloads, and trials.

    Args:
        include_hybrids: Whether to include default hybrid modes as well
        repeated_prefix_variants: Number of repeated-prefix prompt variants
        test_profile: initial, controller, or all

    Returns:
        List of BenchmarkResult objects
    """
    runtime_modes = get_all_runtime_modes(enabled_only=True)

    if include_hybrids:
        runtime_modes.extend(get_default_hybrid_modes())

    runtime_mode_map = {mode.name: mode for mode in runtime_modes}

    test_cases = build_test_plan(
        runtime_modes=runtime_modes,
        include_hybrids=include_hybrids,
        repeated_prefix_variants=repeated_prefix_variants,
        test_profile=test_profile,
    )

    results: List[BenchmarkResult] = []

    total_runs = len(test_cases) * CONFIG.system.num_trials
    timestamp = _timestamp_str()
    run_counter = 0

    print("=" * 80)
    print("Starting curated large benchmark run")
    print(f"Test profile: {test_profile}")
    print(f"Enabled modes: {len(runtime_modes)}")
    print(f"Curated test cases: {len(test_cases)}")
    print(f"Trials per pair: {CONFIG.system.num_trials}")
    print(f"Total runs: {total_runs}")
    print("=" * 80)
    print(f"Per-run logs will be saved under: {LOGS_DIR}")
    print("=" * 80)

    for test_case in test_cases:
        runtime_mode = runtime_mode_map[test_case.mode_name]
        runtime_workload = build_runtime_workload_by_name(
            test_case.workload_name,
            repeated_prefix_variant=test_case.repeated_prefix_variant,
        )

        for trial_index in range(CONFIG.system.num_trials):
            run_counter += 1
            result = _run_test_case_quietly(
                timestamp=timestamp,
                run_index=run_counter,
                total_runs=total_runs,
                runtime_mode=runtime_mode,
                runtime_workload=runtime_workload,
                trial_index=trial_index,
            )
            results.append(result)

    print("\n" + "=" * 80)
    print("Benchmark sweep complete.")
    print(f"Collected results: {len(results)}")
    print("=" * 80)

    annotate_results_with_baseline_similarity(results)

    return results


# =============================================================================
# Main entrypoint
# =============================================================================

def main() -> None:
    """
    Main script entrypoint.

    Runs the full benchmark sweep and saves outputs to timestamped files.
    """
    timestamp = _timestamp_str()

    results = run_full_benchmark(
        include_hybrids=True,
        repeated_prefix_variants=2,
        test_profile="controller",
    )

    json_path = RAW_RESULTS_DIR / f"benchmark_results_{timestamp}.json"
    csv_path = RAW_RESULTS_DIR / f"benchmark_results_{timestamp}.csv"
    summary_csv_path = RAW_RESULTS_DIR / f"benchmark_summary_{timestamp}.csv"
    aggregate_csv_path = RAW_RESULTS_DIR / f"benchmark_aggregates_{timestamp}.csv"
    comparison_csv_path = RAW_RESULTS_DIR / f"benchmark_comparisons_{timestamp}.csv"
    table_md_path = RAW_RESULTS_DIR / f"benchmark_table_{timestamp}.md"
    comparison_md_path = RAW_RESULTS_DIR / f"benchmark_comparisons_{timestamp}.md"

    save_results_json(results, json_path)
    save_results_csv(results, csv_path)
    save_summary_csv(results, summary_csv_path)
    aggregate_rows = build_aggregate_rows(results)
    comparison_rows = build_comparison_rows(results)

    save_aggregate_csv(aggregate_rows, aggregate_csv_path)
    save_aggregate_csv(comparison_rows, comparison_csv_path)
    save_test_table_md(
        rows=build_test_table_rows(results),
        output_path=table_md_path,
        title="Benchmark Table",
        columns=[
            "mode_name",
            "workload_name",
            "workload_cell",
            "task_type",
            "controller_selected_mode_name",
            "controller_phase_label",
            "completed_runs",
            "failed_runs",
            "ttft_ms_mean",
            "total_latency_ms_mean",
            "tokens_per_second_mean",
            "avg_power_w_mean",
            "energy_per_token_j_mean",
            "peak_gpu_memory_mb_mean",
            "reference_rouge_l_f1_mean",
            "benchmark_primary_metric_name",
            "benchmark_primary_metric_value_mean",
            "failure_rate",
        ],
    )
    save_test_table_md(
        rows=comparison_rows,
        output_path=comparison_md_path,
        title="Benchmark Comparisons vs FP16 Baseline",
        columns=[
            "mode_name",
            "workload_name",
            "workload_cell",
            "task_type",
            "controller_selected_mode_name",
            "controller_phase_label",
            "latency_speedup_vs_baseline",
            "throughput_ratio_vs_baseline",
            "energy_per_token_ratio_vs_baseline",
            "peak_gpu_memory_delta_vs_baseline_mb",
            "quality_degradation_vs_baseline_mean",
            "benchmark_primary_metric_delta_vs_baseline",
            "failure_rate",
        ],
    )

    print("\nSaved results:")
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")
    print(f"  Summary CSV : {summary_csv_path}")
    print(f"  Aggregate CSV : {aggregate_csv_path}")
    print(f"  Comparison CSV : {comparison_csv_path}")
    print(f"  Markdown table : {table_md_path}")
    print(f"  Comparison markdown : {comparison_md_path}")

if __name__ == "__main__":
    main()
