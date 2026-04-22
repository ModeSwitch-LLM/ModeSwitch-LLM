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

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List

from config import (
    CONFIG,
    RAW_RESULTS_DIR,
)
from modes import get_all_runtime_modes, get_default_hybrid_modes
from workloads import get_all_runtime_workloads, summarize_workload
from runner import run_single_benchmark
from metrics import BenchmarkResult


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
            "backend": r.backend,
            "trial_index": r.trial_index,
            "total_latency_ms": r.total_latency_ms,
            "tokens_per_second": r.tokens_per_second,
            "output_tokens_generated": r.output_tokens_generated,
            "peak_gpu_memory_mb": r.peak_gpu_memory_mb,
            "reserved_gpu_memory_mb": r.reserved_gpu_memory_mb,
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

# =============================================================================
# Benchmark sweep logic
# =============================================================================

def run_full_benchmark(
    include_hybrids: bool = False,
    repeated_prefix_variants: int = 2,
) -> List[BenchmarkResult]:
    """
    Run the full benchmark sweep across modes, workloads, and trials.

    Args:
        include_hybrids: Whether to include default hybrid modes as well
        repeated_prefix_variants: Number of repeated-prefix prompt variants

    Returns:
        List of BenchmarkResult objects
    """
    runtime_modes = get_all_runtime_modes(enabled_only=True)

    if include_hybrids:
        runtime_modes.extend(get_default_hybrid_modes())

    runtime_workloads = get_all_runtime_workloads(
        repeated_prefix_variants=repeated_prefix_variants
    )

    results: List[BenchmarkResult] = []

    total_runs = len(runtime_modes) * len(runtime_workloads) * CONFIG.system.num_trials
    run_counter = 0

    print("=" * 80)
    print("Starting full benchmark sweep")
    print(f"Enabled modes: {len(runtime_modes)}")
    print(f"Workloads: {len(runtime_workloads)}")
    print(f"Trials per pair: {CONFIG.system.num_trials}")
    print(f"Total runs: {total_runs}")
    print("=" * 80)

    for mode in runtime_modes:
        print(f"\n[MODE] {mode.name} | backend={mode.backend} | phase={mode.primary_phase}")
        print(f"       notes: {mode.notes}")

        for workload in runtime_workloads:
            print(f"  [WORKLOAD] {summarize_workload(workload)}")

            for trial_index in range(CONFIG.system.num_trials):
                run_counter += 1

                print(
                    f"    [RUN {run_counter}/{total_runs}] "
                    f"trial={trial_index} | mode={mode.name} | workload={workload.name}"
                )

                result = run_single_benchmark(
                    runtime_mode=mode,
                    workload=workload,
                    trial_index=trial_index,
                )

                if result.error:
                    print(f"      -> FAILED: {result.error}")
                else:
                    print(
                        f"      -> OK | "
                        f"ttft_ms={result.ttft_ms} | "
                        f"avg_tbt_ms={result.avg_tbt_ms} | "
                        f"total_latency_ms={result.total_latency_ms} | "
                        f"peak_mem_mb={result.peak_gpu_memory_mb}"
                    )

                results.append(result)

    print("\n" + "=" * 80)
    print("Benchmark sweep complete.")
    print(f"Collected results: {len(results)}")
    print("=" * 80)

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
    )

    json_path = RAW_RESULTS_DIR / f"benchmark_results_{timestamp}.json"
    csv_path = RAW_RESULTS_DIR / f"benchmark_results_{timestamp}.csv"
    summary_csv_path = RAW_RESULTS_DIR / f"benchmark_summary_{timestamp}.csv"

    save_results_json(results, json_path)
    save_results_csv(results, csv_path)
    save_summary_csv(results, summary_csv_path)

    print("\nSaved results:")
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")
    print(f"  Summary CSV : {summary_csv_path}")


if __name__ == "__main__":
    main()