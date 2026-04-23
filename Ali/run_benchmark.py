#!/usr/bin/env python3
"""
run_benchmark.py — ModeSwitch-LLM Benchmarking Pipeline
Author: Ali Alshehhi

Entry point for the full benchmarking pipeline.

Quick-start examples:

  # Benchmark FP16 baseline on short-short workloads, 3 repeats
  python run_benchmark.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --modes FP16_BASELINE \
    --cells SS \
    --conditions baseline \
    --repeats 3

  # Full sweep across all modes and cells
  python run_benchmark.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --modes FP16_BASELINE W4A16_AWQ W8A8 SPEC_DECODE KV_COMPRESS_H2O FA2_ONLY \
    --cells SS SL LS LL \
    --conditions baseline mem_pressure_50 mem_pressure_75 \
    --repeats 3

  # Generate a post-hoc report from an existing run
  python run_benchmark.py --report-only results/run_20241115_142301_abc123

  # List available workload samples
  python run_benchmark.py --list-samples

  # Dry run (lists what would be run without executing)
  python run_benchmark.py --dry-run --modes FP16_BASELINE W4A16_AWQ --cells SS SL
"""

import argparse
import sys
import logging
from pathlib import Path

# Ensure project is importable from this script's directory
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.pipeline import BenchmarkPipeline, PipelineConfig, ALL_MODES, InferenceMode
from benchmarks.workloads import WorkloadSuite, SystemCondition
from benchmarks.reporter import generate_full_report
from benchmarks.metrics import print_metric_schema


def parse_args():
    parser = argparse.ArgumentParser(
        description="ModeSwitch-LLM Benchmarking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="HuggingFace model name or local path (default: Llama-2-7b-chat-hf)",
    )
    parser.add_argument(
        "--draft-model",
        default=None,
        help="Draft model for speculative decoding (default: same as --model with 1B variant)",
    )

    # ── Sweep configuration ────────────────────────────────────────────────
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["FP16_BASELINE"],
        choices=ALL_MODES,
        help="Inference modes to benchmark",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["SS"],
        choices=["SS", "SL", "LS", "LL"],
        help="Workload cells: SS=short-short, SL=short-long, LS=long-short, LL=long-long",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline"],
        choices=[c.value for c in SystemCondition],
        help="System conditions to test",
    )

    # ── Run settings ───────────────────────────────────────────────────────
    parser.add_argument("--repeats",      type=int, default=3,
                        help="Number of repeat runs per sample (default: 3)")
    parser.add_argument("--warmup",       type=int, default=3,
                        help="Number of warm-up generations before benchmarking (default: 3)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--results-dir",  default="results",
                        help="Directory to save results (default: ./results)")
    parser.add_argument("--energy-poll-ms", type=float, default=50.0,
                        help="GPU power polling interval in ms (default: 50)")

    # ── Quality metrics ────────────────────────────────────────────────────
    parser.add_argument("--bertscore",    action="store_true",
                        help="Compute BERTScore for quality evaluation (slow; adds ~10s/sample)")
    parser.add_argument("--perplexity",   action="store_true",
                        help="Compute self-perplexity (very slow; use for small subsets)")

    # ── Model architecture (for KV-cache estimation) ───────────────────────
    parser.add_argument("--num-layers",   type=int, default=32)
    parser.add_argument("--num-heads",    type=int, default=32)
    parser.add_argument("--head-dim",     type=int, default=128)

    # ── Special modes ──────────────────────────────────────────────────────
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be run without executing any inference",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="Print all workload samples and exit",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print all metric definitions and exit",
    )
    parser.add_argument(
        "--report-only",
        metavar="RUN_DIR",
        default=None,
        help="Generate report from an existing run directory (skip inference)",
    )

    # ── Logging ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


def dry_run(args) -> None:
    """Print exactly what would be run."""
    suite = WorkloadSuite(seed=args.seed)
    cells = args.cells
    modes = args.modes
    conditions = args.conditions

    total_runs = 0
    print("\n" + "="*70)
    print("DRY RUN — ModeSwitch-LLM Benchmark Sweep")
    print("="*70)
    print(f"Model:       {args.model}")
    print(f"Modes:       {modes}")
    print(f"Cells:       {cells}")
    print(f"Conditions:  {conditions}")
    print(f"Repeats:     {args.repeats}")
    print(f"Warmup runs: {args.warmup}")
    print()

    for mode in modes:
        for cell in cells:
            samples = suite.get_cell(cell)
            for cond in conditions:
                n = len(samples) * args.repeats
                total_runs += n
                print(f"  {mode:<22} × {cell} × {cond:<20} → {len(samples)} samples × {args.repeats} repeats = {n} runs")

    print()
    print(f"  TOTAL INFERENCE RUNS: {total_runs}")
    print(f"  Estimated time (rough): {total_runs * 10 / 60:.0f} – {total_runs * 30 / 60:.0f} minutes")
    print("="*70)


def list_samples(args) -> None:
    suite = WorkloadSuite(seed=args.seed)
    print("\nWorkload Suite Summary")
    print("="*70)
    for cell, info in suite.summary().items():
        print(f"\nCell {cell}: {info['count']} samples | "
              f"max_new_tokens={info['max_new_tokens']} | "
              f"tasks={info['tasks']}")
        for s in suite.get_cell(cell):
            preview = s.prompt[:80].replace("\n", " ")
            print(f"  [{s.sample_id}] ({s.task_type.value}) {preview!r}")
    print(f"\nTotal: {len(suite)} samples across 4 workload cells")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    # ── Special modes ──────────────────────────────────────────────────────
    if args.list_samples:
        list_samples(args)
        return

    if args.list_metrics:
        print_metric_schema()
        return

    if args.report_only:
        print(f"Generating report for run: {args.report_only}")
        generate_full_report(args.report_only)
        return

    if args.dry_run:
        dry_run(args)
        return

    # ── Full benchmark run ─────────────────────────────────────────────────
    config = PipelineConfig(
        model_name=args.model,
        draft_model_name=args.draft_model or args.model,
        modes_to_run=args.modes,
        cells_to_run=args.cells,
        conditions_to_run=args.conditions,
        num_repeats=args.repeats,
        warmup_runs=args.warmup,
        seed=args.seed,
        results_dir=args.results_dir,
        energy_poll_ms=args.energy_poll_ms,
        compute_bertscore=args.bertscore,
        compute_perplexity=args.perplexity,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        log_level=args.log_level,
    )

    print("\n" + "="*70)
    print("ModeSwitch-LLM Benchmarking Pipeline")
    print("="*70)
    print(f"Model:      {config.model_name}")
    print(f"Modes:      {config.modes_to_run}")
    print(f"Cells:      {config.cells_to_run}")
    print(f"Conditions: {config.conditions_to_run}")
    print(f"Repeats:    {config.num_repeats} per sample")
    print(f"Results:    {config.results_dir}/")
    print("="*70 + "\n")

    pipeline = BenchmarkPipeline(config)
    pipeline.run()

    # Auto-generate report after run
    print("\nGenerating post-run report...")
    generate_full_report(pipeline.results_path)
    print(f"\nDone. Results in: {pipeline.results_path}")


if __name__ == "__main__":
    main()
