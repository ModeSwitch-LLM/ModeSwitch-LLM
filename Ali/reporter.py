"""
reporter.py — ModeSwitch-LLM Benchmarking Pipeline
Author: Ali Alshehhi

Post-run analysis and reporting module. Takes the raw per-sample JSON results
produced by the pipeline and generates:

  1. Aggregated statistics per (mode × cell × condition):
       mean, std, median, p95 for every metric

  2. Relative-change tables vs the FP16_BASELINE:
       ΔX% for TTFT, TBT_mean, TBT_p95, decode_tps, energy_per_token_j,
       peak_vram_gb, rougeL_f

  3. Phase dominance analysis:
       For each cell, what fraction of total time is prefill vs decode?
       How does this shift across modes?

  4. Pareto frontier identification:
       Which (mode × condition) combinations are non-dominated on the
       (energy_per_token_j, rougeL_f) and (tbt_mean_ms, rougeL_f) fronts?

  5. ASCII progress tables printed live during runs (via log_live_result)

  6. Markdown report generation (suitable for including in the paper appendix)

  7. Optional matplotlib plots (if matplotlib is available):
       - TTFT vs TBT scatter per mode+cell
       - Energy per token bar charts
       - CDF of TBT per mode
       - Memory usage heatmap
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Plots disabled. pip install matplotlib numpy")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ****************************************************
# Data loading
# ****************************************************

def load_results(run_dir: str | Path) -> List[dict]:
    """
    Load all per-sample JSON results from a benchmark run directory.
    Returns a flat list of result dicts (one per sample).
    """
    run_dir = Path(run_dir)
    results = []
    for json_file in sorted(run_dir.rglob("*.json")):
        if json_file.name in ("config.json", "summary.json"):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            if "sample_id" in data:
                results.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    logger.info(f"Loaded {len(results)} sample results from {run_dir}")
    return results


# ****************************************************
# Statistical aggregation
# ****************************************************

FLOAT_METRICS = [
    "ttft_ms", "tbt_mean_ms", "tbt_median_ms", "tbt_p95_ms", "tbt_p99_ms", "tbt_std_ms",
    "total_decode_ms", "total_inference_ms",
    "prefill_throughput_tps", "decode_throughput_tps",
    "decode_prefill_ratio",
    "peak_vram_gb", "kv_cache_estimated_gb",
    "energy_per_token_j", "total_energy_j", "mean_power_w",
    "rouge1_f", "rouge2_f", "rougeL_f", "bertscore_f1",
    "rep_rate_3gram", "vocab_diversity",
]

LOWER_IS_BETTER = {
    "ttft_ms", "tbt_mean_ms", "tbt_median_ms", "tbt_p95_ms", "tbt_p99_ms", "tbt_std_ms",
    "total_decode_ms", "total_inference_ms", "rep_rate_3gram",
    "energy_per_token_j", "total_energy_j", "peak_vram_gb", "kv_cache_estimated_gb",
}
HIGHER_IS_BETTER = {
    "prefill_throughput_tps", "decode_throughput_tps",
    "rouge1_f", "rouge2_f", "rougeL_f", "bertscore_f1", "vocab_diversity",
}


def _safe_mean(vals: List[float]) -> Optional[float]:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else None


def _safe_std(vals: List[float]) -> Optional[float]:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    if len(valid) < 2:
        return 0.0
    m = sum(valid) / len(valid)
    return math.sqrt(sum((x - m) ** 2 for x in valid) / len(valid))


def _safe_percentile(vals: List[float], p: float) -> Optional[float]:
    valid = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not valid:
        return None
    idx = (p / 100.0) * (len(valid) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(valid) - 1)
    return valid[lo] * (1 - (idx - lo)) + valid[hi] * (idx - lo)


def aggregate_results(
    results: List[dict],
    group_by: Tuple[str, ...] = ("inference_mode", "workload_cell", "system_condition"),
) -> Dict[tuple, dict]:
    """
    Group raw sample results by the specified keys and compute summary statistics
    for every float metric.

    Returns:
        Dict mapping group-key-tuple → {metric: {mean, std, median, p95, n, ...}}
    """
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in results:
        if r.get("status") != "ok":
            continue
        key = tuple(r.get(k, "unknown") for k in group_by)
        groups[key].append(r)

    aggregated = {}
    for key, group in groups.items():
        agg = {"n": len(group), "group_keys": dict(zip(group_by, key))}
        for metric in FLOAT_METRICS:
            vals = [r.get(metric) for r in group]
            agg[metric] = {
                "mean":   _safe_mean(vals),
                "std":    _safe_std(vals),
                "median": _safe_percentile(vals, 50),
                "p95":    _safe_percentile(vals, 95),
                "min":    min((v for v in vals if v is not None), default=None),
                "max":    max((v for v in vals if v is not None), default=None),
            }
        aggregated[key] = agg
    return aggregated


# ****************************************************
# Delta vs baseline
# ****************************************************

def compute_delta_table(
    aggregated: Dict[tuple, dict],
    baseline_mode: str = "FP16_BASELINE",
    metrics_to_compare: Optional[List[str]] = None,
) -> Dict[tuple, dict]:
    """
    Compute relative change (%) of each mode vs the FP16 baseline
    for the same (cell, condition) group.

    Returns:
        Dict mapping (mode, cell, condition) → {metric: delta_pct}
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            "ttft_ms", "tbt_mean_ms", "tbt_p95_ms", "decode_throughput_tps",
            "energy_per_token_j", "peak_vram_gb", "rougeL_f", "rep_rate_3gram",
        ]

    # Build baseline lookup: (cell, condition) → agg_dict
    baseline_lookup = {}
    for key, agg in aggregated.items():
        mode, cell, cond = key
        if mode == baseline_mode:
            baseline_lookup[(cell, cond)] = agg

    deltas = {}
    for key, agg in aggregated.items():
        mode, cell, cond = key
        if mode == baseline_mode:
            continue
        base = baseline_lookup.get((cell, cond))
        if base is None:
            continue
        delta_row = {}
        for metric in metrics_to_compare:
            m_val = (agg.get(metric) or {}).get("mean")
            b_val = (base.get(metric) or {}).get("mean")
            if m_val is not None and b_val is not None and b_val != 0:
                delta_row[metric] = round((m_val - b_val) / abs(b_val) * 100.0, 2)
            else:
                delta_row[metric] = None
        deltas[key] = delta_row
    return deltas


# ****************************************************
# Phase dominance analysis
# ****************************************************

def compute_phase_dominance(aggregated: Dict[tuple, dict]) -> Dict[tuple, dict]:
    """
    For each (mode, cell, condition), compute prefill and decode fractions of
    total inference time.

    Returns:
        Dict mapping key → {"prefill_pct": float, "decode_pct": float,
                             "dominated_by": "prefill"|"decode"|"balanced"}
    """
    dominance = {}
    for key, agg in aggregated.items():
        ttft     = (agg.get("ttft_ms") or {}).get("mean") or 0.0
        dec_ms   = (agg.get("total_decode_ms") or {}).get("mean") or 0.0
        total    = ttft + dec_ms
        if total <= 0:
            continue
        pre_pct  = ttft / total * 100.0
        dec_pct  = dec_ms / total * 100.0
        if pre_pct > 60:
            dominated = "prefill"
        elif dec_pct > 60:
            dominated = "decode"
        else:
            dominated = "balanced"
        dominance[key] = {
            "prefill_ms":   round(ttft, 2),
            "decode_ms":    round(dec_ms, 2),
            "total_ms":     round(total, 2),
            "prefill_pct":  round(pre_pct, 1),
            "decode_pct":   round(dec_pct, 1),
            "dominated_by": dominated,
        }
    return dominance


# ****************************************************
# Pareto frontier
# ****************************************************

def find_pareto_frontier(
    aggregated: Dict[tuple, dict],
    obj1_metric: str = "energy_per_token_j",
    obj2_metric: str = "rougeL_f",
    obj1_minimize: bool = True,
    obj2_minimize: bool = False,
    cell_filter: Optional[str] = None,
) -> List[tuple]:
    """
    Identify Pareto-optimal (mode, cell, condition) combinations on two objectives.

    A point is dominated if another point is strictly better on at least one objective
    and at least as good on both.

    Args:
        obj1_metric:    First objective metric name.
        obj2_metric:    Second objective metric name.
        obj1_minimize:  True if lower is better for obj1.
        obj2_minimize:  True if lower is better for obj2.
        cell_filter:    If set, only consider this workload cell.

    Returns:
        List of keys that are on the Pareto frontier.
    """
    points = []
    for key, agg in aggregated.items():
        _, cell, _ = key
        if cell_filter and cell != cell_filter:
            continue
        v1 = (agg.get(obj1_metric) or {}).get("mean")
        v2 = (agg.get(obj2_metric) or {}).get("mean")
        if v1 is None or v2 is None:
            continue
        # Normalize: we want to minimize both objectives on Pareto
        x = v1 if obj1_minimize else -v1
        y = v2 if obj2_minimize else -v2
        points.append((key, x, y))

    if not points:
        return []

    frontier_keys = []
    for i, (ki, xi, yi) in enumerate(points):
        dominated = False
        for j, (kj, xj, yj) in enumerate(points):
            if i == j:
                continue
            # j dominates i if j is <= i on both and < on at least one
            if xj <= xi and yj <= yi and (xj < xi or yj < yi):
                dominated = True
                break
        if not dominated:
            frontier_keys.append(ki)
    return frontier_keys


# ****************************************************
# Markdown report
# ****************************************************

def generate_markdown_report(
    aggregated: Dict[tuple, dict],
    delta_table: Dict[tuple, dict],
    phase_dominance: Dict[tuple, dict],
    run_id: str,
    output_path: str | Path,
) -> None:
    """
    Write a comprehensive Markdown report summarizing all benchmark results.
    Suitable for including in the paper's appendix or as a standalone benchmark report.
    """
    lines = []
    _l = lines.append

    _l(f"# ModeSwitch-LLM Benchmark Report")
    _l(f"**Run ID:** `{run_id}`\n")
    _l("---\n")

    # ── Section 1: Aggregate metrics per mode × cell ──────────────────────
    _l("## 1. Aggregate Metrics by Mode and Workload Cell\n")
    _l("All latency values in milliseconds (ms). Throughput in tokens/second (tps). "
       "VRAM in GB. Energy in J/token.\n")

    headers = ["Mode", "Cell", "Cond", "n", "TTFT (ms)", "TBT_mn (ms)",
               "TBT_p95 (ms)", "Dec_tps", "VRAM (GB)", "E/tok (J)", "ROUGE-L"]
    _l("| " + " | ".join(headers) + " |")
    _l("| " + " | ".join(["---"] * len(headers)) + " |")

    def _fmt(d, k, decimals=2):
        v = (d.get(k) or {}).get("mean") if isinstance(d.get(k), dict) else d.get(k)
        return f"{v:.{decimals}f}" if isinstance(v, (int, float)) and not math.isnan(v) else "—"

    for key in sorted(aggregated.keys()):
        agg = aggregated[key]
        mode, cell, cond = key
        row = [
            mode, cell, cond, str(agg["n"]),
            _fmt(agg, "ttft_ms", 1),
            _fmt(agg, "tbt_mean_ms", 1),
            _fmt(agg, "tbt_p95_ms", 1),
            _fmt(agg, "decode_throughput_tps", 1),
            _fmt(agg, "peak_vram_gb", 2),
            _fmt(agg, "energy_per_token_j", 4),
            _fmt(agg, "rougeL_f", 4),
        ]
        _l("| " + " | ".join(row) + " |")

    _l("")

    # ── Section 2: Delta vs baseline ──────────────────────────────────────
    _l("## 2. Relative Change vs FP16 Baseline (Δ%)\n")
    _l("Positive values = worse than baseline (for latency/energy/memory). "
       "Negative values = better than baseline. "
       "For ROUGE-L, negative = worse quality.\n")

    delta_headers = ["Mode", "Cell", "ΔTTFT", "ΔTBT_mn", "ΔTBT_p95",
                     "ΔDec_tps", "ΔE/tok", "ΔVRAM", "ΔROUGE-L"]
    _l("| " + " | ".join(delta_headers) + " |")
    _l("| " + " | ".join(["---"] * len(delta_headers)) + " |")

    for key, deltas in sorted(delta_table.items()):
        mode, cell, _ = key
        def dfmt(k):
            v = deltas.get(k)
            if v is None:
                return "—"
            sign = "+" if v > 0 else ""
            return f"{sign}{v:.1f}%"
        row = [
            mode, cell,
            dfmt("ttft_ms"), dfmt("tbt_mean_ms"), dfmt("tbt_p95_ms"),
            dfmt("decode_throughput_tps"), dfmt("energy_per_token_j"),
            dfmt("peak_vram_gb"), dfmt("rougeL_f"),
        ]
        _l("| " + " | ".join(row) + " |")
    _l("")

    # ── Section 3: Phase dominance ─────────────────────────────────────────
    _l("## 3. Phase Dominance Analysis\n")
    _l("Shows what fraction of total inference time is spent in prefill vs decode. "
       "`decode_prefill_ratio > 1` = decode-dominated workload.\n")

    phase_headers = ["Mode", "Cell", "Prefill (ms)", "Decode (ms)", "Pre%", "Dec%", "Dominated by"]
    _l("| " + " | ".join(phase_headers) + " |")
    _l("| " + " | ".join(["---"] * len(phase_headers)) + " |")

    for key, pd_info in sorted(phase_dominance.items()):
        mode, cell, _ = key
        row = [
            mode, cell,
            f"{pd_info['prefill_ms']:.1f}",
            f"{pd_info['decode_ms']:.1f}",
            f"{pd_info['prefill_pct']:.1f}%",
            f"{pd_info['decode_pct']:.1f}%",
            f"**{pd_info['dominated_by'].upper()}**",
        ]
        _l("| " + " | ".join(row) + " |")
    _l("")

    # ── Section 4: Key observations ───────────────────────────────────────
    _l("## 4. Key Observations\n")
    _l("*(Auto-generated from aggregate data)*\n")
    _generate_observations(lines, aggregated, delta_table, phase_dominance)
    _l("")

    # ── Write file ────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"[Reporter] Markdown report saved: {output_path}")


def _generate_observations(
    lines: List[str],
    aggregated: dict,
    delta_table: dict,
    phase_dominance: dict,
) -> None:
    """Auto-generate plain-English observations from the data."""
    obs = []

    # Find best energy mode per cell
    energy_by_cell: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for key, agg in aggregated.items():
        mode, cell, cond = key
        if cond != "baseline":
            continue
        e = (agg.get("energy_per_token_j") or {}).get("mean")
        if e is not None:
            energy_by_cell[cell].append((mode, e))

    for cell, entries in energy_by_cell.items():
        if not entries:
            continue
        best_mode, best_e = min(entries, key=lambda x: x[1])
        baseline_e = next((e for m, e in entries if m == "FP16_BASELINE"), None)
        if baseline_e and baseline_e > 0:
            pct = (baseline_e - best_e) / baseline_e * 100
            obs.append(
                f"- **{cell} cell**: Most energy-efficient mode is `{best_mode}` "
                f"({best_e:.4f} J/tok, {pct:.1f}% reduction vs FP16 baseline)."
            )

    # Find decode-dominated cells
    decode_dominated = []
    for key, pd_info in phase_dominance.items():
        mode, cell, cond = key
        if mode == "FP16_BASELINE" and cond == "baseline":
            if pd_info["dominated_by"] == "decode":
                decode_dominated.append(cell)

    if decode_dominated:
        obs.append(
            f"- Decode-dominated workload cells: **{', '.join(set(decode_dominated))}**. "
            "These cells benefit most from decode-phase optimizations "
            "(quantization, KV compression, speculative decoding)."
        )

    # Find prefill-dominated cells
    prefill_dominated = [
        cell for key, pd_info in phase_dominance.items()
        if key[0] == "FP16_BASELINE" and key[2] == "baseline"
        and pd_info["dominated_by"] == "prefill"
        for cell in [key[1]]
    ]
    if prefill_dominated:
        obs.append(
            f"- Prefill-dominated workload cells: **{', '.join(set(prefill_dominated))}**. "
            "These cells benefit most from FlashAttention-2 and prefill-batching optimizations."
        )

    if not obs:
        obs.append("- Insufficient data for auto-generated observations.")

    lines.extend(obs)


# ****************************************************
# Matplotlib plots
# ****************************************************

def plot_ttft_vs_tbt_scatter(
    aggregated: Dict[tuple, dict],
    output_path: str | Path,
    cell_filter: Optional[str] = None,
) -> None:
    """TTFT vs TBT_mean scatter plot, colored by inference mode."""
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    mode_colors = {
        "FP16_BASELINE": "#1f77b4",
        "W4A16_AWQ":     "#ff7f0e",
        "W8A8":          "#2ca02c",
        "SPEC_DECODE":   "#d62728",
        "KV_COMPRESS_H2O": "#9467bd",
        "FA2_ONLY":      "#8c564b",
    }
    markers = {"SS": "o", "SL": "s", "LS": "^", "LL": "D"}
    plotted_modes = set()

    for key, agg in aggregated.items():
        mode, cell, cond = key
        if cell_filter and cell != cell_filter:
            continue
        if cond != "baseline":
            continue
        ttft = (agg.get("ttft_ms") or {}).get("mean")
        tbt  = (agg.get("tbt_mean_ms") or {}).get("mean")
        if ttft is None or tbt is None:
            continue
        color  = mode_colors.get(mode, "#666")
        marker = markers.get(cell, "o")
        label  = mode if mode not in plotted_modes else "_nolegend_"
        ax.scatter(ttft, tbt, c=color, marker=marker, s=100, label=label, alpha=0.8, zorder=3)
        plotted_modes.add(mode)
        ax.annotate(cell, (ttft, tbt), textcoords="offset points", xytext=(5, 3), fontsize=7)

    ax.set_xlabel("TTFT (ms) — Prefill Latency", fontsize=11)
    ax.set_ylabel("Mean TBT (ms) — Decode Step Latency", fontsize=11)
    title = f"TTFT vs Mean TBT by Mode"
    if cell_filter:
        title += f" (cell={cell_filter})"
    ax.set_title(title, fontsize=13)
    ax.legend(title="Inference Mode", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info(f"[Reporter] TTFT vs TBT scatter saved: {output_path}")


def plot_energy_per_token_bar(
    aggregated: Dict[tuple, dict],
    output_path: str | Path,
) -> None:
    """Bar chart of energy per generated token across modes and cells."""
    if not _MATPLOTLIB_AVAILABLE:
        return

    cells = ["SS", "SL", "LS", "LL"]
    modes = []
    data_by_mode: Dict[str, List[Optional[float]]] = defaultdict(list)

    for key in sorted(aggregated.keys()):
        mode, cell, cond = key
        if cond != "baseline":
            continue
        if mode not in modes:
            modes.append(mode)

    for mode in modes:
        for cell in cells:
            found = None
            for key, agg in aggregated.items():
                m, c, cond = key
                if m == mode and c == cell and cond == "baseline":
                    found = (agg.get("energy_per_token_j") or {}).get("mean")
                    break
            data_by_mode[mode].append(found)

    if not modes or not any(any(v is not None for v in vals) for vals in data_by_mode.values()):
        logger.info("[Reporter] No energy data; skipping energy bar chart.")
        return

    x = np.arange(len(cells))
    width = 0.8 / len(modes)
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, mode in enumerate(modes):
        vals = data_by_mode[mode]
        heights = [v if v is not None else 0.0 for v in vals]
        offset = (i - len(modes) / 2 + 0.5) * width
        ax.bar(x + offset, heights, width * 0.9, label=mode, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cells)
    ax.set_xlabel("Workload Cell", fontsize=11)
    ax.set_ylabel("Energy per Generated Token (J/tok)", fontsize=11)
    ax.set_title("Energy per Generated Token by Mode and Workload Cell", fontsize=13)
    ax.legend(title="Mode", fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info(f"[Reporter] Energy bar chart saved: {output_path}")


def plot_tbt_cdf(
    results: List[dict],
    output_path: str | Path,
    cell_filter: str = "SL",
) -> None:
    """CDF of per-step TBT measurements across modes for a given workload cell."""
    if not _MATPLOTLIB_AVAILABLE:
        return

    # Re-collect raw TBT lists per mode
    tbt_by_mode: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        if r.get("workload_cell") != cell_filter or r.get("status") != "ok":
            continue
        mode = r.get("inference_mode", "unknown")
        # tbt_per_step_ms is not stored in summary JSON (only aggregates are)
        # Use tbt_mean_ms as a proxy single point (real CDF needs raw per-step data)
        v = r.get("tbt_mean_ms")
        if v is not None:
            tbt_by_mode[mode].append(v)

    if not tbt_by_mode:
        logger.info(f"[Reporter] No TBT data for cell={cell_filter}; skipping CDF plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, vals in tbt_by_mode.items():
        sorted_vals = sorted(vals)
        cdf = [(i + 1) / len(sorted_vals) for i in range(len(sorted_vals))]
        ax.plot(sorted_vals, cdf, label=mode, linewidth=2)

    ax.set_xlabel(f"Mean TBT (ms) — cell={cell_filter}", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title(f"CDF of Mean TBT | {cell_filter} Workload Cell", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info(f"[Reporter] TBT CDF plot saved: {output_path}")


def plot_memory_heatmap(
    aggregated: Dict[tuple, dict],
    output_path: str | Path,
) -> None:
    """Heatmap of peak VRAM usage: modes (rows) × cells (cols)."""
    if not _MATPLOTLIB_AVAILABLE:
        return

    cells = ["SS", "SL", "LS", "LL"]
    modes_ordered = [
        "FP16_BASELINE", "W4A16_AWQ", "W8A8",
        "SPEC_DECODE", "KV_COMPRESS_H2O", "FA2_ONLY",
    ]

    matrix = []
    row_labels = []
    for mode in modes_ordered:
        row = []
        has_data = False
        for cell in cells:
            found = None
            for key, agg in aggregated.items():
                m, c, cond = key
                if m == mode and c == cell and cond == "baseline":
                    found = (agg.get("peak_vram_gb") or {}).get("mean")
                    break
            row.append(found if found is not None else float("nan"))
            if found is not None:
                has_data = True
        if has_data:
            matrix.append(row)
            row_labels.append(mode)

    if not matrix:
        return

    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, max(4, len(row_labels) * 0.7)))
    im = ax.imshow(arr, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Peak VRAM (GB)")

    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(cells)):
            val = arr[i, j]
            if not math.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color="black")

    ax.set_title("Peak GPU VRAM Usage (GB) — Mode × Workload Cell", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info(f"[Reporter] Memory heatmap saved: {output_path}")


# ****************************************************
# Full post-run report generation
# ****************************************************

def generate_full_report(run_dir: str | Path) -> None:
    """
    Load all results from a run directory and produce the complete report suite:
    - summary stats JSON
    - delta table JSON
    - phase dominance JSON
    - Markdown report
    - All plots (if matplotlib available)
    """
    run_dir = Path(run_dir)
    results = load_results(run_dir)
    if not results:
        logger.error(f"No results found in {run_dir}")
        return

    run_id = run_dir.name
    report_dir = run_dir / "report"
    report_dir.mkdir(exist_ok=True)

    # Aggregate
    agg = aggregate_results(results)
    with open(report_dir / "aggregated.json", "w") as f:
        # Convert tuple keys to strings for JSON
        json.dump({str(k): v for k, v in agg.items()}, f, indent=2, default=str)

    # Deltas
    deltas = compute_delta_table(agg)
    with open(report_dir / "deltas.json", "w") as f:
        json.dump({str(k): v for k, v in deltas.items()}, f, indent=2, default=str)

    # Phase dominance
    pd_info = compute_phase_dominance(agg)
    with open(report_dir / "phase_dominance.json", "w") as f:
        json.dump({str(k): v for k, v in pd_info.items()}, f, indent=2, default=str)

    # Pareto
    pareto = find_pareto_frontier(agg, "energy_per_token_j", "rougeL_f")
    with open(report_dir / "pareto.json", "w") as f:
        json.dump([str(k) for k in pareto], f, indent=2)

    # Markdown
    generate_markdown_report(agg, deltas, pd_info, run_id, report_dir / "report.md")

    # Plots
    if _MATPLOTLIB_AVAILABLE:
        plot_ttft_vs_tbt_scatter(agg, report_dir / "ttft_vs_tbt_scatter.png")
        plot_energy_per_token_bar(agg, report_dir / "energy_per_token.png")
        plot_tbt_cdf(results, report_dir / "tbt_cdf_SL.png", cell_filter="SL")
        plot_memory_heatmap(agg, report_dir / "memory_heatmap.png")

    print(f"\n[Reporter] Full report written to: {report_dir}")


# ****************************************************
# CLI
# ****************************************************
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate benchmark report from run directory")
    parser.add_argument("run_dir", help="Path to the benchmark run directory")
    args = parser.parse_args()
    generate_full_report(args.run_dir)
