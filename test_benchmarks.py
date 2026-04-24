from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reporter import (
    aggregate_results,
    build_failure_summary,
    compute_delta_table,
    compute_phase_dominance,
    find_pareto_frontier,
    generate_markdown_report,
    prepare_results,
)

from modes import build_runtime_mode_by_name

def _make_fake_result(
    mode_name: str = "fp16_baseline",
    workload_name: str = "short_prompt_short_output",
    prompt_tokens_target: int = 128,
    max_new_tokens: int = 32,
    num_requests_in_batch: int = 1,
    repeated_prefix: bool = False,
    memory_pressure: bool = False,
    ttft_ms: float = 20.0,
    avg_tbt_ms: float = 14.0,
    tbt_p95_ms: float = 16.0,
    prefill_latency_ms: float = 20.0,
    decode_latency_ms: float = 448.0,
    total_latency_ms: float = 468.0,
    tokens_per_second: float = 70.0,
    decode_throughput_tps: float = 71.0,
    prefill_throughput_tps: float = 3200.0,
    peak_gpu_memory_mb: float = 14500.0,
    energy_per_token_j: float = 0.48,
    reference_rouge_l_f1: float = 0.42,
    baseline_similarity_rouge_l_f1: float | None = None,
    quality_degradation_vs_baseline: float | None = None,
    success: bool = True,
    error: str | None = None,
    error_type: str | None = None,
    **extra,
):
    row = {
        "mode_name": mode_name,
        "workload_name": workload_name,
        "backend": "vllm",
        "trial_index": 0,
        "prompt_tokens_target": prompt_tokens_target,
        "max_new_tokens": max_new_tokens,
        "repeated_prefix": repeated_prefix,
        "memory_pressure": memory_pressure,
        "num_requests_in_batch": num_requests_in_batch,
        "ttft_ms": ttft_ms,
        "avg_tbt_ms": avg_tbt_ms,
        "tbt_median_ms": avg_tbt_ms - 0.5,
        "tbt_p95_ms": tbt_p95_ms,
        "tbt_p99_ms": tbt_p95_ms + 1.0,
        "tbt_std_ms": 1.2,
        "prefill_latency_ms": prefill_latency_ms,
        "decode_latency_ms": decode_latency_ms,
        "total_latency_ms": total_latency_ms,
        "tokens_per_second": tokens_per_second,
        "batched_tokens_per_second": None,
        "prefill_throughput_tps": prefill_throughput_tps,
        "decode_throughput_tps": decode_throughput_tps,
        "decode_prefill_ratio": (decode_latency_ms / prefill_latency_ms) if prefill_latency_ms else None,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "reserved_gpu_memory_mb": peak_gpu_memory_mb + 500.0,
        "cpu_ram_peak_mb": 1500.0,
        "cpu_ram_delta_mb": 120.0,
        "gpu_peak_delta_mb": 300.0,
        "gpu_reserved_delta_mb": 400.0,
        "kv_cache_estimate_mb": 512.0,
        "avg_power_w": 180.0,
        "energy_joules": energy_per_token_j * max(max_new_tokens, 1),
        "energy_per_token_j": energy_per_token_j,
        "reference_rouge_l_f1": reference_rouge_l_f1,
        "reference_token_f1": 0.50,
        "baseline_similarity_rouge_l_f1": baseline_similarity_rouge_l_f1,
        "quality_degradation_vs_baseline": quality_degradation_vs_baseline,
        "success": success,
        "error": error,
        "error_type": error_type,
    }
    row.update(extra)
    return row


class TestPrepareResults(unittest.TestCase):

    def test_infers_workload_cell_for_standard_workload(self):
        rows = prepare_results([
            _make_fake_result(
                workload_name="short_prompt_short_output",
                prompt_tokens_target=128,
                max_new_tokens=32,
            )
        ])
        self.assertEqual(rows[0]["workload_cell"], "SS")

    def test_infers_special_cells(self):
        prefix_row = prepare_results([
            _make_fake_result(repeated_prefix=True)
        ])[0]
        mem_row = prepare_results([
            _make_fake_result(memory_pressure=True)
        ])[0]
        self.assertEqual(prefix_row["workload_cell"], "PREFIX")
        self.assertEqual(mem_row["workload_cell"], "MEM")

    def test_infers_system_condition(self):
        base = prepare_results([_make_fake_result(num_requests_in_batch=1)])[0]
        batch = prepare_results([_make_fake_result(num_requests_in_batch=4)])[0]
        mem = prepare_results([_make_fake_result(memory_pressure=True)])[0]
        self.assertEqual(base["system_condition"], "baseline")
        self.assertEqual(batch["system_condition"], "batch_4")
        self.assertEqual(mem["system_condition"], "memory_pressure")


class TestAggregation(unittest.TestCase):

    def setUp(self):
        self.results = [
            _make_fake_result(
                mode_name="fp16_baseline",
                workload_name="short_prompt_short_output",
                ttft_ms=20.0,
                avg_tbt_ms=14.0,
                energy_per_token_j=0.48,
            ),
            _make_fake_result(
                mode_name="fp16_baseline",
                workload_name="short_prompt_short_output",
                ttft_ms=21.0,
                avg_tbt_ms=14.5,
                energy_per_token_j=0.49,
            ),
            _make_fake_result(
                mode_name="awq_4bit",
                workload_name="short_prompt_short_output",
                ttft_ms=22.0,
                avg_tbt_ms=8.5,
                energy_per_token_j=0.29,
                peak_gpu_memory_mb=6200.0,
                reference_rouge_l_f1=0.41,
            ),
            _make_fake_result(
                mode_name="fp16_baseline",
                workload_name="short_prompt_long_output",
                max_new_tokens=128,
                avg_tbt_ms=14.8,
                decode_latency_ms=1800.0,
                total_latency_ms=1820.0,
            ),
        ]

    def test_groups_formed_correctly(self):
        agg = aggregate_results(self.results)
        self.assertEqual(len(agg), 3)

    def test_mean_computed_correctly(self):
        agg = aggregate_results(self.results)
        row = agg[("fp16_baseline", "short_prompt_short_output", "baseline")]
        self.assertAlmostEqual(row["ttft_ms"]["mean"], 20.5, places=3)
        self.assertEqual(row["n"], 2)

    def test_errors_excluded_from_metric_aggregation(self):
        results = self.results + [
            _make_fake_result(
                mode_name="fp16_baseline",
                workload_name="short_prompt_short_output",
                ttft_ms=999.0,
                success=False,
                error="OOM",
                error_type="OutOfMemoryError",
            )
        ]
        agg = aggregate_results(results)
        row = agg[("fp16_baseline", "short_prompt_short_output", "baseline")]
        self.assertEqual(row["n"], 2)


class TestFailureSummary(unittest.TestCase):

    def test_failures_are_counted(self):
        results = [
            _make_fake_result(mode_name="fp16_baseline"),
            _make_fake_result(
                mode_name="fp16_baseline",
                success=False,
                error="OOM",
                error_type="OutOfMemoryError",
            ),
        ]
        summary = build_failure_summary(results)
        row = summary[("fp16_baseline", "short_prompt_short_output", "baseline")]
        self.assertEqual(row["num_runs"], 2)
        self.assertEqual(row["num_failures"], 1)
        self.assertAlmostEqual(row["failure_rate"], 0.5, places=4)


class TestDeltaTable(unittest.TestCase):

    def setUp(self):
        self.results = [
            _make_fake_result(
                mode_name="fp16_baseline",
                workload_name="short_prompt_short_output",
                ttft_ms=20.0,
                avg_tbt_ms=14.0,
                decode_throughput_tps=70.0,
                energy_per_token_j=0.48,
                peak_gpu_memory_mb=14500.0,
                reference_rouge_l_f1=0.42,
            ),
            _make_fake_result(
                mode_name="awq_4bit",
                workload_name="short_prompt_short_output",
                ttft_ms=22.0,
                avg_tbt_ms=8.6,
                decode_throughput_tps=115.0,
                energy_per_token_j=0.29,
                peak_gpu_memory_mb=6200.0,
                reference_rouge_l_f1=0.41,
            ),
        ]
        self.agg = aggregate_results(self.results)

    def test_baseline_not_in_delta_table(self):
        deltas = compute_delta_table(self.agg, baseline_mode="fp16_baseline")
        for key in deltas:
            self.assertNotEqual(key[0], "fp16_baseline")

    def test_energy_delta_is_negative_for_awq(self):
        deltas = compute_delta_table(self.agg, baseline_mode="fp16_baseline")
        row = deltas[("awq_4bit", "short_prompt_short_output", "baseline")]
        self.assertLess(row["energy_per_token_j"], 0.0)


class TestPhaseDominance(unittest.TestCase):

    def test_decode_dominated(self):
        agg = aggregate_results([
            _make_fake_result(
                mode_name="fp16_baseline",
                workload_name="short_prompt_long_output",
                max_new_tokens=128,
                prefill_latency_ms=20.0,
                decode_latency_ms=7000.0,
                total_latency_ms=7020.0,
            )
        ])
        pd = compute_phase_dominance(agg)
        row = pd[("fp16_baseline", "short_prompt_long_output", "baseline")]
        self.assertEqual(row["dominated_by"], "decode")

    def test_prefill_dominated(self):
        agg = aggregate_results([
            _make_fake_result(
                mode_name="fp16_baseline",
                workload_name="long_prompt_short_output",
                prompt_tokens_target=1024,
                max_new_tokens=32,
                prefill_latency_ms=500.0,
                decode_latency_ms=300.0,
                total_latency_ms=800.0,
            )
        ])
        pd = compute_phase_dominance(agg)
        row = pd[("fp16_baseline", "long_prompt_short_output", "baseline")]
        self.assertIn(row["dominated_by"], {"prefill", "balanced"})


class TestParetoFrontier(unittest.TestCase):

    def test_dominated_point_excluded(self):
        agg = aggregate_results([
            _make_fake_result(mode_name="fp16_baseline", energy_per_token_j=0.48, reference_rouge_l_f1=0.42),
            _make_fake_result(mode_name="awq_4bit", energy_per_token_j=0.29, reference_rouge_l_f1=0.41),
            _make_fake_result(mode_name="quant_bad", energy_per_token_j=0.50, reference_rouge_l_f1=0.38),
        ])
        frontier = find_pareto_frontier(
            agg,
            obj1_metric="energy_per_token_j",
            obj2_metric="reference_rouge_l_f1",
            obj1_minimize=True,
            obj2_minimize=False,
        )
        modes = [key[0] for key in frontier]
        self.assertNotIn("quant_bad", modes)


class TestModeResolution(unittest.TestCase):

    def test_hybrid_prefix_mode_can_be_built_by_name(self):
        mode = build_runtime_mode_by_name("gptq_plus_prefix_caching")
        self.assertEqual(mode.name, "gptq_plus_prefix_caching")
        self.assertTrue(mode.prefix_caching)
        self.assertEqual(mode.quantization, "gptq")

    def test_hybrid_cont_batch_mode_can_be_built_by_name(self):
        mode = build_runtime_mode_by_name("int8_plus_continuous_batching")
        self.assertEqual(mode.name, "int8_plus_continuous_batching")
        self.assertTrue(mode.continuous_batching)
        self.assertEqual(mode.quantization, "compressed-tensors")


class TestMarkdownReport(unittest.TestCase):
    def test_markdown_report_generation(self):
        results = [
            _make_fake_result(mode_name="fp16_baseline"),
            _make_fake_result(
                mode_name="awq_4bit",
                energy_per_token_j=0.29,
                peak_gpu_memory_mb=6200.0,
                reference_rouge_l_f1=0.41,
            ),
        ]
        aggregated = aggregate_results(results)
        deltas = compute_delta_table(aggregated, baseline_mode="fp16_baseline")
        phase = compute_phase_dominance(aggregated)
        failures = build_failure_summary(results)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.md"
            generate_markdown_report(
                aggregated=aggregated,
                delta_table=deltas,
                phase_dominance=phase,
                failure_summary=failures,
                run_id="test_run_001",
                output_path=out,
                quality_metric_name="reference_rouge_l_f1",
            )
            self.assertTrue(out.exists())
            content = out.read_text(encoding="utf-8")
            self.assertIn("ModeSwitch-LLM Benchmark Report", content)
            self.assertIn("fp16_baseline", content)
            self.assertIn("awq_4bit", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)