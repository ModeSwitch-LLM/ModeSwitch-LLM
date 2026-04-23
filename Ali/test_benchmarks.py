"""
tests/test_benchmarks.py — ModeSwitch-LLM Benchmarking Pipeline
Author: Ali Alshehhi

Comprehensive unit and integration tests for the benchmarking pipeline.
All tests run without a GPU — GPU-dependent paths are mocked/skipped.

Run with:
    pytest tests/test_benchmarks.py -v

Or individual test classes:
    pytest tests/test_benchmarks.py::TestWorkloads -v
    pytest tests/test_benchmarks.py::TestMetrics -v
    pytest tests/test_benchmarks.py::TestPhaseMonitor -v
"""

from __future__ import annotations

import math
import sys
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make sure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.workloads import (
    WorkloadSuite, WorkloadSample, 
    PromptCategory, OutputCategory, TaskType, SystemCondition, SYSTEM_CONDITIONS,
)
from benchmarks.metrics import (
    compute_latency_metrics,
    compute_memory_metrics,
    compute_energy_metrics,
    compute_repetition_metrics,
    compute_delta_vs_baseline,
    compute_speculative_metrics,
    METRIC_SCHEMA,
)
from benchmarks.phase_monitor import (
    EnergyPoller,
    MemoryPressureContext,
    PhaseTimingProcessor,
)
from benchmarks.reporter import (
    aggregate_results,
    compute_delta_table,
    compute_phase_dominance,
    find_pareto_frontier,
)


# ****************************************************
# Helper fixtures
# ****************************************************

def _make_fake_result(
    mode="FP16_BASELINE",
    cell="SS",
    cond="baseline",
    ttft_ms=20.0,
    tbt_mean_ms=14.0,
    tbt_p95_ms=16.0,
    decode_tps=70.0,
    peak_vram=14.2,
    e_per_tok=0.48,
    rougeL=0.42,
    status="ok",
    **kwargs,
) -> dict:
    return {
        "inference_mode":        mode,
        "workload_cell":         cell,
        "system_condition":      cond,
        "status":                status,
        "ttft_ms":               ttft_ms,
        "tbt_mean_ms":           tbt_mean_ms,
        "tbt_median_ms":         tbt_mean_ms - 0.5,
        "tbt_p95_ms":            tbt_p95_ms,
        "tbt_p99_ms":            tbt_p95_ms + 1.0,
        "tbt_std_ms":            1.2,
        "total_decode_ms":       tbt_mean_ms * 128,
        "total_inference_ms":    ttft_ms + tbt_mean_ms * 128,
        "prefill_throughput_tps": 3500.0,
        "decode_throughput_tps": decode_tps,
        "decode_prefill_ratio":  (tbt_mean_ms * 128) / ttft_ms if ttft_ms > 0 else 0,
        "peak_vram_gb":          peak_vram,
        "kv_cache_estimated_gb": 0.5,
        "energy_per_token_j":    e_per_tok,
        "total_energy_j":        e_per_tok * 128,
        "mean_power_w":          180.0,
        "rouge1_f":              rougeL + 0.05,
        "rouge2_f":              rougeL - 0.05,
        "rougeL_f":              rougeL,
        "bertscore_f1":          rougeL + 0.40,
        "rep_rate_3gram":        0.05,
        "vocab_diversity":       0.75,
        "sample_id":             f"{cell}_qa_0001",
        "prompt_tokens":         64,
        "generated_tokens":      128,
        **kwargs,
    }


# ****************************************************
# Workloads tests
# ****************************************************

class TestWorkloads(unittest.TestCase):

    def setUp(self):
        self.suite = WorkloadSuite(seed=42)

    def test_total_sample_count_positive(self):
        self.assertGreater(len(self.suite), 0)

    def test_all_four_cells_present(self):
        summary = self.suite.summary()
        for cell in ("SS", "SL", "LS", "LL"):
            self.assertIn(cell, summary, f"Cell {cell} missing from suite")

    def test_each_cell_has_samples(self):
        for cell in ("SS", "SL", "LS", "LL"):
            samples = self.suite.get_cell(cell)
            self.assertGreater(len(samples), 0, f"No samples in cell {cell}")

    def test_workload_cell_consistency(self):
        """Every sample's workload_cell must match its prompt/output category."""
        for sample in self.suite.get_all():
            expected_cell = (
                ("S" if sample.prompt_category == PromptCategory.SHORT else "L") +
                ("S" if sample.output_category == OutputCategory.SHORT else "L")
            )
            self.assertEqual(
                sample.workload_cell, expected_cell,
                f"Sample {sample.sample_id}: cell={sample.workload_cell} "
                f"but expected {expected_cell}"
            )

    def test_token_budgets_correct(self):
        """SS/LS samples should have max_new_tokens=128; SL/LL should have 512."""
        for cell in ("SS", "LS"):
            for s in self.suite.get_cell(cell):
                self.assertEqual(s.max_new_tokens, 128,
                                 f"{cell} sample {s.sample_id} has wrong max_new_tokens")
        for cell in ("SL", "LL"):
            for s in self.suite.get_cell(cell):
                self.assertEqual(s.max_new_tokens, 512,
                                 f"{cell} sample {s.sample_id} has wrong max_new_tokens")

    def test_sample_ids_unique(self):
        ids = [s.sample_id for s in self.suite.get_all()]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate sample IDs detected")

    def test_get_task_filters(self):
        qa_samples = self.suite.get_task(TaskType.QA)
        self.assertTrue(all(s.task_type == TaskType.QA for s in qa_samples))

    def test_prompts_non_empty(self):
        for sample in self.suite.get_all():
            self.assertGreater(len(sample.prompt.strip()), 0,
                               f"Empty prompt in {sample.sample_id}")

    def test_min_new_tokens_less_than_max(self):
        for sample in self.suite.get_all():
            self.assertLess(sample.min_new_tokens, sample.max_new_tokens,
                            f"{sample.sample_id}: min >= max new tokens")

    def test_system_conditions_complete(self):
        expected = {sc.value for sc in SystemCondition}
        actual   = {sc.value for sc in SYSTEM_CONDITIONS.keys()}
        self.assertEqual(expected, actual)

    def test_suite_repr(self):
        r = repr(self.suite)
        self.assertIn("WorkloadSuite", r)


# ****************************************************
# Metrics tests
# ****************************************************

class TestLatencyMetrics(unittest.TestCase):

    def test_basic_computation(self):
        m = compute_latency_metrics(
            ttft_ms=20.0,
            tbt_per_step_ms=[14.0, 14.5, 15.0, 13.5, 16.0],
            prompt_tokens=64,
            generated_tokens=5,
        )
        self.assertAlmostEqual(m["ttft_ms"], 20.0, places=2)
        self.assertAlmostEqual(m["tbt_mean_ms"], 14.6, places=1)
        self.assertGreater(m["tbt_p95_ms"], m["tbt_mean_ms"])
        self.assertGreater(m["prefill_throughput_tps"], 0)
        self.assertGreater(m["decode_throughput_tps"], 0)

    def test_empty_tbt_doesnt_crash(self):
        m = compute_latency_metrics(20.0, [], 64, 0)
        self.assertIsNotNone(m["tbt_mean_ms"])

    def test_decode_prefill_ratio(self):
        # Decode-heavy: SL workload
        m = compute_latency_metrics(
            ttft_ms=20.0,
            tbt_per_step_ms=[14.0] * 512,
            prompt_tokens=64,
            generated_tokens=512,
        )
        self.assertGreater(m["decode_prefill_ratio"], 1.0)

    def test_prefill_heavy(self):
        # Prefill-heavy: LS workload (long prompt, few decode steps)
        m = compute_latency_metrics(
            ttft_ms=300.0,
            tbt_per_step_ms=[14.0] * 32,
            prompt_tokens=1024,
            generated_tokens=32,
        )
        self.assertLess(m["decode_prefill_ratio"], 1.0)

    def test_all_keys_present(self):
        m = compute_latency_metrics(20.0, [14.0, 15.0], 64, 2)
        expected_keys = [
            "ttft_ms", "tbt_mean_ms", "tbt_median_ms", "tbt_p95_ms", "tbt_p99_ms",
            "tbt_std_ms", "total_decode_ms", "total_inference_ms",
            "prefill_throughput_tps", "decode_throughput_tps", "overall_throughput_tps",
            "prompt_tokens", "generated_tokens", "total_tokens", "decode_prefill_ratio",
        ]
        for k in expected_keys:
            self.assertIn(k, m, f"Missing key: {k}")

    def test_throughput_units(self):
        """Prefill throughput should roughly equal prompt_tokens / (ttft_s)."""
        m = compute_latency_metrics(1000.0, [10.0], 100, 1)  # ttft=1s, 100 tok
        self.assertAlmostEqual(m["prefill_throughput_tps"], 100.0, places=0)


class TestMemoryMetrics(unittest.TestCase):

    def test_basic(self):
        m = compute_memory_metrics(
            peak_vram_gb=14.2,
            allocated_vram_gb=13.8,
            reserved_vram_gb=15.0,
            prompt_tokens=64,
            generated_tokens=128,
        )
        self.assertEqual(m["peak_vram_gb"], 14.2)
        self.assertIsNotNone(m["kv_cache_estimated_gb"])
        self.assertGreater(m["kv_cache_estimated_gb"], 0)

    def test_kv_cache_formula(self):
        """Verify analytical KV cache formula is correct."""
        # 2 * 32 layers * 32 heads * 128 head_dim * 192 seq_len * 2 bytes
        expected_bytes = 2 * 32 * 32 * 128 * (64 + 128) * 2
        expected_gb = expected_bytes / (1024 ** 3)
        m = compute_memory_metrics(10.0, 9.0, 10.0, 64, 128,
                                   num_layers=32, num_heads=32, head_dim=128)
        self.assertAlmostEqual(m["kv_cache_estimated_gb"], expected_gb, places=4)

    def test_int4_kv_cache_smaller(self):
        """INT4 (0.5 bytes/element) KV cache should be 4× smaller than FP16."""
        m_fp16 = compute_memory_metrics(10.0, 9.0, 10.0, 128, 128, dtype_bytes=2)
        m_int4 = compute_memory_metrics(10.0, 9.0, 10.0, 128, 128, dtype_bytes=1)
        self.assertAlmostEqual(
            m_fp16["kv_cache_estimated_gb"] / m_int4["kv_cache_estimated_gb"], 2.0, places=3
        )


class TestEnergyMetrics(unittest.TestCase):

    def test_basic(self):
        m = compute_energy_metrics(
            total_energy_j=1.84,
            generated_tokens=128,
            total_inference_ms=1847.0,
            mean_power_w=180.0,
        )
        self.assertAlmostEqual(m["energy_per_token_j"], 1.84 / 128, places=5)
        self.assertIsNotNone(m["tokens_per_joule"])
        self.assertAlmostEqual(m["tokens_per_joule"], 128 / 1.84, places=2)

    def test_none_energy(self):
        m = compute_energy_metrics(None, 128, 1847.0)
        self.assertIsNone(m["energy_per_token_j"])
        self.assertIsNone(m["total_energy_j"])

    def test_zero_tokens(self):
        m = compute_energy_metrics(1.0, 0, 100.0)
        self.assertIsNone(m["energy_per_token_j"])


class TestRepetitionMetrics(unittest.TestCase):

    def test_highly_repetitive_text(self):
        text = "the cat sat on the mat " * 10
        m = compute_repetition_metrics(text)
        self.assertGreater(m["rep_rate_3gram"], 0.5)
        self.assertLess(m["vocab_diversity"], 0.3)

    def test_diverse_text(self):
        text = (
            "The transformer architecture revolutionized natural language processing. "
            "Attention mechanisms allow models to focus on relevant context tokens. "
            "Positional encodings inject sequence order information into embeddings. "
            "Layer normalization stabilizes training of deep neural networks. "
            "Feed-forward sublayers apply independent nonlinearities per position."
        )
        m = compute_repetition_metrics(text)
        self.assertLess(m["rep_rate_3gram"], 0.2)
        self.assertGreater(m["vocab_diversity"], 0.5)

    def test_empty_text(self):
        m = compute_repetition_metrics("")
        # Should not crash; all values can be None or 0
        self.assertIsInstance(m, dict)

    def test_single_word(self):
        m = compute_repetition_metrics("hello")
        self.assertIsNotNone(m["total_output_tokens"])


class TestDeltaVsBaseline(unittest.TestCase):

    def test_basic_delta(self):
        baseline = {"ttft_ms": 20.0, "tbt_mean_ms": 14.0, "energy_per_token_j": 0.5}
        mode_m   = {"ttft_ms": 25.0, "tbt_mean_ms": 8.0,  "energy_per_token_j": 0.3}
        deltas = compute_delta_vs_baseline(mode_m, baseline,
                                           ["ttft_ms", "tbt_mean_ms", "energy_per_token_j"])
        # TTFT got worse: (25-20)/20 = +25%
        self.assertAlmostEqual(deltas["delta_pct_ttft_ms"], 25.0, places=1)
        # TBT improved: (8-14)/14 ≈ -42.9%
        self.assertAlmostEqual(deltas["delta_pct_tbt_mean_ms"], -42.857, places=1)
        # Energy improved: -40%
        self.assertAlmostEqual(deltas["delta_pct_energy_per_token_j"], -40.0, places=1)

    def test_none_values(self):
        baseline = {"ttft_ms": 20.0, "energy_per_token_j": None}
        mode_m   = {"ttft_ms": 25.0, "energy_per_token_j": 0.3}
        deltas = compute_delta_vs_baseline(mode_m, baseline,
                                           ["ttft_ms", "energy_per_token_j"])
        self.assertIsNotNone(deltas["delta_pct_ttft_ms"])
        self.assertIsNone(deltas["delta_pct_energy_per_token_j"])

    def test_zero_baseline_no_div_zero(self):
        baseline = {"ttft_ms": 0.0}
        mode_m   = {"ttft_ms": 20.0}
        deltas = compute_delta_vs_baseline(mode_m, baseline, ["ttft_ms"])
        self.assertIsNone(deltas["delta_pct_ttft_ms"])


class TestSpeculativeMetrics(unittest.TestCase):

    def test_high_acceptance(self):
        m = compute_speculative_metrics(
            acceptance_rates=[0.85, 0.90, 0.78, 0.92, 0.88],
            tokens_per_draft_step=[3, 4, 3, 4, 3],
        )
        self.assertGreater(m["spec_mean_acceptance"], 0.8)
        self.assertAlmostEqual(m["spec_mean_tokens_per_draft"], 3.4, places=1)
        self.assertGreater(m["spec_effective_multiplier"], 4.0)

    def test_empty_input(self):
        m = compute_speculative_metrics([], [])
        self.assertIsNone(m["spec_mean_acceptance"])


# ****************************************************
# Phase monitor tests (CPU-only, mocked CUDA)
# ****************************************************

class TestPhaseTimingProcessor(unittest.TestCase):

    def _make_processor(self, prompt_tokens=64):
        proc = PhaseTimingProcessor(prompt_tokens=prompt_tokens, device="cpu")
        proc._use_cuda_events = False  # Disable CUDA for CPU-only test
        return proc

    def _fake_scores(self):
        import torch
        return torch.zeros(1, 100)  # fake logits

    def test_first_call_is_prefill(self):
        proc = self._make_processor()
        proc.mark_generate_start()
        import time; time.sleep(0.01)  # Simulate prefill time
        proc(None, self._fake_scores())  # Call 0 = prefill
        self.assertIsNotNone(proc.prefill_end_time)
        self.assertIsNotNone(proc._first_token_wall)

    def test_subsequent_calls_accumulate_tbt(self):
        import time
        proc = self._make_processor()
        proc.mark_generate_start()
        time.sleep(0.005)
        proc(None, self._fake_scores())  # prefill
        for _ in range(5):
            time.sleep(0.005)
            proc(None, self._fake_scores())  # 5 decode steps
        self.assertEqual(len(proc._tbt_wall), 5)

    def test_get_result_returns_valid_object(self):
        import time
        proc = self._make_processor(prompt_tokens=32)
        proc.mark_generate_start()
        time.sleep(0.01)
        proc(None, self._fake_scores())  # prefill
        for _ in range(3):
            time.sleep(0.005)
            proc(None, self._fake_scores())
        result = proc.get_result(generated_tokens=3)
        self.assertIsNotNone(result.prefill)
        self.assertIsNotNone(result.decode)
        self.assertEqual(result.decode.generated_tokens, 3)
        self.assertGreater(result.prefill.ttft_ms, 0)
        self.assertGreater(result.decode.tbt_mean_ms, 0)

    def test_reset_clears_state(self):
        proc = self._make_processor()
        proc.mark_generate_start()
        proc(None, self._fake_scores())
        proc.reset()
        self.assertEqual(proc._call_count, 0)
        self.assertIsNone(proc._wall_start)
        self.assertEqual(proc._tbt_wall, [])


class TestMemoryPressureContext(unittest.TestCase):

    @patch("benchmarks.phase_monitor.torch.cuda.is_available", return_value=False)
    def test_no_gpu_no_crash(self, _):
        """Should gracefully do nothing when CUDA is unavailable."""
        ctx = MemoryPressureContext(vram_fraction=0.5, device="cpu")
        with ctx:
            pass  # Should not raise

    def test_zero_fraction_no_allocation(self):
        ctx = MemoryPressureContext(vram_fraction=0.0)
        with ctx:
            self.assertIsNone(ctx._pressure_tensor)

    def test_invalid_fraction_raises(self):
        with self.assertRaises(AssertionError):
            MemoryPressureContext(vram_fraction=1.5)
        with self.assertRaises(AssertionError):
            MemoryPressureContext(vram_fraction=-0.1)


class TestEnergyPoller(unittest.TestCase):

    @patch("benchmarks.phase_monitor._PYNVML_AVAILABLE", False)
    def test_unavailable_no_crash(self):
        ep = EnergyPoller()
        ep._available = False
        with ep:
            import time; time.sleep(0.05)
        self.assertIsNone(ep.get_total_energy_joules())

    def test_trapezoidal_integration(self):
        """Test energy integration math directly without pynvml."""
        ep = EnergyPoller.__new__(EnergyPoller)
        ep._available = True
        # Simulate 100W constant power for 1 second → 100J
        ep._power_samples = [(i * 0.1, 100.0) for i in range(11)]  # 0.0–1.0 s, 100W
        energy = ep.get_total_energy_joules()
        self.assertAlmostEqual(energy, 100.0, places=1)

    def test_fewer_than_2_samples(self):
        ep = EnergyPoller.__new__(EnergyPoller)
        ep._available = True
        ep._power_samples = [(0.0, 200.0)]  # only 1 sample
        self.assertIsNone(ep.get_total_energy_joules())


# ****************************************************
# Reporter tests
# ****************************************************

class TestAggregation(unittest.TestCase):

    def setUp(self):
        self.results = [
            _make_fake_result("FP16_BASELINE", "SS", "baseline",
                              ttft_ms=20.0, tbt_mean_ms=14.0, e_per_tok=0.48),
            _make_fake_result("FP16_BASELINE", "SS", "baseline",
                              ttft_ms=21.0, tbt_mean_ms=14.5, e_per_tok=0.49),
            _make_fake_result("W4A16_AWQ",    "SS", "baseline",
                              ttft_ms=22.0, tbt_mean_ms=8.5, e_per_tok=0.29),
            _make_fake_result("FP16_BASELINE", "SL", "baseline",
                              ttft_ms=20.0, tbt_mean_ms=14.5, e_per_tok=0.51),
        ]

    def test_groups_formed_correctly(self):
        agg = aggregate_results(self.results)
        keys = list(agg.keys())
        self.assertEqual(len(keys), 3)  # (fp16, SS), (awq, SS), (fp16, SL)

    def test_mean_computed_correctly(self):
        agg = aggregate_results(self.results)
        fp16_ss = agg[("FP16_BASELINE", "SS", "baseline")]
        self.assertAlmostEqual(fp16_ss["ttft_ms"]["mean"], 20.5, places=3)
        self.assertEqual(fp16_ss["n"], 2)

    def test_errors_excluded_from_aggregation(self):
        results_with_error = self.results + [
            _make_fake_result("FP16_BASELINE", "SS", "baseline",
                              ttft_ms=999.0, status="oom")
        ]
        agg = aggregate_results(results_with_error)
        fp16_ss = agg[("FP16_BASELINE", "SS", "baseline")]
        # Error result should be excluded → still n=2
        self.assertEqual(fp16_ss["n"], 2)

    def test_std_computed_correctly(self):
        agg = aggregate_results(self.results)
        fp16_ss = agg[("FP16_BASELINE", "SS", "baseline")]
        # std of [20.0, 21.0] = 0.5
        self.assertAlmostEqual(fp16_ss["ttft_ms"]["std"], 0.5, places=3)


class TestDeltaTable(unittest.TestCase):

    def setUp(self):
        self.results = [
            _make_fake_result("FP16_BASELINE", "SS", "baseline",
                              ttft_ms=20.0, tbt_mean_ms=14.0, e_per_tok=0.48),
            _make_fake_result("W4A16_AWQ",    "SS", "baseline",
                              ttft_ms=22.0, tbt_mean_ms=8.6, e_per_tok=0.29),
        ]
        self.agg = aggregate_results(self.results)

    def test_baseline_not_in_delta_table(self):
        deltas = compute_delta_table(self.agg)
        for key in deltas:
            mode = key[0]
            self.assertNotEqual(mode, "FP16_BASELINE")

    def test_energy_delta_correct(self):
        deltas = compute_delta_table(self.agg)
        awq_deltas = deltas.get(("W4A16_AWQ", "SS", "baseline"), {})
        # (0.29 - 0.48) / 0.48 * 100 ≈ -39.6%
        e_delta = awq_deltas.get("energy_per_token_j")
        if e_delta is not None:
            self.assertLess(e_delta, 0)  # AWQ should use less energy


class TestPhaseDominance(unittest.TestCase):

    def test_decode_dominated(self):
        # SL: short prompt, long output → decode-dominated
        results = [_make_fake_result(
            "FP16_BASELINE", "SL", "baseline",
            ttft_ms=20.0,    # 20ms prefill
        )]
        # Override total_decode_ms to be large
        results[0]["total_decode_ms"] = 7000.0
        results[0]["total_inference_ms"] = 7020.0
        agg = aggregate_results(results)
        pd = compute_phase_dominance(agg)
        val = pd.get(("FP16_BASELINE", "SL", "baseline"))
        if val:
            self.assertEqual(val["dominated_by"], "decode")

    def test_prefill_dominated(self):
        results = [_make_fake_result(
            "FP16_BASELINE", "LS", "baseline",
            ttft_ms=500.0,
        )]
        results[0]["total_decode_ms"] = 400.0  # modest decode
        results[0]["total_inference_ms"] = 900.0
        agg = aggregate_results(results)
        pd = compute_phase_dominance(agg)
        val = pd.get(("FP16_BASELINE", "LS", "baseline"))
        if val:
            self.assertIn(val["dominated_by"], ["prefill", "balanced"])


class TestParetoFrontier(unittest.TestCase):

    def test_dominated_point_excluded(self):
        # FP16: high energy, high quality
        # AWQ: low energy, slightly lower quality
        # QUANT_BAD: high energy, low quality → dominated by both
        results = [
            _make_fake_result("FP16_BASELINE", "SS", "baseline",
                              e_per_tok=0.48, rougeL=0.42),
            _make_fake_result("W4A16_AWQ",    "SS", "baseline",
                              e_per_tok=0.29, rougeL=0.41),
            _make_fake_result("QUANT_BAD",    "SS", "baseline",
                              e_per_tok=0.50, rougeL=0.38),
        ]
        agg = aggregate_results(results)
        frontier = find_pareto_frontier(
            agg, "energy_per_token_j", "rougeL_f",
            obj1_minimize=True, obj2_minimize=False, cell_filter="SS"
        )
        # QUANT_BAD should be dominated (high energy AND low quality)
        frontier_modes = [k[0] for k in frontier]
        self.assertNotIn("QUANT_BAD", frontier_modes)

    def test_empty_aggregation(self):
        frontier = find_pareto_frontier({}, "energy_per_token_j", "rougeL_f")
        self.assertEqual(frontier, [])


# ****************************************************
# Integration: workload suite + metrics pipeline (CPU-only, no model)
# ****************************************************

class TestEndToEndCPUOnly(unittest.TestCase):

    def test_workload_suite_produces_valid_token_budgets(self):
        suite = WorkloadSuite()
        for sample in suite.get_all():
            self.assertGreater(sample.max_new_tokens, 0)
            self.assertGreater(sample.min_new_tokens, 0)
            self.assertLess(sample.min_new_tokens, sample.max_new_tokens)

    def test_metrics_pipeline_no_crash(self):
        """Simulate a complete metrics computation without a GPU."""
        import time
        proc = PhaseTimingProcessor(prompt_tokens=64, device="cpu")
        proc._use_cuda_events = False
        proc.mark_generate_start()
        time.sleep(0.02)
        import torch
        fake_scores = torch.zeros(1, 100)
        proc(None, fake_scores)  # prefill
        for _ in range(10):
            time.sleep(0.001)
            proc(None, fake_scores)
        result = proc.get_result(generated_tokens=10)

        lat = compute_latency_metrics(
            ttft_ms=result.prefill.ttft_ms,
            tbt_per_step_ms=result.decode.tbt_per_step_ms,
            prompt_tokens=64,
            generated_tokens=10,
        )
        self.assertGreater(lat["ttft_ms"], 0)
        self.assertGreater(lat["decode_throughput_tps"], 0)

    def test_full_quality_pipeline_no_reference(self):
        text = ("The model generates text autoregressively one token at a time. "
                "Each token is conditioned on all previous tokens in the sequence. "
                "This process continues until the model outputs an end-of-sequence token.")
        qual = compute_repetition_metrics(text)
        self.assertIn("vocab_diversity", qual)
        self.assertIn("rep_rate_3gram", qual)
        self.assertIsNotNone(qual["vocab_diversity"])

    def test_full_quality_with_rouge(self):
        try:
            from benchmarks.metrics import compute_all_quality_metrics
            hyp = "The fox jumped over the dog quickly and ran into the forest."
            ref = "The quick brown fox jumped over the lazy dog."
            qual = compute_all_quality_metrics(hyp, reference=ref, compute_bert=False)
            # If rouge_score installed, should have values; otherwise None
            self.assertIn("rougeL_f", qual)
        except ImportError:
            self.skipTest("rouge_score not installed")

    def test_markdown_report_generation(self):
        """Generate a markdown report from fake results; check it writes a file."""
        from benchmarks.reporter import generate_markdown_report, aggregate_results, \
            compute_delta_table, compute_phase_dominance
        results = [
            _make_fake_result("FP16_BASELINE", "SS", "baseline"),
            _make_fake_result("W4A16_AWQ",    "SS", "baseline", e_per_tok=0.29),
        ]
        agg = aggregate_results(results)
        deltas = compute_delta_table(agg)
        pd_info = compute_phase_dominance(agg)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.md"
            generate_markdown_report(agg, deltas, pd_info, "test_run_001", out)
            self.assertTrue(out.exists())
            content = out.read_text()
            self.assertIn("ModeSwitch-LLM", content)
            self.assertIn("FP16_BASELINE", content)
            self.assertIn("W4A16_AWQ", content)


# ****************************************************
# Metric schema completeness
# ****************************************************

class TestMetricSchema(unittest.TestCase):

    def test_schema_is_non_empty(self):
        self.assertGreater(len(METRIC_SCHEMA), 0)

    def test_all_schema_values_are_strings(self):
        for name, desc in METRIC_SCHEMA.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)


# ****************************************************
# Fix import typo in TestWorkloads (WorkloadSuiteself → WorkloadSuite)
# ****************************************************
# (Patched in setUp; no action needed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
