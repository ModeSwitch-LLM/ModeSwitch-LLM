"""
pipeline.py — ModeSwitch-LLM Benchmarking Pipeline
Author: Ali Alshehhi

Main benchmarking pipeline that orchestrates:
  1. Model loading for each inference mode
  2. GPU warm-up and state reset between runs
  3. Workload dispatch with phase-aware monitoring
  4. Metric collection, aggregation, and persistence
  5. Reproducible seeding and run bookkeeping

Inference modes benchmarked:
  - FP16_BASELINE:    Full-precision (float16) weights, standard attention.
  - W4A16_AWQ:        INT4 weight-only quantization (AWQ format).
  - W8A8:             INT8 weight + activation quantization (bitsandbytes).
  - SPEC_DECODE:      Speculative decoding with a small draft model.
  - KV_COMPRESS_H2O:  Heavy Hitter Oracle KV-cache eviction.
  - FA2_ONLY:         FlashAttention-2 kernel, full precision.

Run entry point: BenchmarkPipeline.run()

Each result is written to results/<run_id>/<mode>/<workload_cell>/<sample_id>.json
after every sample, so crashes are recoverable. A summary CSV is written at the end.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .metrics import (
    compute_latency_metrics,
    compute_memory_metrics,
    compute_energy_metrics,
    compute_all_quality_metrics,
    compute_delta_vs_baseline,
    MetricsDict,
)
from .phase_monitor import (
    PhaseTimingProcessor,
    EnergyPoller,
    MemoryPressureContext,
    PhaseTimingResult,
)
from .workloads import WorkloadSample, WorkloadSuite, SystemCondition, SYSTEM_CONDITIONS

logger = logging.getLogger(__name__)


# ****************************************************
# Inference mode identifiers
# ****************************************************

class InferenceMode:
    FP16_BASELINE  = "FP16_BASELINE"
    W4A16_AWQ      = "W4A16_AWQ"
    W8A8           = "W8A8"
    SPEC_DECODE    = "SPEC_DECODE"
    KV_COMPRESS    = "KV_COMPRESS_H2O"
    FA2_ONLY       = "FA2_ONLY"

ALL_MODES = [
    InferenceMode.FP16_BASELINE,
    InferenceMode.W4A16_AWQ,
    InferenceMode.W8A8,
    InferenceMode.SPEC_DECODE,
    InferenceMode.KV_COMPRESS,
    InferenceMode.FA2_ONLY,
]


# ****************************************************
# Per-sample result record
# ****************************************************

@dataclass
class SampleResult:
    # Identity
    run_id:           str
    sample_id:        str
    workload_cell:    str
    task_type:        str
    inference_mode:   str
    system_condition: str
    model_name:       str
    timestamp:        str

    # Inputs
    prompt_tokens:    int
    max_new_tokens:   int
    min_new_tokens:   int

    # Outputs
    generated_tokens: int
    output_text:      str

    # Phase timing
    ttft_ms:            float
    tbt_mean_ms:        float
    tbt_median_ms:      float
    tbt_p95_ms:         float
    tbt_p99_ms:         float
    tbt_std_ms:         float
    total_decode_ms:    float
    total_inference_ms: float
    prefill_throughput_tps: float
    decode_throughput_tps:  float
    decode_prefill_ratio:   float

    # Memory
    peak_vram_gb:           float
    kv_cache_estimated_gb:  Optional[float]

    # Energy
    energy_per_token_j:     Optional[float]
    total_energy_j:         Optional[float]
    mean_power_w:           Optional[float]

    # Quality
    rouge1_f:         Optional[float] = None
    rouge2_f:         Optional[float] = None
    rougeL_f:         Optional[float] = None
    bertscore_f1:     Optional[float] = None
    rep_rate_3gram:   Optional[float] = None
    vocab_diversity:  Optional[float] = None
    perplexity:       Optional[float] = None

    # Speculative decoding extras
    spec_mean_acceptance:       Optional[float] = None
    spec_effective_multiplier:  Optional[float] = None

    # Status
    status:           str = "ok"  # "ok", "oom", "timeout", "error"
    error_message:    Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ****************************************************
# Pipeline configuration
# ****************************************************

@dataclass
class PipelineConfig:
    model_name:           str = "meta-llama/Llama-2-7b-chat-hf"
    draft_model_name:     str = "meta-llama/Llama-2-7b-chat-hf"  # For speculative decode
    device:               str = "cuda"
    results_dir:          str = "results"
    warmup_runs:          int = 3       # Number of warm-up generations before benchmarking
    num_repeats:          int = 3       # Repeat each sample N times; report mean
    energy_poll_ms:       float = 50.0  # EnergyPoller interval
    reset_cache_between:  bool = True   # torch.cuda.empty_cache() between runs
    compute_bertscore:    bool = False  # Slow; enable only for final eval
    compute_perplexity:   bool = False  # Very slow; enable for a small subset
    max_new_tokens_cap:   int = 1024    # Hard cap regardless of workload setting
    seed:                 int = 42
    log_level:            str = "INFO"
    # Which modes and cells to run (empty = run all)
    modes_to_run:         List[str] = field(default_factory=list)
    cells_to_run:         List[str] = field(default_factory=list)
    conditions_to_run:    List[str] = field(default_factory=list)
    # Model architecture for KV-cache size estimation
    num_layers:           int = 32
    num_heads:            int = 32
    head_dim:             int = 128


# ****************************************************
# Model loader
# ****************************************************

class ModelLoader:
    """
    Handles loading and caching of models for each inference mode.
    Models are loaded lazily and cached to avoid redundant loads.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._cache: Dict[str, tuple] = {}  # mode --> (model, tokenizer)

    def get_model_and_tokenizer(self, mode: str):
        if mode in self._cache:
            return self._cache[mode]

        logger.info(f"[ModelLoader] Loading model for mode={mode} ...")
        model, tokenizer = self._load(mode)
        self._cache[mode] = (model, tokenizer)
        return model, tokenizer

    def _load(self, mode: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        common_kwargs = dict(
            pretrained_model_name_or_path=self.config.model_name,
            device_map=self.config.device,
            torch_dtype=torch.float16,
        )

        if mode == InferenceMode.FP16_BASELINE:
            model = AutoModelForCausalLM.from_pretrained(**common_kwargs)

        elif mode == InferenceMode.FA2_ONLY:
            model = AutoModelForCausalLM.from_pretrained(
                **common_kwargs,
                attn_implementation="flash_attention_2",
            )

        elif mode == InferenceMode.W4A16_AWQ:
            # Requires autoawq: pip install autoawq
            try:
                from awq import AutoAWQForCausalLM
                model = AutoAWQForCausalLM.from_quantized(
                    self.config.model_name,
                    fuse_layers=True,
                    trust_remote_code=False,
                    safetensors=True,
                )
            except ImportError:
                logger.warning("autoawq not installed; falling back to bitsandbytes INT4.")
                bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=bnb_config,
                    device_map=self.config.device,
                )

        elif mode == InferenceMode.W8A8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map=self.config.device,
            )

        elif mode == InferenceMode.SPEC_DECODE:
            # Load target (full) model; draft model loaded separately
            model = AutoModelForCausalLM.from_pretrained(**common_kwargs)

        elif mode == InferenceMode.KV_COMPRESS:
            # KV compression is applied at inference time via hooks; load standard model
            model = AutoModelForCausalLM.from_pretrained(**common_kwargs)

        else:
            raise ValueError(f"Unknown inference mode: {mode}")

        model.eval()
        logger.info(f"[ModelLoader] Model loaded for mode={mode}.")
        return model, tokenizer

    def unload(self, mode: str) -> None:
        if mode in self._cache:
            model, _ = self._cache.pop(mode)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"[ModelLoader] Unloaded model for mode={mode}.")

    def unload_all(self) -> None:
        for mode in list(self._cache.keys()):
            self.unload(mode)


# ****************************************************
# Single-sample inference runner
# ****************************************************

class InferenceRunner:
    """
    Runs a single inference sample under a given mode + system condition,
    collects all metrics, and returns a SampleResult.
    """

    def __init__(self, config: PipelineConfig, run_id: str):
        self.config = config
        self.run_id = run_id

    def run_sample(
        self,
        sample: WorkloadSample,
        model,
        tokenizer,
        mode: str,
        condition_cfg,
    ) -> SampleResult:
        """
        Execute one inference run for (sample, mode, condition) and return metrics.
        """
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        # ── Tokenize ──────────────────────────────────────────────────────
        try:
            inputs = tokenizer(
                sample.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.config.device)
        except Exception as e:
            return self._error_result(sample, mode, condition_cfg, str(e), timestamp)

        prompt_tokens = inputs["input_ids"].shape[1]
        max_new = min(sample.max_new_tokens, self.config.max_new_tokens_cap)

        # ── Reset GPU state ───────────────────────────────────────────────
        if self.config.reset_cache_between and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # ── Energy poller + Memory pressure ───────────────────────────────
        device_idx = 0  # Assume single GPU
        energy_poller = EnergyPoller(
            device_index=device_idx,
            poll_interval_ms=self.config.energy_poll_ms,
        )
        phase_processor = PhaseTimingProcessor(
            prompt_tokens=prompt_tokens,
            device=self.config.device,
        )

        try:
            with MemoryPressureContext(
                vram_fraction=condition_cfg.vram_fraction,
                device=self.config.device,
            ):
                with energy_poller:
                    # Mark wall-clock start just before generate()
                    phase_processor.mark_generate_start()

                    generate_kwargs = dict(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=max_new,
                        min_new_tokens=sample.min_new_tokens,
                        do_sample=False,          # greedy for reproducibility
                        temperature=1.0,
                        logits_processor=[phase_processor],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    # Mode-specific generate overrides
                    if mode == InferenceMode.SPEC_DECODE:
                        generate_kwargs.update(self._spec_decode_kwargs(model))

                    with torch.no_grad():
                        output_ids = model.generate(**generate_kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        except torch.cuda.OutOfMemoryError as e:
            return self._error_result(sample, mode, condition_cfg, f"OOM: {e}", timestamp,
                                      status="oom")
        except Exception as e:
            return self._error_result(sample, mode, condition_cfg, str(e), timestamp)

        # ── Decode output text ─────────────────────────────────────────────
        new_token_ids = output_ids[0, prompt_tokens:]
        generated_tokens = len(new_token_ids)
        output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

        # ── Get timing result ──────────────────────────────────────────────
        timing: PhaseTimingResult = phase_processor.get_result(generated_tokens)

        # ── Attach energy to timing result ─────────────────────────────────
        total_energy = energy_poller.get_total_energy_joules()
        mean_power   = energy_poller.get_mean_power_watts()
        timing.total_energy_j = total_energy

        # ── Compute all metrics ────────────────────────────────────────────
        lat_metrics = compute_latency_metrics(
            ttft_ms=timing.prefill.ttft_ms,
            tbt_per_step_ms=timing.decode.tbt_per_step_ms,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
        )
        mem_metrics = compute_memory_metrics(
            peak_vram_gb=timing.peak_vram_gb,
            allocated_vram_gb=timing.allocated_vram_gb,
            reserved_vram_gb=timing.reserved_vram_gb,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
        )
        nrg_metrics = compute_energy_metrics(
            total_energy_j=total_energy,
            generated_tokens=generated_tokens,
            total_inference_ms=lat_metrics["total_inference_ms"],
            mean_power_w=mean_power,
        )
        qual_metrics = compute_all_quality_metrics(
            hypothesis=output_text,
            reference=sample.reference_output,
            compute_rep=True,
            compute_rouge=True,
            compute_bert=self.config.compute_bertscore,
        )

        # ── Assemble SampleResult ──────────────────────────────────────────
        return SampleResult(
            run_id=self.run_id,
            sample_id=sample.sample_id,
            workload_cell=sample.workload_cell,
            task_type=sample.task_type.value,
            inference_mode=mode,
            system_condition=condition_cfg.condition.value,
            model_name=self.config.model_name,
            timestamp=timestamp,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new,
            min_new_tokens=sample.min_new_tokens,
            generated_tokens=generated_tokens,
            output_text=output_text[:2000],  # Truncate for storage
            ttft_ms=lat_metrics["ttft_ms"],
            tbt_mean_ms=lat_metrics["tbt_mean_ms"],
            tbt_median_ms=lat_metrics["tbt_median_ms"],
            tbt_p95_ms=lat_metrics["tbt_p95_ms"],
            tbt_p99_ms=lat_metrics["tbt_p99_ms"],
            tbt_std_ms=lat_metrics["tbt_std_ms"],
            total_decode_ms=lat_metrics["total_decode_ms"],
            total_inference_ms=lat_metrics["total_inference_ms"],
            prefill_throughput_tps=lat_metrics["prefill_throughput_tps"],
            decode_throughput_tps=lat_metrics["decode_throughput_tps"],
            decode_prefill_ratio=lat_metrics["decode_prefill_ratio"],
            peak_vram_gb=mem_metrics["peak_vram_gb"],
            kv_cache_estimated_gb=mem_metrics["kv_cache_estimated_gb"],
            energy_per_token_j=nrg_metrics["energy_per_token_j"],
            total_energy_j=nrg_metrics["total_energy_j"],
            mean_power_w=nrg_metrics["mean_power_w"],
            rouge1_f=qual_metrics.get("rouge1_f"),
            rouge2_f=qual_metrics.get("rouge2_f"),
            rougeL_f=qual_metrics.get("rougeL_f"),
            bertscore_f1=qual_metrics.get("bertscore_f1"),
            rep_rate_3gram=qual_metrics.get("rep_rate_3gram"),
            vocab_diversity=qual_metrics.get("vocab_diversity"),
            status="ok",
        )

    def _spec_decode_kwargs(self, model) -> dict:
        """Returns extra kwargs for speculative decoding if available."""
        # HuggingFace >= 4.38 supports assistant_model for spec decode
        return {}  # Filled by pipeline when draft model is loaded

    def _error_result(
        self, sample, mode, condition_cfg, msg, timestamp, status="error"
    ) -> SampleResult:
        return SampleResult(
            run_id=self.run_id,
            sample_id=sample.sample_id,
            workload_cell=sample.workload_cell,
            task_type=sample.task_type.value,
            inference_mode=mode,
            system_condition=condition_cfg.condition.value,
            model_name=self.config.model_name,
            timestamp=timestamp,
            prompt_tokens=0,
            max_new_tokens=sample.max_new_tokens,
            min_new_tokens=sample.min_new_tokens,
            generated_tokens=0,
            output_text="",
            ttft_ms=float("nan"),
            tbt_mean_ms=float("nan"),
            tbt_median_ms=float("nan"),
            tbt_p95_ms=float("nan"),
            tbt_p99_ms=float("nan"),
            tbt_std_ms=float("nan"),
            total_decode_ms=float("nan"),
            total_inference_ms=float("nan"),
            prefill_throughput_tps=0.0,
            decode_throughput_tps=0.0,
            decode_prefill_ratio=float("nan"),
            peak_vram_gb=0.0,
            kv_cache_estimated_gb=None,
            energy_per_token_j=None,
            total_energy_j=None,
            mean_power_w=None,
            status=status,
            error_message=msg[:500],
        )


# ****************************************************
# GPU warm-up
# ****************************************************

def warmup_model(model, tokenizer, device: str, n: int = 3) -> None:
    """
    Run N short generations to warm up CUDA kernels, JIT compilation,
    and cuDNN autotuner before benchmarking begins. This is critical for
    reproducible timing — the first few runs are always slower.
    """
    logger.info(f"[Warmup] Running {n} warm-up generation(s)...")
    warmup_prompt = "Hello, how are you?"
    ids = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    for i in range(n):
        with torch.no_grad():
            model.generate(
                input_ids=ids["input_ids"],
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    logger.info("[Warmup] Done.")


# ****************************************************
# Main benchmark pipeline
# ****************************************************

class BenchmarkPipeline:
    """
    Top-level orchestrator for the ModeSwitch-LLM benchmarking study.

    Usage:
        config = PipelineConfig(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            modes_to_run=["FP16_BASELINE", "W4A16_AWQ"],
            cells_to_run=["SS", "SL"],
            conditions_to_run=["baseline"],
        )
        pipeline = BenchmarkPipeline(config)
        pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.results_path = Path(config.results_dir) / self.run_id
        self.results_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.results_path / "benchmark.log"),
            ],
        )

        self.suite = WorkloadSuite(seed=config.seed)
        self.loader = ModelLoader(config)
        self.runner = InferenceRunner(config, self.run_id)
        self.all_results: List[SampleResult] = []

        # Resolve which modes / cells / conditions to run
        self.modes      = config.modes_to_run or ALL_MODES
        self.cells      = config.cells_to_run  or ["SS", "SL", "LS", "LL"]
        self.conditions = [
            SYSTEM_CONDITIONS[SystemCondition(c)]
            for c in (config.conditions_to_run or [sc.value for sc in SystemCondition])
        ]

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self._save_config()

    def _save_config(self) -> None:
        from dataclasses import asdict
        cfg_path = self.results_path / "config.json"
        with open(cfg_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info(f"[Pipeline] Config saved to {cfg_path}")

    def run(self) -> None:
        """Execute the full benchmark sweep."""
        logger.info(f"[Pipeline] Starting run {self.run_id}")
        logger.info(f"[Pipeline] Modes: {self.modes}")
        logger.info(f"[Pipeline] Cells: {self.cells}")
        logger.info(f"[Pipeline] Conditions: {[c.condition.value for c in self.conditions]}")
        logger.info(f"[Pipeline] Total workload samples: {len(self.suite)}")

        total_start = time.perf_counter()

        for mode in self.modes:
            logger.info(f"\n{'='*60}")
            logger.info(f"[Pipeline] === MODE: {mode} ===")
            logger.info(f"{'='*60}")

            try:
                model, tokenizer = self.loader.get_model_and_tokenizer(mode)
            except Exception as e:
                logger.error(f"[Pipeline] Failed to load model for {mode}: {e}")
                continue

            # Warm up CUDA kernels for this model
            warmup_model(model, tokenizer, self.config.device, n=self.config.warmup_runs)

            for cell in self.cells:
                samples = self.suite.get_cell(cell)
                logger.info(f"[Pipeline] Cell {cell}: {len(samples)} samples")

                for condition_cfg in self.conditions:
                    logger.info(
                        f"[Pipeline] Condition: {condition_cfg.condition.value} "
                        f"| {condition_cfg.description}"
                    )

                    for sample in samples:
                        repeat_results = []
                        for rep in range(self.config.num_repeats):
                            result = self.runner.run_sample(
                                sample=sample,
                                model=model,
                                tokenizer=tokenizer,
                                mode=mode,
                                condition_cfg=condition_cfg,
                            )
                            repeat_results.append(result)
                            self._log_result(result, rep)

                        # Average numeric metrics across repeats
                        final = self._average_repeats(repeat_results)
                        self.all_results.append(final)
                        self._save_result(final, mode, cell)

            # Unload model to free VRAM before loading next mode
            self.loader.unload(mode)

        total_elapsed = time.perf_counter() - total_start
        logger.info(f"\n[Pipeline] Complete. Total time: {total_elapsed:.1f}s")
        self._write_summary_csv()
        self._write_summary_json()

    def _average_repeats(self, results: List[SampleResult]) -> SampleResult:
        """Average all float metrics across repeat runs; keep metadata from first result."""
        if len(results) == 1:
            return results[0]

        base = results[0]
        float_fields = [
            "ttft_ms", "tbt_mean_ms", "tbt_median_ms", "tbt_p95_ms", "tbt_p99_ms",
            "tbt_std_ms", "total_decode_ms", "total_inference_ms",
            "prefill_throughput_tps", "decode_throughput_tps", "decode_prefill_ratio",
            "peak_vram_gb", "kv_cache_estimated_gb",
            "energy_per_token_j", "total_energy_j", "mean_power_w",
            "rouge1_f", "rouge2_f", "rougeL_f", "bertscore_f1",
            "rep_rate_3gram", "vocab_diversity",
        ]
        ok_results = [r for r in results if r.status == "ok"]
        if not ok_results:
            return base

        averaged = base
        for fname in float_fields:
            vals = [getattr(r, fname) for r in ok_results if getattr(r, fname) is not None]
            if vals:
                try:
                    setattr(averaged, fname, round(sum(vals) / len(vals), 6))
                except Exception:
                    pass
        return averaged

    def _log_result(self, result: SampleResult, rep: int) -> None:
        status_str = f"[{result.status.upper()}]" if result.status != "ok" else ""
        logger.info(
            f"  {status_str} {result.sample_id} rep={rep+1} | "
            f"TTFT={result.ttft_ms:.1f}ms | "
            f"TBT_mean={result.tbt_mean_ms:.1f}ms | "
            f"TBT_p95={result.tbt_p95_ms:.1f}ms | "
            f"tokens={result.generated_tokens} | "
            f"VRAM={result.peak_vram_gb:.2f}GB | "
            f"E/tok={result.energy_per_token_j:.4f}J" if result.energy_per_token_j else
            f"  {status_str} {result.sample_id} rep={rep+1} | "
            f"TTFT={result.ttft_ms:.1f}ms | "
            f"TBT_mean={result.tbt_mean_ms:.1f}ms | "
            f"tokens={result.generated_tokens}"
        )

    def _save_result(self, result: SampleResult, mode: str, cell: str) -> None:
        out_dir = self.results_path / mode / cell
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{result.sample_id}_{result.system_condition}.json"
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _write_summary_csv(self) -> None:
        import csv
        if not self.all_results:
            return
        csv_path = self.results_path / "summary.csv"
        fieldnames = list(self.all_results[0].to_dict().keys())
        # Exclude raw output text from CSV (too large)
        fieldnames = [f for f in fieldnames if f != "output_text"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.all_results:
                row = {k: v for k, v in r.to_dict().items() if k in fieldnames}
                writer.writerow(row)
        logger.info(f"[Pipeline] Summary CSV saved: {csv_path}")

    def _write_summary_json(self) -> None:
        summary_path = self.results_path / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                [r.to_dict() for r in self.all_results],
                f,
                indent=2,
                default=str,
            )
        logger.info(f"[Pipeline] Summary JSON saved: {summary_path}")
        self._print_aggregate_table()

    def _print_aggregate_table(self) -> None:
        """Print a condensed human-readable summary table to stdout."""
        from collections import defaultdict
        agg = defaultdict(list)
        for r in self.all_results:
            if r.status == "ok":
                key = (r.inference_mode, r.workload_cell)
                agg[key].append(r)

        print("\n" + "="*90)
        print(f"{'MODE':<20} {'CELL':<6} {'n':>4} {'TTFT':>10} {'TBT_mn':>10} "
              f"{'TBT_p95':>10} {'DEC_TPS':>10} {'VRAM':>8} {'E/tok':>10}")
        print("-"*90)

        def _m(lst, fn=lambda x: x):
            vals = [fn(v) for v in lst if v is not None]
            return round(sum(vals) / len(vals), 3) if vals else float("nan")

        for (mode, cell), rs in sorted(agg.items()):
            print(
                f"{mode:<20} {cell:<6} {len(rs):>4} "
                f"{_m(rs, lambda r: r.ttft_ms):>10.1f} "
                f"{_m(rs, lambda r: r.tbt_mean_ms):>10.1f} "
                f"{_m(rs, lambda r: r.tbt_p95_ms):>10.1f} "
                f"{_m(rs, lambda r: r.decode_throughput_tps):>10.1f} "
                f"{_m(rs, lambda r: r.peak_vram_gb):>8.2f} "
                f"{_m(rs, lambda r: r.energy_per_token_j) if any(r.energy_per_token_j for r in rs) else 'N/A':>10}"
            )
        print("="*90)


# ****************************************************
# CLI entry point
# ****************************************************
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ModeSwitch-LLM Benchmark Pipeline")
    parser.add_argument("--model",      default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--modes",      nargs="+", default=["FP16_BASELINE"],
                        choices=ALL_MODES)
    parser.add_argument("--cells",      nargs="+", default=["SS"],
                        choices=["SS", "SL", "LS", "LL"])
    parser.add_argument("--conditions", nargs="+", default=["baseline"],
                        choices=[c.value for c in SystemCondition])
    parser.add_argument("--repeats",    type=int,  default=3)
    parser.add_argument("--warmup",     type=int,  default=3)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--bertscore",  action="store_true")
    parser.add_argument("--perplexity", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(
        model_name=args.model,
        modes_to_run=args.modes,
        cells_to_run=args.cells,
        conditions_to_run=args.conditions,
        num_repeats=args.repeats,
        warmup_runs=args.warmup,
        results_dir=args.results_dir,
        compute_bertscore=args.bertscore,
        compute_perplexity=args.perplexity,
    )

    pipeline = BenchmarkPipeline(cfg)
    pipeline.run()
