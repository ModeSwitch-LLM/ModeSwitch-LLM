"""
phase_monitor.py — ModeSwitch-LLM Benchmarking Pipeline
Author: Ali Alshehhi

Provides fine-grained, phase-separated timing for LLM inference using a combination
of CUDA events (for GPU-side timing) and Python-side wall-clock bookkeeping.

Two fundamental phases of autoregressive inference are tracked separately:

  PREFILL (a.k.a. "time to first token"):
    The model processes the full input prompt in a single forward pass,
    populating the KV cache. This phase is compute-bound.
    We measure: TTFT = wall time from tokenization complete → first token sampled.

  DECODE (a.k.a. "time between tokens"):
    The model generates tokens autoregressively, one step at a time.
    This phase is memory-bandwidth-bound on most hardware.
    We measure per-step TBT and report: mean, median, p95, p99, std.

Architecture:
  - PhaseMonitor attaches to HuggingFace generate() via a custom LogitsProcessor.
    The processor fires on every forward pass; we detect the phase boundary by
    counting calls (call 0 = prefill, calls 1..N = decode steps).
  - CUDA events provide sub-millisecond GPU-side timing. We record a pair of
    (start, end) CUDA events around each phase and synchronize after generation.
  - An EnergyPoller background thread samples GPU power draw via pynvml at
    configurable intervals to compute energy per generated token.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# pynvml is optional; if unavailable, energy metrics will be None
try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False
    logger.warning(
        "pynvml not found. Energy metrics will be unavailable. "
        "Install with: pip install nvidia-ml-py3"
    )


# ****************************************************
# Data classes
# ****************************************************

@dataclass
class PrefillMetrics:
    """All metrics collected during the prefill phase."""
    ttft_ms:            float     # Wall-clock time to first token (ms)
    ttft_cuda_ms:       float     # GPU-side time to first token via CUDA events (ms)
    prompt_tokens:      int       # Number of input tokens processed
    prefill_throughput: float     # tokens/second during prefill (prompt_tokens / ttft_s)
    prefill_energy_j:   Optional[float] = None  # Joules consumed during prefill


@dataclass
class DecodeMetrics:
    """All metrics collected during the decode phase."""
    tbt_per_step_ms:    List[float]    # Per-token wall-clock TBT in ms
    tbt_cuda_per_step:  List[float]    # Per-token GPU-side TBT via CUDA events (ms)
    generated_tokens:   int
    decode_throughput:  float          # tokens/second during decode
    total_decode_ms:    float          # Sum of all TBT steps
    tbt_mean_ms:        float
    tbt_median_ms:      float
    tbt_p95_ms:         float
    tbt_p99_ms:         float
    tbt_std_ms:         float
    decode_energy_j:    Optional[float] = None
    energy_per_token_j: Optional[float] = None  # decode_energy_j / generated_tokens


@dataclass
class PhaseTimingResult:
    """Combined result from a single monitored inference call."""
    prefill:            PrefillMetrics
    decode:             DecodeMetrics
    total_wall_ms:      float          # End-to-end wall time
    total_energy_j:     Optional[float] = None
    peak_vram_gb:       float = 0.0
    allocated_vram_gb:  float = 0.0
    reserved_vram_gb:   float = 0.0
    error:              Optional[str] = None


# ****************************************************
# Energy poller
# ****************************************************

class EnergyPoller:
    """
    Background thread that polls GPU power draw via pynvml at a fixed interval.

    Usage:
        with EnergyPoller(device_index=0, poll_interval_ms=50) as ep:
            # run inference here
            energy_j = ep.get_total_energy_joules()
    """

    def __init__(self, device_index: int = 0, poll_interval_ms: float = 50.0):
        if not _PYNVML_AVAILABLE:
            self._available = False
            return
        self._available = True
        self._device_index = device_index
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._power_samples: List[Tuple[float, float]] = []  # (timestamp, watts)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            logger.warning(f"pynvml init failed: {e}. Energy metrics disabled.")
            self._available = False

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)  # milliwatts
                self._power_samples.append((time.perf_counter(), mw / 1000.0))
            except Exception:
                pass
            self._stop_event.wait(timeout=self._poll_interval_s)

    def __enter__(self) -> "EnergyPoller":
        if not self._available:
            return self
        self._power_samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        if not self._available:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_total_energy_joules(self) -> Optional[float]:
        """Integrate power over time using the trapezoidal rule."""
        if not self._available or len(self._power_samples) < 2:
            return None
        total = 0.0
        for i in range(1, len(self._power_samples)):
            t0, w0 = self._power_samples[i - 1]
            t1, w1 = self._power_samples[i]
            dt = t1 - t0
            total += 0.5 * (w0 + w1) * dt  # trapezoid rule --> joules
        return total

    def get_mean_power_watts(self) -> Optional[float]:
        if not self._available or not self._power_samples:
            return None
        return sum(w for _, w in self._power_samples) / len(self._power_samples)

    def get_samples(self) -> List[Tuple[float, float]]:
        return list(self._power_samples)


# ****************************************************
# HuggingFace LogitsProcessor for phase-boundary detection
# ****************************************************

class PhaseTimingProcessor:
    """
    A HuggingFace-compatible LogitsProcessor that hooks into model.generate()
    to record TTFT and per-step TBT using both wall-clock and CUDA event timing.

    Usage:
        processor = PhaseTimingProcessor(prompt_tokens=64)
        output = model.generate(
            ...,
            logits_processor=[processor],
        )
        result = processor.get_result(
            generated_tokens=output.shape[1] - input_ids.shape[1]
        )

    Note on CUDA events:
        We record a CUDA event at the START of each logits processor call.
        The gap between event[0] and event[1] is the prefill GPU time.
        The gap between event[i] and event[i+1] for i>=1 is the i-th TBT.
        This captures GPU compute time but excludes Python scheduling overhead;
        combining with wall-clock gives us a complete picture.
    """

    def __init__(self, prompt_tokens: int, device: str = "cuda"):
        self._prompt_tokens = prompt_tokens
        self._device = device
        self._call_count = 0

        # Wall-clock timestamps (perf_counter for sub-ms precision)
        self._wall_start: Optional[float] = None       # Set just before generate()
        self._first_token_wall: Optional[float] = None
        self._tbt_wall: List[float] = []               # Per-decode-step durations
        self._last_decode_wall: Optional[float] = None

        # CUDA events
        self._use_cuda_events = torch.cuda.is_available()
        self._cuda_events: List[torch.cuda.Event] = []

        # Prefill energy split time (so EnergyPoller can be split)
        self.prefill_end_time: Optional[float] = None

    def mark_generate_start(self) -> None:
        """Call this immediately before model.generate() is invoked."""
        if self._use_cuda_events:
            torch.cuda.synchronize()
        self._wall_start = time.perf_counter()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Called by HuggingFace generate() after each forward pass.
        Call 0: end of prefill → record TTFT.
        Calls 1+: end of each decode step → record TBT.
        """
        now = time.perf_counter()

        # Record CUDA event
        if self._use_cuda_events:
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self._cuda_events.append(evt)

        if self._call_count == 0:
            # First call: prefill just completed, first token being sampled
            if self._wall_start is not None:
                self._first_token_wall = now
            self.prefill_end_time = now
            self._last_decode_wall = now
        else:
            # Subsequent calls: a decode step just completed
            if self._last_decode_wall is not None:
                self._tbt_wall.append((now - self._last_decode_wall) * 1000.0)  # → ms
            self._last_decode_wall = now

        self._call_count += 1
        return scores  # pass-through; no modification

    def get_result(self, generated_tokens: int) -> PhaseTimingResult:
        """
        Build the full PhaseTimingResult after generation is complete.
        Call torch.cuda.synchronize() before calling this.
        """
        if self._use_cuda_events and self._cuda_events:
            torch.cuda.synchronize()

        # ── TTFT ──────────────────────────────────────────────────────────
        if self._wall_start and self._first_token_wall:
            ttft_wall_ms = (self._first_token_wall - self._wall_start) * 1000.0
        else:
            ttft_wall_ms = float("nan")

        ttft_cuda_ms = float("nan")
        if self._use_cuda_events and len(self._cuda_events) >= 1:
            # Time from just before generate() is hard to record as CUDA event
            # (no forward pass yet). We approximate via event[0] elapsed from
            # the first event; note this underestimates by overhead before first pass.
            # For now report as NaN; in practice users can use wall_start + sync trick.
            ttft_cuda_ms = float("nan")

        prefill_throughput = (
            self._prompt_tokens / (ttft_wall_ms / 1000.0)
            if ttft_wall_ms > 0 and not (ttft_wall_ms != ttft_wall_ms)
            else 0.0
        )

        prefill = PrefillMetrics(
            ttft_ms=ttft_wall_ms,
            ttft_cuda_ms=ttft_cuda_ms,
            prompt_tokens=self._prompt_tokens,
            prefill_throughput=prefill_throughput,
        )

        # ── TBT / Decode ──────────────────────────────────────────────────
        tbts = self._tbt_wall if self._tbt_wall else [0.0]
        total_decode_ms = sum(tbts)
        decode_throughput = (
            generated_tokens / (total_decode_ms / 1000.0) if total_decode_ms > 0 else 0.0
        )

        # CUDA per-step TBT: consecutive event pairs starting from index 1
        cuda_tbts: List[float] = []
        if self._use_cuda_events and len(self._cuda_events) >= 2:
            for i in range(1, len(self._cuda_events)):
                try:
                    dt = self._cuda_events[i - 1].elapsed_time(self._cuda_events[i])
                    cuda_tbts.append(dt)
                except Exception:
                    cuda_tbts.append(float("nan"))

        sorted_tbts = sorted(tbts)
        n = len(sorted_tbts)

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return float("nan")
            idx = (p / 100.0) * (len(data) - 1)
            lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
            frac = idx - lo
            return data[lo] * (1 - frac) + data[hi] * frac

        mean_tbt   = sum(tbts) / n
        median_tbt = percentile(sorted_tbts, 50)
        p95_tbt    = percentile(sorted_tbts, 95)
        p99_tbt    = percentile(sorted_tbts, 99)
        std_tbt    = (sum((x - mean_tbt) ** 2 for x in tbts) / n) ** 0.5 if n > 1 else 0.0

        decode = DecodeMetrics(
            tbt_per_step_ms=tbts,
            tbt_cuda_per_step=cuda_tbts,
            generated_tokens=generated_tokens,
            decode_throughput=decode_throughput,
            total_decode_ms=total_decode_ms,
            tbt_mean_ms=mean_tbt,
            tbt_median_ms=median_tbt,
            tbt_p95_ms=p95_tbt,
            tbt_p99_ms=p99_tbt,
            tbt_std_ms=std_tbt,
        )

        # ── Memory ────────────────────────────────────────────────────────
        peak_vram = alloc_vram = reserv_vram = 0.0
        if torch.cuda.is_available():
            peak_vram   = torch.cuda.max_memory_allocated() / (1024 ** 3)
            alloc_vram  = torch.cuda.memory_allocated()     / (1024 ** 3)
            reserv_vram = torch.cuda.memory_reserved()      / (1024 ** 3)

        total_wall_ms = (
            (ttft_wall_ms + total_decode_ms)
            if not (ttft_wall_ms != ttft_wall_ms)
            else total_decode_ms
        )

        return PhaseTimingResult(
            prefill=prefill,
            decode=decode,
            total_wall_ms=total_wall_ms,
            peak_vram_gb=peak_vram,
            allocated_vram_gb=alloc_vram,
            reserved_vram_gb=reserv_vram,
        )

    def reset(self) -> None:
        self._call_count = 0
        self._wall_start = None
        self._first_token_wall = None
        self._tbt_wall.clear()
        self._last_decode_wall = None
        self._cuda_events.clear()
        self.prefill_end_time = None


# ****************************************************
# CUDA-event-based precise prefill timer (alternative approach)
# ****************************************************

class CUDAPrefillTimer:
    """
    Records two CUDA events — one immediately before and one immediately after
    the first forward pass of generate() — to get a hardware-accurate TTFT.

    Requires monkey-patching the model's forward() method before generation.
    Use within PhaseMonitor.run_monitored() for transparent wrapping.
    """

    def __init__(self):
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event:   Optional[torch.cuda.Event] = None
        self._call_count = 0

    def pre_forward_hook(self, module, args, kwargs):
        """Register as a forward pre-hook on the model."""
        if self._call_count == 0 and torch.cuda.is_available():
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        return None

    def post_forward_hook(self, module, args, kwargs, output):
        """Register as a forward hook on the model."""
        if self._call_count == 0 and torch.cuda.is_available():
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._end_event.record()
        self._call_count += 1
        return output

    def get_prefill_cuda_ms(self) -> Optional[float]:
        if self._start_event and self._end_event:
            torch.cuda.synchronize()
            return self._start_event.elapsed_time(self._end_event)
        return None


# ****************************************************
# Memory pressure context manager
# ****************************************************

class MemoryPressureContext:
    """
    Allocates a large dummy CUDA tensor to simulate memory pressure before
    inference, then releases it afterward.

    Args:
        vram_fraction: Fraction of currently FREE VRAM to consume (0.0–0.9).
        device: Target CUDA device.

    Usage:
        with MemoryPressureContext(vram_fraction=0.5) as ctx:
            stats = ctx.get_stats()   # dict with pre/post VRAM info
            output = model.generate(...)
    """

    def __init__(self, vram_fraction: float = 0.0, device: str = "cuda"):
        assert 0.0 <= vram_fraction <= 0.9, "vram_fraction must be in [0.0, 0.9]"
        self._fraction = vram_fraction
        self._device = device
        self._pressure_tensor: Optional[torch.Tensor] = None
        self._stats: dict = {}

    def __enter__(self) -> "MemoryPressureContext":
        if self._fraction <= 0.0 or not torch.cuda.is_available():
            return self

        free_bytes = (
            torch.cuda.get_device_properties(self._device).total_memory
            - torch.cuda.memory_allocated(self._device)
        )
        target_bytes = int(free_bytes * self._fraction)
        # Round to float16 elements (2 bytes each)
        n_elements = target_bytes // 2

        self._stats["free_before_gb"] = free_bytes / (1024 ** 3)
        self._stats["target_allocation_gb"] = target_bytes / (1024 ** 3)

        try:
            self._pressure_tensor = torch.empty(
                n_elements, dtype=torch.float16, device=self._device
            )
            # Touch the tensor to ensure physical allocation
            self._pressure_tensor.fill_(0.0)
            actual_free = (
                torch.cuda.get_device_properties(self._device).total_memory
                - torch.cuda.memory_allocated(self._device)
            )
            self._stats["free_after_gb"] = actual_free / (1024 ** 3)
            self._stats["actual_allocated_gb"] = self._pressure_tensor.element_size() \
                * self._pressure_tensor.numel() / (1024 ** 3)
            logger.info(
                f"[MemPressure] Allocated {self._stats['actual_allocated_gb']:.2f} GB "
                f"(target fraction={self._fraction:.0%})"
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning("[MemPressure] OOM during pressure allocation; proceeding without.")
            self._pressure_tensor = None

        return self

    def __exit__(self, *_) -> None:
        if self._pressure_tensor is not None:
            del self._pressure_tensor
            self._pressure_tensor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[MemPressure] Pressure tensor released; cache cleared.")

    def get_stats(self) -> dict:
        return dict(self._stats)


# ****************************************************
# Standalone test / demonstration
# ****************************************************
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demonstrate EnergyPoller
    if _PYNVML_AVAILABLE and torch.cuda.is_available():
        print("Testing EnergyPoller for 0.5 seconds...")
        with EnergyPoller(poll_interval_ms=50) as ep:
            time.sleep(0.5)
        e = ep.get_total_energy_joules()
        w = ep.get_mean_power_watts()
        print(f"  Energy: {e:.3f} J | Mean power: {w:.1f} W | Samples: {len(ep.get_samples())}")
    else:
        print("pynvml or CUDA not available; skipping EnergyPoller test.")

    # Demonstrate MemoryPressureContext
    if torch.cuda.is_available():
        print("\nTesting MemoryPressureContext at 50% pressure...")
        with MemoryPressureContext(vram_fraction=0.5) as mpc:
            stats = mpc.get_stats()
            for k, v in stats.items():
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nPhaseMonitor module loaded successfully.")
