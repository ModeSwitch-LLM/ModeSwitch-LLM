"""
workloads.py — ModeSwitch-LLM Benchmarking Pipeline
Author: Ali Alshehhi

Defines all workload groups, prompt datasets, and system condition configurations
used in the benchmarking study. Workloads are categorized along two axes:

  Prompt Length:  Short (S) ~64–128 tokens  |  Long (L) ~512–2048 tokens
  Output Length:  Short (S) ~64–128 tokens  |  Long (L) ~512–1024 tokens

This gives four canonical workload cells: SS, SL, LS, LL
Each cell includes diverse task types (QA, summarization, reasoning, code, chat)
to ensure the benchmark captures real-world LLM usage patterns.

System Conditions tested:
  - BASELINE:        No additional memory pressure; GPU in idle state before run
  - MEM_PRESSURE_50: ~50% of remaining GPU VRAM pre-allocated by a dummy tensor
  - MEM_PRESSURE_75: ~75% of remaining GPU VRAM pre-allocated
  - BATCH_SIZE_*:    Multi-sequence batching (batch size 1, 2, 4, 8)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ****************************************************
# Enums
# ****************************************************

class PromptCategory(Enum):
    SHORT = "short"   # ~64–128 tokens
    LONG  = "long"    # ~512–2048 tokens


class OutputCategory(Enum):
    SHORT = "short"   # ~64–128 tokens
    LONG  = "long"    # ~512–1024 tokens


class TaskType(Enum):
    QA           = "qa"
    SUMMARIZE    = "summarization"
    REASONING    = "reasoning"
    CODE         = "code"
    CHAT         = "chat"
    EXTRACTION   = "extraction"


class SystemCondition(Enum):
    BASELINE        = "baseline"
    MEM_PRESSURE_50 = "mem_pressure_50"
    MEM_PRESSURE_75 = "mem_pressure_75"
    BATCH_2         = "batch_2"
    BATCH_4         = "batch_4"
    BATCH_8         = "batch_8"


# ****************************************************
# Data classes
# ****************************************************

@dataclass
class WorkloadSample:
    """A single benchmark sample with full metadata."""
    sample_id:        str
    prompt:           str
    reference_output: Optional[str]          # For quality metrics; None if open-ended
    task_type:        TaskType
    prompt_category:  PromptCategory
    output_category:  OutputCategory
    max_new_tokens:   int                    # Hard cap passed to model.generate()
    min_new_tokens:   int                    # Forces minimum decode steps
    workload_cell:    str                    # "SS", "SL", "LS", "LL"
    tags:             List[str] = field(default_factory=list)


@dataclass
class SystemConditionConfig:
    """Runtime system condition to apply before a benchmark run."""
    condition:           SystemCondition
    vram_fraction:       float  # Fraction of free VRAM to pre-allocate (0.0 = none)
    batch_size:          int    # Number of sequences in a batch
    description:         str


# ****************************************************
# System condition registry
# ****************************************************

SYSTEM_CONDITIONS: dict[SystemCondition, SystemConditionConfig] = {
    SystemCondition.BASELINE: SystemConditionConfig(
        condition=SystemCondition.BASELINE,
        vram_fraction=0.0,
        batch_size=1,
        description="No memory pressure; single sequence; clean GPU state"
    ),
    SystemCondition.MEM_PRESSURE_50: SystemConditionConfig(
        condition=SystemCondition.MEM_PRESSURE_50,
        vram_fraction=0.50,
        batch_size=1,
        description="~50% of free VRAM consumed by dummy tensor before inference"
    ),
    SystemCondition.MEM_PRESSURE_75: SystemConditionConfig(
        condition=SystemCondition.MEM_PRESSURE_75,
        vram_fraction=0.75,
        batch_size=1,
        description="~75% of free VRAM consumed by dummy tensor before inference"
    ),
    SystemCondition.BATCH_2: SystemConditionConfig(
        condition=SystemCondition.BATCH_2,
        vram_fraction=0.0,
        batch_size=2,
        description="Batch size 2; no extra memory pressure"
    ),
    SystemCondition.BATCH_4: SystemConditionConfig(
        condition=SystemCondition.BATCH_4,
        vram_fraction=0.0,
        batch_size=4,
        description="Batch size 4; no extra memory pressure"
    ),
    SystemCondition.BATCH_8: SystemConditionConfig(
        condition=SystemCondition.BATCH_8,
        vram_fraction=0.0,
        batch_size=8,
        description="Batch size 8; no extra memory pressure"
    ),
}


# ****************************************************
# Prompt corpus
# Each entry: (prompt_text, reference_output_or_None, task_type)
# Prompts are written to be roughly the right token count for each category.
# Token estimates assume ~4 chars/token (GPT/Llama tokenizer approximation).
# ****************************************************

# ── SHORT PROMPTS (target: 64–128 tokens ≈ 256–512 chars) ──────────────────

_SHORT_QA = [
    (
        "What is the difference between supervised and unsupervised learning in machine learning? "
        "Give a brief explanation with one concrete example of each.",
        "Supervised learning uses labeled data to train a model (e.g., image classification with "
        "labeled cat/dog images). Unsupervised learning finds patterns in unlabeled data "
        "(e.g., clustering customer purchase histories into segments).",
        TaskType.QA
    ),
    (
        "Explain the transformer attention mechanism in two to three sentences. "
        "Focus on what queries, keys, and values represent conceptually.",
        "In attention, queries represent what the current token is looking for, keys represent "
        "what each token has to offer, and values are the actual content retrieved. "
        "The dot-product of queries and keys produces attention weights, which then form "
        "a weighted sum over the values.",
        TaskType.QA
    ),
    (
        "What is gradient checkpointing and why is it useful when training large neural networks "
        "on limited GPU memory?",
        "Gradient checkpointing recomputes intermediate activations during the backward pass "
        "instead of storing them all during the forward pass, trading computation time for "
        "reduced peak memory usage. This allows training larger models or using larger batch sizes.",
        TaskType.QA
    ),
    (
        "What does KV-cache compression mean in the context of LLM inference, and what are "
        "the main tradeoffs involved?",
        "KV-cache compression reduces the memory used to store past key-value pairs during "
        "autoregressive decoding, typically via quantization, eviction, or pooling. "
        "The tradeoff is between memory savings and potential degradation in output quality "
        "due to lost context.",
        TaskType.QA
    ),
    (
        "In one paragraph, explain what speculative decoding is and how it can speed up "
        "LLM inference without changing the output distribution.",
        "Speculative decoding uses a small draft model to propose several tokens at once, "
        "which the large target model then verifies in parallel. Tokens accepted up to the "
        "first rejection are kept; the output distribution is mathematically equivalent to "
        "sampling from the target model alone, so quality is preserved while throughput improves.",
        TaskType.QA
    ),
]

_SHORT_CHAT = [
    (
        "I'm debugging a Python script that uses multiprocessing. My worker processes seem to "
        "be hanging silently. What are the three most common causes of this and how would I "
        "diagnose each one?",
        None,
        TaskType.CHAT
    ),
    (
        "I have a CUDA out-of-memory error during the backward pass of a training loop. "
        "Walk me through a systematic debugging checklist I should follow.",
        None,
        TaskType.CHAT
    ),
    (
        "What is the intuition behind why AdamW generally outperforms vanilla Adam for "
        "fine-tuning large language models?",
        None,
        TaskType.CHAT
    ),
    (
        "Briefly compare LoRA, QLoRA, and full fine-tuning in terms of compute cost, "
        "memory footprint, and final model quality. Which would you recommend for a "
        "single A100 with 40 GB VRAM?",
        None,
        TaskType.CHAT
    ),
]

_SHORT_CODE = [
    (
        "Write a Python function that takes a list of integers and returns a new list "
        "containing only the prime numbers. Optimize for readability, not raw speed.",
        None,
        TaskType.CODE
    ),
    (
        "Write a Python context manager that measures and prints the elapsed wall-clock "
        "time of any code block it wraps.",
        None,
        TaskType.CODE
    ),
    (
        "Implement a simple token-bucket rate limiter class in Python. It should support "
        "a `consume(tokens)` method that returns True if the request can proceed.",
        None,
        TaskType.CODE
    ),
]

_SHORT_REASONING = [
    (
        "A train leaves City A at 60 mph. Another train leaves City B, 300 miles away, "
        "at 90 mph, heading toward City A. They leave at the same time. "
        "At what time and at what distance from City A do they meet? Show your reasoning.",
        "They approach each other at 60+90=150 mph. Distance 300 mi / 150 mph = 2 hours. "
        "Meeting point: 60 * 2 = 120 miles from City A.",
        TaskType.REASONING
    ),
    (
        "There are three boxes: one labeled 'Apples', one 'Oranges', one 'Mixed'. "
        "All labels are wrong. You can pull one fruit from one box without looking inside. "
        "How do you correctly label all three boxes? Explain your reasoning step by step.",
        "Pull from 'Mixed'. Since all labels are wrong, 'Mixed' contains only one fruit type. "
        "If you pull an apple, that box is 'Apples'. Then the box labeled 'Apples' must be "
        "'Oranges' or 'Mixed'. Since it can't be 'Apples' and can't be 'Mixed' (we found "
        "the mixed is actually apples), it's 'Oranges'. The remaining box is 'Mixed'.",
        TaskType.REASONING
    ),
]


# ── LONG PROMPTS (target: 512–2048 tokens ≈ 2000–8000 chars) ───────────────

_LONG_SUMMARIZATION_TEXT_1 = """\
The following is an excerpt from a technical report on efficient transformer inference. Please
provide a concise but complete summary covering: (1) the main problem identified, (2) the
proposed solution, and (3) the reported empirical results.

---

Abstract:
Large language models (LLMs) such as GPT-4 and LLaMA-2 have demonstrated remarkable capabilities
across a wide range of tasks. However, deploying these models in production systems remains
challenging due to their substantial computational and memory requirements. During inference, LLMs
must perform two distinct phases: prefill, where the entire input prompt is processed in parallel
to populate the key-value (KV) cache, and decoding, where new tokens are generated autoregressively
one step at a time. These two phases exhibit fundamentally different computational characteristics.
Prefill is compute-bound, processing large input matrices in parallel with high arithmetic intensity,
while decoding is memory-bandwidth-bound, performing sequential single-token operations that stress
DRAM bandwidth rather than FLOP capacity.

This divergence between phases creates a fundamental tension: inference configurations that are
optimal for prefill (e.g., full-precision computation, larger batch sizes that increase FLOP
utilization) may be suboptimal for decoding (where bandwidth-saving techniques such as quantization
can provide greater benefit). Existing inference systems largely treat the two phases with a single
uniform configuration, leaving efficiency gains on the table.

In this work, we propose PhaseSplit, a phase-aware inference system that independently configures
the prefill and decode phases using different precision levels, attention implementations, and
memory layouts. Specifically, PhaseSplit uses full-precision FlashAttention-2 during prefill to
maximize throughput on compute-intensive operations, and switches to INT4 weight-only quantization
with paged KV-cache during decoding to maximize memory bandwidth efficiency.

We evaluate PhaseSplit on an NVIDIA A100 80 GB GPU across six model families ranging from 7B to
70B parameters, using workloads that vary prompt length from 128 to 4096 tokens and output length
from 128 to 2048 tokens. Our key results are as follows. For short-prompt, long-output workloads,
which are decode-dominated, PhaseSplit achieves a 2.3× reduction in time-between-tokens (TBT) and
a 1.8× reduction in energy per generated token compared to an FP16 baseline, with less than 0.5
points degradation in downstream task accuracy on MMLU and HellaSwag. For long-prompt, short-output
workloads, which are prefill-dominated, PhaseSplit's gains are more modest at 1.15× speedup in
time-to-first-token (TTFT), since the decode-phase optimizations are less relevant. For the
balanced long-prompt, long-output setting, PhaseSplit achieves 1.6× end-to-end throughput improvement.

We further find that a static configuration chosen by examining only aggregate throughput would miss
these phase-specific gains entirely, motivating the need for phase-aware evaluation protocols when
benchmarking inference systems. We release our benchmarking suite and PhaseSplit runtime as
open-source software.

Background:
Modern transformer-based LLMs follow an autoregressive generation paradigm. Given an input token
sequence (the prompt), the model first processes the entire prompt in a single forward pass — the
prefill phase — filling a key-value (KV) cache with intermediate representations. Subsequent tokens
are then generated one at a time in the decode phase, each requiring a forward pass that attends
over all previously cached keys and values. The KV cache grows linearly with sequence length and can
consume tens of gigabytes of GPU memory for long contexts.

The hardware efficiency of these two phases differs substantially. During prefill, the operation is
essentially a large batched matrix multiplication, which maps well onto the high-FLOP throughput of
modern GPUs. Arithmetic intensity — measured as FLOPs per byte of memory transfer — is high,
meaning the GPU compute units are well-utilized. Techniques such as FlashAttention-2 and tensor
parallelism further improve prefill throughput by minimizing memory re-reads.

During decoding, each step generates a single new token and performs a single-row attention
operation: the query is one vector, while keys and values span the entire context. This is a
matrix-vector multiplication with extremely low arithmetic intensity. The bottleneck is not
computation but DRAM bandwidth: the model weights and KV cache must be read from GPU memory for
each token, even though only a tiny fraction of FLOP capacity is exercised. This is why
quantizing weights from FP16 to INT4 — which cuts the memory footprint and bandwidth cost by 4× —
can provide a much larger inference speedup during decode than during prefill.

This observation is the key motivation for PhaseSplit.
---
"""

_LONG_SUMMARIZATION_TEXT_2 = """\
You are a senior ML engineer reviewing the following design document for an inference serving
system. Please identify: (1) all design decisions that have performance implications, (2) any
potential bottlenecks that should be benchmarked before committing to the design, and (3) three
specific experiments you would run to validate the design.

---

Design Document: Adaptive Inference Router v0.3

Overview:
We are building an inference router that sits between client applications and a pool of backend
model replicas. The router is responsible for: (1) accepting inference requests over a REST API,
(2) selecting the most appropriate backend replica and inference configuration for each request,
(3) forwarding the request, and (4) streaming the response back to the client with minimal added
latency.

The core novel component is the Adaptive Configuration Selector (ACS), a lightweight module that
chooses among a fixed library of inference configurations based on features extracted from each
incoming request. The configuration library currently includes:

Config A — FP16 Baseline: Full-precision weights, standard attention, no quantization.
Config B — W4A16: INT4 weight-only quantization (AWQ), FP16 activations, suitable for decode-heavy
  workloads where bandwidth is the bottleneck.
Config C — W8A8: INT8 weight and activation quantization, suitable for throughput-optimized
  serving with modest quality tradeoffs.
Config D — Speculative Decoding (SD): Draft model is a 1B parameter version of the target model.
  Acceptance rate is expected to degrade for highly creative or domain-specific prompts.
Config E — KV-Cache Compression: H2O (Heavy Hitter Oracle) eviction policy, retaining the top 20%
  most-attended KV pairs per layer. Best for long-context inputs.
Config F — FlashAttention-2 Only: No quantization; memory-efficient attention kernel.
  Expected to help primarily during prefill for long prompts.

Request Feature Extraction:
The ACS extracts the following features from each incoming request synchronously before routing:
- Prompt token count (measured via fast tokenization, ~0.5 ms overhead)
- Predicted output length (using a small linear regression model trained on request logs,
  ~0.1 ms overhead)
- Current GPU memory utilization (polled from nvidia-smi cache, ~0.05 ms overhead)
- Latency target SLO (extracted from request header if present, otherwise set to default)

Routing Logic:
The ACS uses a deterministic rule tree trained via offline regression on historical request logs.
The tree has a maximum depth of 4 and a maximum of 15 leaf nodes. Inference through the tree
takes approximately 0.02 ms on CPU.

Backend Pool:
Each backend replica runs a single model instance on a dedicated A100 SXM4 80 GB GPU. Replicas
are isolated processes communicating via shared-memory ring buffers. Each replica maintains a
persistent model in GPU memory, loading the appropriate CUDA kernels for the selected
configuration lazily on the first request of each type.

Streaming and Batching:
Responses are streamed back token by token via server-sent events (SSE). The router aggregates
tokens from the backend and forwards them with a target added latency of less than 1 ms per token.
Continuous batching is used within each backend replica to improve GPU utilization during
decode-heavy workloads.

Open Questions:
1. Should quantized configs be warmed up at startup to avoid cold-start kernel loading latency?
2. Is the predicted output length feature accurate enough to drive config selection, or should
   we fall back to a simpler prompt-length-only heuristic?
3. How should the router handle tail latency spikes caused by KV-cache eviction in Config E?
---
"""

_LONG_REASONING_1 = """\
Below is a complex algorithmic problem. Work through it step by step, showing all intermediate
reasoning. Do not skip steps.

Problem: You are given a directed weighted graph with N nodes (numbered 1 to N) and M edges.
Each edge has a positive integer weight representing the travel time in minutes. You also have
a list of K "priority nodes" that must be visited in order (not necessarily consecutively) during
any valid path. You want to find the shortest path from node 1 to node N such that the priority
nodes are visited in the specified order.

Constraints: N ≤ 500, M ≤ 10,000, K ≤ 10.

Example:
Nodes: 1, 2, 3, 4, 5, 6
Edges (directed): 1→2 (4), 1→3 (2), 2→4 (5), 3→4 (1), 3→5 (8), 4→6 (3), 5→6 (2)
Priority nodes in order: [3, 4]
Start: 1, End: 6

Questions:
(a) What algorithm or algorithmic framework would you use and why?
(b) What is the state space for the dynamic programming or graph search formulation?
(c) What is the time and space complexity of your approach?
(d) Solve the example instance by hand, showing each step of your algorithm.
(e) How would your approach change if the priority node ordering constraint were relaxed
    to: "visit all K priority nodes in any order"?
"""

_LONG_CODE_1 = """\
You are an expert Python engineer. Write complete, production-quality code for the following task.
Include docstrings, type annotations, error handling, and example usage.

Task: Implement a GPU memory pressure simulator for benchmarking LLM inference. The class should:

1. Accept a target fraction of free VRAM to pre-allocate (e.g., 0.5 for 50%).
2. Allocate a dummy CUDA tensor of the appropriate size using PyTorch.
3. Support use as a context manager: entering allocates the pressure tensor, exiting releases it
   and clears the CUDA cache.
4. Provide a `get_memory_stats()` method that returns a dictionary containing:
   - total_vram_gb
   - free_vram_before_pressure_gb
   - allocated_pressure_gb
   - free_vram_after_pressure_gb
   - pressure_fraction_actual (may differ slightly from target due to rounding)
5. Handle edge cases: GPU not available, requested fraction exceeds available memory,
   fraction out of [0, 1] range.
6. Log all allocation and deallocation events using Python's logging module.

After the class implementation, also write:
- A unit test class using pytest that tests the context manager, the memory stats,
  and all edge cases.
- A benchmark harness function `run_pressured_inference(model, tokenizer, prompt, fraction)`
  that runs a single inference call under the specified memory pressure and returns
  a dictionary with the inference result and memory stats.
"""

_LONG_EXTRACTION_1 = """\
Below is a raw log output from an LLM inference benchmarking run. Parse this log carefully and
extract all benchmark results into structured form. For each inference mode, report: mode name,
workload cell (SS/SL/LS/LL), TTFT (ms), mean TBT (ms), peak GPU memory (GB), tokens per second
(prefill and decode separately), and any quality scores present.

Log:
[2024-11-15 14:23:01] Starting benchmark suite: ModeSwitch-LLM v0.1
[2024-11-15 14:23:01] Device: NVIDIA A100 SXM4 80GB | CUDA 12.2 | PyTorch 2.1.0
[2024-11-15 14:23:01] Model: meta-llama/Llama-2-7b-chat-hf
[2024-11-15 14:23:45] === Mode: FP16_BASELINE | Workload: SS (prompt=64tok, output=128tok) ===
[2024-11-15 14:23:45] Prefill: 64 tokens in 18.4 ms → 3478.3 tok/s
[2024-11-15 14:23:45] Decode: 128 tokens in 1847.2 ms → 69.3 tok/s (TBT mean=14.4ms, p95=16.1ms)
[2024-11-15 14:23:45] TTFT: 18.4 ms | Peak VRAM: 14.2 GB | Energy: 0.48 J/tok
[2024-11-15 14:23:45] Quality — ROUGE-L: 0.421 | BERTScore-F1: 0.832
[2024-11-15 14:25:12] === Mode: FP16_BASELINE | Workload: SL (prompt=64tok, output=512tok) ===
[2024-11-15 14:25:12] Prefill: 64 tokens in 17.9 ms → 3575.1 tok/s
[2024-11-15 14:25:12] Decode: 512 tokens in 7512.8 ms → 68.2 tok/s (TBT mean=14.7ms, p95=16.8ms)
[2024-11-15 14:25:12] TTFT: 17.9 ms | Peak VRAM: 14.8 GB | Energy: 0.51 J/tok
[2024-11-15 14:25:12] Quality — ROUGE-L: 0.318 | BERTScore-F1: 0.801
[2024-11-15 14:27:03] === Mode: W4A16_AWQ | Workload: SS (prompt=64tok, output=128tok) ===
[2024-11-15 14:27:03] Prefill: 64 tokens in 22.1 ms → 2895.9 tok/s
[2024-11-15 14:27:03] Decode: 128 tokens in 1102.4 ms → 116.1 tok/s (TBT mean=8.6ms, p95=9.4ms)
[2024-11-15 14:27:03] TTFT: 22.1 ms | Peak VRAM: 5.8 GB | Energy: 0.29 J/tok
[2024-11-15 14:27:03] Quality — ROUGE-L: 0.408 | BERTScore-F1: 0.821
[2024-11-15 14:28:55] === Mode: W4A16_AWQ | Workload: LS (prompt=1024tok, output=128tok) ===
[2024-11-15 14:28:55] Prefill: 1024 tokens in 198.3 ms → 5163.9 tok/s
[2024-11-15 14:28:55] Decode: 128 tokens in 1134.7 ms → 112.8 tok/s (TBT mean=8.9ms, p95=10.1ms)
[2024-11-15 14:28:55] TTFT: 198.3 ms | Peak VRAM: 7.1 GB | Energy: 0.31 J/tok
[2024-11-15 14:28:55] Quality — ROUGE-L: 0.389 | BERTScore-F1: 0.814
[2024-11-15 14:30:41] === Mode: SPEC_DECODE | Workload: SL (prompt=64tok, output=512tok) ===
[2024-11-15 14:30:41] Prefill: 64 tokens in 41.2 ms → 1553.4 tok/s
[2024-11-15 14:30:41] Decode: 512 tokens in 3891.3 ms → 131.6 tok/s (TBT mean=7.6ms, p95=11.2ms)
[2024-11-15 14:30:41] Acceptance rate: 0.73 | Mean accepted tokens/draft: 2.74
[2024-11-15 14:30:41] TTFT: 41.2 ms | Peak VRAM: 16.9 GB | Energy: 0.38 J/tok
[2024-11-15 14:30:41] Quality — ROUGE-L: 0.321 | BERTScore-F1: 0.803
[2024-11-15 14:33:20] === Mode: KV_COMPRESS_H2O | Workload: LL (prompt=1024tok, output=512tok) ===
[2024-11-15 14:33:20] Prefill: 1024 tokens in 201.7 ms → 5079.3 tok/s
[2024-11-15 14:33:20] Decode: 512 tokens in 4821.4 ms → 106.2 tok/s (TBT mean=9.4ms, p95=13.7ms)
[2024-11-15 14:33:20] KV retention: 20% | Eviction events: 388
[2024-11-15 14:33:20] TTFT: 201.7 ms | Peak VRAM: 9.3 GB | Energy: 0.33 J/tok
[2024-11-15 14:33:20] Quality — ROUGE-L: 0.295 | BERTScore-F1: 0.789
[2024-11-15 14:35:58] Benchmark suite complete. Total wall time: 752.4 s
"""

_LONG_QA_1 = """\
I am designing an LLM inference benchmarking study for a graduate research project. The goal is
to compare multiple efficient inference modes (FP16 baseline, INT4 quantization, speculative
decoding, KV-cache compression) across different workload types on a single GPU. Please answer
each of the following questions in detail:

1. What are the key confounders I should control for when comparing time-to-first-token (TTFT)
   across different inference modes? Think carefully about GPU state, thermal throttling,
   CUDA kernel warm-up, and memory fragmentation.

2. For measuring energy per generated token, I plan to poll GPU power draw using pynvml in a
   background thread at 100 ms intervals. What are the limitations of this approach? What
   would a more accurate measurement methodology look like?

3. When measuring time-between-tokens (TBT), should I measure wall-clock time between
   successive calls to the generation loop, or should I use CUDA events with
   torch.cuda.Event(enable_timing=True)? What are the tradeoffs?

4. How should I define "output quality" for open-ended generation tasks where there is no
   single reference answer? What metrics would you recommend beyond ROUGE and BERTScore?

5. I want to simulate "memory pressure" conditions to test how each inference mode degrades
   as available VRAM shrinks. What is the cleanest way to do this without actually running
   another model or process? Are there any pitfalls I should be aware of?

6. For speculative decoding specifically, the acceptance rate varies with the prompt/output
   distribution. How should I report speculative decoding performance in a way that is fair
   and interpretable compared to non-speculative baselines?
"""


# Collect all prompt materials into lists by category
_SHORT_PROMPTS_ALL = (
    _SHORT_QA
    + _SHORT_CHAT
    + _SHORT_CODE
    + _SHORT_REASONING
)

_LONG_PROMPTS_ALL = [
    (_LONG_SUMMARIZATION_TEXT_1, None, TaskType.SUMMARIZE),
    (_LONG_SUMMARIZATION_TEXT_2, None, TaskType.SUMMARIZE),
    (_LONG_REASONING_1,          None, TaskType.REASONING),
    (_LONG_CODE_1,               None, TaskType.CODE),
    (_LONG_EXTRACTION_1,         None, TaskType.EXTRACTION),
    (_LONG_QA_1,                 None, TaskType.QA),
]


# ****************************************************
# Token budget configuration per workload cell
# ****************************************************

WORKLOAD_TOKEN_BUDGETS = {
    "SS": dict(min_new_tokens=32,  max_new_tokens=128),
    "SL": dict(min_new_tokens=256, max_new_tokens=512),
    "LS": dict(min_new_tokens=32,  max_new_tokens=128),
    "LL": dict(min_new_tokens=256, max_new_tokens=512),
}


# ****************************************************
# WorkloadSuite builder
# ****************************************************

class WorkloadSuite:
    """
    Builds and manages the complete set of benchmark workload samples.
    Samples are deterministically generated from the prompt corpus above.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._samples: List[WorkloadSample] = []
        self._build()

    def _make_sample(
        self,
        idx: int,
        prompt: str,
        reference: Optional[str],
        task: TaskType,
        prompt_cat: PromptCategory,
        output_cat: OutputCategory,
    ) -> WorkloadSample:
        cell = f"{'S' if prompt_cat == PromptCategory.SHORT else 'L'}" \
               f"{'S' if output_cat == OutputCategory.SHORT else 'L'}"
        budgets = WORKLOAD_TOKEN_BUDGETS[cell]
        return WorkloadSample(
            sample_id=f"{cell}_{task.value}_{idx:04d}",
            prompt=prompt.strip(),
            reference_output=reference,
            task_type=task,
            prompt_category=prompt_cat,
            output_category=output_cat,
            max_new_tokens=budgets["max_new_tokens"],
            min_new_tokens=budgets["min_new_tokens"],
            workload_cell=cell,
            tags=[task.value, cell],
        )

    def _build(self) -> None:
        # SS cell: short prompt → short output
        for i, (p, r, t) in enumerate(_SHORT_PROMPTS_ALL):
            self._samples.append(self._make_sample(
                i, p, r, t, PromptCategory.SHORT, OutputCategory.SHORT
            ))

        # SL cell: short prompt → long output
        for i, (p, r, t) in enumerate(_SHORT_PROMPTS_ALL):
            self._samples.append(self._make_sample(
                i, p, r, t, PromptCategory.SHORT, OutputCategory.LONG
            ))

        # LS cell: long prompt → short output
        for i, (p, r, t) in enumerate(_LONG_PROMPTS_ALL):
            self._samples.append(self._make_sample(
                i, p, r, t, PromptCategory.LONG, OutputCategory.SHORT
            ))

        # LL cell: long prompt → long output
        for i, (p, r, t) in enumerate(_LONG_PROMPTS_ALL):
            self._samples.append(self._make_sample(
                i, p, r, t, PromptCategory.LONG, OutputCategory.LONG
            ))

    def get_all(self) -> List[WorkloadSample]:
        return list(self._samples)

    def get_cell(self, cell: str) -> List[WorkloadSample]:
        """Return samples for a specific workload cell: 'SS', 'SL', 'LS', 'LL'."""
        assert cell in ("SS", "SL", "LS", "LL"), f"Unknown cell: {cell}"
        return [s for s in self._samples if s.workload_cell == cell]

    def get_task(self, task: TaskType) -> List[WorkloadSample]:
        return [s for s in self._samples if s.task_type == task]

    def summary(self) -> dict:
        cells = {}
        for s in self._samples:
            cells.setdefault(s.workload_cell, []).append(s)
        return {
            cell: {
                "count": len(slist),
                "tasks": list({s.task_type.value for s in slist}),
                "max_new_tokens": slist[0].max_new_tokens,
                "min_new_tokens": slist[0].min_new_tokens,
            }
            for cell, slist in cells.items()
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __repr__(self) -> str:
        return f"WorkloadSuite(samples={len(self._samples)}, cells={list(self.summary().keys())})"


# ****************************************************
# Quick smoke test
# ****************************************************
if __name__ == "__main__":
    suite = WorkloadSuite()
    print(f"Total samples: {len(suite)}")
    for cell, info in suite.summary().items():
        print(f"  {cell}: {info['count']} samples | tasks={info['tasks']} "
              f"| max_new_tokens={info['max_new_tokens']}")
    sample = suite.get_cell("LS")[0]
    print(f"\nSample LS[0]: id={sample.sample_id}, task={sample.task_type.value}")
    print(f"  Prompt (first 120 chars): {sample.prompt[:120]!r}")
