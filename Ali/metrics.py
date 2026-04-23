"""
metrics.py — ModeSwitch-LLM Benchmarking Pipeline
Author: Ali Alshehhi

Defines all evaluation metrics used in the benchmarking study:

  LATENCY METRICS (phase-specific):
    - TTFT (Time to First Token): prefill latency
    - TBT (Time Between Tokens): mean, median, p95, p99 decode step latency
    - End-to-end generation latency

  THROUGHPUT METRICS:
    - Prefill throughput: tokens/second during prefill
    - Decode throughput: tokens/second during decode
    - Overall tokens/second

  MEMORY METRICS:
    - Peak GPU VRAM allocation (GB)
    - Reserved VRAM (GB)
    - KV-cache estimated size (GB), computed analytically

  ENERGY METRICS:
    - Total inference energy (Joules)
    - Energy per generated token (J/token)
    - Mean GPU power draw (Watts)

  OUTPUT QUALITY METRICS:
    - ROUGE-1, ROUGE-2, ROUGE-L (for tasks with reference outputs)
    - BERTScore Precision, Recall, F1
    - Self-BLEU (for diversity / repetition measurement, no reference needed)
    - Perplexity under a reference model (optional; requires a second model pass)
    - Repetition rate: fraction of generated n-grams that are repeated

Design principle: all metric functions are pure functions that take strings or
lists of numbers as input and return flat dictionaries of float values.
This makes them easy to serialize to JSON/CSV and easy to unit-test.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Optional heavy dependencies — gracefully degraded if missing
try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    logger.warning("rouge_score not installed. ROUGE metrics disabled. pip install rouge-score")

try:
    from bert_score import score as bert_score_fn
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False
    logger.warning("bert_score not installed. BERTScore metrics disabled. pip install bert-score")


# ****************************************************
# Type aliases
# ****************************************************
MetricsDict = Dict[str, Optional[float]]


# ****************************************************
# Latency metrics
# ****************************************************

def compute_latency_metrics(
    ttft_ms: float,
    tbt_per_step_ms: List[float],
    prompt_tokens: int,
    generated_tokens: int,
) -> MetricsDict:
    """
    Compute all latency and throughput metrics from raw timing data.

    Args:
        ttft_ms:           Time to first token in milliseconds.
        tbt_per_step_ms:   List of per-step decode latencies in milliseconds.
        prompt_tokens:     Number of prompt (input) tokens.
        generated_tokens:  Number of tokens generated during decode.

    Returns:
        Flat dictionary of metric name ---> float value.
    """
    if not tbt_per_step_ms:
        tbt_per_step_ms = [0.0]

    n = len(tbt_per_step_ms)
    total_decode_ms = sum(tbt_per_step_ms)
    total_ms = ttft_ms + total_decode_ms

    sorted_tbts = sorted(tbt_per_step_ms)

    def _pct(data: List[float], p: float) -> float:
        if not data:
            return float("nan")
        idx = (p / 100.0) * (len(data) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(data) - 1)
        return data[lo] * (1 - (idx - lo)) + data[hi] * (idx - lo)

    mean_tbt = total_decode_ms / n
    std_tbt = math.sqrt(sum((x - mean_tbt) ** 2 for x in tbt_per_step_ms) / n) if n > 1 else 0.0

    prefill_throughput = (prompt_tokens / (ttft_ms / 1000.0)) if ttft_ms > 0 else 0.0
    decode_throughput  = (generated_tokens / (total_decode_ms / 1000.0)) if total_decode_ms > 0 else 0.0
    overall_throughput = ((prompt_tokens + generated_tokens) / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    return {
        # Core latency
        "ttft_ms":              round(ttft_ms, 3),
        "tbt_mean_ms":          round(mean_tbt, 3),
        "tbt_median_ms":        round(_pct(sorted_tbts, 50), 3),
        "tbt_p95_ms":           round(_pct(sorted_tbts, 95), 3),
        "tbt_p99_ms":           round(_pct(sorted_tbts, 99), 3),
        "tbt_std_ms":           round(std_tbt, 3),
        "tbt_min_ms":           round(sorted_tbts[0], 3),
        "tbt_max_ms":           round(sorted_tbts[-1], 3),
        "total_decode_ms":      round(total_decode_ms, 3),
        "total_inference_ms":   round(total_ms, 3),
        # Throughput
        "prefill_throughput_tps": round(prefill_throughput, 2),
        "decode_throughput_tps":  round(decode_throughput, 2),
        "overall_throughput_tps": round(overall_throughput, 2),
        # Token counts
        "prompt_tokens":        prompt_tokens,
        "generated_tokens":     generated_tokens,
        "total_tokens":         prompt_tokens + generated_tokens,
        # Phase dominance ratio (>1 = decode dominated, <1 = prefill dominated)
        "decode_prefill_ratio": round(total_decode_ms / ttft_ms, 3) if ttft_ms > 0 else float("inf"),
    }


# ****************************************************
# Memory metrics
# ****************************************************

def compute_memory_metrics(
    peak_vram_gb: float,
    allocated_vram_gb: float,
    reserved_vram_gb: float,
    prompt_tokens: int,
    generated_tokens: int,
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    dtype_bytes: int = 2,  # fp16 = 2 bytes
) -> MetricsDict:
    """
    Compute memory metrics and analytically estimate KV-cache size.

    KV-cache formula:
        kv_cache_bytes = 2 * num_layers * num_heads * head_dim * seq_len * dtype_bytes
        (factor of 2 for keys and values)

    Args:
        peak_vram_gb:      Peak GPU memory allocated, from torch.cuda.max_memory_allocated().
        allocated_vram_gb: Current allocated VRAM after generation.
        reserved_vram_gb:  Current reserved VRAM (PyTorch memory pool).
        prompt_tokens:     Input sequence length.
        generated_tokens:  Number of generated tokens.
        num_layers:        Model depth (default: 32, matches 7B models).
        num_heads:         Number of KV heads.
        head_dim:          Head dimension.
        dtype_bytes:       Bytes per element (2 = fp16, 1 = int8, 0.5 = int4).

    Returns:
        Dictionary of memory metric name → value in GB or float.
    """
    max_seq_len = prompt_tokens + generated_tokens
    kv_bytes = 2 * num_layers * num_heads * head_dim * max_seq_len * dtype_bytes
    kv_gb = kv_bytes / (1024 ** 3)

    return {
        "peak_vram_gb":            round(peak_vram_gb, 4),
        "allocated_vram_gb":       round(allocated_vram_gb, 4),
        "reserved_vram_gb":        round(reserved_vram_gb, 4),
        "kv_cache_estimated_gb":   round(kv_gb, 4),
        "kv_cache_pct_of_peak":    round(kv_gb / peak_vram_gb * 100, 2) if peak_vram_gb > 0 else None,
        "model_weights_estimated_gb": round(peak_vram_gb - kv_gb, 4) if peak_vram_gb > kv_gb else None,
    }


# ****************************************************
# Energy metrics
# ****************************************************

def compute_energy_metrics(
    total_energy_j: Optional[float],
    generated_tokens: int,
    total_inference_ms: float,
    mean_power_w: Optional[float] = None,
) -> MetricsDict:
    """
    Compute energy efficiency metrics.

    Args:
        total_energy_j:    Total Joules consumed during inference (from EnergyPoller).
        generated_tokens:  Number of output tokens.
        total_inference_ms: Total inference time in milliseconds.
        mean_power_w:      Mean GPU power draw in Watts (from EnergyPoller).

    Returns:
        Dictionary: energy_per_token_j, total_energy_j, efficiency_tokens_per_joule.
    """
    energy_per_token = (
        total_energy_j / generated_tokens
        if (total_energy_j is not None and generated_tokens > 0)
        else None
    )
    tokens_per_joule = (
        generated_tokens / total_energy_j
        if (total_energy_j is not None and total_energy_j > 0)
        else None
    )

    return {
        "total_energy_j":         round(total_energy_j, 4) if total_energy_j is not None else None,
        "energy_per_token_j":     round(energy_per_token, 6) if energy_per_token is not None else None,
        "tokens_per_joule":       round(tokens_per_joule, 3) if tokens_per_joule is not None else None,
        "mean_power_w":           round(mean_power_w, 2) if mean_power_w is not None else None,
        "inferred_power_w":       round(
            (total_energy_j / (total_inference_ms / 1000.0)), 2
        ) if (total_energy_j and total_inference_ms > 0) else None,
    }


# ****************************************************
# Output quality metrics
# ****************************************************

def compute_rouge(
    hypothesis: str,
    reference: str,
) -> MetricsDict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    Requires: pip install rouge-score

    Args:
        hypothesis: Model-generated text.
        reference:  Reference/ground-truth text.

    Returns:
        Dictionary with rouge1_f, rouge2_f, rougeL_f and precision/recall variants.
    """
    if not _ROUGE_AVAILABLE:
        return {k: None for k in ["rouge1_f", "rouge1_p", "rouge1_r",
                                   "rouge2_f", "rouge2_p", "rouge2_r",
                                   "rougeL_f", "rougeL_p", "rougeL_r"]}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge1_p": round(scores["rouge1"].precision, 4),
        "rouge1_r": round(scores["rouge1"].recall, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rouge2_p": round(scores["rouge2"].precision, 4),
        "rouge2_r": round(scores["rouge2"].recall, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
        "rougeL_p": round(scores["rougeL"].precision, 4),
        "rougeL_r": round(scores["rougeL"].recall, 4),
    }


def compute_bertscore(
    hypotheses: List[str],
    references: List[str],
    lang: str = "en",
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 8,
    device: Optional[str] = None,
) -> List[MetricsDict]:
    """
    Compute BERTScore for a batch of (hypothesis, reference) pairs.
    BERTScore correlates better with human judgment than ROUGE for open-ended text.
    Requires: pip install bert-score

    Args:
        hypotheses:   List of model outputs.
        references:   List of reference outputs (same length as hypotheses).
        lang:         Language code.
        model_type:   BERT variant to use (distilbert is fast; roberta-large is best).
        batch_size:   Processing batch size for the BERTScore model.
        device:       "cuda" or "cpu"; auto-detected if None.

    Returns:
        List of dicts with bertscore_p, bertscore_r, bertscore_f1.
    """
    if not _BERTSCORE_AVAILABLE:
        return [{"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}
                for _ in hypotheses]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        P, R, F = bert_score_fn(
            hypotheses,
            references,
            lang=lang,
            model_type=model_type,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )
        return [
            {
                "bertscore_p":  round(float(P[i]), 4),
                "bertscore_r":  round(float(R[i]), 4),
                "bertscore_f1": round(float(F[i]), 4),
            }
            for i in range(len(hypotheses))
        ]
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return [{"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}
                for _ in hypotheses]


def compute_repetition_metrics(text: str, ngram_sizes: Tuple[int, ...] = (2, 3, 4)) -> MetricsDict:
    """
    Measure repetition in generated text — a proxy for degenerate output.
    A high repetition rate signals mode collapse or greedy/low-temperature artifacts.

    For each n-gram size, computes:
        repetition_rate_N = 1 - (unique_ngrams / total_ngrams)
    where higher values → more repetitive text.

    Also computes:
        - vocab_diversity: unique words / total words (type-token ratio)
        - avg_sentence_length: mean tokens per sentence
    """
    tokens = text.lower().split()
    words = re.findall(r"\b[a-z]+\b", text.lower())

    results: MetricsDict = {}

    # N-gram repetition rates
    for n in ngram_sizes:
        if len(tokens) < n:
            results[f"rep_rate_{n}gram"] = None
            continue
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        results[f"rep_rate_{n}gram"] = round(1.0 - unique / total, 4) if total > 0 else None

    # Vocabulary diversity (type-token ratio)
    results["vocab_diversity"] = round(len(set(words)) / len(words), 4) if words else None

    # Sentence length
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)
        results["avg_sentence_length"] = round(avg_sent_len, 2)
        results["num_sentences"] = len(sentences)
    else:
        results["avg_sentence_length"] = None
        results["num_sentences"] = 0

    results["total_output_tokens"] = len(tokens)
    results["unique_output_tokens"] = len(set(tokens))

    return results


def compute_perplexity(
    text: str,
    model,
    tokenizer,
    device: str = "cuda",
    stride: int = 512,
    max_length: int = 2048,
) -> MetricsDict:
    """
    Compute perplexity of generated text under the model itself (self-perplexity).
    Lower perplexity = text is more coherent/predictable under the model's distribution.
    Very high perplexity after quantization may indicate quality degradation.

    Uses a sliding-window approach to handle long texts within the model's context window.

    NOTE: This requires a full forward pass and is compute-intensive.
    Only run this for a small subset of samples (e.g., 5–10 per cell).

    Args:
        text:       Generated text string to evaluate.
        model:      HuggingFace causal LM (already on `device`).
        tokenizer:  Corresponding tokenizer.
        device:     CUDA device.
        stride:     Sliding window stride.
        max_length: Maximum sequence length for the model.

    Returns:
        {"perplexity": float or None, "nll_mean": float or None}
    """
    try:
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        if seq_len < 2:
            return {"perplexity": None, "nll_mean": None}

        nlls = []
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            trg_len = end - prev_end
            chunk = input_ids[:, begin:end]
            target = chunk.clone()
            target[:, :-trg_len] = -100  # ignore non-target positions

            with torch.no_grad():
                out = model(chunk, labels=target)
                nll = out.loss * trg_len
            nlls.append(nll.float())
            prev_end = end
            if end == seq_len:
                break

        mean_nll = torch.stack(nlls).sum() / seq_len
        ppl = torch.exp(mean_nll).item()
        return {
            "perplexity": round(ppl, 4),
            "nll_mean":   round(mean_nll.item(), 6),
        }
    except Exception as e:
        logger.warning(f"Perplexity computation failed: {e}")
        return {"perplexity": None, "nll_mean": None}


# ****************************************************
# Aggregate quality: combine all available metrics
# ****************************************************

def compute_all_quality_metrics(
    hypothesis: str,
    reference: Optional[str] = None,
    compute_rep: bool = True,
    compute_rouge: bool = True,
    compute_bert: bool = False,  # slow; enable selectively
    bertscore_model: str = "distilbert-base-uncased",
) -> MetricsDict:
    """
    Compute all applicable quality metrics for a single generated output.

    Args:
        hypothesis:      Generated text.
        reference:       Reference text (if None, ROUGE and BERTScore are skipped).
        compute_rep:     Whether to compute repetition metrics (fast, always useful).
        compute_rouge:   Whether to compute ROUGE (requires reference).
        compute_bert:    Whether to compute BERTScore (slow; use sparingly).
        bertscore_model: BERT variant for BERTScore.

    Returns:
        Merged flat dictionary of all available quality metrics.
    """
    all_metrics: MetricsDict = {}

    if compute_rep:
        all_metrics.update(compute_repetition_metrics(hypothesis))

    if reference is not None:
        if compute_rouge and _ROUGE_AVAILABLE:
            all_metrics.update(compute_rouge(hypothesis, reference))
        elif compute_rouge:
            all_metrics.update({k: None for k in ["rouge1_f", "rouge2_f", "rougeL_f"]})

        if compute_bert and _BERTSCORE_AVAILABLE:
            bert_results = compute_bertscore(
                [hypothesis], [reference], model_type=bertscore_model
            )
            all_metrics.update(bert_results[0])
        elif compute_bert:
            all_metrics.update({"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None})
    else:
        # No reference: fill with None
        if compute_rouge:
            all_metrics.update({k: None for k in ["rouge1_f", "rouge2_f", "rougeL_f",
                                                    "rouge1_p", "rouge1_r", "rouge2_p",
                                                    "rouge2_r", "rougeL_p", "rougeL_r"]})
        if compute_bert:
            all_metrics.update({"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None})

    return all_metrics


# ****************************************************
# Speculative decoding specific metrics
# ****************************************************

def compute_speculative_metrics(
    acceptance_rates: List[float],
    tokens_per_draft_step: List[int],
) -> MetricsDict:
    """
    Metrics specific to speculative decoding runs.

    Args:
        acceptance_rates:      Per-draft-block acceptance rates [0, 1].
        tokens_per_draft_step: Number of tokens accepted per speculative proposal.

    Returns:
        Summary statistics including mean acceptance rate and efficiency multiplier.
    """
    if not acceptance_rates:
        return {"spec_mean_acceptance": None, "spec_p10_acceptance": None,
                "spec_effective_multiplier": None}

    n = len(acceptance_rates)
    sorted_ar = sorted(acceptance_rates)
    mean_ar = sum(acceptance_rates) / n

    def _pct(data, p):
        idx = (p / 100.0) * (len(data) - 1)
        lo = int(idx); hi = min(lo + 1, len(data) - 1)
        return data[lo] * (1 - (idx - lo)) + data[hi] * (idx - lo)

    mean_tpd = (sum(tokens_per_draft_step) / len(tokens_per_draft_step)
                if tokens_per_draft_step else 0.0)

    return {
        "spec_mean_acceptance":    round(mean_ar, 4),
        "spec_p10_acceptance":     round(_pct(sorted_ar, 10), 4),
        "spec_p50_acceptance":     round(_pct(sorted_ar, 50), 4),
        "spec_p90_acceptance":     round(_pct(sorted_ar, 90), 4),
        "spec_std_acceptance":     round(
            math.sqrt(sum((x - mean_ar) ** 2 for x in acceptance_rates) / n), 4
        ) if n > 1 else 0.0,
        "spec_mean_tokens_per_draft": round(mean_tpd, 3),
        # Effective speedup relative to 1-token-at-a-time: E[tokens accepted] + 1 / 1
        "spec_effective_multiplier":  round(mean_tpd + 1, 3) if mean_tpd else None,
    }


# ****************************************************
# Comparative / cross-mode delta metrics
# ****************************************************

def compute_delta_vs_baseline(
    mode_metrics: MetricsDict,
    baseline_metrics: MetricsDict,
    keys_to_compare: Optional[List[str]] = None,
) -> MetricsDict:
    """
    Compute relative change of each metric versus the FP16 baseline.

    delta = (mode_value - baseline_value) / baseline_value * 100
    Positive delta = mode is worse (for latency/energy); interpretation is caller's.

    Returns:
        Dict with "delta_pct_<key>" entries.
    """
    if keys_to_compare is None:
        keys_to_compare = [
            "ttft_ms", "tbt_mean_ms", "tbt_p95_ms",
            "decode_throughput_tps", "energy_per_token_j",
            "peak_vram_gb", "rouge1_f", "rougeL_f", "bertscore_f1",
        ]
    result: MetricsDict = {}
    for key in keys_to_compare:
        m_val = mode_metrics.get(key)
        b_val = baseline_metrics.get(key)
        if m_val is not None and b_val is not None and b_val != 0:
            delta = (m_val - b_val) / abs(b_val) * 100.0
            result[f"delta_pct_{key}"] = round(delta, 2)
        else:
            result[f"delta_pct_{key}"] = None
    return result


# ****************************************************
# Metric schema / documentation
# ****************************************************

METRIC_SCHEMA = {
    # Latency
    "ttft_ms":               "Time to first token (ms). Lower is better.",
    "tbt_mean_ms":           "Mean time between tokens during decode (ms). Lower is better.",
    "tbt_median_ms":         "Median TBT (ms). Robust to outliers.",
    "tbt_p95_ms":            "95th-percentile TBT (ms). Tail latency indicator.",
    "tbt_p99_ms":            "99th-percentile TBT (ms). Worst-case tail latency.",
    "tbt_std_ms":            "Standard deviation of TBT (ms). Consistency indicator.",
    "total_inference_ms":    "Total wall-clock inference time (ms).",
    "decode_prefill_ratio":  "Ratio of decode time to prefill time. >1 = decode-dominated.",
    # Throughput
    "prefill_throughput_tps":"Tokens/second during prefill.",
    "decode_throughput_tps": "Tokens/second during decode.",
    "overall_throughput_tps":"Overall tokens/second (prefill + decode).",
    # Memory
    "peak_vram_gb":          "Peak GPU VRAM allocation during inference (GB).",
    "kv_cache_estimated_gb": "Analytically estimated KV-cache size (GB).",
    # Energy
    "energy_per_token_j":   "Joules per generated token. Lower is better.",
    "total_energy_j":       "Total energy consumed during inference (J).",
    "mean_power_w":         "Mean GPU power draw during inference (W).",
    # Quality
    "rouge1_f":             "ROUGE-1 F1 score [0,1]. Higher is better.",
    "rouge2_f":             "ROUGE-2 F1 score [0,1]. Higher is better.",
    "rougeL_f":             "ROUGE-L F1 score [0,1]. Higher is better.",
    "bertscore_f1":         "BERTScore F1 [0,1]. Higher is better.",
    "rep_rate_3gram":       "3-gram repetition rate [0,1]. Lower is better.",
    "vocab_diversity":      "Type-token ratio [0,1]. Higher = more diverse output.",
    "perplexity":           "Self-perplexity under the model. Lower = more coherent.",
}

def print_metric_schema() -> None:
    print("ModeSwitch-LLM Metric Definitions")
    print("=" * 60)
    for name, desc in METRIC_SCHEMA.items():
        print(f"  {name:<30} {desc}")


# ****************************************************
# Quick test
# ****************************************************
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Latency
    lat = compute_latency_metrics(
        ttft_ms=22.5,
        tbt_per_step_ms=[14.2, 14.8, 13.9, 15.1, 14.5, 16.2, 13.8],
        prompt_tokens=64,
        generated_tokens=7,
    )
    print("Latency metrics:")
    for k, v in lat.items():
        print(f"  {k}: {v}")

    # Repetition
    hyp = ("The quick brown fox jumps over the lazy dog. "
           "The quick brown fox jumps over the lazy dog again. "
           "This is a test of repetition detection in generated text.")
    rep = compute_repetition_metrics(hyp)
    print("\nRepetition metrics:")
    for k, v in rep.items():
        print(f"  {k}: {v}")

    # ROUGE
    if _ROUGE_AVAILABLE:
        ref = "The fox jumped over the dog."
        rouge = compute_rouge(hyp[:50], ref)
        print("\nROUGE metrics:")
        for k, v in rouge.items():
            print(f"  {k}: {v}")

    # Energy
    nrg = compute_energy_metrics(
        total_energy_j=1.84,
        generated_tokens=128,
        total_inference_ms=1847.0,
        mean_power_w=180.0,
    )
    print("\nEnergy metrics:")
    for k, v in nrg.items():
        print(f"  {k}: {v}")

    print("\nAll metrics module loaded successfully.")
