from __future__ import annotations

from dataclasses import dataclass

from .features import RequestFeatures


@dataclass(frozen=True)
class ControllerClassification:
    """
    High-level regime classification used by the fixed-mode router.
    """

    label: str
    estimated_prefill_share_pct: float
    reason: str


def estimate_prefill_share_pct(features: RequestFeatures) -> float:
    """
    Cheap proxy for whether a request is likely prefill-heavy.

    This is not a learned model; it is a hand-fit rule derived from the
    benchmark sweep. The goal is to separate requests with MCQ-style
    long-prompt / tiny-output behavior from generative decode-heavy behavior.
    """

    score = 0.0

    if features.prompt_tokens >= 384:
        score += 15.0
    if features.prompt_tokens >= 768:
        score += 10.0

    if features.expected_output_tokens <= 8:
        score += 30.0
    elif features.expected_output_tokens <= 16:
        score += 20.0

    ratio = features.output_to_prompt_ratio
    if ratio <= 0.02:
        score += 20.0
    elif ratio <= 0.05:
        score += 10.0

    if features.expected_output_tokens >= 64:
        score -= 20.0
    if features.expected_output_tokens >= 128:
        score -= 15.0

    if features.memory_pressure:
        score += 5.0

    return max(0.0, min(100.0, score))


def classify_request(features: RequestFeatures) -> ControllerClassification:
    """
    Map static request features to a coarse routing regime.
    """

    if features.batch_pressure == "high":
        return ControllerClassification(
            label="batched",
            estimated_prefill_share_pct=0.0,
            reason="High concurrent request pressure favors the throughput-oriented batching branch.",
        )

    if features.shared_prefix:
        return ControllerClassification(
            label="chat_shared_prefix",
            estimated_prefill_share_pct=0.0,
            reason="A real shared-prefix signal is present, so the chat prefix-caching branch applies.",
        )

    estimated_prefill_share_pct = estimate_prefill_share_pct(features)
    if estimated_prefill_share_pct > 40.0:
        return ControllerClassification(
            label="prefill_heavy",
            estimated_prefill_share_pct=estimated_prefill_share_pct,
            reason=(
                "Longer prompt plus very short expected output suggests MMLU/GPQA-style "
                "prefill dominance."
            ),
        )

    return ControllerClassification(
        label="decode_heavy",
        estimated_prefill_share_pct=estimated_prefill_share_pct,
        reason="Expected generation length is large enough that decode should dominate total latency.",
    )
