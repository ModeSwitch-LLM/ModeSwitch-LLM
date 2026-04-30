from __future__ import annotations

from dataclasses import dataclass

from workloads import RuntimeWorkload

from .classifier import classify_request
from .features import RequestFeatures, extract_request_features_from_workload


@dataclass(frozen=True)
class ControllerDecision:
    """
    Final fixed-mode routing decision made before inference starts.
    """

    selected_mode_name: str
    classification_label: str
    estimated_prefill_share_pct: float
    reason: str


def route_request(features: RequestFeatures) -> ControllerDecision:
    """
    Route the entire request to one fixed inference mode.

    The policy follows the revised project direction:
    - no prefill/decode swapping inside a request
    - use phase dominance only to choose one mode up front
    """

    classification = classify_request(features)

    if classification.label == "batched":
        return ControllerDecision(
            selected_mode_name="int8_plus_continuous_batching",
            classification_label=classification.label,
            estimated_prefill_share_pct=classification.estimated_prefill_share_pct,
            reason=(
                "High batch pressure: send the full request through the INT8 + continuous batching serving path."
            ),
        )

    if classification.label == "chat_shared_prefix":
        return ControllerDecision(
            selected_mode_name="gptq_plus_prefix_caching",
            classification_label=classification.label,
            estimated_prefill_share_pct=classification.estimated_prefill_share_pct,
            reason=(
                "Shared-prefix chat request: use the measured best chat-style combo, GPTQ + prefix caching."
            ),
        )

    if classification.label == "prefill_heavy":
        return ControllerDecision(
            selected_mode_name="fp16_baseline",
            classification_label=classification.label,
            estimated_prefill_share_pct=classification.estimated_prefill_share_pct,
            reason=(
                "Prefill-heavy request: keep the whole request on the conservative FP16 branch to preserve quality."
            ),
        )

    if features.memory_pressure:
        return ControllerDecision(
            selected_mode_name="prefix_caching",
            classification_label=classification.label,
            estimated_prefill_share_pct=classification.estimated_prefill_share_pct,
            reason=(
                "Decode-heavy under memory pressure: use conservative prefix caching as the lossless default branch."
            ),
        )

    return ControllerDecision(
        selected_mode_name="gptq_4bit",
        classification_label=classification.label,
        estimated_prefill_share_pct=classification.estimated_prefill_share_pct,
        reason="Decode-heavy request without shared prefix: use GPTQ as the best single-mode decode optimization.",
    )


def route_runtime_workload(workload: RuntimeWorkload) -> ControllerDecision:
    """
    Convenience wrapper for benchmarking code that already operates on
    RuntimeWorkload objects.
    """

    features = extract_request_features_from_workload(workload)
    return route_request(features)
