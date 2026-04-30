from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from workloads import RuntimeWorkload


@dataclass(frozen=True)
class RequestFeatures:
    """
    Lightweight static request features available before inference starts.

    The controller intentionally uses only cheap-to-compute signals so it can
    choose a fixed mode before any tokens are generated.
    """

    prompt_tokens: int
    expected_output_tokens: int
    shared_prefix: bool
    batch_pressure: str
    memory_pressure: bool
    workload_tag: Optional[str] = None
    system_condition: Optional[str] = None

    @property
    def output_to_prompt_ratio(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return float(self.expected_output_tokens) / float(self.prompt_tokens)


def normalize_batch_pressure(
    batch_pressure: Optional[str] = None,
    num_requests_in_batch: Optional[int] = None,
    system_condition_name: Optional[str] = None,
) -> str:
    """
    Normalize a few different batching hints into a compact controller signal.
    """

    if batch_pressure:
        lowered = str(batch_pressure).strip().lower()
        if lowered in {"high", "heavy", "batched"}:
            return "high"
        if lowered in {"medium", "moderate"}:
            return "medium"
        return "normal"

    if num_requests_in_batch is not None:
        try:
            batch_size = int(num_requests_in_batch)
        except Exception:
            batch_size = 1
        if batch_size >= 4:
            return "high"
        if batch_size >= 2:
            return "medium"

    if system_condition_name:
        lowered = str(system_condition_name).strip().lower()
        if lowered.startswith("batch_"):
            try:
                batch_size = int(lowered.split("_", 1)[1])
            except Exception:
                batch_size = 1
            if batch_size >= 4:
                return "high"
            if batch_size >= 2:
                return "medium"

    return "normal"


def extract_request_features_from_workload(
    workload: RuntimeWorkload,
    *,
    batch_pressure: Optional[str] = None,
    num_requests_in_batch: Optional[int] = None,
    workload_tag: Optional[str] = None,
) -> RequestFeatures:
    """
    Build controller features from an existing runtime workload object.
    """

    metadata = workload.metadata or {}
    resolved_workload_tag = (
        workload_tag
        or metadata.get("workload_family")
        or workload.benchmark_suite
        or workload.task_type
    )

    return RequestFeatures(
        prompt_tokens=int(getattr(workload, "prompt_tokens_target", 0) or 0),
        expected_output_tokens=int(getattr(workload, "max_new_tokens", 0) or 0),
        shared_prefix=bool(getattr(workload, "repeated_prefix", False)),
        batch_pressure=normalize_batch_pressure(
            batch_pressure=batch_pressure,
            num_requests_in_batch=num_requests_in_batch,
            system_condition_name=getattr(workload, "system_condition_name", None),
        ),
        memory_pressure=bool(getattr(workload, "memory_pressure", False)),
        workload_tag=str(resolved_workload_tag) if resolved_workload_tag is not None else None,
        system_condition=getattr(workload, "system_condition_name", None),
    )
