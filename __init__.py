"""
# ============================================================================
# ModeSwitch-LLM Controller Package
# ============================================================================
# This package exposes the public request-routing API for ModeSwitch-LLM.
#
# Main tasks:
# - Exports the request feature schema.
# - Exports the coarse request classifier.
# - Exports the final controller decision schema.
# - Exports helper functions for routing raw request features or RuntimeWorkload objects.
# - Keeps controller imports clean for runner.py and other benchmark scripts.
# ============================================================================
"""

from .classifier import ControllerClassification, classify_request
from .features import RequestFeatures, extract_request_features_from_workload
from .router import ControllerDecision, route_request, route_runtime_workload

__all__ = [
    "ControllerClassification",
    "ControllerDecision",
    "RequestFeatures",
    "classify_request",
    "extract_request_features_from_workload",
    "route_request",
    "route_runtime_workload",
]
