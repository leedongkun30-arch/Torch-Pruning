"""Integration helpers for external model ecosystems."""

from .hf import (
    HFModelFlavor,
    HFDensePruningConfig,
    detect_hf_model_flavor,
    prune_hf_model,
)

__all__ = [
    "HFModelFlavor",
    "HFDensePruningConfig",
    "detect_hf_model_flavor",
    "prune_hf_model",
]
