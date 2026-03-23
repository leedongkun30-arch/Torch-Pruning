"""Minimal Mixtral patch helpers inspired by Expert_Sparsity integration points."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn


def is_mixtral_like_model(model: nn.Module) -> bool:
    name = model.__class__.__name__.lower()
    config_type = getattr(getattr(model, "config", None), "model_type", "")
    return "mixtral" in name or config_type == "mixtral"


def patch_mixtral_for_dynamic_skipping(model: nn.Module) -> Dict[str, Any]:
    patched_modules = []
    for module_name, module in model.named_modules():
        if hasattr(module, "experts") and hasattr(module, "router"):
            setattr(module, "dynamic_skipping_enabled", True)
            patched_modules.append(module_name)
    return {
        "patched_modules": patched_modules,
        "is_mixtral_like": is_mixtral_like_model(model),
    }
