"""Expert pruning methods for merged Expert_Sparsity workflows."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch.nn as nn

from torch_pruning.hf.moe import ExpertSparsityConfig, prune_moe_hf_model


def _resolve_r(args: Any, default: int = 1) -> int:
    value = getattr(args, "r", None)
    return int(value if value is not None else default)


def layerwise_pruning(model: nn.Module, calib_loader, args: Optional[Any] = None):
    config = ExpertSparsityConfig(
        method="layerwise_pruning",
        preserve_experts=_resolve_r(args, 1),
        score_metric=getattr(args, "score_metric", "l1"),
    )
    info = prune_moe_hf_model(model, config=config)
    info["n_calibration_batches"] = len(calib_loader)
    return model, info


def progressive_pruning(model: nn.Module, calib_loader, args: Optional[Any] = None):
    total_keep = _resolve_r(args, 1)
    info: Dict[str, Any] = {"path": "moe", "method": "progressive_pruning", "steps": []}
    current_model = model
    start = max(total_keep, 1)
    for preserve_experts in range(start + 1, start - 1, -1):
        preserve_experts = max(total_keep, preserve_experts)
        config = ExpertSparsityConfig(
            method="progressive_pruning",
            preserve_experts=preserve_experts,
            score_metric=getattr(args, "score_metric", "l1"),
        )
        step_info = prune_moe_hf_model(current_model, config=config)
        info["steps"].append(step_info)
        if preserve_experts == total_keep:
            break
    info["n_calibration_batches"] = len(calib_loader)
    return current_model, info
