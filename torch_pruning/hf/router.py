# Acknowledgement: This routing layer unifies Torch-Pruning for dense/CNN
# Hugging Face models and Expert_Sparsity-style methods for MoE models.

"""Automatic routing for Hugging Face dense and MoE pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch.nn as nn

from .dense import prune_dense_hf_model
from .moe import ExpertSparsityConfig, find_moe_blocks, prune_moe_hf_model
from .runtime import HFModelKind, build_hf_example_inputs, infer_hf_model_kind


@dataclass
class HFUnifiedPruningConfig:
    pruning_ratio: float = 0.5
    moe_method: str = "layerwise_pruning"
    preserve_experts: int = 1
    score_metric: str = "l1"
    beta: float = 0.2


def is_moe_model(model: nn.Module) -> bool:
    return infer_hf_model_kind(model) == HFModelKind.MOE


def prune_hf_model(model: nn.Module, example_inputs: Any = None, *, config: Optional[HFUnifiedPruningConfig] = None):
    resolved = config or HFUnifiedPruningConfig()
    if is_moe_model(model):
        if resolved.moe_method == "dynamic_skipping":
            moe_config = ExpertSparsityConfig(
                method=resolved.moe_method,
                preserve_experts=resolved.preserve_experts,
                score_metric=resolved.score_metric,
                dynamic_skipping=True,
                beta=resolved.beta,
            )
            return prune_moe_hf_model(model, config=moe_config)
        moe_config = ExpertSparsityConfig(
            method=resolved.moe_method,
            preserve_experts=resolved.preserve_experts,
            score_metric=resolved.score_metric,
        )
        return prune_moe_hf_model(model, config=moe_config)

    if example_inputs is None:
        example_inputs = build_hf_example_inputs(model)
    return prune_dense_hf_model(model, example_inputs, pruning_ratio=resolved.pruning_ratio)
