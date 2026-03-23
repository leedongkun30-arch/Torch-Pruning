# Acknowledgement: This MoE Hugging Face pruning path is merged for public use
# with an Expert_Sparsity-inspired workflow boundary for expert pruning and
# dynamic skipping, while remaining lightweight enough for downstream patching.

"""Expert_Sparsity-style helpers for Hugging Face MoE models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class ExpertSparsityConfig:
    method: str = "layerwise_pruning"
    preserve_experts: int = 1
    score_metric: str = "l1"
    dynamic_skipping: bool = False
    beta: Optional[float] = None


@dataclass
class MoEBlockSpec:
    name: str
    module: nn.Module
    router_name: Optional[str]
    router: Optional[nn.Module]
    num_experts: int


def _locate_router(module: nn.Module) -> tuple[Optional[str], Optional[nn.Module]]:
    for attr_name in ("gate", "router"):
        if hasattr(module, attr_name):
            router = getattr(module, attr_name)
            if isinstance(router, nn.Module):
                return attr_name, router
    for child_name, child in module.named_children():
        lowered = child_name.lower()
        if lowered in {"gate", "router"}:
            return child_name, child
    return None, None


def find_moe_blocks(model: nn.Module) -> List[MoEBlockSpec]:
    """Find HF-style MoE blocks that expose expert lists and routers."""

    blocks: List[MoEBlockSpec] = []
    for name, module in model.named_modules():
        experts = getattr(module, "experts", None)
        if isinstance(experts, nn.ModuleList) and len(experts) > 0:
            router_name, router = _locate_router(module)
            blocks.append(
                MoEBlockSpec(
                    name=name,
                    module=module,
                    router_name=router_name,
                    router=router,
                    num_experts=len(experts),
                )
            )
    return blocks


def _score_expert(expert: nn.Module, metric: str) -> float:
    values: List[torch.Tensor] = []
    for parameter in expert.parameters():
        if parameter is None:
            continue
        if metric == "l2":
            values.append(parameter.detach().float().pow(2).sum())
        else:
            values.append(parameter.detach().float().abs().sum())
    if not values:
        return 0.0
    score = torch.stack(values).sum().item()
    return float(score)


def _keep_top_experts(module: nn.Module, keep_indices: List[int]) -> None:
    experts = getattr(module, "experts")
    module.experts = nn.ModuleList([experts[idx] for idx in keep_indices])
    if hasattr(module, "num_experts"):
        setattr(module, "num_experts", len(keep_indices))
    if hasattr(module, "num_local_experts"):
        setattr(module, "num_local_experts", len(keep_indices))

    router_name, router = _locate_router(module)
    if isinstance(router, nn.Linear):
        keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=router.weight.device)
        router.weight = nn.Parameter(router.weight.data.index_select(0, keep_tensor).clone())
        if router.bias is not None:
            router.bias = nn.Parameter(router.bias.data.index_select(0, keep_tensor).clone())
        router.out_features = len(keep_indices)
        if router_name is not None:
            setattr(module, router_name, router)


def apply_dynamic_skipping(model: nn.Module, beta: float) -> Dict[str, Any]:
    """Attach a dynamic-skipping hint to every detected MoE block."""

    blocks = find_moe_blocks(model)
    for block in blocks:
        setattr(block.module, "expert_skipping_beta", beta)
        if hasattr(block.module, "beta"):
            setattr(block.module, "beta", beta)
    return {
        "path": "moe",
        "mode": "dynamic_skipping",
        "beta": beta,
        "num_blocks": len(blocks),
        "blocks": [block.name for block in blocks],
    }


def prune_moe_hf_model(
    model: nn.Module,
    *,
    config: Optional[ExpertSparsityConfig] = None,
) -> Dict[str, Any]:
    """Apply a lightweight Expert_Sparsity-style expert pruning workflow."""

    resolved = config or ExpertSparsityConfig()
    if resolved.dynamic_skipping and resolved.beta is not None:
        dynamic_summary = apply_dynamic_skipping(model, resolved.beta)
    else:
        dynamic_summary = None

    blocks = find_moe_blocks(model)
    if not blocks:
        raise ValueError("No MoE blocks with `experts` were detected in the model.")

    block_summaries: List[Dict[str, Any]] = []
    for block in blocks:
        experts = getattr(block.module, "experts")
        scores = [_score_expert(expert, resolved.score_metric) for expert in experts]
        keep = min(resolved.preserve_experts, len(scores))
        keep_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:keep]
        keep_indices.sort()
        _keep_top_experts(block.module, keep_indices)
        block_summaries.append(
            {
                "block": block.name,
                "router": block.router_name,
                "original_experts": len(scores),
                "kept_experts": len(keep_indices),
                "kept_indices": keep_indices,
            }
        )

    return {
        "path": "moe",
        "method": resolved.method,
        "score_metric": resolved.score_metric,
        "preserve_experts": resolved.preserve_experts,
        "dynamic_skipping": dynamic_summary,
        "blocks": block_summaries,
    }
