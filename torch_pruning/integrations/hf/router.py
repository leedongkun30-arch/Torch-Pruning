# Acknowledgement: This integration layer is organized around Torch-Pruning's
# dependency-graph pruning workflow for CNN and dense Hugging Face models.
# Acknowledgement: MoE-specific entry points are intentionally separated so a
# downstream Expert_Sparsity-derived implementation can be dropped in cleanly.

"""Routing helpers for pruning Hugging Face-style dense and MoE models.

This module is intentionally conservative:
- CNN / dense models are pruned directly with Torch-Pruning.
- MoE models are routed through an adapter boundary so downstream projects can
  plug in their local Expert_Sparsity modifications without changing the public
  package structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch.nn as nn

from ...pruner import BasePruner, importance as importance_module
from ...utils import count_ops_and_params


class HFModelFlavor(str, Enum):
    """High-level Hugging Face model categories supported by this repo."""

    CNN_DENSE = "cnn_or_dense"
    MOE = "moe"
    UNKNOWN = "unknown"


@dataclass
class HFDensePruningConfig:
    """Configuration for pruning Hugging Face CNN/dense models with TP."""

    pruning_ratio: float = 0.5
    global_pruning: bool = False
    iterative_steps: int = 1
    round_to: Optional[int] = None
    importance: Optional[importance_module.Importance] = None
    ignored_modules: Sequence[nn.Module] = field(default_factory=tuple)
    root_module_types: Sequence[type[nn.Module]] = field(
        default_factory=lambda: (nn.Conv2d, nn.Linear)
    )
    output_transform: Optional[Any] = None

    def resolved_importance(self) -> importance_module.Importance:
        return self.importance or importance_module.GroupMagnitudeImportance(p=2)


@dataclass
class HFMoEPruningPlan:
    """Descriptor returned for MoE models.

    The plan captures module names that a downstream Expert_Sparsity integration
    may want to target. The actual pruning implementation is left as an adapter
    boundary because this repository intentionally does not vendor that code.
    """

    expert_modules: List[str]
    router_modules: List[str]
    shared_modules: List[str]


class ExpertSparsityAdapter:
    """Extension point for local Expert_Sparsity-based implementations."""

    def plan(self, model: nn.Module) -> HFMoEPruningPlan:
        expert_modules: List[str] = []
        router_modules: List[str] = []
        shared_modules: List[str] = []
        for name, module in model.named_modules():
            lower_name = name.lower()
            if any(token in lower_name for token in ("expert", "experts")):
                expert_modules.append(name)
            elif any(token in lower_name for token in ("gate", "router")):
                router_modules.append(name)
            elif any(token in lower_name for token in ("shared_expert", "shared")):
                shared_modules.append(name)
        return HFMoEPruningPlan(
            expert_modules=expert_modules,
            router_modules=router_modules,
            shared_modules=shared_modules,
        )

    def prune(self, model: nn.Module, example_inputs: Any, **_: Any) -> HFMoEPruningPlan:
        plan = self.plan(model)
        raise NotImplementedError(
            "MoE pruning is intentionally routed through an adapter boundary. "
            "Integrate your local Expert_Sparsity implementation here and use "
            f"the generated plan as a starting point: {plan}."
        )


def detect_hf_model_flavor(model: nn.Module) -> HFModelFlavor:
    """Infer whether a Hugging Face model should use dense or MoE pruning."""

    module_names = [name.lower() for name, _ in model.named_modules()]
    if any(token in name for name in module_names for token in ("experts", "expert", "router", "gate")):
        return HFModelFlavor.MOE
    if any(isinstance(module, nn.Conv2d) for module in model.modules()):
        return HFModelFlavor.CNN_DENSE
    if any(isinstance(module, nn.Linear) for module in model.modules()):
        return HFModelFlavor.CNN_DENSE
    return HFModelFlavor.UNKNOWN


def default_ignored_modules(model: nn.Module) -> List[nn.Module]:
    """Heuristic to skip task heads / classifiers during dense pruning."""

    ignored: List[nn.Module] = []
    suffixes = (
        "classifier",
        "lm_head",
        "score",
        "qa_outputs",
        "embed_out",
        "generator_lm_head",
    )
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(suffixes):
            ignored.append(module)
    return ignored


def _materialize_example_inputs(example_inputs: Any) -> Any:
    if isinstance(example_inputs, Mapping):
        return dict(example_inputs)
    if isinstance(example_inputs, (list, tuple)):
        return tuple(example_inputs)
    return example_inputs


def prune_hf_dense_model(
    model: nn.Module,
    example_inputs: Any,
    config: Optional[HFDensePruningConfig] = None,
) -> Dict[str, Any]:
    """Prune a dense Hugging Face model with Torch-Pruning."""

    resolved_config = config or HFDensePruningConfig()
    example_inputs = _materialize_example_inputs(example_inputs)
    ignored_layers = list(dict.fromkeys([
        *default_ignored_modules(model),
        *resolved_config.ignored_modules,
    ]))

    pruner = BasePruner(
        model=model,
        example_inputs=example_inputs,
        importance=resolved_config.resolved_importance(),
        global_pruning=resolved_config.global_pruning,
        iterative_steps=resolved_config.iterative_steps,
        pruning_ratio=resolved_config.pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=resolved_config.round_to,
        root_module_types=resolved_config.root_module_types,
        output_transform=resolved_config.output_transform,
    )

    base_macs, base_params = count_ops_and_params(model, example_inputs)
    for group in pruner.step(interactive=True):
        group.prune()
    pruned_macs, pruned_params = count_ops_and_params(model, example_inputs)

    return {
        "backend": HFModelFlavor.CNN_DENSE.value,
        "ignored_layers": [name for name, module in model.named_modules() if module in ignored_layers],
        "base_macs": base_macs,
        "base_params": base_params,
        "pruned_macs": pruned_macs,
        "pruned_params": pruned_params,
    }


def prune_hf_model(
    model: nn.Module,
    example_inputs: Any,
    *,
    flavor: Optional[HFModelFlavor] = None,
    dense_config: Optional[HFDensePruningConfig] = None,
    moe_adapter: Optional[ExpertSparsityAdapter] = None,
) -> Any:
    """Dispatch Hugging Face pruning to the dense or MoE path."""

    resolved_flavor = flavor or detect_hf_model_flavor(model)
    if resolved_flavor == HFModelFlavor.CNN_DENSE:
        return prune_hf_dense_model(model, example_inputs, config=dense_config)
    if resolved_flavor == HFModelFlavor.MOE:
        adapter = moe_adapter or ExpertSparsityAdapter()
        return adapter.plan(model)
    raise ValueError(
        "Unable to infer a supported Hugging Face pruning flavor for this model. "
        "Pass `flavor=HFModelFlavor.CNN_DENSE` or `HFModelFlavor.MOE` explicitly."
    )
