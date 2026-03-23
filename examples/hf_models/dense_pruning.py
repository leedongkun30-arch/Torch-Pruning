# Acknowledgement: This dense/CNN pruning entry point is organized around
# Torch-Pruning and is intended for Hugging Face models that do not use MoE.

"""Torch-Pruning workflow for dense or CNN Hugging Face models."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

import torch_pruning as tp


def default_dense_ignored_layers(model: nn.Module) -> List[nn.Module]:
    """Skip task-specific heads by default."""

    ignored_layers: List[nn.Module] = []
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
            ignored_layers.append(module)
    return ignored_layers


def prune_dense_hf_model(
    model: nn.Module,
    example_inputs: Any,
    *,
    pruning_ratio: float = 0.5,
    importance: Optional[tp.importance.Importance] = None,
    ignored_layers: Optional[Sequence[nn.Module]] = None,
    root_module_types: Sequence[type[nn.Module]] = (nn.Conv2d, nn.Linear),
    round_to: Optional[int] = None,
    output_transform: Optional[Any] = None,
) -> Dict[str, Any]:
    """Apply Torch-Pruning to a dense/CNN Hugging Face model."""

    if isinstance(example_inputs, Mapping):
        example_inputs = dict(example_inputs)
    elif isinstance(example_inputs, (list, tuple)):
        example_inputs = tuple(example_inputs)

    ignored = list(default_dense_ignored_layers(model))
    if ignored_layers is not None:
        ignored.extend(ignored_layers)
    ignored = list(dict.fromkeys(ignored))

    imp = importance or tp.importance.GroupMagnitudeImportance(p=2)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    pruner = tp.pruner.BasePruner(
        model=model,
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored,
        root_module_types=root_module_types,
        round_to=round_to,
        output_transform=output_transform,
    )

    for group in pruner.step(interactive=True):
        group.prune()

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    ignored_names = [name for name, module in model.named_modules() if module in ignored]
    return {
        "path": "dense",
        "ignored_layers": ignored_names,
        "base_macs": base_macs,
        "base_params": base_params,
        "pruned_macs": pruned_macs,
        "pruned_params": pruned_params,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense/CNN HF pruning with Torch-Pruning")
    parser.add_argument("--model", required=True, help="HF model identifier")
    parser.add_argument("--pruning-ratio", type=float, default=0.5)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise SystemExit("transformers is required for this example.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    pixel_values = processor(images=torch.rand(3, 224, 224), return_tensors="pt")["pixel_values"].to(device)
    summary = prune_dense_hf_model(
        model,
        {"pixel_values": pixel_values},
        pruning_ratio=args.pruning_ratio,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
