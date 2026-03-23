# Acknowledgement: This unified entry point routes dense/CNN Hugging Face models
# to Torch-Pruning and MoE models to the merged Expert_Sparsity workflow.

"""Unified Hugging Face pruning entry point."""

from __future__ import annotations

import argparse
import json

import torch

from torch_pruning.hf import HFUnifiedPruningConfig, load_pretrained_hf_model, prune_hf_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified HF pruning entry point")
    parser.add_argument("--model", required=True, help="HF model identifier")
    parser.add_argument("--pruning-ratio", type=float, default=0.5)
    parser.add_argument("--moe-method", default="layerwise_pruning", choices=["layerwise_pruning", "progressive_pruning", "dynamic_skipping"])
    parser.add_argument("--r", type=int, default=1, help="Experts to preserve for MoE")
    parser.add_argument("--score-metric", default="l1", choices=["l1", "l2"])
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = HFUnifiedPruningConfig(
        pruning_ratio=args.pruning_ratio,
        moe_method=args.moe_method,
        preserve_experts=args.r,
        score_metric=args.score_metric,
        beta=args.beta,
    )

    loaded = load_pretrained_hf_model(args.model, trust_remote_code=args.trust_remote_code)
    summary = prune_hf_model(loaded.model, loaded.example_inputs, config=config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
