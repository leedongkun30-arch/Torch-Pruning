# Acknowledgement: This unified entry point routes dense/CNN Hugging Face models
# to Torch-Pruning and MoE models to the merged Expert_Sparsity workflow.

"""Unified Hugging Face pruning entry point."""

from __future__ import annotations

import argparse
import json

import torch

from torch_pruning.hf import HFUnifiedPruningConfig, is_moe_model, prune_hf_model


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
    from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM

    config = HFUnifiedPruningConfig(
        pruning_ratio=args.pruning_ratio,
        moe_method=args.moe_method,
        preserve_experts=args.r,
        score_metric=args.score_metric,
        beta=args.beta,
    )

    dense_model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).eval()
    if is_moe_model(dense_model):
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).eval()
        summary = prune_hf_model(model, config=config)
    else:
        processor = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        pixel_values = processor(images=torch.rand(3, 224, 224), return_tensors="pt")["pixel_values"]
        summary = prune_hf_model(dense_model, {"pixel_values": pixel_values}, config=config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
