# Acknowledgement: This dense/CNN pruning entry point is organized around
# Torch-Pruning and is intended for Hugging Face models that do not use MoE.

"""Torch-Pruning workflow for dense or CNN Hugging Face models."""

from __future__ import annotations

import argparse
import json

import torch

from torch_pruning.hf import prune_dense_hf_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense/CNN HF pruning with Torch-Pruning")
    parser.add_argument("--model", required=True, help="HF model identifier")
    parser.add_argument("--pruning-ratio", type=float, default=0.5)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoImageProcessor, AutoModel

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
