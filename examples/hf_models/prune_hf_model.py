"""Minimal Hugging Face pruning entry point for public repo documentation.

Examples
--------
Dense / CNN models:
    python examples/hf_models/prune_hf_model.py \
        --model google/vit-base-patch16-224 \
        --model-kind dense

MoE models:
    python examples/hf_models/prune_hf_model.py \
        --model mistralai/Mixtral-8x7B-v0.1 \
        --model-kind moe
"""

from __future__ import annotations

import argparse
import json

import torch

import torch_pruning as tp
from torch_pruning.integrations.hf import (
    HFDensePruningConfig,
    HFModelFlavor,
    prune_hf_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune Hugging Face models")
    parser.add_argument("--model", required=True, help="HF model identifier")
    parser.add_argument(
        "--model-kind",
        choices=[HFModelFlavor.CNN_DENSE.value, HFModelFlavor.MOE.value],
        required=True,
        help="Use `cnn_or_dense` for CNN/dense models and `moe` for MoE models.",
    )
    parser.add_argument("--pruning-ratio", type=float, default=0.5)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required for this example. Install it manually before running."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flavor = HFModelFlavor(args.model_kind)

    if flavor == HFModelFlavor.CNN_DENSE:
        processor = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).eval().to(device)
        pixel_values = processor(images=torch.rand(3, 224, 224), return_tensors="pt")["pixel_values"].to(device)
        summary = prune_hf_model(
            model,
            {"pixel_values": pixel_values},
            flavor=flavor,
            dense_config=HFDensePruningConfig(pruning_ratio=args.pruning_ratio),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).eval().to(device)
        encoded = tokenizer("Torch pruning router demo.", return_tensors="pt")
        example_inputs = {key: value.to(device) for key, value in encoded.items()}
        summary = prune_hf_model(model, example_inputs, flavor=flavor)

    print(json.dumps(summary, indent=2, default=lambda x: x if isinstance(x, (int, float, str, list, dict)) else repr(x)))


if __name__ == "__main__":
    main()
