# Acknowledgement: This MoE pruning entry point merges an Expert_Sparsity-style
# expert pruning workflow into the repository for Hugging Face MoE models.

"""Expert_Sparsity-style workflow for Hugging Face MoE models."""

from __future__ import annotations

import argparse
import json

from torch_pruning.hf import ExpertSparsityConfig, load_pretrained_hf_model, prune_moe_hf_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoE HF pruning with Expert_Sparsity-style workflow")
    parser.add_argument("--model", required=True, help="HF model identifier")
    parser.add_argument(
        "--method",
        default="layerwise_pruning",
        choices=["layerwise_pruning", "progressive_pruning", "dynamic_skipping"],
    )
    parser.add_argument("--r", type=int, default=1, help="Number of experts to preserve")
    parser.add_argument("--score-metric", default="l1", choices=["l1", "l2"])
    parser.add_argument("--beta", type=float, default=None, help="Dynamic skipping threshold")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_pretrained_hf_model(args.model, trust_remote_code=args.trust_remote_code)
    summary = prune_moe_hf_model(
        loaded.model,
        config=ExpertSparsityConfig(
            method=args.method,
            preserve_experts=args.r,
            score_metric=args.score_metric,
            dynamic_skipping=args.method == "dynamic_skipping",
            beta=args.beta,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
