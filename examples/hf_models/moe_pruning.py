# Acknowledgement: This MoE pruning entry point is reserved for a local
# Expert_Sparsity-based workflow and is intentionally kept separate from the
# Torch-Pruning dense/CNN path.

"""Entry point placeholder for MoE Hugging Face pruning workflows."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

import torch.nn as nn


def build_moe_workflow_plan(
    model: nn.Module,
    *,
    model_name: str,
    output_dir: str,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Describe the separate MoE workflow without merging Expert_Sparsity code."""

    expert_like_modules = []
    router_like_modules = []
    for name, _ in model.named_modules():
        lower_name = name.lower()
        if "expert" in lower_name:
            expert_like_modules.append(name)
        if "router" in lower_name or "gate" in lower_name:
            router_like_modules.append(name)

    return {
        "path": "moe",
        "model_name": model_name,
        "output_dir": output_dir,
        "note": note,
        "expert_like_modules": expert_like_modules,
        "router_like_modules": router_like_modules,
        "status": "attach_local_expert_sparsity_code_here",
    }


def run_moe_pruning_workflow(*_: Any, **__: Any) -> None:
    raise NotImplementedError(
        "MoE pruning stays on a separate Expert_Sparsity-based path in this repo. "
        "Attach your local implementation inside examples/hf_models/moe_pruning.py."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoE HF pruning workflow placeholder")
    parser.add_argument("--model", required=True, help="HF model identifier")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--note", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = {
        "path": "moe",
        "model_name": args.model,
        "output_dir": args.output_dir,
        "note": args.note,
        "status": "attach_local_expert_sparsity_code_here",
    }
    print(json.dumps(summary, indent=2))
    raise SystemExit(
        "This public entry point documents the MoE path only. Insert your local Expert_Sparsity workflow here."
    )


if __name__ == "__main__":
    main()
