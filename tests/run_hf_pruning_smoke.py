"""Standalone smoke test for the unified Hugging Face pruning repository."""

from __future__ import annotations

from types import SimpleNamespace

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn

from expert_sparsity.data import build_calib_loader
from expert_sparsity.method import METHODS
from expert_sparsity.model import patch_mixtral_for_dynamic_skipping
from torch_pruning.hf import HFUnifiedPruningConfig, build_hf_example_inputs, prune_hf_model


class TinyTokenizer:
    def __call__(self, text, truncation, max_length, padding, return_tensors):
        token_ids = torch.arange(max_length).unsqueeze(0)
        return {
            "input_ids": token_ids,
            "attention_mask": torch.ones_like(token_ids),
        }


class TinyDenseHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, 4)
        self.config = SimpleNamespace(model_type="vit", image_size=16, num_channels=3)

    def forward(self, pixel_values: torch.Tensor):
        x = self.backbone(pixel_values)
        x = x.flatten(1)
        return SimpleNamespace(logits=self.classifier(x))


class TinyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class TinyMoEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(8, 3, bias=False)
        self.experts = nn.ModuleList([TinyExpert(), TinyExpert(), TinyExpert()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.router(x), dim=-1)
        outputs = []
        for idx, expert in enumerate(self.experts):
            outputs.append(weights[:, idx : idx + 1] * expert(x))
        return sum(outputs)


class TinyMoEHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.block_sparse_moe = TinyMoEBlock()
        self.lm_head = nn.Linear(8, 32)
        self.config = SimpleNamespace(model_type="mixtral", vocab_size=32)

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        x = self.embed(input_ids).mean(dim=1)
        hidden = self.block_sparse_moe(x)
        return SimpleNamespace(logits=self.lm_head(hidden))


def main() -> None:
    print("[1/4] Building calibration loader...")
    loader = build_calib_loader(
        "c4",
        TinyTokenizer(),
        max_block_size=8,
        n_blocks_for_stat=2,
        batch_size=1,
        num_workers=0,
        seed=0,
    )
    batch = next(iter(loader))
    assert "input_ids" in batch
    assert set(METHODS.keys()) == {"layerwise_pruning", "progressive_pruning", "dynamic_skipping"}

    print("[2/4] Running dense Torch-Pruning path...")
    dense_model = TinyDenseHFModel().eval()
    dense_summary = prune_hf_model(
        dense_model,
        build_hf_example_inputs(dense_model, image_size=16),
        config=HFUnifiedPruningConfig(pruning_ratio=0.25),
    )
    assert dense_summary["path"] == "dense"

    print("[3/4] Running MoE Expert_Sparsity path...")
    moe_model = TinyMoEHFModel().eval()
    patch_info = patch_mixtral_for_dynamic_skipping(moe_model)
    assert patch_info["is_mixtral_like"] is True
    moe_summary = prune_hf_model(
        moe_model,
        config=HFUnifiedPruningConfig(moe_method="dynamic_skipping", preserve_experts=2, beta=0.3),
    )
    assert moe_summary["path"] == "moe"
    assert len(moe_model.block_sparse_moe.experts) == 3  # dynamic skipping should not prune experts

    print("[4/4] Running MoE pruning mode...")
    moe_prune_model = TinyMoEHFModel().eval()
    moe_prune_summary = prune_hf_model(
        moe_prune_model,
        config=HFUnifiedPruningConfig(moe_method="layerwise_pruning", preserve_experts=2),
    )
    assert moe_prune_summary["path"] == "moe"
    assert len(moe_prune_model.block_sparse_moe.experts) == 2

    print("Smoke test passed.")
    print({
        "dense": dense_summary["path"],
        "moe_dynamic": moe_summary["path"],
        "moe_pruned_experts": len(moe_prune_model.block_sparse_moe.experts),
    })


if __name__ == "__main__":
    main()
