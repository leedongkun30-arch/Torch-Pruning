"""Regression tests for unified HF dense and MoE pruning helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from expert_sparsity.data import build_calib_loader
from expert_sparsity.method import METHODS
from expert_sparsity.model import patch_mixtral_for_dynamic_skipping
from torch_pruning.hf import (
    HFUnifiedPruningConfig,
    apply_dynamic_skipping,
    default_dense_ignored_layers,
    find_moe_blocks,
    is_moe_model,
    prune_dense_hf_model,
    prune_hf_model,
    prune_moe_hf_model,
)


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
        self.config = SimpleNamespace(model_type="vit")

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
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
        self.num_experts = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.router(x), dim=-1)
        outputs = []
        for idx, expert in enumerate(self.experts):
            outputs.append(weights[:, idx : idx + 1] * expert(x))
        return sum(outputs)


class TinyMoEHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_sparse_moe = TinyMoEBlock()
        self.config = SimpleNamespace(model_type="mixtral")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block_sparse_moe(x)


def test_default_dense_ignored_layers_finds_classifier() -> None:
    ignored = default_dense_ignored_layers(TinyDenseHFModel())
    assert len(ignored) == 1
    assert isinstance(ignored[0], nn.Linear)


def test_prune_dense_hf_model_keeps_output_shape() -> None:
    model = TinyDenseHFModel().eval()
    example_inputs = {"pixel_values": torch.randn(1, 3, 16, 16)}
    before = model(**example_inputs).logits
    summary = prune_dense_hf_model(
        model,
        example_inputs,
        pruning_ratio=0.25,
        output_transform=lambda out: out.logits.sum(),
    )
    after = model(**example_inputs).logits
    assert before.shape == after.shape == (1, 4)
    assert summary["path"] == "dense"
    assert summary["pruned_params"] < summary["base_params"]


def test_expert_sparsity_data_and_methods() -> None:
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


def test_find_and_prune_moe_hf_model() -> None:
    model = TinyMoEHFModel().eval()
    blocks = find_moe_blocks(model)
    assert len(blocks) == 1
    summary = prune_moe_hf_model(model)
    assert summary["path"] == "moe"
    assert len(model.block_sparse_moe.experts) == 1
    assert model.block_sparse_moe.router.out_features == 1


def test_dynamic_skipping_and_mixtral_patch() -> None:
    model = TinyMoEHFModel().eval()
    patch_info = patch_mixtral_for_dynamic_skipping(model)
    summary = apply_dynamic_skipping(model, beta=0.2)
    assert patch_info["is_mixtral_like"] is True
    assert summary["mode"] == "dynamic_skipping"
    assert model.block_sparse_moe.expert_skipping_beta == 0.2


def test_unified_router_dispatches_dense_and_moe() -> None:
    dense_model = TinyDenseHFModel().eval()
    dense_summary = prune_hf_model(
        dense_model,
        {"pixel_values": torch.randn(1, 3, 16, 16)},
        config=HFUnifiedPruningConfig(pruning_ratio=0.25),
    )
    assert dense_summary["path"] == "dense"
    assert is_moe_model(dense_model) is False

    moe_model = TinyMoEHFModel().eval()
    moe_summary = prune_hf_model(
        moe_model,
        config=HFUnifiedPruningConfig(moe_method="dynamic_skipping", preserve_experts=2, beta=0.3),
    )
    assert moe_summary["path"] == "moe"
    assert is_moe_model(moe_model) is True
