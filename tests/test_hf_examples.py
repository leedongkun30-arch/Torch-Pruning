"""Regression tests for merged HF dense and MoE pruning helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from torch_pruning.hf import (
    ExpertSparsityConfig,
    apply_dynamic_skipping,
    default_dense_ignored_layers,
    find_moe_blocks,
    prune_dense_hf_model,
    prune_moe_hf_model,
)


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
    assert "classifier" in summary["ignored_layers"]


def test_find_and_prune_moe_hf_model() -> None:
    model = TinyMoEHFModel().eval()
    blocks = find_moe_blocks(model)
    assert len(blocks) == 1
    assert blocks[0].router_name == "router"

    summary = prune_moe_hf_model(
        model,
        config=ExpertSparsityConfig(
            method="layerwise_pruning",
            preserve_experts=2,
            score_metric="l1",
        ),
    )

    assert summary["path"] == "moe"
    assert summary["blocks"][0]["original_experts"] == 3
    assert summary["blocks"][0]["kept_experts"] == 2
    assert len(model.block_sparse_moe.experts) == 2
    assert model.block_sparse_moe.router.out_features == 2


def test_apply_dynamic_skipping_annotations() -> None:
    model = TinyMoEHFModel().eval()
    summary = apply_dynamic_skipping(model, beta=0.2)
    assert summary["mode"] == "dynamic_skipping"
    assert summary["num_blocks"] == 1
    assert model.block_sparse_moe.expert_skipping_beta == 0.2
