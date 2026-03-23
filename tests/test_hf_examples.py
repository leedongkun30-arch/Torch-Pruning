"""Regression tests for the separated HF pruning example entry points."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from examples.hf_models.dense_pruning import default_dense_ignored_layers, prune_dense_hf_model
from examples.hf_models.moe_pruning import build_moe_workflow_plan, run_moe_pruning_workflow


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


class TinyMoEHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(8, 2)
        self.experts = nn.ModuleList([TinyExpert(), TinyExpert()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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


def test_moe_workflow_plan_and_placeholder() -> None:
    model = TinyMoEHFModel()
    plan = build_moe_workflow_plan(
        model,
        model_name="toy-moe",
        output_dir="outputs/toy",
        note="separate workflow",
    )

    assert plan["path"] == "moe"
    assert plan["router_like_modules"] == ["router"]
    assert any(name.startswith("experts") for name in plan["expert_like_modules"])

    with pytest.raises(NotImplementedError):
        run_moe_pruning_workflow()
