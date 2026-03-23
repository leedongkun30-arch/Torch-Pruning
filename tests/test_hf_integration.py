"""Tests for the Hugging Face pruning integration scaffolding."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from torch_pruning.integrations.hf import (
    HFDensePruningConfig,
    HFModelFlavor,
    detect_hf_model_flavor,
    prune_hf_model,
)


class TinyHFVisionModel(nn.Module):
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
        logits = self.classifier(x)
        return SimpleNamespace(logits=logits)


class TinyExpert(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.relu(self.w1(x)))


class TinyHFMoEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(8, 2)
        self.experts = nn.ModuleList([TinyExpert(8), TinyExpert(8)])

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = input_ids.float()
        weights = torch.softmax(self.router(hidden), dim=-1)
        outputs = sum(weights[:, i : i + 1] * expert(hidden) for i, expert in enumerate(self.experts))
        return SimpleNamespace(last_hidden_state=outputs)


def test_detect_hf_model_flavor() -> None:
    assert detect_hf_model_flavor(TinyHFVisionModel()) == HFModelFlavor.CNN_DENSE
    assert detect_hf_model_flavor(TinyHFMoEModel()) == HFModelFlavor.MOE


def test_prune_hf_dense_model() -> None:
    model = TinyHFVisionModel().eval()
    example_inputs = {"pixel_values": torch.randn(1, 3, 16, 16)}
    baseline_out = model(**example_inputs).logits

    summary = prune_hf_model(
        model,
        example_inputs,
        flavor=HFModelFlavor.CNN_DENSE,
        dense_config=HFDensePruningConfig(pruning_ratio=0.25, output_transform=lambda out: out.logits.sum()),
    )

    pruned_out = model(**example_inputs).logits
    assert baseline_out.shape == pruned_out.shape == (1, 4)
    assert summary["backend"] == HFModelFlavor.CNN_DENSE.value
    assert summary["pruned_params"] < summary["base_params"]
    assert "classifier" in summary["ignored_layers"]


def test_prune_hf_moe_model_returns_plan() -> None:
    model = TinyHFMoEModel().eval()
    plan = prune_hf_model(
        model,
        {"input_ids": torch.randn(2, 8)},
        flavor=HFModelFlavor.MOE,
    )

    assert plan.expert_modules
    assert plan.router_modules == ["router"]
