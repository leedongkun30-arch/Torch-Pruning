# Acknowledgement: This runtime layer centralizes Hugging Face model loading,
# example input creation, forward execution, and output reduction.

"""Shared Hugging Face runtime utilities for dense and MoE pruning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class HFModelKind(str, Enum):
    DENSE = "dense"
    MOE = "moe"


KNOWN_MOE_MODEL_TYPES = {"mixtral", "qwen2_moe", "deepseek_v2", "deepseek_v3", "jamba"}
KNOWN_VISION_MODEL_TYPES = {"vit", "deit", "beit", "swin", "convnext", "resnet"}


@dataclass
class HFLoadedModel:
    model: nn.Module
    example_inputs: Dict[str, torch.Tensor]
    model_kind: HFModelKind


def infer_hf_model_kind(model: nn.Module) -> HFModelKind:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "")
    if model_type in KNOWN_MOE_MODEL_TYPES:
        return HFModelKind.MOE
    for _, module in model.named_modules():
        experts = getattr(module, "experts", None)
        if isinstance(experts, nn.ModuleList) and len(experts) > 0:
            return HFModelKind.MOE
    return HFModelKind.DENSE


def infer_hf_input_mode(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "")
    if model_type in KNOWN_VISION_MODEL_TYPES or any(isinstance(m, nn.Conv2d) for m in model.modules()):
        return "vision"
    return "text"


def build_hf_example_inputs(
    model: nn.Module,
    *,
    batch_size: int = 1,
    seq_len: int = 16,
    image_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    config = getattr(model, "config", None)
    input_mode = infer_hf_input_mode(model)
    device = device or next(model.parameters()).device
    if input_mode == "vision":
        channels = getattr(config, "num_channels", 3)
        size = image_size or getattr(config, "image_size", 224)
        if isinstance(size, (tuple, list)):
            height, width = size
        else:
            height = width = int(size)
        return {"pixel_values": torch.randn(batch_size, channels, height, width, device=device)}

    vocab_size = max(8, int(getattr(config, "vocab_size", 128)))
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def hf_forward(model: nn.Module, example_inputs: Dict[str, torch.Tensor]) -> Any:
    return model(**example_inputs)


def hf_output_transform(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.sum()
    if hasattr(output, "logits"):
        return output.logits.sum()
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.sum()
    if hasattr(output, "pooler_output"):
        return output.pooler_output.sum()
    if isinstance(output, dict):
        for key in ("logits", "last_hidden_state", "pooler_output"):
            if key in output:
                return output[key].sum()
        return next(iter(output.values())).sum()
    if isinstance(output, (tuple, list)):
        return hf_output_transform(output[0])
    raise TypeError(f"Unsupported HF output type: {type(output)!r}")


def load_pretrained_hf_model(model_name: str, *, trust_remote_code: bool = False) -> HFLoadedModel:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if config.model_type in KNOWN_MOE_MODEL_TYPES:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        encoded = tokenizer("Structural pruning for Mixture-of-Experts models.", return_tensors="pt")
        return HFLoadedModel(model=model, example_inputs=dict(encoded), model_kind=HFModelKind.MOE)

    if config.model_type in KNOWN_VISION_MODEL_TYPES:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).eval()
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        pixel_values = processor(images=torch.rand(3, getattr(config, "image_size", 224), getattr(config, "image_size", 224)), return_tensors="pt")["pixel_values"]
        return HFLoadedModel(model=model, example_inputs={"pixel_values": pixel_values}, model_kind=HFModelKind.DENSE)

    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    encoded = tokenizer("Structural pruning for dense Hugging Face models.", return_tensors="pt")
    return HFLoadedModel(model=model, example_inputs=dict(encoded), model_kind=HFModelKind.DENSE)
