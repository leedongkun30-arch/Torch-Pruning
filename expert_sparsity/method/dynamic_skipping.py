"""Dynamic skipping helper for merged Expert_Sparsity workflows."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from expert_sparsity.model import patch_mixtral_for_dynamic_skipping
from torch_pruning.hf.moe import apply_dynamic_skipping


def dynamic_skipping(model: nn.Module, calib_loader, args: Optional[Any] = None):
    beta = getattr(args, "beta", 0.2)
    patch_info = patch_mixtral_for_dynamic_skipping(model)
    skip_info = apply_dynamic_skipping(model, beta=beta)
    info = {
        "path": "moe",
        "method": "dynamic_skipping",
        "beta": beta,
        "n_calibration_batches": len(calib_loader),
        "patch_info": patch_info,
        "skip_info": skip_info,
    }
    return model, info
