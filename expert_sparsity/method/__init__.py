"""MoE pruning methods inspired by Expert_Sparsity."""

from . import dynamic_skipping, expert_pruning

METHODS = {
    "layerwise_pruning": expert_pruning.layerwise_pruning,
    "progressive_pruning": expert_pruning.progressive_pruning,
    "dynamic_skipping": dynamic_skipping.dynamic_skipping,
}

__all__ = ["METHODS", "dynamic_skipping", "expert_pruning"]
