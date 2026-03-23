"""Model patch helpers for Expert_Sparsity-style workflows."""

from .modeling_mixtral import is_mixtral_like_model, patch_mixtral_for_dynamic_skipping

__all__ = ["is_mixtral_like_model", "patch_mixtral_for_dynamic_skipping"]
