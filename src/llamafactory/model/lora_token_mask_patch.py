from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from peft.tuners.lora.layer import Linear as LoraLinear

if TYPE_CHECKING:
    from torch.nn import Module

_PATCHED = False

logger = logging.getLogger(__name__)


def set_lora_token_mask(model: "Module", mask: Optional[torch.Tensor]) -> None:
    """
    Set token mask on all LoRA linear layers in the model.
    This is more reliable than ContextVar in distributed training scenarios.
    
    Args:
        model: The model containing LoRA layers (e.g., language_model).
        mask: The token mask tensor, or None to disable masking.
    """
    mask_to_set = mask.detach() if mask is not None else None
    for module in model.modules():
        if isinstance(module, LoraLinear):
            module._lora_token_mask = mask_to_set


def clear_lora_token_mask(model: "Module") -> None:
    """
    Clear token mask from all LoRA linear layers in the model.
    
    Args:
        model: The model containing LoRA layers.
    """
    for module in model.modules():
        if isinstance(module, LoraLinear):
            module._lora_token_mask = None


@contextmanager
def use_lora_token_mask_on_model(model: "Module", mask: Optional[torch.Tensor]):
    """
    Context manager to apply a token mask to all LoRA linear layers in a model.
    This replaces the ContextVar-based approach for better stability in distributed training.
    
    Args:
        model: The model containing LoRA layers.
        mask: The token mask tensor.
    """
    set_lora_token_mask(model, mask)
    try:
        yield
    finally:
        clear_lora_token_mask(model)


def enable_lora_token_mask_patch() -> None:
    """
    Monkey patch peft.tuners.lora.layer.Linear.forward once to support token-level masking.
    """
    global _PATCHED
    if _PATCHED:
        return

    LoraLinear.forward = _mask_aware_linear_forward  # type: ignore[assignment]
    _PATCHED = True


def _mask_aware_linear_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        
        # Retrieve mask from instance attribute (set by set_lora_token_mask)
        mask = getattr(self, '_lora_token_mask', None)
        prepared_mask = _prepare_token_mask(mask, result)
        lora_A_keys = self.lora_A.keys()
        for active_adapter in self.active_adapters:
            if active_adapter not in lora_A_keys:
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            x = self._cast_input_dtype(x, lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                addition = lora_B(lora_A(dropout(x))) * scaling
            else:
                if isinstance(dropout, nn.Identity) or not self.training:
                    base_result = result
                else:
                    x = dropout(x)
                    base_result = None

                addition = self.lora_magnitude_vector[active_adapter](
                    x,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    base_layer=self.get_base_layer(),
                    base_result=base_result,
                )

            addition = _apply_mask(addition, prepared_mask)
            result = result + addition

        result = result.to(torch_result_dtype)

    return result


def _prepare_token_mask(
    mask: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> Optional[torch.Tensor]:
    if mask is None or not isinstance(mask, torch.Tensor):
        return None

    prepared = mask
    # Expand trailing singleton dimensions so that mask can broadcast to reference
    while prepared.dim() < reference.dim():
        prepared = prepared.unsqueeze(-1)

    try:
        prepared = prepared.to(device=reference.device, dtype=reference.dtype)
        prepared = prepared.expand(reference.shape)
    except RuntimeError:
        # As a fallback, reshape the mask to align with all but the last dim if sizes match.
        leading_elems = reference.shape[:-1]
        trailing = 1
        if prepared.numel() == 0:
            return None
        try:
            prepared = prepared.view(*leading_elems, trailing).to(reference.device, reference.dtype)
            prepared = prepared.expand(reference.shape)
        except Exception:
            # If expansion fails (e.g. shape mismatch), return None.
            # This means masking will be disabled for this layer.
            # CAUTION: This might be intended for layers where mask doesn't apply (e.g. Vision),
            # but risky if it's a Text layer with mismatch.
            return None

    return prepared


def _apply_mask(addition: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return addition
    
    return addition * mask


__all__ = [
    "enable_lora_token_mask_patch",
    "set_lora_token_mask",
    "clear_lora_token_mask",
    "use_lora_token_mask_on_model",
]
