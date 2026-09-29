from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...extras import logging

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


logger = logging.get_logger(__name__)


def _mask_vocab_row_grads(param: torch.nn.Parameter, original_vocab_size: int) -> None:
    """
    Mask gradients for rows [0:original_vocab_size).
    This should be used after resizing vocab, to only train new token rows.
    """

    def _hook(grad: torch.Tensor | None) -> torch.Tensor | None:
        if grad is None:
            return None
        if grad.ndim == 0:
            return grad
        if grad.size(0) <= original_vocab_size:
            return grad
        masked = grad.clone()
        masked[:original_vocab_size].zero_()
        return masked

    param.register_hook(_hook)


def _mask_vocab_token_id_grads(param: torch.nn.Parameter, train_token_ids: torch.Tensor) -> None:
    """
    Mask gradients for all vocab rows except `train_token_ids`.
    """

    def _hook(grad: torch.Tensor | None) -> torch.Tensor | None:
        if grad is None:
            return None
        if grad.ndim == 0:
            return grad
        vocab_size = grad.size(0)
        if train_token_ids.numel() == 0:
            return grad
        if train_token_ids.max().item() >= vocab_size or train_token_ids.min().item() < 0:
            return grad
        masked = grad.clone()
        masked.zero_()
        masked.index_copy_(0, train_token_ids, grad.index_select(0, train_token_ids))
        return masked

    param.register_hook(_hook)


def apply_train_new_tokens_only_hooks(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    train_tokens: list[str] | None = None,
) -> None:
    """
    Apply gradient-masking hooks so only newly added token rows are trainable for:
      - input embedding weight (nn.Embedding.weight)
      - output embedding / lm_head weight (nn.Linear.weight)

    Notes:
      - This does NOT automatically add these modules to LoRA `modules_to_save`.
        You still need to set `additional_target` (or let adapter.py infer it).
      - To prevent AdamW from modifying frozen rows via weight decay, we also mark
        these parameters for no weight decay and let the trainer zero out group wd.
    """
    # Prefer explicit token-id masking (works even if the base model already includes the tokens).
    token_ids: list[int] = []
    missing: list[str] = []
    if tokenizer is not None and train_tokens:
        for tok in train_tokens:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None or (hasattr(tokenizer, "unk_token_id") and tid == tokenizer.unk_token_id):
                missing.append(tok)
            else:
                token_ids.append(int(tid))

        token_ids = sorted(set(token_ids))
        if missing:
            logger.warning_rank0(
                "train_new_tokens_only: some provided tokens are not in tokenizer vocab and will be ignored: "
                + ", ".join(missing[:20])
                + (" ..." if len(missing) > 20 else "")
            )

    use_token_id_mask = len(token_ids) > 0
    if use_token_id_mask:
        train_token_ids = torch.tensor(token_ids, dtype=torch.long)
        logger.info_rank0(
            f"train_new_tokens_only: using token-id masking for {len(token_ids)} token(s). "
            f"min_id={token_ids[0]}, max_id={token_ids[-1]}."
        )
    else:
        original_vocab_size = getattr(model, "_lf_original_vocab_size", None)
        resized_vocab_size = getattr(model, "_lf_resized_vocab_size", None)
        if original_vocab_size is None or resized_vocab_size is None:
            logger.warning_rank0(
                "train_new_tokens_only is enabled, but cannot determine which vocab rows to train. "
                "Provide `add_tokens` / `add_special_tokens` (even if already present) so we can select by token id."
            )
            return

        original_vocab_size = int(original_vocab_size)
        resized_vocab_size = int(resized_vocab_size)
        if resized_vocab_size <= original_vocab_size:
            logger.warning_rank0(
                "train_new_tokens_only is enabled, but no vocab expansion detected and no valid train_tokens provided. "
                "Skipping hooks."
            )
            return

    marked: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # PEFT "modules_to_save" typically nests trainable copies under ".modules_to_save".
        is_embed = ("embed_tokens" in name and name.endswith(".weight")) or (
            "embed_tokens" in name and ".modules_to_save" in name and name.endswith(".weight")
        )
        is_lm_head = ("lm_head" in name and name.endswith(".weight")) or (
            "lm_head" in name and ".modules_to_save" in name and name.endswith(".weight")
        )
        if not (is_embed or is_lm_head):
            continue
        if param.ndim != 2:
            continue

        if use_token_id_mask:
            _mask_vocab_token_id_grads(param, train_token_ids=train_token_ids.to(device=param.device))
        else:
            if param.size(0) != resized_vocab_size:
                continue
            _mask_vocab_row_grads(param, original_vocab_size=original_vocab_size)
        setattr(param, "_lf_no_weight_decay", True)
        marked.append(param)

    if marked:
        setattr(model, "_lf_no_weight_decay_params", marked)
        if use_token_id_mask:
            logger.info_rank0(f"Enabled train_new_tokens_only: token-id grad masking on {len(marked)} parameter(s).")
        else:
            logger.info_rank0(
                f"Enabled train_new_tokens_only: range grad masking for vocab rows [0:{original_vocab_size}) "
                f"on {len(marked)} parameter(s)."
            )
    else:
        logger.warning_rank0(
            "train_new_tokens_only is enabled, but no matching embedding/lm_head parameters were found. "
            "Did you set `additional_target` to include input/output embeddings when using LoRA?"
        )


def set_no_weight_decay_for_marked_params(optimizer: "Optimizer", model: "PreTrainedModel") -> None:
    """
    Ensure parameters marked with `_lf_no_weight_decay` have weight_decay=0.0 in the optimizer param groups.
    This is necessary because AdamW applies weight decay even when gradients are masked to zero.
    """
    marked = getattr(model, "_lf_no_weight_decay_params", None)
    if not marked:
        return

    marked_ids = {id(p) for p in marked}
    changed = 0
    for group in optimizer.param_groups:
        params = group.get("params", [])
        if any(id(p) in marked_ids for p in params):
            if group.get("weight_decay", 0.0) != 0.0:
                group["weight_decay"] = 0.0
                changed += 1

    if changed:
        logger.info_rank0(f"Set weight_decay=0.0 for {changed} optimizer param group(s) (train_new_tokens_only).")


