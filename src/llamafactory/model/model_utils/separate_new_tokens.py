from __future__ import annotations

import os
from contextlib import contextmanager
from types import MethodType
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras import logging
from .checkpoint import load_safetensors_weight_map


if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedModel, PreTrainedTokenizerBase


logger = logging.get_logger(__name__)


def install_separate_new_token_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    is_trainable: bool,
    init_from_base: bool = True,
) -> bool:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or not isinstance(input_embeddings, nn.Embedding):
        logger.warning_rank0("Separate new-token embeddings require an nn.Embedding input layer.")
        return False
    if output_embeddings is None or not isinstance(output_embeddings, nn.Linear):
        logger.warning_rank0("Separate new-token lm_head requires an nn.Linear output layer.")
        return False

    embedding_vocab_size = int(input_embeddings.num_embeddings)
    tokenizer_size = int(len(tokenizer))
    new_token_ids = _get_separate_new_token_ids(tokenizer, embedding_vocab_size, tokenizer_size)
    if not new_token_ids:
        setattr(model, "_lf_original_vocab_size", embedding_vocab_size)
        setattr(model, "_lf_resized_vocab_size", embedding_vocab_size)
        setattr(model, "_lf_num_new_tokens", 0)
        return False

    output_vocab_size = int(output_embeddings.out_features)
    if output_vocab_size != embedding_vocab_size:
        raise ValueError(
            "Separate new-token embeddings expect input/output vocab sizes to match before installation, "
            f"got input={embedding_vocab_size}, output={output_vocab_size}."
        )

    max_new_token_id = max(new_token_ids)
    if max_new_token_id >= embedding_vocab_size and max_new_token_id + 1 > tokenizer_size:
        raise ValueError(
            "Separate new-token ids exceed both the model embedding vocabulary and tokenizer size, "
            f"max_new_token_id={max_new_token_id}, embedding_vocab={embedding_vocab_size}, tokenizer_vocab={tokenizer_size}."
        )

    num_new_tokens = len(new_token_ids)
    hidden_size = int(input_embeddings.embedding_dim)
    output_hidden_size = int(output_embeddings.in_features)
    if output_hidden_size != hidden_size:
        raise ValueError(
            "Separate new-token embeddings expect input/output hidden sizes to match, "
            f"got input={hidden_size}, output={output_hidden_size}."
        )

    device = input_embeddings.weight.device
    dtype = input_embeddings.weight.dtype
    head_device = output_embeddings.weight.device
    head_dtype = output_embeddings.weight.dtype

    if not isinstance(getattr(model, "separate_new_token_embeddings", None), nn.Embedding):
        model.separate_new_token_embeddings = nn.Embedding(
            num_new_tokens,
            hidden_size,
            device=device,
            dtype=dtype,
        )
    if not isinstance(getattr(model, "separate_new_token_lm_head", None), nn.Linear):
        model.separate_new_token_lm_head = nn.Linear(
            hidden_size,
            num_new_tokens,
            bias=False,
            device=head_device,
            dtype=head_dtype,
        )

    if init_from_base:
        _init_extra_weight_from_base(model.separate_new_token_embeddings.weight, input_embeddings.weight)
        _init_extra_weight_from_base(model.separate_new_token_lm_head.weight, output_embeddings.weight)

    input_embeddings.weight.requires_grad_(False)
    output_embeddings.weight.requires_grad_(False)
    model.separate_new_token_embeddings.weight.requires_grad_(is_trainable)
    model.separate_new_token_lm_head.weight.requires_grad_(is_trainable)

    original_vocab_size = min(new_token_ids)
    separate_vocab_size = max(embedding_vocab_size, max_new_token_id + 1)

    setattr(model, "_lf_original_vocab_size", original_vocab_size)
    setattr(model, "_lf_resized_vocab_size", tokenizer_size)
    setattr(model, "_lf_num_new_tokens", num_new_tokens)
    setattr(model, "_lf_use_separate_new_token_embeddings", True)
    setattr(model, "_lf_separate_new_token_base_vocab_size", original_vocab_size)
    setattr(model, "_lf_separate_new_token_embedding_vocab_size", embedding_vocab_size)
    setattr(model, "_lf_separate_new_token_vocab_size", separate_vocab_size)
    setattr(model, "_lf_separate_new_token_ids", tuple(new_token_ids))

    _patch_input_embedding_forward(model, input_embeddings)
    _patch_output_embedding_forward(model, output_embeddings)

    logger.info_rank0(
        "Installed separate new-token embeddings/head: "
        f"embedding_vocab={embedding_vocab_size}, new_tokens={num_new_tokens}, "
        f"new_token_id_range=[{min(new_token_ids)}, {max_new_token_id}], tokenizer_vocab={tokenizer_size}."
    )
    return True


def load_separate_new_token_weights(
    model: nn.Module,
    checkpoint_dir: str,
    strict: bool = False,
) -> bool:
    """Load separate mesh embeddings and head from the complete model checkpoint."""
    state_dict = _load_separate_new_token_state_from_safetensors(model, checkpoint_dir)
    if not state_dict:
        return False
    model.load_state_dict(state_dict, strict=strict)
    logger.info_rank0(f"Loaded separate new-token weights from model safetensors: {checkpoint_dir}")
    return True


def _patch_input_embedding_forward(model: nn.Module, input_embeddings: nn.Embedding) -> None:
    if not hasattr(input_embeddings, "_lf_original_forward"):
        input_embeddings._lf_original_forward = input_embeddings.forward

    def routed_forward(self: nn.Embedding, input_ids: torch.Tensor) -> torch.Tensor:
        extra_embeddings = getattr(model, "separate_new_token_embeddings", None)
        if extra_embeddings is None or input_ids is None:
            return self._lf_original_forward(input_ids)

        new_token_ids = tuple(getattr(model, "_lf_separate_new_token_ids"))
        first_new_token_id = int(new_token_ids[0])
        last_new_token_id = int(new_token_ids[-1])
        ids_are_contiguous = len(new_token_ids) == last_new_token_id - first_new_token_id + 1
        if ids_are_contiguous:
            extra_mask = input_ids.ge(first_new_token_id) & input_ids.le(last_new_token_id)
        else:
            token_id_tensor = torch.tensor(new_token_ids, device=input_ids.device, dtype=input_ids.dtype)
            extra_mask = torch.isin(input_ids, token_id_tensor)

        if not bool(extra_mask.any()):
            return self._lf_original_forward(input_ids)

        embedding_vocab_size = int(getattr(model, "_lf_separate_new_token_embedding_vocab_size"))
        if bool((input_ids[~extra_mask] >= embedding_vocab_size).any()):
            bad_id = int(input_ids[~extra_mask][input_ids[~extra_mask] >= embedding_vocab_size][0].item())
            raise IndexError(f"Token id {bad_id} exceeds base embedding vocab size {embedding_vocab_size}.")

        safe_input_ids = input_ids.masked_fill(extra_mask, 0)
        base_outputs = self._lf_original_forward(safe_input_ids)
        outputs = base_outputs.clone()
        if ids_are_contiguous:
            extra_ids = input_ids[extra_mask] - first_new_token_id
        else:
            remap = input_ids.new_full((last_new_token_id + 1,), fill_value=-1)
            token_id_tensor = torch.tensor(new_token_ids, device=input_ids.device, dtype=input_ids.dtype)
            remap[token_id_tensor] = torch.arange(len(new_token_ids), device=input_ids.device, dtype=input_ids.dtype)
            extra_ids = remap[input_ids[extra_mask]]

        extra_outputs = extra_embeddings(extra_ids.to(device=extra_embeddings.weight.device))
        outputs[extra_mask] = extra_outputs.to(device=outputs.device, dtype=outputs.dtype)
        return outputs

    input_embeddings.forward = MethodType(routed_forward, input_embeddings)


def _patch_output_embedding_forward(model: nn.Module, output_embeddings: nn.Linear) -> None:
    if not hasattr(output_embeddings, "_lf_original_forward"):
        output_embeddings._lf_original_forward = output_embeddings.forward

    def routed_forward(self: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
        base_logits = self._lf_original_forward(hidden_states)
        extra_lm_head = getattr(model, "separate_new_token_lm_head", None)
        if extra_lm_head is None:
            return base_logits

        extra_logits = extra_lm_head(hidden_states.to(dtype=extra_lm_head.weight.dtype))
        extra_logits = extra_logits.to(dtype=base_logits.dtype)
        new_token_ids = tuple(getattr(model, "_lf_separate_new_token_ids"))
        max_new_token_id = int(new_token_ids[-1])
        if max_new_token_id < base_logits.shape[-1]:
            logits = base_logits.clone()
        else:
            logits_shape = (*base_logits.shape[:-1], max_new_token_id + 1)
            logits = base_logits.new_full(logits_shape, torch.finfo(base_logits.dtype).min)
            logits[..., : base_logits.shape[-1]] = base_logits

        token_id_tensor = torch.tensor(new_token_ids, device=logits.device, dtype=torch.long)
        logits.index_copy_(dim=-1, index=token_id_tensor, source=extra_logits)
        return logits

    output_embeddings.forward = MethodType(routed_forward, output_embeddings)


def _get_separate_new_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    embedding_vocab_size: int,
    tokenizer_size: int,
) -> list[int]:
    explicit_token_ids = getattr(tokenizer, "_lf_separate_new_token_ids", None)
    if explicit_token_ids:
        return sorted({int(token_id) for token_id in explicit_token_ids})

    if tokenizer_size > embedding_vocab_size:
        return list(range(embedding_vocab_size, tokenizer_size))

    return []


def _init_extra_weight_from_base(extra_weight: torch.Tensor, base_weight: torch.Tensor) -> None:
    target_is_zero3 = _is_zero3_partitioned_param(extra_weight)
    modifier_rank = 0 if target_is_zero3 else None
    with _maybe_gather_zero3_params([base_weight, extra_weight], modifier_rank=modifier_rank), torch.no_grad():
        if target_is_zero3 and _distributed_rank() != 0:
            return
        if base_weight.numel() == 0 or extra_weight.numel() == 0:
            return

        embedding_dim = base_weight.shape[1]
        avg_weight = (
            base_weight.detach()
            .float()
            .mean(dim=0, keepdim=True)
            .to(
                device=extra_weight.device,
                dtype=extra_weight.dtype,
            )
        )
        noise = torch.empty_like(extra_weight)
        noise.normal_(mean=0.0, std=1.0 / (embedding_dim**0.5))
        extra_weight.copy_(avg_weight + noise)


def _load_separate_new_token_state_from_safetensors(
    model: nn.Module,
    checkpoint_dir: str,
) -> dict[str, torch.Tensor]:
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return {}

    try:
        from safetensors import safe_open
    except ImportError:
        return {}

    target_keys = [
        key
        for key in model.state_dict().keys()
        if key in {"separate_new_token_embeddings.weight", "separate_new_token_lm_head.weight"}
    ]
    if not target_keys:
        return {}

    weight_map = load_safetensors_weight_map(checkpoint_dir)
    if not weight_map:
        return {}

    loaded: dict[str, torch.Tensor] = {}
    for target_key in target_keys:
        source_key = _match_separate_new_token_key(target_key, weight_map.keys())
        if source_key is None:
            continue

        shard_path = os.path.join(checkpoint_dir, weight_map[source_key])
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                loaded[target_key] = shard.get_tensor(source_key)
        except Exception as exc:
            logger.warning_rank0(f"Failed to read `{source_key}` from `{shard_path}`: {exc}")

    return loaded


def _match_separate_new_token_key(target_key: str, source_keys) -> str | None:
    candidates = [
        target_key,
        f"base_model.model.{target_key}",
        f"model.{target_key}",
    ]
    source_key_set = set(source_keys)
    for candidate in candidates:
        if candidate in source_key_set:
            return candidate

    matches = [source_key for source_key in source_key_set if source_key.endswith(target_key)]
    if len(matches) == 1:
        return matches[0]

    return None


def _is_zero3_partitioned_param(param: torch.nn.Parameter | torch.Tensor) -> bool:
    return hasattr(param, "ds_id") or hasattr(param, "ds_status")


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


@contextmanager
def _maybe_gather_zero3_params(
    params: Sequence[torch.nn.Parameter | torch.Tensor],
    *,
    modifier_rank: int | None = None,
):
    zero3_params = [param for param in params if _is_zero3_partitioned_param(param)]
    if not zero3_params or not is_deepspeed_zero3_enabled():
        yield
        return

    try:
        import deepspeed
    except ImportError:
        yield
        return

    with deepspeed.zero.GatheredParameters(zero3_params, modifier_rank=modifier_rank):
        yield
