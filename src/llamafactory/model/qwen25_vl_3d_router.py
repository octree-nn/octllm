from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Sequence
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLSdpaAttention,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


from .model_utils.checkpoint import load_safetensors_weight_map


ROUTER_CONFIG_KEY = "llamafactory_3d_router_config"
ATTN_PROJ_MODES = {"none", "o", "qkv", "qkvo"}
LAYER_SCOPES = {"all", "last_n", "explicit"}

_ORIGINAL_DECODER_LAYER_FORWARD = Qwen2_5_VLDecoderLayer.forward
_ORIGINAL_SDPA_ATTENTION_FORWARD = Qwen2_5_VLSdpaAttention.forward
_PATCHED = False


class Qwen2MeshMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        act_fn,
        *,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.act_fn = act_fn

    @classmethod
    def from_base(cls, base_mlp: nn.Module, mlp_ratio: float = 1.0, init_from_base: bool = True) -> Qwen2MeshMLP:
        hidden_size = int(base_mlp.gate_proj.in_features)
        base_intermediate_size = int(base_mlp.gate_proj.out_features)
        intermediate_size = max(1, int(round(base_intermediate_size * mlp_ratio)))
        bias = base_mlp.gate_proj.bias is not None
        device = base_mlp.gate_proj.weight.device
        dtype = base_mlp.gate_proj.weight.dtype
        mesh_mlp = cls(
            hidden_size,
            intermediate_size,
            base_mlp.act_fn,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        if init_from_base:
            _copy_mlp_weights(mesh_mlp, base_mlp)

        return mesh_mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def enable_qwen25_vl_3d_router_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    Qwen2_5_VLDecoderLayer.forward = _routed_decoder_layer_forward  # type: ignore[assignment]
    Qwen2_5_VLSdpaAttention.forward = _routed_sdpa_attention_forward  # type: ignore[assignment]
    _PATCHED = True


def install_qwen25_vl_3d_router(
    model: nn.Module,
    *,
    replace_ffn: bool = True,
    replace_attn_proj: bool = False,
    attn_proj_mode: str = "none",
    layer_scope: str = "last_n",
    last_n_layers: int = 0,
    layer_ids: Sequence[int] | str | None = None,
    mlp_ratio: float = 1.0,
    init_from_base: bool = True,
    freeze_base: bool = True,
) -> list[int]:
    if not replace_ffn and not replace_attn_proj:
        raise ValueError("3D token router needs at least one branch: replace_ffn or replace_attn_proj.")

    if attn_proj_mode not in ATTN_PROJ_MODES:
        raise ValueError(f"route_3d_attn_proj_mode must be one of {sorted(ATTN_PROJ_MODES)}, got {attn_proj_mode}.")
    if layer_scope not in LAYER_SCOPES:
        raise ValueError(f"route_3d_layer_scope must be one of {sorted(LAYER_SCOPES)}, got {layer_scope}.")
    if replace_attn_proj and attn_proj_mode == "none":
        raise ValueError("route_3d_attn_proj_mode must be o/qkv/qkvo when replace_attn_proj is enabled.")
    if not replace_attn_proj and attn_proj_mode != "none":
        raise ValueError("route_3d_attn_proj_mode must be none when replace_attn_proj is disabled.")
    if mlp_ratio <= 0:
        raise ValueError(f"route_3d_mlp_ratio must be > 0, got {mlp_ratio}.")

    enable_qwen25_vl_3d_router_patch()
    language_model = get_qwen25_vl_language_model(model)
    _validate_sdpa_language_model(language_model)

    layers = _get_decoder_layers(language_model)
    selected_layer_ids = resolve_route_layer_ids(
        len(layers),
        layer_scope=layer_scope,
        last_n_layers=last_n_layers,
        layer_ids=layer_ids,
    )
    selected = set(selected_layer_ids)

    for idx, layer in enumerate(layers):
        layer._3d_router_enabled = idx in selected
        layer._3d_router_replace_ffn = bool(replace_ffn and idx in selected)
        layer._3d_route_mask = None
        if hasattr(layer, "self_attn"):
            layer.self_attn._3d_router_enabled = idx in selected
            layer.self_attn._3d_router_replace_attn_proj = bool(replace_attn_proj and idx in selected)
            layer.self_attn._3d_attn_proj_mode = attn_proj_mode if idx in selected else "none"
            layer.self_attn._3d_route_mask = None

    for idx in selected_layer_ids:
        layer = layers[idx]
        if freeze_base:
            _freeze_base_layer_parameters(layer)

        if replace_ffn:
            if not isinstance(getattr(layer, "mesh_mlp", None), nn.Module):
                layer.mesh_mlp = Qwen2MeshMLP.from_base(
                    layer.mlp,
                    mlp_ratio=mlp_ratio,
                    init_from_base=init_from_base,
                )
            _set_module_trainable(layer.mesh_mlp, True)

        if replace_attn_proj:
            _install_attention_projection_branches(layer.self_attn, attn_proj_mode, init_from_base=init_from_base)

    router_config = {
        "replace_ffn": bool(replace_ffn),
        "replace_attn_proj": bool(replace_attn_proj),
        "attn_proj_mode": attn_proj_mode,
        "layer_scope": layer_scope,
        "last_n_layers": int(last_n_layers),
        "layer_ids": selected_layer_ids,
        "mlp_ratio": float(mlp_ratio),
        "init_from_base": bool(init_from_base),
        "freeze_base": bool(freeze_base),
    }
    setattr(language_model, "_3d_router_layer_ids", selected_layer_ids)
    setattr(language_model, "_3d_router_config", router_config)
    _store_3d_router_config(model, language_model, router_config)

    return selected_layer_ids


def set_3d_route_mask(language_model: nn.Module, mask: torch.Tensor | None) -> None:
    route_mask = mask.detach() if isinstance(mask, torch.Tensor) else None
    setattr(language_model, "_3d_route_mask", route_mask)

    layers = getattr(language_model, "layers", None)
    if layers is None:
        return

    for layer in layers:
        if not getattr(layer, "_3d_router_enabled", False):
            continue
        layer._3d_route_mask = route_mask
        if hasattr(layer, "self_attn"):
            layer.self_attn._3d_route_mask = route_mask


def clear_3d_route_mask(language_model: nn.Module) -> None:
    set_3d_route_mask(language_model, None)


def load_3d_router_weights(
    model: nn.Module,
    checkpoint_dir: str,
    strict: bool = False,
) -> bool:
    """Load installed 3D branches from the complete model checkpoint."""
    state_dict = _load_3d_router_state_from_safetensors(model, checkpoint_dir)
    if not state_dict:
        return False
    model.load_state_dict(state_dict, strict=strict)
    return True


def _is_3d_router_state_key(name: str) -> bool:
    router_module_names = {"mesh_mlp", "mesh_q_proj", "mesh_k_proj", "mesh_v_proj", "mesh_o_proj"}
    return any(part in router_module_names for part in name.split("."))


def _load_3d_router_state_from_safetensors(model: nn.Module, checkpoint_dir: str) -> dict[str, torch.Tensor]:
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return {}

    try:
        from safetensors import safe_open
    except ImportError:
        print("safetensors is not available; cannot recover 3D router weights from model shards.")
        return {}

    router_target_keys = [name for name in model.state_dict().keys() if _is_3d_router_state_key(name)]
    if not router_target_keys:
        return {}

    weight_map = load_safetensors_weight_map(checkpoint_dir)
    if not weight_map:
        return {}

    source_keys = [name for name in weight_map.keys() if _is_3d_router_state_key(name)]
    if not source_keys:
        return {}

    loaded: dict[str, torch.Tensor] = {}
    for target_key in router_target_keys:
        source_key = _match_safetensors_router_key(target_key, source_keys)
        if source_key is None:
            continue

        shard_path = os.path.join(checkpoint_dir, weight_map[source_key])
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                loaded[target_key] = shard.get_tensor(source_key)
        except Exception as exc:
            print(f"failed to read 3D router tensor `{source_key}` from `{shard_path}`: {exc}")

    missing = sorted(set(router_target_keys) - set(loaded.keys()))
    if missing:
        print(f"missing 3D router tensors in model safetensors: {missing[:10]}")

    return loaded


def _match_safetensors_router_key(target_key: str, source_keys: Sequence[str]) -> str | None:
    candidates = [
        target_key,
        target_key.replace(".language_model.", "."),
        target_key.replace("base_model.model.", ""),
        target_key.replace("base_model.model.", "").replace(".language_model.", "."),
    ]
    for candidate in candidates:
        if candidate in source_keys:
            return candidate

    target_suffix = _router_key_suffix(target_key)
    matches = [source_key for source_key in source_keys if _router_key_suffix(source_key) == target_suffix]
    if len(matches) == 1:
        return matches[0]

    return None


def _router_key_suffix(key: str) -> str:
    for marker in ("layers.", "mesh_"):
        idx = key.find(marker)
        if idx >= 0:
            return key[idx:]

    return key


def get_qwen25_vl_3d_router_config(model: nn.Module) -> dict:
    try:
        language_model = get_qwen25_vl_language_model(model)
    except ValueError:
        language_model = None

    if language_model is not None:
        config = getattr(language_model, "_3d_router_config", None)
        if isinstance(config, dict):
            return copy.deepcopy(config)

    for config_obj in _iter_config_candidates(model, language_model):
        config = getattr(config_obj, ROUTER_CONFIG_KEY, None)
        if isinstance(config, dict):
            return copy.deepcopy(config)

    return {}


def _store_3d_router_config(model: nn.Module, language_model: nn.Module, router_config: dict) -> None:
    for config_obj in _iter_config_candidates(model, language_model):
        setattr(config_obj, ROUTER_CONFIG_KEY, copy.deepcopy(router_config))


def _iter_config_candidates(model: nn.Module, language_model: nn.Module | None = None) -> list[object]:
    candidates: list[object] = []
    module_candidates: list[object] = [model]
    if language_model is not None:
        module_candidates.append(language_model)

    for attr in ("module", "base_model", "model"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            module_candidates.append(module)

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base_model = get_base_model()
            if isinstance(base_model, nn.Module):
                module_candidates.append(base_model)
        except Exception:
            pass

    for module in module_candidates:
        config = getattr(module, "config", None)
        if config is not None and config not in candidates:
            candidates.append(config)
        text_config = getattr(config, "text_config", None)
        if text_config is not None and text_config not in candidates:
            candidates.append(text_config)

    return candidates


def get_qwen25_vl_language_model(model: nn.Module) -> nn.Module:
    candidates: list[nn.Module] = [model]
    for attr in ("module", "base_model", "model"):
        obj = getattr(model, attr, None)
        if isinstance(obj, nn.Module) and obj not in candidates:
            candidates.append(obj)

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base_model = get_base_model()
            if isinstance(base_model, nn.Module) and base_model not in candidates:
                candidates.append(base_model)
        except Exception:
            pass

    for candidate in candidates:
        for path in ("model.language_model", "language_model"):
            resolved = _resolve_attr_path(candidate, path)
            if isinstance(resolved, nn.Module) and hasattr(resolved, "layers"):
                return resolved

    raise ValueError("Cannot find Qwen2.5-VL language_model on the provided model.")


def resolve_route_layer_ids(
    num_layers: int,
    *,
    layer_scope: str,
    last_n_layers: int = 0,
    layer_ids: Sequence[int] | str | None = None,
) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")

    if layer_scope == "all":
        selected = list(range(num_layers))
    elif layer_scope == "last_n":
        if last_n_layers <= 0 or last_n_layers >= num_layers:
            selected = list(range(num_layers))
        else:
            selected = list(range(num_layers - last_n_layers, num_layers))
    elif layer_scope == "explicit":
        parsed = _parse_layer_ids(layer_ids)
        if not parsed:
            raise ValueError("route_3d_layer_ids must be provided when route_3d_layer_scope is explicit.")
        selected = parsed
    else:
        raise ValueError(f"Unknown route_3d_layer_scope: {layer_scope}")

    invalid = [idx for idx in selected if idx < 0 or idx >= num_layers]
    if invalid:
        raise ValueError(f"route_3d_layer_ids contain invalid layer ids for {num_layers} layers: {invalid}.")

    return sorted(set(selected))


def _routed_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value=None,
    output_attentions: bool | None = False,
    use_cache: bool | None = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
):
    if not getattr(self, "_3d_router_enabled", False):
        return _ORIGINAL_DECODER_LAYER_FORWARD(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    route_mask = getattr(self, "_3d_route_mask", None)
    if hasattr(self, "self_attn"):
        self.self_attn._3d_route_mask = route_mask

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    if getattr(self, "_3d_router_replace_ffn", False) and isinstance(getattr(self, "mesh_mlp", None), nn.Module):
        prepared_mask = _prepare_route_mask(route_mask, hidden_states)
        hidden_states = _routed_sparse_mlp(hidden_states, self.mlp, self.mesh_mlp, prepared_mask)
    else:
        hidden_states = self.mlp(hidden_states)

    hidden_states = residual + hidden_states
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)

    return outputs


def _routed_sdpa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    if not getattr(self, "_3d_router_replace_attn_proj", False):
        return _ORIGINAL_SDPA_ATTENTION_FORWARD(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    if output_attentions:
        raise ValueError("3D token router only supports Qwen2.5-VL SDPA attention with output_attentions=False.")

    route_mask = _prepare_route_mask(getattr(self, "_3d_route_mask", None), hidden_states)
    if route_mask is None:
        return _ORIGINAL_SDPA_ATTENTION_FORWARD(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()
    mode = getattr(self, "_3d_attn_proj_mode", "none")

    if mode in {"qkv", "qkvo"}:
        query_states = _routed_sparse_linear(hidden_states, self.q_proj, self.mesh_q_proj, route_mask)
        key_states = _routed_sparse_linear(hidden_states, self.k_proj, self.mesh_k_proj, route_mask)
        value_states = _routed_sparse_linear(hidden_states, self.v_proj, self.mesh_v_proj, route_mask)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and q_len > 1 else False
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    if mode in {"o", "qkvo"}:
        attn_output = _routed_sparse_linear(attn_output, self.o_proj, self.mesh_o_proj, route_mask)
    else:
        attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def _copy_mlp_weights(mesh_mlp: Qwen2MeshMLP, base_mlp: nn.Module) -> None:
    rows = min(mesh_mlp.intermediate_size, base_mlp.gate_proj.out_features)
    source_params = [
        base_mlp.gate_proj.weight,
        base_mlp.up_proj.weight,
        base_mlp.down_proj.weight,
    ]
    target_params = [
        mesh_mlp.gate_proj.weight,
        mesh_mlp.up_proj.weight,
        mesh_mlp.down_proj.weight,
    ]
    if mesh_mlp.gate_proj.bias is not None and base_mlp.gate_proj.bias is not None:
        source_params.extend([base_mlp.gate_proj.bias, base_mlp.up_proj.bias])
        target_params.extend([mesh_mlp.gate_proj.bias, mesh_mlp.up_proj.bias])
    if mesh_mlp.down_proj.bias is not None and base_mlp.down_proj.bias is not None:
        source_params.append(base_mlp.down_proj.bias)
        target_params.append(mesh_mlp.down_proj.bias)

    target_is_zero3 = any(_is_zero3_partitioned_param(param) for param in target_params)
    modifier_rank = 0 if target_is_zero3 else None
    with _maybe_gather_zero3_params(source_params + target_params, modifier_rank=modifier_rank), torch.no_grad():
        if target_is_zero3 and _distributed_rank() != 0:
            return

        mesh_mlp.gate_proj.weight[:rows].copy_(base_mlp.gate_proj.weight[:rows])
        mesh_mlp.up_proj.weight[:rows].copy_(base_mlp.up_proj.weight[:rows])
        mesh_mlp.down_proj.weight[:, :rows].copy_(base_mlp.down_proj.weight[:, :rows])
        if mesh_mlp.gate_proj.bias is not None and base_mlp.gate_proj.bias is not None:
            mesh_mlp.gate_proj.bias[:rows].copy_(base_mlp.gate_proj.bias[:rows])
            mesh_mlp.up_proj.bias[:rows].copy_(base_mlp.up_proj.bias[:rows])
        if mesh_mlp.down_proj.bias is not None and base_mlp.down_proj.bias is not None:
            mesh_mlp.down_proj.bias.copy_(base_mlp.down_proj.bias)


def _install_attention_projection_branches(attn: nn.Module, mode: str, *, init_from_base: bool) -> None:
    if mode in {"qkv", "qkvo"}:
        if not isinstance(getattr(attn, "mesh_q_proj", None), nn.Module):
            attn.mesh_q_proj = _clone_linear(attn.q_proj, init_from_base=init_from_base)
        if not isinstance(getattr(attn, "mesh_k_proj", None), nn.Module):
            attn.mesh_k_proj = _clone_linear(attn.k_proj, init_from_base=init_from_base)
        if not isinstance(getattr(attn, "mesh_v_proj", None), nn.Module):
            attn.mesh_v_proj = _clone_linear(attn.v_proj, init_from_base=init_from_base)
        _set_module_trainable(attn.mesh_q_proj, True)
        _set_module_trainable(attn.mesh_k_proj, True)
        _set_module_trainable(attn.mesh_v_proj, True)

    if mode in {"o", "qkvo"}:
        if not isinstance(getattr(attn, "mesh_o_proj", None), nn.Module):
            attn.mesh_o_proj = _clone_linear(attn.o_proj, init_from_base=init_from_base)
        _set_module_trainable(attn.mesh_o_proj, True)


def _clone_linear(base: nn.Linear, *, init_from_base: bool) -> nn.Linear:
    cloned = nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )
    if init_from_base:
        _copy_linear_weights(cloned, base)

    return cloned


def _copy_linear_weights(target: nn.Linear, source: nn.Linear) -> None:
    source_params = [source.weight]
    target_params = [target.weight]
    if target.bias is not None and source.bias is not None:
        source_params.append(source.bias)
        target_params.append(target.bias)

    target_is_zero3 = any(_is_zero3_partitioned_param(param) for param in target_params)
    modifier_rank = 0 if target_is_zero3 else None
    with _maybe_gather_zero3_params(source_params + target_params, modifier_rank=modifier_rank), torch.no_grad():
        if target_is_zero3 and _distributed_rank() != 0:
            return

        target.weight.copy_(source.weight)
        if target.bias is not None and source.bias is not None:
            target.bias.copy_(source.bias)


def _is_zero3_partitioned_param(param: torch.nn.Parameter) -> bool:
    return hasattr(param, "ds_id") or hasattr(param, "ds_status")


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


@contextmanager
def _maybe_gather_zero3_params(
    params: Sequence[torch.nn.Parameter],
    *,
    modifier_rank: int | None = None,
):
    zero3_params = [param for param in params if _is_zero3_partitioned_param(param)]
    if not zero3_params:
        yield
        return

    try:
        import deepspeed
    except ImportError:
        yield
        return

    with deepspeed.zero.GatheredParameters(zero3_params, modifier_rank=modifier_rank):
        yield


def _freeze_base_layer_parameters(layer: nn.Module) -> None:
    for name, param in layer.named_parameters():
        if name.startswith("mesh_") or ".mesh_" in name:
            continue
        param.requires_grad_(False)


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters(recurse=True):
        param.requires_grad_(trainable)


def _routed_sparse_mlp(
    hidden_states: torch.Tensor,
    base_mlp: nn.Module,
    mesh_mlp: nn.Module,
    route_mask: torch.Tensor | None,
) -> torch.Tensor:
    if route_mask is None:
        return base_mlp(hidden_states) + _zero_use_module(mesh_mlp, hidden_states)

    route_mask = route_mask.bool()
    route_flat = route_mask.reshape(-1)
    route_count = int(route_flat.sum().item())
    total_count = route_flat.numel()

    if route_count == 0:
        return base_mlp(hidden_states) + _zero_use_module(mesh_mlp, hidden_states)

    if route_count == total_count:
        return mesh_mlp(hidden_states) + _zero_use_module(base_mlp, hidden_states)

    hidden_flat = hidden_states.reshape(total_count, hidden_states.shape[-1])
    output_flat = torch.empty_like(hidden_flat)
    base_mask = ~route_flat
    output_flat[base_mask] = base_mlp(hidden_flat[base_mask])
    output_flat[route_flat] = mesh_mlp(hidden_flat[route_flat])
    return output_flat.reshape_as(hidden_states)


def _routed_sparse_linear(
    hidden_states: torch.Tensor,
    base_proj: nn.Module,
    mesh_proj: nn.Module,
    route_mask: torch.Tensor | None,
) -> torch.Tensor:
    if route_mask is None:
        return base_proj(hidden_states) + _zero_use_module(mesh_proj, hidden_states)

    route_mask = route_mask.bool()
    route_flat = route_mask.reshape(-1)
    route_count = int(route_flat.sum().item())
    total_count = route_flat.numel()

    if route_count == 0:
        return base_proj(hidden_states) + _zero_use_module(mesh_proj, hidden_states)

    if route_count == total_count:
        return mesh_proj(hidden_states) + _zero_use_module(base_proj, hidden_states)

    hidden_flat = hidden_states.reshape(total_count, hidden_states.shape[-1])
    base_mask = ~route_flat
    base_output = base_proj(hidden_flat[base_mask])
    mesh_output = mesh_proj(hidden_flat[route_flat])
    output_flat = base_output.new_empty((total_count, base_output.shape[-1]))
    output_flat[base_mask] = base_output
    output_flat[route_flat] = mesh_output
    return output_flat.reshape(*hidden_states.shape[:-1], output_flat.shape[-1])


def _zero_use_module(module: nn.Module, reference: torch.Tensor) -> torch.Tensor:
    zero = reference.new_zeros(())
    if not module.training:
        return zero

    for param in module.parameters(recurse=True):
        if param.requires_grad:
            zero = zero + param.reshape(-1)[:1].sum() * 0.0

    return zero


def _prepare_route_mask(mask: torch.Tensor | None, hidden_states: torch.Tensor) -> torch.Tensor | None:
    if mask is None or not isinstance(mask, torch.Tensor):
        return None

    batch_size, seq_len = hidden_states.shape[:2]
    prepared = mask.to(device=hidden_states.device)
    if prepared.dim() == 1:
        if batch_size != 1:
            return None
        prepared = prepared.unsqueeze(0)
    elif prepared.dim() > 2:
        prepared = prepared.view(prepared.shape[0], -1)

    if prepared.shape[0] != batch_size:
        return None
    if prepared.shape[1] != seq_len:
        if prepared.shape[1] > seq_len:
            prepared = prepared[:, -seq_len:]
        else:
            return None

    return prepared.bool()


def _validate_sdpa_language_model(language_model: nn.Module) -> None:
    config = getattr(language_model, "config", None)
    attn_impl = getattr(config, "_attn_implementation", None)
    layers = _get_decoder_layers(language_model)
    has_non_sdpa = any(not isinstance(layer.self_attn, Qwen2_5_VLSdpaAttention) for layer in layers)
    if attn_impl != "sdpa" or has_non_sdpa:
        raise ValueError(
            "3D token router only supports Qwen2.5-VL SDPA attention. "
            "Set `flash_attn: sdpa` in the training config."
        )


def _get_decoder_layers(language_model: nn.Module) -> nn.ModuleList:
    layers = getattr(language_model, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("Cannot find decoder layers on Qwen2.5-VL language_model.")
    return layers


def _resolve_attr_path(obj: object, path: str) -> object | None:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _parse_layer_ids(layer_ids: Sequence[int] | str | None) -> list[int]:
    if layer_ids is None:
        return []
    if isinstance(layer_ids, str):
        return [int(item.strip()) for item in layer_ids.split(",") if item.strip()]
    if isinstance(layer_ids, Iterable):
        return [int(item) for item in layer_ids]
    return []


__all__ = [
    "ROUTER_CONFIG_KEY",
    "Qwen2MeshMLP",
    "clear_3d_route_mask",
    "enable_qwen25_vl_3d_router_patch",
    "get_qwen25_vl_3d_router_config",
    "get_qwen25_vl_language_model",
    "install_qwen25_vl_3d_router",
    "load_3d_router_weights",
    "resolve_route_layer_ids",
    "set_3d_route_mask",
]
