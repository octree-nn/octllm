from __future__ import annotations

import pytest
import torch
from torch import nn


router = pytest.importorskip("llamafactory.model.qwen25_vl_3d_router")
modeling_qwen25_vl = pytest.importorskip("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
configuration_qwen25_vl = pytest.importorskip("transformers.models.qwen2_5_vl.configuration_qwen2_5_vl")


class WrappedQwen25VL(nn.Module):
    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = language_model


class DummyAttention(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        return torch.zeros_like(hidden_states), None, None


class CountingMLP(nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset
        self.weight = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, hidden_states):
        self.calls.append(tuple(hidden_states.shape))
        return hidden_states + self.offset + self.weight * 0.0


def _tiny_text_model(
    attn_implementation: str = "sdpa",
    num_layers: int = 3,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    rope_scaling: dict | None = None,
):
    config = configuration_qwen25_vl.Qwen2_5_VLTextConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        rope_scaling=rope_scaling,
    )
    config._attn_implementation = attn_implementation
    return modeling_qwen25_vl.Qwen2_5_VLTextModel(config)


def test_resolve_route_layer_ids_supports_all_last_n_and_explicit():
    assert router.resolve_route_layer_ids(4, layer_scope="all") == [0, 1, 2, 3]
    assert router.resolve_route_layer_ids(4, layer_scope="last_n", last_n_layers=2) == [2, 3]
    assert router.resolve_route_layer_ids(4, layer_scope="last_n", last_n_layers=0) == [0, 1, 2, 3]
    assert router.resolve_route_layer_ids(4, layer_scope="explicit", layer_ids="3, 1, 1") == [1, 3]

    with pytest.raises(ValueError, match="invalid layer ids"):
        router.resolve_route_layer_ids(4, layer_scope="explicit", layer_ids="4")


def test_install_router_adds_mesh_mlp_only_to_selected_layers():
    model = WrappedQwen25VL(_tiny_text_model(num_layers=3))

    selected = router.install_qwen25_vl_3d_router(
        model,
        replace_ffn=True,
        replace_attn_proj=False,
        attn_proj_mode="none",
        layer_scope="last_n",
        last_n_layers=1,
    )

    layers = model.model.language_model.layers
    assert selected == [2]
    assert not hasattr(layers[0], "mesh_mlp")
    assert hasattr(layers[2], "mesh_mlp")
    assert all(not p.requires_grad for p in layers[2].mlp.parameters())
    assert all(p.requires_grad for p in layers[2].mesh_mlp.parameters())
    assert model.model.language_model.config.llamafactory_3d_router_config["layer_ids"] == [2]
    assert "model.language_model.layers.2.mesh_mlp.gate_proj.weight" in model.state_dict()


def test_install_router_adds_requested_attention_projection_branches():
    model = WrappedQwen25VL(_tiny_text_model(num_layers=2))

    router.install_qwen25_vl_3d_router(
        model,
        replace_ffn=False,
        replace_attn_proj=True,
        attn_proj_mode="o",
        layer_scope="all",
    )

    attn = model.model.language_model.layers[0].self_attn
    assert hasattr(attn, "mesh_o_proj")
    assert not hasattr(attn, "mesh_q_proj")

    model = WrappedQwen25VL(_tiny_text_model(num_layers=2))
    router.install_qwen25_vl_3d_router(
        model,
        replace_ffn=False,
        replace_attn_proj=True,
        attn_proj_mode="qkvo",
        layer_scope="all",
    )

    attn = model.model.language_model.layers[0].self_attn
    assert hasattr(attn, "mesh_q_proj")
    assert hasattr(attn, "mesh_k_proj")
    assert hasattr(attn, "mesh_v_proj")
    assert hasattr(attn, "mesh_o_proj")


def test_install_router_rejects_non_sdpa_attention():
    model = WrappedQwen25VL(_tiny_text_model(attn_implementation="eager", num_layers=1))

    with pytest.raises(ValueError, match="SDPA attention"):
        router.install_qwen25_vl_3d_router(
            model,
            replace_ffn=True,
            replace_attn_proj=False,
            attn_proj_mode="none",
            layer_scope="all",
        )


def test_routed_text_model_forward_supports_qkvo_attention_projection():
    text_model = _tiny_text_model(
        num_layers=1,
        hidden_size=24,
        intermediate_size=48,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 1]},
    )
    model = WrappedQwen25VL(text_model)
    router.install_qwen25_vl_3d_router(
        model,
        replace_ffn=True,
        replace_attn_proj=True,
        attn_proj_mode="qkvo",
        layer_scope="all",
    )
    router.set_3d_route_mask(text_model, torch.tensor([[0.0, 1.0, 1.0, 0.0]]))

    output = text_model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)

    assert tuple(output.last_hidden_state.shape) == (1, 4, 24)


def test_sparse_mlp_route_only_executes_selected_token_paths():
    hidden_states = torch.zeros(1, 4, 3)
    base_mlp = CountingMLP(offset=1.0)
    mesh_mlp = CountingMLP(offset=2.0)
    route_mask = torch.tensor([[False, True, False, True]])

    output = router._routed_sparse_mlp(hidden_states, base_mlp, mesh_mlp, route_mask)

    assert base_mlp.calls == [(2, 3)]
    assert mesh_mlp.calls == [(2, 3)]
    assert torch.equal(output[0, :, 0], torch.tensor([1.0, 2.0, 1.0, 2.0]))


def test_sparse_mlp_skips_mesh_forward_when_no_tokens_are_routed():
    hidden_states = torch.zeros(1, 4, 3)
    base_mlp = CountingMLP(offset=1.0)
    mesh_mlp = CountingMLP(offset=2.0)
    route_mask = torch.zeros(1, 4, dtype=torch.bool)

    output = router._routed_sparse_mlp(hidden_states, base_mlp, mesh_mlp, route_mask)

    assert base_mlp.calls == [(1, 4, 3)]
    assert mesh_mlp.calls == []
    assert torch.equal(output[0, :, 0], torch.ones(4))


def test_sparse_mlp_skips_base_forward_when_all_tokens_are_routed():
    hidden_states = torch.zeros(1, 4, 3)
    base_mlp = CountingMLP(offset=1.0)
    mesh_mlp = CountingMLP(offset=2.0)
    route_mask = torch.ones(1, 4, dtype=torch.bool)

    output = router._routed_sparse_mlp(hidden_states, base_mlp, mesh_mlp, route_mask)

    assert base_mlp.calls == []
    assert mesh_mlp.calls == [(1, 4, 3)]
    assert torch.equal(output[0, :, 0], torch.full((4,), 2.0))


def test_routed_decoder_layer_selects_mesh_mlp_per_token():
    model = WrappedQwen25VL(_tiny_text_model(num_layers=1))
    router.install_qwen25_vl_3d_router(
        model,
        replace_ffn=True,
        replace_attn_proj=False,
        attn_proj_mode="none",
        layer_scope="all",
    )
    layer = model.model.language_model.layers[0]
    layer.self_attn = DummyAttention()

    hidden_states = torch.randn(1, 3, 16)
    route_mask = torch.tensor([[0, 1, 0]], dtype=torch.float32)
    router.set_3d_route_mask(model.model.language_model, route_mask)

    normed = layer.post_attention_layernorm(hidden_states)
    expected = hidden_states + torch.where(
        route_mask.bool().unsqueeze(-1),
        layer.mesh_mlp(normed),
        layer.mlp(normed),
    )

    output = layer(hidden_states, use_cache=False)[0]

    assert torch.allclose(output, expected, atol=1e-5)


def test_save_and_load_router_weights_round_trip(model_checkpoint):
    from types import SimpleNamespace

    from llamafactory.hparams import FinetuningArguments
    from scripts.generate_octree import _install_and_load_qwen25_3d_router

    model = WrappedQwen25VL(_tiny_text_model(num_layers=3))
    router.install_qwen25_vl_3d_router(
        model,
        replace_ffn=True,
        replace_attn_proj=True,
        attn_proj_mode="qkvo",
        layer_scope="explicit",
        layer_ids=[0, 2],
    )
    source_weight = model.model.language_model.layers[0].mesh_mlp.gate_proj.weight
    with torch.no_grad():
        source_weight.fill_(0.25)

    checkpoint = model_checkpoint(model.state_dict())

    restored = WrappedQwen25VL(_tiny_text_model(num_layers=3))
    setattr(
        restored.model.language_model.config,
        router.ROUTER_CONFIG_KEY,
        router.get_qwen25_vl_3d_router_config(model),
    )
    _install_and_load_qwen25_3d_router(
        restored,
        SimpleNamespace(model_name_or_path=checkpoint, adapter_name_or_path=None),
        FinetuningArguments(use_3d_token_router=True, use_3d_position_embedding=True, finetuning_type="freeze"),
    )
    assert not hasattr(restored.model.language_model.layers[1], "mesh_mlp")
    assert hasattr(restored.model.language_model.layers[2].self_attn, "mesh_q_proj")
    restored_weight = restored.model.language_model.layers[0].mesh_mlp.gate_proj.weight
    assert torch.allclose(restored_weight, source_weight)


def test_load_router_weights_from_model_safetensors(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    model = WrappedQwen25VL(_tiny_text_model(num_layers=1))
    router.install_qwen25_vl_3d_router(
        model,
        replace_ffn=True,
        replace_attn_proj=False,
        attn_proj_mode="none",
        layer_scope="all",
        init_from_base=False,
    )
    source_key = "model.language_model.layers.0.mesh_mlp.gate_proj.weight"
    expected = torch.full_like(model.model.language_model.layers[0].mesh_mlp.gate_proj.weight, 0.5)
    safetensors_torch.save_file({source_key: expected}, tmp_path / "model.safetensors")

    assert router.load_3d_router_weights(model, str(tmp_path), strict=False)

    restored_weight = model.model.language_model.layers[0].mesh_mlp.gate_proj.weight
    assert torch.allclose(restored_weight, expected)
