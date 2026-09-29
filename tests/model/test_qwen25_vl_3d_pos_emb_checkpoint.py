import pytest
import torch


qwen3d = pytest.importorskip("llamafactory.model.qwen25_vl_3d_replace")


def _reset_pos_emb_state():
    qwen3d.clear_pos_emb_model_cache()
    qwen3d._pos_emb_loaded_state_dicts.clear()


def test_load_depth_pos_emb_from_safetensors(model_checkpoint):
    _reset_pos_emb_state()
    expected = torch.full((4, 6), 0.75)
    checkpoint = model_checkpoint({"mesh_abs_pos_emb.depth_emb.weight": expected})

    assert qwen3d.load_pos_emb_weights(checkpoint, strict=False)
    model = qwen3d.get_or_create_pos_emb_model(6, "cpu", full_depth=3, max_depth=6)

    assert torch.allclose(model.depth_emb.weight, expected)
    _reset_pos_emb_state()


def test_attach_depth_pos_emb_registers_in_model_state_dict():
    _reset_pos_emb_state()
    model = torch.nn.Module()

    assert qwen3d.attach_pos_emb_to_model(model, 6, "cpu", full_depth=3, max_depth=6)
    assert "mesh_abs_pos_emb.depth_emb.weight" in model.state_dict()
    _reset_pos_emb_state()
