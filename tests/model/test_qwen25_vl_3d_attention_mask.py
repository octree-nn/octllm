import pytest
import torch


qwen25_vl_3d_replace = pytest.importorskip("llamafactory.model.qwen25_vl_3d_replace")


def _is_blocked(mask: torch.Tensor, batch_idx: int, query_idx: int, key_idx: int) -> bool:
    return bool(mask[batch_idx, 0, query_idx, key_idx] == torch.finfo(mask.dtype).min)


def _is_allowed(mask: torch.Tensor, batch_idx: int, query_idx: int, key_idx: int) -> bool:
    return bool(mask[batch_idx, 0, query_idx, key_idx] == 0)


def test_mesh_suppressed_4d_attention_mask_blocks_historical_mask_keys():
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0],
        ]
    )
    inserted_mask_positions = torch.tensor(
        [
            [False, True, False, True, False, False],
            [False, False, True, False, False, False],
        ]
    )

    mask = qwen25_vl_3d_replace._build_mesh_suppressed_4d_attention_mask(
        attention_mask,
        inserted_mask_positions,
        torch.float32,
    )

    assert list(mask.shape) == [2, 1, 6, 6]
    assert _is_allowed(mask, 0, 4, 2)
    assert _is_blocked(mask, 0, 2, 4)  # causal future key
    assert _is_blocked(mask, 0, 4, 5)  # padding key
    assert _is_blocked(mask, 0, 2, 1)  # historical inserted <MASK> key
    assert _is_blocked(mask, 0, 4, 1)
    assert _is_blocked(mask, 0, 4, 3)
    assert _is_allowed(mask, 0, 1, 1)  # current inserted <MASK> keeps self-attention
    assert _is_allowed(mask, 0, 3, 3)
    assert _is_allowed(mask, 1, 2, 2)
    assert _is_blocked(mask, 1, 3, 2)


def test_mesh_suppressed_4d_attention_mask_supports_missing_2d_mask():
    inserted_mask_positions = torch.tensor([[False, True, False]])

    mask = qwen25_vl_3d_replace._build_mesh_suppressed_4d_attention_mask(
        None,
        inserted_mask_positions,
        torch.bfloat16,
    )

    assert mask.dtype == torch.bfloat16
    assert list(mask.shape) == [1, 1, 3, 3]
    assert _is_allowed(mask, 0, 1, 1)
    assert _is_blocked(mask, 0, 2, 1)


def test_mesh_suppressed_4d_attention_mask_rejects_unsupported_shapes():
    inserted_mask_positions = torch.zeros(1, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="same shape"):
        qwen25_vl_3d_replace._build_mesh_suppressed_4d_attention_mask(
            torch.ones(1, 4),
            inserted_mask_positions,
            torch.float32,
        )

    with pytest.raises(ValueError, match="2D attention_mask"):
        qwen25_vl_3d_replace._build_mesh_suppressed_4d_attention_mask(
            torch.zeros(1, 1, 3, 3),
            inserted_mask_positions,
            torch.float32,
        )
