from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch


pytest.importorskip("ocnn")

from llamafactory.model.bytes_to_split import binary_to_split_tensor, bytes_to_binary_sequence
from llamafactory.model.incremental_octree_state import IncrementalOctreeState
from llamafactory.model.models.positional_embedding import DepthPosEmb
from llamafactory.model.split_to_position_embedding import (
    create_template_octree,
    determine_current_layer_status,
    split_to_next_position_embedding,
)


qwen25_vl_3d_replace = pytest.importorskip("llamafactory.model.qwen25_vl_3d_replace")


def _split_tensor(byte_values: Sequence[int]) -> torch.Tensor:
    if not byte_values:
        return torch.empty(0, dtype=torch.float32)
    return binary_to_split_tensor(bytes_to_binary_sequence(byte_values), dtype=torch.float32)


def _legacy_position_result(
    byte_values: Sequence[int],
    *,
    max_depth: int,
    full_depth: int,
    pos_emb_model: DepthPosEmb,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    result = split_to_next_position_embedding(
        _split_tensor(byte_values),
        depth=max_depth,
        full_depth=full_depth,
        threshold=0.0,
        device="cpu",
        pos_emb_model=pos_emb_model,
    )
    assert result is not None
    return result


def _assert_position_parity(
    state: IncrementalOctreeState,
    byte_values: Sequence[int],
    pos_emb_model: DepthPosEmb,
) -> None:
    legacy = _legacy_position_result(
        byte_values,
        max_depth=state.max_depth,
        full_depth=state.full_depth,
        pos_emb_model=pos_emb_model,
    )
    incremental = qwen25_vl_3d_replace._resolve_incremental_octree_positions(
        state.position_metadata,
        pos_emb_model,
        "cpu",
    )
    for legacy_value, incremental_value in zip(legacy, incremental):
        assert (legacy_value is None) == (incremental_value is None)
        if legacy_value is not None:
            assert torch.equal(legacy_value, incremental_value)


def _assert_status_parity(
    state: IncrementalOctreeState,
    byte_values: Sequence[int],
    incremental_status,
) -> None:
    template = create_template_octree(state.max_depth, state.full_depth, "cpu")
    legacy_status = determine_current_layer_status(
        _split_tensor(byte_values),
        state.max_depth,
        state.full_depth,
        template,
        0.0,
    )
    assert (
        incremental_status.layer,
        incremental_status.complete,
        incremental_status.remaining,
    ) == legacy_status


@pytest.mark.parametrize(
    ("max_depth", "full_depth", "byte_stream"),
    [
        # Every layer is empty: OCNN forces the first node to split.
        (4, 2, [0] * 10),
        # Sparse parents at opposite ends exercise ordering across boundaries.
        (4, 2, [0x80, 0, 0, 0, 0, 0, 0, 0x01, 0x80, 0x01, 0xA5, 0x5A]),
        # A dense first layer followed by deterministic mixed bytes.
        (3, 1, [0xFF] + [0x80] * 8 + [0x00, 0x11, 0x22, 0x44, 0x88, 0xAA, 0x55, 0xFF]),
    ],
)
def test_incremental_positions_and_layer_status_are_exactly_equal_to_legacy(
    max_depth: int,
    full_depth: int,
    byte_stream: list[int],
):
    pos_emb_model = DepthPosEmb(num_embed=17, full_depth=full_depth, max_depth=max_depth)
    generator = torch.Generator().manual_seed(20260830 + max_depth * 10 + full_depth)
    with torch.no_grad():
        pos_emb_model.depth_emb.weight.copy_(torch.randn(pos_emb_model.depth_emb.weight.shape, generator=generator))

    state = IncrementalOctreeState(max_depth=max_depth, full_depth=full_depth)
    consumed: list[int] = []
    for byte_value in byte_stream:
        _assert_position_parity(state, consumed, pos_emb_model)
        consumed.append(byte_value)
        status = state.consume_byte(byte_value)
        _assert_status_parity(state, consumed, status)

    assert state.complete


def test_default_depth_boundary_position_matches_legacy():
    """Exercise the benchmark's full_depth=3/max_depth=6 geometry."""
    state = IncrementalOctreeState(max_depth=6, full_depth=3)
    pos_emb_model = DepthPosEmb(num_embed=5, full_depth=3, max_depth=6)
    _assert_position_parity(state, [], pos_emb_model)

    consumed = [0] * 63 + [1]
    for byte_value in consumed:
        status = state.consume_byte(byte_value)

    assert (status.layer, status.complete, status.remaining) == (4, False, 8)
    _assert_position_parity(state, consumed, pos_emb_model)
    _assert_status_parity(state, consumed, status)
