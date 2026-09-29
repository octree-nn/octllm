from __future__ import annotations

import pytest

from llamafactory.model.incremental_octree_state import IncrementalOctreeState


def test_all_zero_layers_match_ocnn_forced_growth_contract():
    state = IncrementalOctreeState(max_depth=4, full_depth=2)

    statuses = [state.consume_byte(0) for _ in range(8)]
    assert [(status.layer, status.complete, status.remaining) for status in statuses] == [
        (2, False, 56),
        (2, False, 48),
        (2, False, 40),
        (2, False, 32),
        (2, False, 24),
        (2, False, 16),
        (2, False, 8),
        (3, False, 8),
    ]
    assert state.position_metadata.previous_position.depth_index == 0
    assert state.position_metadata.next_position.depth_index == 1

    status = state.consume_byte(0)
    assert (status.layer, status.complete, status.remaining) == (4, False, 8)
    assert state.position_metadata.previous_position.depth_index == 1
    assert state.position_metadata.next_position.depth_index == 2

    status = state.consume_byte(0)
    assert (status.layer, status.complete, status.remaining) == (4, True, 0)
    assert state.complete
    assert state.split_length == 80
    assert state.position_metadata.next_position is None


def test_incremental_state_rejects_invalid_configuration_and_updates():
    with pytest.raises(ValueError, match="full_depth"):
        IncrementalOctreeState(max_depth=4, full_depth=0)
    with pytest.raises(ValueError, match="max_depth"):
        IncrementalOctreeState(max_depth=2, full_depth=3)

    state = IncrementalOctreeState(max_depth=1, full_depth=1)
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        state.consume_byte(-1)
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        state.consume_byte(256)
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        state.consume_byte(True)

    assert state.consume_byte(0).complete
    with pytest.raises(RuntimeError, match="complete"):
        state.consume_byte(0)
