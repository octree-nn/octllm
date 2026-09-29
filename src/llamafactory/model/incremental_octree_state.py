from __future__ import annotations

from dataclasses import dataclass


Coordinate = tuple[int, int, int]
NodeGroup = tuple[Coordinate, Coordinate, Coordinate, Coordinate, Coordinate, Coordinate, Coordinate, Coordinate]


@dataclass(frozen=True)
class OctreeTokenPosition:
    """Position information consumed by one generated mesh-byte token."""

    depth_index: int
    xyz: Coordinate


@dataclass(frozen=True)
class OctreeDecodeMetadata:
    """Positions for the previous mesh token and its following MASK token."""

    previous_position: OctreeTokenPosition | None
    next_position: OctreeTokenPosition | None


@dataclass(frozen=True)
class OctreeLayerStatus:
    """Incremental equivalent of ``determine_current_layer_status``."""

    layer: int
    complete: bool
    remaining: int


def _decode_morton_key(key: int, depth: int) -> Coordinate:
    """Decode OCNN's shuffled key for one batch-free node."""
    x = 0
    y = 0
    z = 0
    for bit_index in range(depth):
        key_offset = bit_index * 3
        x |= ((key >> (key_offset + 2)) & 1) << bit_index
        y |= ((key >> (key_offset + 1)) & 1) << bit_index
        z |= ((key >> key_offset) & 1) << bit_index
    return x, y, z


def _children(parent: Coordinate) -> NodeGroup:
    x, y, z = parent
    return tuple(
        (
            x * 2 + ((child_index >> 2) & 1),
            y * 2 + ((child_index >> 1) & 1),
            z * 2 + (child_index & 1),
        )
        for child_index in range(8)
    )  # type: ignore[return-value]


def _round_half_to_even_after_divide_by_two(value: int) -> int:
    """Match ``torch.round(float(value) / 2)`` exactly for small integers."""
    quotient, remainder = divmod(value, 2)
    if remainder == 0 or quotient % 2 == 0:
        return quotient
    return quotient + 1


class IncrementalOctreeState:
    """Track octree topology and next-token positions without rebuilding a tree.

    Mesh byte ``b`` represents the split labels for one consecutive group of
    eight OCNN nodes, most-significant bit first. The state retains only the
    current layer and the groups selected for the following layer.
    """

    def __init__(self, max_depth: int, full_depth: int) -> None:
        if full_depth < 1:
            raise ValueError("full_depth must be at least 1 because one mesh token encodes eight nodes")
        if max_depth < full_depth:
            raise ValueError("max_depth must be greater than or equal to full_depth")

        self.max_depth = int(max_depth)
        self.full_depth = int(full_depth)
        self._current_depth = self.full_depth
        self._current_parent_keys = self._build_full_depth_parent_keys()
        self._next_parent_keys: list[int] = []
        self._group_index = 0
        self._split_length = 0
        self._complete = False
        self._previous_position: OctreeTokenPosition | None = None
        self._next_position_cache: OctreeTokenPosition | None = None

    @property
    def split_length(self) -> int:
        return self._split_length

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def position_metadata(self) -> OctreeDecodeMetadata:
        next_position = None
        if not self._complete:
            if self._next_position_cache is None:
                self._next_position_cache = self._position_for_parent_key(
                    self._current_parent_keys[self._group_index],
                    self._current_depth,
                )
            next_position = self._next_position_cache
        return OctreeDecodeMetadata(
            previous_position=self._previous_position,
            next_position=next_position,
        )

    def consume_byte(self, byte_value: int) -> OctreeLayerStatus:
        """Consume one mesh byte and return the same layer status as the old rebuild path."""
        if self._complete:
            raise RuntimeError("cannot append a mesh byte after the octree is complete")
        if not isinstance(byte_value, int) or isinstance(byte_value, bool) or not 0 <= byte_value <= 255:
            raise ValueError("byte_value must be an integer in [0, 255]")

        consumed_parent_key = self._current_parent_keys[self._group_index]
        consumed_depth = self._current_depth
        self._previous_position = self._next_position_cache or self._position_for_parent_key(
            consumed_parent_key,
            consumed_depth,
        )
        self._next_position_cache = None

        if consumed_depth < self.max_depth:
            child_key_prefix = consumed_parent_key << 3
            for node_index in range(8):
                if byte_value & (1 << (7 - node_index)):
                    self._next_parent_keys.append(child_key_prefix + node_index)

        self._group_index += 1
        self._split_length += 8

        if self._group_index < len(self._current_parent_keys):
            return OctreeLayerStatus(
                layer=self._current_depth,
                complete=False,
                remaining=(len(self._current_parent_keys) - self._group_index) * 8,
            )

        if self._current_depth == self.max_depth:
            self._complete = True
            return OctreeLayerStatus(layer=self._current_depth, complete=True, remaining=0)

        # OCNN's octree_split keeps the first node when an entire layer is
        # empty, so the following depth always contains at least one group.
        if not self._next_parent_keys:
            self._next_parent_keys.append(self._current_parent_keys[0] << 3)

        self._current_depth += 1
        self._current_parent_keys = self._next_parent_keys
        self._next_parent_keys = []
        self._group_index = 0
        return OctreeLayerStatus(
            layer=self._current_depth,
            complete=False,
            remaining=len(self._current_parent_keys) * 8,
        )

    def _build_full_depth_parent_keys(self) -> list[int]:
        parent_depth = self.full_depth - 1
        parent_count = 1 << (3 * parent_depth)
        return list(range(parent_count))

    def _position_for_parent_key(self, parent_key: int, depth: int) -> OctreeTokenPosition:
        # OctreeT.build_xyz rescales each node to a 2 ** (max_depth + 1)
        # coordinate grid, averages each eight-node token, divides by two, and
        # applies round-to-nearest-even. Every group here consists of siblings,
        # so the pre-division average is integral and can be reproduced exactly
        # without allocating tensors.
        parent = _decode_morton_key(parent_key, depth - 1)
        group = _children(parent)
        cell_size = 1 << (self.max_depth + 1 - depth)
        xyz = []
        for axis in range(3):
            scaled_sum = sum(node[axis] * cell_size + cell_size // 2 for node in group)
            scaled_mean = scaled_sum // 8
            xyz.append(_round_half_to_even_after_divide_by_two(scaled_mean))

        return OctreeTokenPosition(
            depth_index=depth - self.full_depth,
            xyz=(xyz[0], xyz[1], xyz[2]),
        )
