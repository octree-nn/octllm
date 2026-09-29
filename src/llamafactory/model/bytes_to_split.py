#!/usr/bin/env python3
"""Unpack mesh bytes into binary octree split sequences."""

from typing import List

import torch


def bytes_to_binary_sequence(byte_sequence: List[int]) -> List[int]:
    """Unpack byte values in [0, 255] into bits, most significant bit first."""
    binary_sequence = []

    for byte_val in byte_sequence:
        # Serialize the most significant bit first to match the octree tokenizer.
        binary_str = format(byte_val, "08b")

        eight_bits = [int(bit) for bit in binary_str]
        binary_sequence.extend(eight_bits)

    return binary_sequence


def binary_to_split_tensor(
    binary_sequence: List[int], shape: tuple = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert binary split values to a tensor, optionally reshaping it.

    Args:
        binary_sequence: Sequence of binary occupancy values.
        shape: Output shape; None leaves the tensor one-dimensional.
        dtype: Output tensor dtype.

    Returns:
        Tensor containing the split values.

    Raises:
        ValueError: The requested shape does not match the sequence length.
    """
    split_tensor = torch.tensor(binary_sequence, dtype=dtype)

    if shape is not None:
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        if total_elements != len(binary_sequence):
            raise ValueError(f"Shape mismatch: expected {total_elements} elements, got {len(binary_sequence)}")

        split_tensor = split_tensor.reshape(shape)

    return split_tensor
