#!/usr/bin/env python3
"""Compute positional embeddings from partial octree split sequences.

Adapted from the OctGPT octree reconstruction method.
"""

from typing import Optional, Tuple

import ocnn
import torch

from .models.octformer import OctreeT
from .utils.utils import seq2octree


def create_template_octree(depth: int, full_depth: int, device: str = "cpu"):
    """Initialize an octree template for reconstruction; return None on failure."""
    try:
        octree = ocnn.octree.init_octree(depth=depth, full_depth=full_depth, batch_size=1, device=device)
        return octree
    except Exception as e:
        print(f"Failed to initialize octree template: {e}")
        return None


def determine_current_layer_status(
    split_sequence: torch.Tensor,
    depth: int,
    full_depth: int,
    octree_template,
    threshold: float = 0.0,
) -> tuple:
    """Find the last reached octree layer and whether its split data is complete.

    Args:
        split_sequence: Binary split values generated so far.
        depth: Maximum octree depth.
        full_depth: Depth through which all octree nodes are allocated.
        octree_template: Initial octree to copy for traversal.
        threshold: Threshold used to binarize split values.

    Returns:
        A tuple of layer index, completion flag, and missing node count.
        Return (full_depth, False, 0) if traversal cannot be initialized.
    """
    import copy

    try:
        temp_octree = copy.deepcopy(octree_template)
        current_pos = 0
        current_layer = full_depth
        layer_complete = False
        remaining_in_layer = 0

        for d in range(full_depth, depth + 1):
            nodes_needed = temp_octree.nnum[d].item()

            if nodes_needed == 0:
                current_layer = d
                layer_complete = True
                continue

            if current_pos + nodes_needed <= len(split_sequence):
                layer_data = split_sequence[current_pos : current_pos + nodes_needed]
                current_pos += nodes_needed
                current_layer = d
                layer_complete = True
                remaining_in_layer = 0

                # Grow the next layer using the occupancy predicted for this layer.
                if d < depth:
                    discrete_layer = (layer_data > threshold).long()
                    try:
                        temp_octree.octree_split(discrete_layer, depth=d)
                        temp_octree.octree_grow(d + 1)
                    except Exception:
                        break
            else:
                available = len(split_sequence) - current_pos
                remaining_in_layer = nodes_needed - available
                current_layer = d
                layer_complete = False

                break

        return current_layer, layer_complete, remaining_in_layer

    except Exception:
        return full_depth, False, 0


def _handle_empty_sequence(
    depth: int, full_depth: int, threshold: float, device: str, pos_emb_model: Optional[torch.nn.Module]
) -> Optional[Tuple[torch.Tensor, torch.Tensor, None, None]]:
    """Initialize the first byte position using a fully occupied octree.

    Average the first eight node embeddings and their coordinates. Return
    (embedding, coordinates, None, None), or None if reconstruction fails.
    """
    try:
        octree_template = create_template_octree(depth, full_depth, device)
        if octree_template is None:
            return None

        # Count the split values needed for a fully occupied octree.
        import copy

        temp_octree = copy.deepcopy(octree_template)
        total_needed_length = 0

        for d in range(full_depth, depth + 1):
            nodes_needed = temp_octree.nnum[d].item()
            total_needed_length += nodes_needed

            if nodes_needed > 0:
                layer_data = torch.ones(nodes_needed, device=device, dtype=torch.float32)

                # Grow the next layer to determine how many split values it needs.
                if d < depth:
                    discrete_layer = (layer_data > threshold).long()
                    try:
                        temp_octree.octree_split(discrete_layer, depth=d)
                        temp_octree.octree_grow(d + 1)
                    except Exception:
                        return None

        full_sequence = torch.ones(total_needed_length, device=device, dtype=torch.float32)

        octree = seq2octree(octree_template, full_sequence, full_depth, depth, threshold)
        if octree is None:
            return None

        octree.to(device)
        octree_t = OctreeT(octree=octree)

        pos_emb_model.to(device)

        position_embeddings = pos_emb_model(octree_t)
        xyz = octree_t.xyz

        first_8_positions = position_embeddings[: min(8, len(position_embeddings))]
        first_8_xyz = xyz[: min(8, len(xyz))]

        if len(first_8_positions) == 0:
            return None

        # Repeat the last embedding if fewer than eight nodes are available.
        if len(first_8_positions) < 8:
            last_position = first_8_positions[-1:].repeat(8 - len(first_8_positions), 1)
            first_8_positions = torch.cat([first_8_positions, last_position], dim=0)

        averaged_position_embedding = first_8_positions.mean(dim=0)
        averaged_xyz = first_8_xyz.mean(dim=0)
        averaged_xyz = averaged_xyz.div(2)
        averaged_xyz = torch.round(averaged_xyz).to(dtype=torch.long, device=device)

        try:
            del octree_t
            del octree
            del octree_template
            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

        return averaged_position_embedding, averaged_xyz, None, None

    except Exception:
        return None


def split_to_next_position_embedding(
    split_sequence,
    depth: int = 6,
    full_depth: int = 3,
    threshold: float = 0.0,
    device: str = "cpu",
    pos_emb_model: Optional[torch.nn.Module] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """Compute the next mesh byte position from a partial split sequence.

    Fill missing splits with occupied nodes through the requested depth, then
    average the next eight node embeddings. An empty sequence uses the first
    eight nodes of a fully occupied octree at that same depth.

    Args:
        split_sequence: Split values generated so far; may be empty.
        depth: Maximum octree depth.
        full_depth: Depth through which all octree nodes are allocated.
        threshold: Threshold used to binarize split values.
        device: Device for reconstruction and positional embeddings.
        pos_emb_model: Positional embedding module; required for reconstruction.

    Returns:
        Next embedding, next coordinates, previous embedding, and previous
        coordinates. Previous values are None for an empty input. Return None
        if the sequence is complete or reconstruction cannot provide a position.
    """
    if isinstance(split_sequence, torch.Tensor):
        split_sequence = split_sequence.to(device)
    else:
        split_sequence = torch.tensor(split_sequence, device=device)

    original_length = len(split_sequence)

    if original_length == 0:
        return _handle_empty_sequence(depth, full_depth, threshold, device, pos_emb_model)

    octree_template = create_template_octree(depth, full_depth, device)
    if octree_template is None:
        return None

    # Count the split values needed to extend the partial tree to the target depth.
    import copy

    temp_octree = copy.deepcopy(octree_template)
    current_pos = 0
    total_needed_length = 0

    for d in range(full_depth, depth + 1):
        nodes_needed = temp_octree.nnum[d].item()
        total_needed_length += nodes_needed

        if nodes_needed > 0:
            if current_pos + nodes_needed <= len(split_sequence):
                layer_data = split_sequence[current_pos : current_pos + nodes_needed]
            else:
                # Treat ungenerated nodes as occupied when extending the partial layer.
                available_data = max(0, len(split_sequence) - current_pos)
                if available_data > 0:
                    existing_data = split_sequence[current_pos : current_pos + available_data]
                    padding_data = torch.ones(nodes_needed - available_data, device=device, dtype=split_sequence.dtype)
                    layer_data = torch.cat([existing_data, padding_data], dim=0)
                else:
                    layer_data = torch.ones(nodes_needed, device=device, dtype=split_sequence.dtype)

            current_pos += nodes_needed

            # Grow the next layer to determine how many split values it needs.
            if d < depth:
                discrete_layer = (layer_data > threshold).long()
                try:
                    temp_octree.octree_split(discrete_layer, depth=d)
                    temp_octree.octree_grow(d + 1)
                except Exception:
                    return None

    # A complete tree has no next position to predict.
    if original_length >= total_needed_length:
        return None
    else:
        padding_length = total_needed_length - original_length
        padding = torch.ones(padding_length, device=device, dtype=split_sequence.dtype)
        complete_sequence = torch.cat([split_sequence, padding], dim=0)
        next_8_start = original_length  # Start immediately after the observed sequence.

    # Reserve eight split values for the next mesh byte.
    if len(complete_sequence) - next_8_start < 8:
        additional_needed = 8 - (len(complete_sequence) - next_8_start)
        additional_padding = torch.ones(additional_needed, device=device, dtype=split_sequence.dtype)
        complete_sequence = torch.cat([complete_sequence, additional_padding], dim=0)

    octree = seq2octree(octree_template, complete_sequence, full_depth, depth, threshold)
    if octree is None:
        return None

    octree.to(device)
    octree_t = OctreeT(octree=octree)
    xyz = octree_t.xyz

    pos_emb_model.to(device)

    try:
        position_embeddings = pos_emb_model(octree_t)

        # Each mesh byte pools the embeddings of eight consecutive octree nodes.
        next_8_end = min(next_8_start + 8, len(position_embeddings))
        next_8_positions = position_embeddings[next_8_start:next_8_end]
        next_8_xyz = xyz[next_8_start:next_8_end]
        pre_8_positions = position_embeddings[next_8_start - 8 : next_8_start]
        pre_8_xyz = xyz[next_8_start - 8 : next_8_start]
        if len(next_8_positions) == 0:
            raise ValueError("No position embeddings are available for the next mesh byte.")

        # A mesh byte requires exactly eight node embeddings.
        if len(next_8_positions) < 8:
            raise ValueError("At least eight node embeddings are required for the next mesh byte.")

        averaged_position_embedding = next_8_positions.mean(dim=0)
        averaged_pre_position_embedding = pre_8_positions.mean(dim=0)
        averaged_xyz = next_8_xyz.mean(dim=0)
        averaged_pre_xyz = pre_8_xyz.mean(dim=0)
        averaged_xyz = averaged_xyz.div(2)
        averaged_xyz = torch.round(averaged_xyz).to(dtype=torch.long, device=device)
        averaged_pre_xyz = averaged_pre_xyz.div(2)
        averaged_pre_xyz = torch.round(averaged_pre_xyz).to(dtype=torch.long, device=device)

        return averaged_position_embedding, averaged_xyz, averaged_pre_position_embedding, averaged_pre_xyz

    except Exception:
        return None

    finally:
        try:
            del octree_t
            del octree
            del octree_template
            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
