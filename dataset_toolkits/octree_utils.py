"""Shared mesh normalization, octree pruning, and occupancy-byte conversion."""

from typing import List, Optional

import numpy as np
import torch
import trimesh
from ocnn.nn import octree2voxel
from ocnn.octree import Octree, Points

from llamafactory.model.utils.utils import octree2seq, seq2octree


def scale_to_unit_cube(mesh):
    """Normalize mesh into the unit cube [-1, 1]^3."""
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def sample_points_from_mesh(mesh, num_samples: int = 100000):
    """Sample surface points and normals from mesh."""
    try:
        points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
        normals = mesh.face_normals[face_indices]
        return points.astype(np.float32), normals.astype(np.float32)
    except Exception as exc:
        print(f"Failed to sample points: {exc}")
        return None, None


def build_octree_from_points(
    points: np.ndarray,
    normals: np.ndarray,
    depth: int = 8,
    full_depth: int = 4,
    device: str = "cpu",
) -> Optional[Octree]:
    """Build an octree from point cloud samples."""
    try:
        points_tensor = torch.from_numpy(points).to(device)
        normals_tensor = torch.from_numpy(normals).to(device)

        points_obj = Points(points=points_tensor, normals=normals_tensor)
        points_obj.clip(min=-1.0, max=1.0)

        octree = Octree(depth=depth, full_depth=full_depth)
        octree.build_octree(points_obj)
        return octree
    except Exception as exc:
        print(f"Failed to build octree: {exc}")
        return None


def convert_to_binary_sequence(split_sequence: torch.Tensor, threshold: float = 0.5) -> List[int]:
    """Binarize a split sequence using the occupancy threshold."""
    flat_sequence = split_sequence.flatten()
    binary_sequence = (flat_sequence > threshold).int().tolist()
    return binary_sequence


def binary_to_bytes(binary_sequence: List[int]) -> List[int]:
    """Pack each group of eight bits into one byte, most significant bit first."""
    if len(binary_sequence) % 8 != 0:
        raise ValueError("Binary sequence length must be a multiple of eight.")

    byte_sequence = []

    for i in range(0, len(binary_sequence), 8):
        eight_bits = binary_sequence[i : i + 8]
        decimal_value = 0
        for j, bit in enumerate(eight_bits):
            decimal_value += bit * (2 ** (7 - j))
        byte_sequence.append(decimal_value)

    return byte_sequence


def save_byte_sequence_to_txt(byte_sequence: List[int], output_path: str):
    """Write space-separated byte values to a text file."""
    with open(output_path, "w") as f:
        f.write(" ".join(map(str, byte_sequence)))


def octree_to_voxel(octree, depth: int):
    """Convert a single octree to a dense occupancy grid."""
    try:
        batch_id = octree.batch_id(depth=depth, nempty=True)
        data = torch.ones((len(batch_id), 1), device=octree.device)

        voxel_data = octree2voxel(data=data, octree=octree, depth=depth, nempty=True)
        voxel_data = voxel_data.permute(0, 4, 1, 2, 3).contiguous()

        # Return the single mesh in the batch.
        voxel = voxel_data[0].squeeze().cpu().numpy()
        return voxel
    except Exception as e:
        print(f"Failed to convert octree to voxels: {e}")
        return None


def prune_octree_at_depth(octree, target_depth: int, drop_prob: float) -> Optional["Octree"]:
    if target_depth >= octree.depth:
        print(f"Target depth {target_depth} must be less than the maximum octree depth {octree.depth}")
        return None

    depth_low = 0
    depth_high = octree.depth + 1
    seq = octree2seq(octree, depth_low=depth_low, depth_high=depth_high).clone()

    nnum_list = [int(val) for val in octree.nnum.tolist()]

    start = sum(nnum_list[depth_low:target_depth])
    end = start + nnum_list[target_depth]
    target_slice = seq[start:end].clone()

    non_empty_mask = target_slice == 1
    if not non_empty_mask.any():
        print(f"Depth {target_depth} has no nonempty nodes to prune")
        return octree

    non_empty_indices = torch.nonzero(non_empty_mask, as_tuple=False).squeeze(-1)
    random_tensor = torch.rand(non_empty_indices.shape[0], device=seq.device)
    drop_selection = random_tensor < drop_prob

    if drop_selection.all():
        keep_idx = torch.randint(0, non_empty_indices.shape[0], (1,), device=seq.device)
        drop_selection[keep_idx] = False

    drop_indices = non_empty_indices[drop_selection]
    target_slice[drop_indices] = 0
    seq[start:end] = target_slice

    keep_selection = ~drop_selection

    if target_depth + 1 < depth_high:
        if target_depth + 1 < len(nnum_list):
            total_children = nnum_list[target_depth + 1]
        else:
            total_children = 0
        if total_children > 0:
            next_start = end
            next_end = next_start + total_children
            next_slice = seq[next_start:next_end].clone()

            num_non_empty = non_empty_indices.shape[0]
            if num_non_empty > 0 and total_children == num_non_empty * 8:
                next_blocks = next_slice.view(num_non_empty, 8)
                kept_blocks = next_blocks[keep_selection]
                new_next_slice = kept_blocks.reshape(-1)

                seq[next_start : next_start + new_next_slice.numel()] = new_next_slice
                if new_next_slice.numel() < next_slice.numel():
                    seq[next_start + new_next_slice.numel() : next_end] = 0
            else:
                seq[next_start:next_end] = 0

    pruned_octree = seq2octree(octree, seq, depth_low=depth_low, depth_high=depth_high)
    pruned_octree.points[pruned_octree.depth] = None
    pruned_octree.normals[pruned_octree.depth] = None
    pruned_octree.features[pruned_octree.depth] = None
    return pruned_octree
