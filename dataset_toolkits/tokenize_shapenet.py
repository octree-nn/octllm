#!/usr/bin/env python3
"""Convert ShapeNet OBJ meshes into octree or rasterized voxel byte sequences."""

import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from typing import List, Optional

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from dataset_toolkits.octree_utils import (
    binary_to_bytes as binary_to_bytes,
)
from dataset_toolkits.octree_utils import (
    build_octree_from_points as build_octree_from_points,
)
from dataset_toolkits.octree_utils import (
    convert_to_binary_sequence as convert_to_binary_sequence,
)
from dataset_toolkits.octree_utils import (
    prune_octree_at_depth as prune_octree_at_depth,
)
from dataset_toolkits.octree_utils import (
    sample_points_from_mesh as sample_points_from_mesh,
)
from dataset_toolkits.octree_utils import (
    save_byte_sequence_to_txt as save_byte_sequence_to_txt,
)
from dataset_toolkits.octree_utils import (
    scale_to_unit_cube as scale_to_unit_cube,
)


shapenet_classes = {
    "chair": "03001627",
    "table": "04379243",
    "airplane": "02691156",
    "car": "02958343",
    "rifle": "04090263",
}


from ocnn.nn import octree2voxel

from llamafactory.model.utils.utils import octree2seq, octree2split


def process_single_model(
    obj_path: str,
    num_samples: int = 100000,
    depth: int = 6,
    full_depth: int = 3,
    mesh_scale: float = 1.0,
    shift: bool = False,
    threshold: float = 0.5,
    device: str = "cpu",
    sequence_mode: str = "octree",
    target_depth: int = 5,
    drop_prob: float = 0.5,
    prune: bool = False,
) -> Optional[List[int]]:
    """Encode one OBJ mesh as occupancy bytes with values from 0 to 255.

    Normalize and sample the mesh, then optionally prune its octree. The
    sequence mode selects octree traversal or C-order voxel rasterization.
    Return None if loading, sampling, or octree construction fails.
    """

    if not os.path.exists(obj_path):
        print(f"File does not exist: {obj_path}")
        return None

    mesh = trimesh.load(obj_path, force="mesh")

    mesh = scale_to_unit_cube(mesh)
    mesh.vertices *= mesh_scale

    points, normals = sample_points_from_mesh(mesh, num_samples)
    if points is None:
        return None

    octree = build_octree_from_points(points, normals, depth, full_depth, device)
    if octree is None:
        return None

    if prune:
        octree = prune_octree_at_depth(octree, target_depth, drop_prob)

    if sequence_mode == "octree":
        split_sequence_full = octree2split(octree, depth_low=full_depth, depth_high=full_depth + 1, shift=shift)
        split_sequence_full = split_sequence_full.flatten()

        if split_sequence_full is None:
            raise ValueError("split_sequence_full is None")

        if depth > full_depth:
            split_sequence = octree2seq(octree, depth_low=full_depth + 1, depth_high=depth + 1, shift=shift)
            split_sequence = split_sequence.flatten()
        else:
            split_sequence = None

        if split_sequence is not None:
            split_sequence_all = torch.cat([split_sequence_full, split_sequence], dim=0)
        else:
            split_sequence_all = split_sequence_full

        binary_sequence = convert_to_binary_sequence(split_sequence_all, threshold)

    elif sequence_mode == "voxel":
        voxel = octree_to_voxel_full(octree, depth)
        if voxel is None:
            return None
        binary_sequence = voxel_to_raster_sequence(voxel)

    else:
        raise ValueError(f"Unsupported sequence mode: {sequence_mode}")

    byte_sequence = binary_to_bytes(binary_sequence)

    return byte_sequence


def octree_to_voxel_full(octree, depth: int):
    """Convert an octree to a dense occupancy grid at the requested depth."""
    batch_id = octree.batch_id(depth=depth, nempty=True)
    data = torch.ones((len(batch_id), 1), device=octree.device)

    voxel_data = octree2voxel(data=data, octree=octree, depth=depth, nempty=True)
    voxel_data = voxel_data.permute(0, 4, 1, 2, 3).contiguous()

    # Return the single mesh in the batch.
    voxel = voxel_data[0].squeeze()
    return voxel.cpu().numpy()


def voxel_to_raster_sequence(voxel: np.ndarray) -> List[int]:
    """Flatten voxel occupancy in C order (the z axis changes fastest)."""
    raster_sequence = voxel.flatten(order="C").astype(int).tolist()
    return raster_sequence


def get_all_subfolders(root_dir: str) -> List[str]:
    """Return sorted names of the immediate subdirectories."""
    subfolders = []
    if not os.path.exists(root_dir):
        print(f"Error: root directory does not exist: {root_dir}")
        return subfolders

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            subfolders.append(item)

    return sorted(subfolders)


def process_shapenet_dataset(
    input_root_dir: str,
    output_root_dir: str,
    num_samples: int = 100000,
    depth: int = 6,
    full_depth: int = 3,
    mesh_scale: float = 1.0,
    shift: bool = False,
    threshold: float = 0.5,
    device: str = "cpu",
    sequence_mode: str = "octree",
    mesh_type: str = "chair",
    target_depth: int = 5,
    drop_prob: float = 0.5,
    prune: bool = False,
):
    """Encode each mesh in a ShapeNet category and save one text file per model.

    Forward the sampling, octree, and pruning options to process_single_model.
    Skip sequences outside the supported length range and report skipped assets.
    """

    print("Processing ShapeNet dataset")
    print(f"Input directory: {input_root_dir}")
    print(f"Output directory: {output_root_dir}")
    print(f"Sequence mode: {sequence_mode}")
    print(f"Pruning: {prune}")

    max_seq_length = 0

    subfolders = get_all_subfolders(input_root_dir)
    if not subfolders:
        print(f"Error: no subdirectories found in the input directory: {input_root_dir}")
        return

    print(f"Found {len(subfolders)} subdirectories to process")

    os.makedirs(output_root_dir, exist_ok=True)

    success_count = 0
    skipped_assets = []

    for subfolder in tqdm(subfolders, desc="Processing subdirectories"):
        input_subfolder_path = os.path.join(input_root_dir, subfolder)
        obj_path = os.path.join(input_subfolder_path, "model.obj")

        if not os.path.exists(obj_path):
            print(f"Skipping missing model.obj: {obj_path}")
            raise ValueError(f"Model {subfolder} does not exist")

        output_subfolder_path = os.path.join(output_root_dir, subfolder)
        os.makedirs(output_subfolder_path, exist_ok=True)

        txt_filename = f"{subfolder}_bytes.txt"
        txt_path = os.path.join(output_subfolder_path, txt_filename)

        try:
            byte_sequence = process_single_model(
                obj_path=obj_path,
                num_samples=num_samples,
                depth=depth,
                full_depth=full_depth,
                mesh_scale=mesh_scale,
                shift=shift,
                threshold=threshold,
                device=device,
                sequence_mode=sequence_mode,
                target_depth=target_depth,
                drop_prob=drop_prob,
                prune=prune,
            )
        except Exception as e:
            print(f"Processing model {subfolder} failed: {e}")
            skipped_assets.append(subfolder)
            continue
        if len(byte_sequence) < 100 or len(byte_sequence) > 6000:
            print(f"Warning: model {subfolder} has an unsupported sequence length: {len(byte_sequence)}; skipped")
            skipped_assets.append(subfolder)
            continue
        if len(byte_sequence) > max_seq_length:
            max_seq_length = len(byte_sequence)
        save_byte_sequence_to_txt(byte_sequence, txt_path)
        success_count += 1

    print("\nProcessing complete.")
    print(f"Successfully processed: {success_count} models")
    print(f"Maximum sequence length: {max_seq_length}")
    print(f"Skipped models: {len(skipped_assets)}")
    # Save skipped asset names into a txt file, each asset on a new line
    skipped_assets_path = os.path.join(output_root_dir, f"skipped_assets_{mesh_type}.txt")
    with open(skipped_assets_path, "w") as f:
        for asset in skipped_assets:
            f.write(f"{asset}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert ShapeNet OBJ meshes into byte sequences.")
    parser.add_argument(
        "--input_dir", "-i", default="datasets/shapenet/fixed_mesh", help="Root directory containing input meshes."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="datasets/shapenet/fixed_mesh_tokens_0.5_chair",
        help="Root directory for generated files.",
    )
    parser.add_argument(
        "--num_samples", "-n", type=int, default=100000, help="Number of points to sample from each mesh."
    )
    parser.add_argument("--depth", "-d", type=int, default=6, help="Maximum octree depth.")
    parser.add_argument(
        "--full_depth", "-f", type=int, default=3, help="Depth through which all octree nodes are allocated."
    )
    parser.add_argument(
        "--mesh_scale", "-s", type=float, default=1.0, help="Scale factor applied after mesh normalization."
    )
    parser.add_argument("--shift", action="store_true", help="Rescale split values from [0, 1] to [-1, 1].")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Occupancy binarization threshold.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Device used for preprocessing.")
    parser.add_argument(
        "--sequence_mode",
        "-m",
        default="octree",
        choices=["octree", "voxel"],
        help="Sequence mode: octree traversal (Z-order) or voxel rasterization (C-order).",
    )
    parser.add_argument(
        "--target_depth", "-td", type=int, default=5, help="Octree depth at which to randomly prune nodes."
    )
    parser.add_argument("--drop_prob", "-dp", type=float, default=0.5, help="Probability of pruning each node.")
    parser.add_argument("--prune", action="store_true", default=True, help="Enable octree pruning.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is unavailable; using CPU.")
        args.device = "cpu"

    for mesh_type in shapenet_classes.keys():
        input_root_dir = os.path.join(args.input_dir, shapenet_classes[mesh_type])
        output_root_dir = os.path.join(args.output_dir, shapenet_classes[mesh_type])
        print(f"Now processing {mesh_type}")
        process_shapenet_dataset(
            input_root_dir=input_root_dir,
            output_root_dir=output_root_dir,
            num_samples=args.num_samples,
            depth=args.depth,
            full_depth=args.full_depth,
            mesh_scale=args.mesh_scale,
            shift=args.shift,
            threshold=args.threshold,
            device=args.device,
            sequence_mode=args.sequence_mode,
            mesh_type=mesh_type,
            target_depth=args.target_depth,
            drop_prob=args.drop_prob,
            prune=args.prune,
        )


if __name__ == "__main__":
    main()
