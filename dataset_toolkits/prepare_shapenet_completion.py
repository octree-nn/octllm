#!/usr/bin/env python3
"""Build paired complete and pruned ShapeNet occupancy grids for completion training."""

import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from typing import List

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from dataset_toolkits.octree_utils import (
    build_octree_from_points as build_octree_from_points,
)
from dataset_toolkits.octree_utils import (
    octree_to_voxel as octree_to_voxel,
)
from dataset_toolkits.octree_utils import (
    prune_octree_at_depth as prune_octree_at_depth,
)
from dataset_toolkits.octree_utils import (
    sample_points_from_mesh as sample_points_from_mesh,
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


def process_single_model(
    obj_path: str,
    num_samples: int = 100000,
    depth: int = 6,
    target_depth: int = 5,
    drop_prob: float = 0.5,
    full_depth: int = 3,
    mesh_scale: float = 1.0,
    device: str = "cpu",
    output_root_dir_ori: str = None,
    output_root_dir_pruned: str = None,
    mesh_type: str = "chair",
) -> None:
    """Write matching complete and pruned occupancy grids for one mesh."""

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

    pruned_octree = prune_octree_at_depth(octree, target_depth, drop_prob)
    if pruned_octree is None:
        return None

    ori_voxel = octree_to_voxel(octree, depth)
    pruned_voxel = octree_to_voxel(pruned_octree, depth)
    if ori_voxel is not None:
        np.save(os.path.join(output_root_dir_ori, f"{obj_path.split('/')[-2]}_{mesh_type}.npy"), ori_voxel)
    if pruned_voxel is not None:
        np.save(os.path.join(output_root_dir_pruned, f"{obj_path.split('/')[-2]}_{mesh_type}.npy"), pruned_voxel)


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
    output_root_dir_ori: str,
    output_root_dir_pruned: str,
    num_samples: int = 100000,
    depth: int = 6,
    target_depth: int = 5,
    drop_prob: float = 0.5,
    full_depth: int = 3,
    mesh_scale: float = 1.0,
    device: str = "cpu",
    mesh_type: str = "chair",
):
    """Write complete and pruned voxel grids for every mesh in a ShapeNet category.

    The two output directories share filenames so each pruned input can be
    matched to its complete target.
    """

    subfolders = get_all_subfolders(input_root_dir)
    if not subfolders:
        print(f"Error: no subdirectories found in the input directory: {input_root_dir}")
        return

    print(f"Found {len(subfolders)} subdirectories to process")

    os.makedirs(output_root_dir_ori, exist_ok=True)
    os.makedirs(output_root_dir_pruned, exist_ok=True)

    for subfolder in tqdm(subfolders, desc="Processing subdirectories"):
        input_subfolder_path = os.path.join(input_root_dir, subfolder)
        obj_path = os.path.join(input_subfolder_path, "model.obj")

        if not os.path.exists(obj_path):
            print(f"Skipping missing model.obj: {obj_path}")
            raise ValueError(f"Model {subfolder} does not exist")

        process_single_model(
            obj_path=obj_path,
            num_samples=num_samples,
            depth=depth,
            target_depth=target_depth,
            drop_prob=drop_prob,
            full_depth=full_depth,
            mesh_scale=mesh_scale,
            device=device,
            output_root_dir_ori=output_root_dir_ori,
            output_root_dir_pruned=output_root_dir_pruned,
            mesh_type=mesh_type,
        )


def main():
    parser = argparse.ArgumentParser(description="Build paired complete and pruned ShapeNet occupancy grids.")
    parser.add_argument(
        "--input_dir", "-i", default="datasets/shapenet/fixed_mesh", help="Root directory containing input meshes."
    )
    parser.add_argument(
        "--output_dir_ori", "-oo", default="datasets/completion/ori/", help="Root directory for generated files."
    )
    parser.add_argument(
        "--output_dir_pruned", "-op", default="datasets/completion/pruned/", help="Root directory for generated files."
    )
    parser.add_argument(
        "--num_samples", "-n", type=int, default=100000, help="Number of points to sample from each mesh."
    )
    parser.add_argument("--depth", "-d", type=int, default=6, help="Maximum octree depth.")
    parser.add_argument(
        "--target_depth", "-td", type=int, default=5, help="Octree depth at which to randomly prune nodes."
    )
    parser.add_argument("--drop_prob", "-dp", type=float, default=0.5, help="Probability of pruning each node.")
    parser.add_argument(
        "--full_depth", "-f", type=int, default=3, help="Depth through which all octree nodes are allocated."
    )
    parser.add_argument(
        "--mesh_scale", "-s", type=float, default=1.0, help="Scale factor applied after mesh normalization."
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device used for preprocessing.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is unavailable; using CPU.")
        args.device = "cpu"

    for mesh_type in shapenet_classes.keys():
        print(f"Now processing {mesh_type}")
        input_root_dir = os.path.join(args.input_dir, shapenet_classes[mesh_type])

        process_shapenet_dataset(
            input_root_dir=input_root_dir,
            output_root_dir_ori=args.output_dir_ori,
            output_root_dir_pruned=args.output_dir_pruned,
            num_samples=args.num_samples,
            depth=args.depth,
            target_depth=args.target_depth,
            drop_prob=args.drop_prob,
            full_depth=args.full_depth,
            mesh_scale=args.mesh_scale,
            device=args.device,
            mesh_type=mesh_type,
        )


if __name__ == "__main__":
    main()
