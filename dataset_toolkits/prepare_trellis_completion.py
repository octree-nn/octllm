#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process TRELLIS-500K dataset and export voxel completion pairs.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import gc
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm


from dataset_toolkits.octree_utils import (
    build_octree_from_points as build_octree_from_points,
    octree_to_voxel as octree_to_voxel,
    prune_octree_at_depth as prune_octree_at_depth,
    sample_points_from_mesh as sample_points_from_mesh,
    scale_to_unit_cube as scale_to_unit_cube,
)

GC_INTERVAL = 50


def process_single_model(
    glb_path: str,
    num_samples: int = 100000,
    depth: int = 6,
    target_depth: int = 5,
    drop_prob: float = 0.5,
    full_depth: int = 3,
    mesh_scale: float = 1.0,
    device: str = "cpu",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert one GLB mesh into original and pruned voxel grids."""
    if not os.path.exists(glb_path):
        print(f"File not found: {glb_path}")
        return None, None

    mesh = trimesh.load(glb_path, force="mesh")
    mesh = scale_to_unit_cube(mesh)
    mesh.vertices *= mesh_scale

    points, normals = sample_points_from_mesh(mesh, num_samples)
    del mesh
    if points is None:
        return None, None

    octree = build_octree_from_points(points, normals, depth, full_depth, device)
    del points, normals
    if octree is None:
        return None, None

    pruned_octree = prune_octree_at_depth(octree, target_depth, drop_prob)
    if pruned_octree is None:
        del octree
        return None, None

    ori_voxel = octree_to_voxel(octree, depth)
    pruned_voxel = octree_to_voxel(pruned_octree, depth)
    del octree, pruned_octree
    return ori_voxel, pruned_voxel


def get_report_dir(output_dir_ori: str, output_dir_pruned: str) -> str:
    """Choose a stable directory for summary files."""
    abs_ori = os.path.abspath(output_dir_ori)
    abs_pruned = os.path.abspath(output_dir_pruned)
    report_dir = os.path.commonpath([abs_ori, abs_pruned])

    if report_dir == os.path.sep:
        return abs_pruned
    return report_dir


def batch_process_dataset(
    category: str,
    csv_path: str,
    data_root: str,
    output_dir_ori: str,
    output_dir_pruned: str,
    num_samples: int = 100000,
    depth: int = 6,
    target_depth: int = 5,
    drop_prob: float = 0.5,
    full_depth: int = 3,
    mesh_scale: float = 1.0,
    device: str = "cpu",
    local_path_column: str = "local_path",
):
    """Process TRELLIS-500K metadata row by row with low memory usage."""
    file_prefix = f"{category}_"
    print(f"CSV path: {csv_path}")
    print(f"Data root: {data_root}")
    print(f"Category: {category}")
    print(f"Original voxel output: {output_dir_ori}")
    print(f"Pruned voxel output: {output_dir_pruned}")

    os.makedirs(output_dir_ori, exist_ok=True)
    os.makedirs(output_dir_pruned, exist_ok=True)

    with open(csv_path, "r") as file:
        total = sum(1 for _ in file) - 1
    print(f"Found {total} models to process")

    success_count = 0
    resumed_count = 0
    fail_count = 0
    skipped_assets = []

    with open(csv_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        if local_path_column not in reader.fieldnames:
            raise ValueError(f"Column not found in CSV: {local_path_column}")

        for index, row in enumerate(tqdm(reader, total=total, desc="Processing models")):
            local_path = row[local_path_column]
            glb_path = os.path.join(data_root, local_path)
            sample_name = Path(glb_path).stem
            output_name = f"{file_prefix}{sample_name}"

            ori_path = os.path.join(output_dir_ori, f"{output_name}.npy")
            pruned_path = os.path.join(output_dir_pruned, f"{output_name}.npy")

            if os.path.exists(ori_path) and os.path.exists(pruned_path):
                resumed_count += 1
                continue

            try:
                ori_voxel, pruned_voxel = process_single_model(
                    glb_path=glb_path,
                    num_samples=num_samples,
                    depth=depth,
                    target_depth=target_depth,
                    drop_prob=drop_prob,
                    full_depth=full_depth,
                    mesh_scale=mesh_scale,
                    device=device,
                )
            except Exception as exc:
                print(f"Failed to process {output_name}: {exc}")
                skipped_assets.append(output_name)
                fail_count += 1
                continue

            if ori_voxel is None or pruned_voxel is None:
                print(f"Failed to export voxels for {output_name}")
                skipped_assets.append(output_name)
                fail_count += 1
                continue

            np.save(ori_path, ori_voxel)
            np.save(pruned_path, pruned_voxel)
            success_count += 1

            if (index + 1) % GC_INTERVAL == 0:
                gc.collect()

    print("\nProcessing complete")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exists): {resumed_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped assets: {len(skipped_assets)}")

    report_dir = get_report_dir(output_dir_ori, output_dir_pruned)
    skipped_assets_path = os.path.join(report_dir, "skipped_assets.txt")
    with open(skipped_assets_path, "w") as file:
        for asset in skipped_assets:
            file.write(f"{asset}\n")


def main():
    parser = argparse.ArgumentParser(description="Process TRELLIS-500K dataset and export voxel completion pairs")
    parser.add_argument(
        "--category",
        default="ObjaverseXL_sketchfab",
        help="TRELLIS-500K category name used for CSV/data lookup and output prefix",
    )
    parser.add_argument(
        "--trellis_root",
        default="datasets/trellis",
        help="Root directory that contains category metadata folders",
    )
    parser.add_argument(
        "--asset_root",
        default="datasets/meshes",
        help="Root directory that contains category asset folders",
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        help="Optional explicit path to the TRELLIS-500K metadata CSV",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Optional explicit root directory for local GLB assets",
    )
    parser.add_argument(
        "--output_dir_ori",
        default="datasets/completion/ori",
        help="Output directory for original voxel grids",
    )
    parser.add_argument(
        "--output_dir_pruned",
        default="datasets/completion/pruned",
        help="Output directory for pruned voxel grids",
    )
    parser.add_argument("--num_samples", "-n", type=int, default=100000, help="Number of sampled points")
    parser.add_argument("--depth", "-d", type=int, default=6, help="Maximum octree depth")
    parser.add_argument(
        "--target_depth",
        "-td",
        type=int,
        default=5,
        help="Depth where non-empty octree nodes are randomly dropped",
    )
    parser.add_argument("--drop_prob", "-dp", type=float, default=0.5, help="Drop probability at target depth")
    parser.add_argument("--full_depth", "-f", type=int, default=3, help="Full octree depth")
    parser.add_argument("--mesh_scale", "-s", type=float, default=1.0, help="Mesh scale factor")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Compute device")
    parser.add_argument(
        "--local_path_column",
        default="local_path",
        help="CSV column that stores relative GLB paths",
    )
    args = parser.parse_args()

    if args.csv_path is None:
        args.csv_path = os.path.join(args.asset_root, args.category, "metadata.csv")

    if args.data_root is None:
        args.data_root = os.path.join(args.asset_root, args.category)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = "cpu"

    batch_process_dataset(
        category=args.category,
        csv_path=args.csv_path,
        data_root=args.data_root,
        output_dir_ori=args.output_dir_ori,
        output_dir_pruned=args.output_dir_pruned,
        num_samples=args.num_samples,
        depth=args.depth,
        target_depth=args.target_depth,
        drop_prob=args.drop_prob,
        full_depth=args.full_depth,
        mesh_scale=args.mesh_scale,
        device=args.device,
        local_path_column=args.local_path_column,
    )


if __name__ == "__main__":
    main()
