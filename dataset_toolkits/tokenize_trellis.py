#!/usr/bin/env python3
"""Convert GLB meshes listed in a CSV manifest into octree byte sequences."""

import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import gc
from typing import List, Optional

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
from llamafactory.model.utils.utils import octree2seq, octree2split


GC_INTERVAL = 50


def process_single_model(
    glb_path: str,
    num_samples: int = 100000,
    depth: int = 6,
    full_depth: int = 3,
    mesh_scale: float = 1.0,
    shift: bool = False,
    threshold: float = 0.5,
    device: str = "cpu",
    target_depth: int = 5,
    drop_prob: float = 0.5,
    prune: bool = False,
) -> Optional[List[int]]:
    """Normalize and sample one GLB mesh, then encode its octree as bytes.

    Return a list of integers in [0, 255], or None if preprocessing fails.
    """

    if not os.path.exists(glb_path):
        print(f"File does not exist: {glb_path}")
        return None

    mesh = trimesh.load(glb_path, force="mesh")

    mesh = scale_to_unit_cube(mesh)
    mesh.vertices *= mesh_scale

    points, normals = sample_points_from_mesh(mesh, num_samples)
    del mesh
    if points is None:
        return None

    octree = build_octree_from_points(points, normals, depth, full_depth, device)
    del points, normals
    if octree is None:
        return None

    if prune:
        octree = prune_octree_at_depth(octree, target_depth, drop_prob)

    split_sequence_full = octree2split(octree, depth_low=full_depth, depth_high=full_depth + 1, shift=shift)
    split_sequence_full = split_sequence_full.flatten()

    if split_sequence_full is None:
        raise ValueError("split_sequence_full is None")

    if depth > full_depth:
        split_sequence = octree2seq(octree, depth_low=full_depth + 1, depth_high=depth + 1, shift=shift)
        split_sequence = split_sequence.flatten()
    else:
        split_sequence = None

    del octree

    if split_sequence is not None:
        split_sequence_all = torch.cat([split_sequence_full, split_sequence], dim=0)
        del split_sequence_full, split_sequence
    else:
        split_sequence_all = split_sequence_full
        del split_sequence_full

    binary_sequence = convert_to_binary_sequence(split_sequence_all, threshold)
    del split_sequence_all

    byte_sequence = binary_to_bytes(binary_sequence)

    return byte_sequence


DATA_ROOT = "."


def batch_process_dataset(
    csv_path: str,
    output_dir: str,
    num_samples: int = 100000,
    depth: int = 6,
    full_depth: int = 3,
    mesh_scale: float = 1.0,
    shift: bool = False,
    threshold: float = 0.5,
    device: str = "cpu",
    local_path_column: str = "local_path",
    target_depth: int = 5,
    drop_prob: float = 0.5,
    prune: bool = False,
    data_root: str = DATA_ROOT,
):
    """Stream a CSV manifest and save one octree byte sequence per mesh.

    Read mesh paths from local_path_column and skip existing output files.
    Forward the remaining options to process_single_model.
    """
    print(f"CSV manifest: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Pruning: {prune}")

    os.makedirs(output_dir, exist_ok=True)

    # Count rows for the progress bar without loading the manifest into memory.
    with open(csv_path, "r") as f:
        total = sum(1 for _ in f) - 1
    print(f"Found {total} models to process")

    success_count = 0
    resumed_count = 0
    fail_count = 0
    max_seq_length = 0
    skipped_assets = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if local_path_column not in reader.fieldnames:
            raise ValueError(f"CSV column does not exist: {local_path_column}")

        for i, row in enumerate(tqdm(reader, total=total, desc="Processing models")):
            local_path = row[local_path_column]
            glb_path = os.path.join(data_root, local_path)
            model_name = row["sha256"]
            txt_filename = f"{model_name}_bytes.txt"
            txt_path = os.path.join(output_dir, txt_filename)

            # Resume processing by skipping outputs that already exist.
            if os.path.exists(txt_path):
                resumed_count += 1
                continue

            try:
                byte_sequence = process_single_model(
                    glb_path=glb_path,
                    num_samples=num_samples,
                    depth=depth,
                    full_depth=full_depth,
                    mesh_scale=mesh_scale,
                    shift=shift,
                    threshold=threshold,
                    device=device,
                    target_depth=target_depth,
                    drop_prob=drop_prob,
                    prune=prune,
                )
            except Exception as e:
                print(f"Processing model {model_name} failed: {e}")
                skipped_assets.append(model_name)
                fail_count += 1
                continue

            if byte_sequence is None:
                fail_count += 1
                skipped_assets.append(model_name)
                print(f"Failed: {model_name}")
                continue

            if len(byte_sequence) < 100 or len(byte_sequence) > 4000:
                print(f"Warning: model {model_name} has an unsupported sequence length: {len(byte_sequence)}; skipped")
                skipped_assets.append(model_name)
                continue

            if len(byte_sequence) > max_seq_length:
                max_seq_length = len(byte_sequence)

            save_byte_sequence_to_txt(byte_sequence, txt_path)
            success_count += 1

            if (i + 1) % GC_INTERVAL == 0:
                gc.collect()

    print("\nProcessing complete.")
    print(f"Successfully processed: {success_count} models")
    print(f"Skipped (already exists): {resumed_count} models")
    print(f"Failed: {fail_count} models")
    print(f"Maximum sequence length: {max_seq_length}")
    print(f"Skipped models: {len(skipped_assets)}")

    skipped_assets_path = os.path.join(output_dir, "skipped_assets.txt")
    with open(skipped_assets_path, "w") as f:
        for asset in skipped_assets:
            f.write(f"{asset}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GLB meshes listed in a CSV manifest into octree byte sequences."
    )
    parser.add_argument("--csv_path", help="Path to the CSV manifest.", default="datasets/trellis/HSSD/metadata.csv")
    parser.add_argument("--output_dir", help="Output directory.", default="datasets/trellis/HSSD/tokenized_bytes")
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
    parser.add_argument("--local_path_column", default="local_path", help="CSV column containing local mesh paths.")
    parser.add_argument(
        "--target_depth", "-td", type=int, default=5, help="Octree depth at which to randomly prune nodes."
    )
    parser.add_argument("--drop_prob", "-dp", type=float, default=0.5, help="Probability of pruning each node.")
    parser.add_argument("--prune", action="store_true", default=True, help="Enable octree pruning.")

    parser.add_argument("--data_root", required=True, help="Root directory for metadata local_path values.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is unavailable; using CPU.")
        args.device = "cpu"

    batch_process_dataset(
        data_root=args.data_root,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        depth=args.depth,
        full_depth=args.full_depth,
        mesh_scale=args.mesh_scale,
        shift=args.shift,
        threshold=args.threshold,
        device=args.device,
        local_path_column=args.local_path_column,
        target_depth=args.target_depth,
        drop_prob=args.drop_prob,
        prune=args.prune,
    )


if __name__ == "__main__":
    main()
