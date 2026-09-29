#!/usr/bin/env python3
"""Convert one GLB or OBJ mesh to the S-Octree tokens used by OctLLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def mesh_to_octree_sequence(
    mesh_path: str | Path,
    *,
    num_samples: int = 100000,
    depth: int = 6,
    full_depth: int = 3,
    device: str = "cpu",
    target_depth: int = 5,
    drop_prob: float = 0.5,
    prune: bool = True,
) -> str:
    """Reuse the dataset tokenizer and wrap its occupancy bytes as mesh tokens.

    GLB and OBJ use the same normalization, surface sampling, pruning, and
    octree traversal. Sampling and pruning are stochastic; save the returned
    sequence to reuse an exact input. No model checkpoint is loaded here.
    """
    mesh_path = Path(mesh_path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Input mesh not found: {mesh_path}")
    if mesh_path.suffix.lower() not in {".glb", ".obj"}:
        raise ValueError("Input mesh must be a .glb or .obj file.")
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if not 0 <= full_depth <= depth:
        raise ValueError("Require 0 <= full_depth <= depth.")
    if prune and not full_depth <= target_depth < depth:
        raise ValueError("Pruning requires full_depth <= target_depth < depth.")
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError("drop_prob must be in [0, 1].")

    # Import only for mesh input, so text/image chat has no geometry imports.
    # The dataset loader uses Trimesh's format detection for both GLB and OBJ.
    from dataset_toolkits.tokenize_trellis import process_single_model

    try:
        values = process_single_model(
            glb_path=str(mesh_path),
            num_samples=num_samples,
            depth=depth,
            full_depth=full_depth,
            mesh_scale=1.0,
            shift=False,
            threshold=0.5,
            device=device,
            target_depth=target_depth,
            drop_prob=drop_prob,
            prune=prune,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to tokenize input mesh {mesh_path}: {exc}") from exc
    if not values:
        raise RuntimeError(f"Mesh preprocessing returned no octree sequence: {mesh_path}")
    if any(value < 0 or value > 255 for value in values):
        raise ValueError("Mesh preprocessing produced a byte outside [0, 255].")
    return "<mesh_bos>" + "".join(f"<mesh{value}>" for value in values) + "<mesh_eos>"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Input .glb or .obj file.")
    parser.add_argument("--output", required=True, help="Output UTF-8 file containing S-Octree mesh tokens.")
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--full-depth", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--target-depth", type=int, default=5)
    parser.add_argument("--drop-prob", type=float, default=0.5)
    parser.add_argument("--no-prune", action="store_true", help="Disable the dataset's sparse-octree pruning.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        sequence = mesh_to_octree_sequence(
            args.mesh,
            num_samples=args.num_samples,
            depth=args.depth,
            full_depth=args.full_depth,
            device=args.device,
            target_depth=args.target_depth,
            drop_prob=args.drop_prob,
            prune=not args.no_prune,
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(sequence + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Octree tokens: {output}")


if __name__ == "__main__":
    main()
