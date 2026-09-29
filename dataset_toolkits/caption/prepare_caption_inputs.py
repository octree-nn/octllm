#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from tqdm import tqdm


EXPECTED_VIEW_FILES = ("014.png", "015.png", "016.png", "017.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multiview image dataset manifest for geometry caption generation."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("datasets/trellis/ObjaverseXL_sketchfab/renders_cond"),
        help="Root directory that contains one subdirectory per 3D model.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to save the output dataset JSON. Defaults to <root-dir>/multiview_geometry_caption_dataset.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of valid samples to include. Use 0 for all samples.",
    )
    return parser.parse_args()


def build_sample(model_dir: Path) -> dict | None:
    image_paths = [model_dir / filename for filename in EXPECTED_VIEW_FILES]
    if not all(path.is_file() for path in image_paths):
        return None

    return {
        "name": model_dir.name,
        "images": [str(path.resolve()) for path in image_paths],
    }


def collect_samples(root_dir: Path, limit: int) -> tuple[list[dict], int]:
    total_dirs = 0
    samples: list[dict] = []

    for child in tqdm(sorted(root_dir.iterdir()), desc="Collecting samples"):
        if not child.is_dir():
            continue

        total_dirs += 1
        sample = build_sample(child)
        if sample is None:
            continue

        samples.append(sample)
        if limit > 0 and len(samples) >= limit:
            break

    return samples, total_dirs


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    output_path = args.output_path or (root_dir / "multiview_geometry_caption_dataset.json")

    if not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    samples, total_dirs = collect_samples(root_dir, args.limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    skipped_dirs = total_dirs - len(samples)
    print(f"Scanned model directories: {total_dirs}")
    print(f"Valid samples written: {len(samples)}")
    print(f"Skipped directories: {skipped_dirs}")
    print(f"Output dataset path: {output_path}")


if __name__ == "__main__":
    main()
