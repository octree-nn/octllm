#!/usr/bin/env python3
"""Convert per-asset caption text files to the CSV consumed by SFT preparation."""

import argparse
import csv
from pathlib import Path


def export_captions(input_dir: Path, output_csv: Path) -> int:
    rows = []
    for path in sorted(input_dir.glob("*.txt")):
        caption = path.read_text(encoding="utf-8").strip()
        if caption:
            rows.append({"asset_id": path.stem, "caption": caption})
    if not rows:
        raise ValueError(f"No non-empty caption text files in {input_dir}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("asset_id", "caption"))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    count = export_captions(args.input_dir, args.output_csv)
    print(f"Exported {count} captions to {args.output_csv}")


if __name__ == "__main__":
    main()
