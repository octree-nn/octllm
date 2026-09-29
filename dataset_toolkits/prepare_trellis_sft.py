#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate TRELLIS LLM finetuning data from token, render, and caption assets.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any


DATASET_TYPES = ("description", "understanding", "image")
VALID_IMAGE_INDEXES = tuple(f"{index:03d}.png" for index in range(6, 24))


def load_templates(template_path: str) -> list[dict[str, Any]]:
    """Load templates from a JSON file."""
    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_token_sequence(txt_path: Path) -> str | None:
    """Read token sequence and convert to mesh token format."""
    if not txt_path.exists():
        return None

    try:
        content = txt_path.read_text(encoding="utf-8").strip()
    except Exception as error:
        print(f"Failed to read token file: {txt_path}, error: {error}")
        return None

    return convert_to_mesh_tokens(content)


def convert_to_mesh_tokens(content: str) -> str | None:
    """Convert integer sequence to mesh token format."""
    if not content:
        return None

    try:
        mesh_tokens = []
        for integer_str in content.split():
            integer_str = integer_str.strip()
            if not integer_str:
                continue

            try:
                int(integer_str)
            except ValueError:
                print(f"Warning: found non-integer token: {integer_str}")
                continue

            mesh_tokens.append(f"<mesh{integer_str}>")

        if mesh_tokens:
            return "<mesh_bos>" + "".join(mesh_tokens) + "<mesh_eos>"
        return None
    except Exception as error:
        print(f"Failed to convert token sequence: {error}")
        return None


def normalize_caption(caption: str) -> str:
    """Normalize caption text for dataset generation."""
    normalized = caption.lower().strip()
    if normalized.endswith("."):
        normalized = normalized[:-1]
    return normalized


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate TRELLIS finetuning datasets from paired assets."
    )
    parser.add_argument(
        "--token_dir",
        type=Path,
        required=True,
        help="Directory containing token sequence files named <asset_id>_bytes.txt.",
    )
    parser.add_argument(
        "--render_dir",
        type=Path,
        required=True,
        help="Directory containing per-asset render folders.",
    )
    parser.add_argument(
        "--caption_csv",
        type=Path,
        required=True,
        help="Caption CSV path with columns asset_id and caption.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for generated JSON files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Maximum number of conversations to generate per dataset type.",
    )
    parser.add_argument(
        "--max_file_size",
        type=int,
        default=1 * 1024 * 1024 * 1024,
        help="Maximum JSON file size in bytes before splitting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for template and image sampling.",
    )
    parser.add_argument(
        "--text_template_path",
        type=str,
        default="data/templates/text_to_3d.json",
        help="Template path for text-to-3D dataset type.",
    )
    parser.add_argument(
        "--understanding_template_path",
        type=str,
        default="data/templates/understanding.json",
        help="Template path for 3D-to-text dataset type.",
    )
    parser.add_argument(
        "--image_template_path",
        type=str,
        default="data/templates/image_to_3d.json",
        help="Template path for image-to-3D dataset type.",
    )
    return parser.parse_args()


def load_caption_map(caption_csv: Path) -> dict[str, str]:
    """Load asset_id -> caption mapping from caption CSV."""
    caption_map: dict[str, str] = {}

    with caption_csv.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row_index, row in enumerate(reader, start=2):
            asset_id = (row.get("asset_id") or "").strip()
            caption = (row.get("caption") or "").strip()

            if not asset_id or not caption:
                continue

            if asset_id in caption_map:
                print(f"Warning: duplicate caption row for asset_id={asset_id} at CSV line {row_index}, keep first")
                continue

            caption_map[asset_id] = caption

    return caption_map


def load_token_map(token_dir: Path) -> dict[str, Path]:
    """Load asset_id -> token file path mapping."""
    token_map: dict[str, Path] = {}
    for path in token_dir.glob("*_bytes.txt"):
        if not path.is_file():
            continue
        asset_id = path.name.removesuffix("_bytes.txt")
        if asset_id:
            token_map[asset_id] = path
    return token_map


def load_render_map(render_dir: Path) -> dict[str, list[Path]]:
    """Load asset_id -> usable render image paths mapping."""
    render_map: dict[str, list[Path]] = {}
    for asset_dir in render_dir.iterdir():
        if not asset_dir.is_dir():
            continue

        usable_images = []
        for image_name in VALID_IMAGE_INDEXES:
            image_path = asset_dir / image_name
            if image_path.is_file():
                usable_images.append(image_path)

        render_map[asset_dir.name] = usable_images

    return render_map


def get_template_path(dataset_type: str, args: argparse.Namespace) -> str:
    """Get template path by dataset type."""
    if dataset_type == "description":
        return args.text_template_path
    if dataset_type == "understanding":
        return args.understanding_template_path
    if dataset_type == "image":
        return args.image_template_path
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def generate_conversation(
    template: dict[str, Any],
    caption: str,
    token_sequence: str,
    dataset_type: str,
    image_path: str | None = None,
) -> dict[str, Any]:
    """Generate one conversation sample by dataset type."""
    conversation = {"messages": []}
    object_name = normalize_caption(caption)

    for message in template["messages"]:
        new_message = {"role": message["role"], "content": message["content"]}

        if dataset_type == "description":
            if message["role"] == "user":
                new_message["content"] = new_message["content"].replace("#object_name#", object_name)
            elif message["role"] == "assistant":
                new_message["content"] = new_message["content"].replace("#response#", token_sequence)
        elif dataset_type == "understanding":
            if message["role"] == "user":
                new_message["content"] = new_message["content"].replace("#response#", token_sequence)
            elif message["role"] == "assistant":
                new_message["content"] = new_message["content"].replace("#object_name#", object_name)
        elif dataset_type == "image":
            if message["role"] == "assistant":
                new_message["content"] = new_message["content"].replace("#response#", token_sequence)

        conversation["messages"].append(new_message)

    if dataset_type == "image":
        if image_path is None:
            raise ValueError("image_path is required for image dataset generation")
        conversation["images"] = [image_path]

    return conversation


class DatasetWriter:
    """Write one dataset type into sharded JSON files."""

    def __init__(self, output_dir: Path, dataset_type: str, max_file_size: int) -> None:
        self.output_dir = output_dir
        self.dataset_type = dataset_type
        self.max_file_size = max_file_size
        self.current_data: list[dict[str, Any]] = []
        self.file_counter = 1
        self.total_conversations = 0

    def add(self, conversation: dict[str, Any]) -> None:
        self.current_data.append(conversation)
        self.total_conversations += 1

        if self.total_conversations % 100 == 0:
            self._check_file_size_and_save()

    def finalize(self) -> None:
        self._save_current_file()

    def _output_path(self) -> Path:
        return self.output_dir / f"llm_trellis_dataset_{self.file_counter}_{self.dataset_type}.json"

    def _save_current_file(self) -> None:
        if not self.current_data:
            return

        output_path = self._output_path()
        print(f"Saving {self.dataset_type} results to {output_path}...")
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.current_data, f, ensure_ascii=False, indent=2)

        file_size = output_path.stat().st_size
        print(f"File {output_path} size: {file_size / (1024 * 1024):.2f} MB")

        self.file_counter += 1
        self.current_data = []

    def _check_file_size_and_save(self) -> bool:
        if not self.current_data:
            return False

        estimated_size = len(json.dumps(self.current_data, ensure_ascii=False, indent=2).encode("utf-8"))
        if estimated_size >= self.max_file_size * 0.9:
            self._save_current_file()
            return True
        return False


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate required input paths."""
    if not args.token_dir.is_dir():
        raise FileNotFoundError(f"token_dir does not exist: {args.token_dir}")
    if not args.render_dir.is_dir():
        raise FileNotFoundError(f"render_dir does not exist: {args.render_dir}")
    if not args.caption_csv.is_file():
        raise FileNotFoundError(f"caption_csv does not exist: {args.caption_csv}")


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    random.seed(args.seed)

    print("Loading templates...")
    templates = {dataset_type: load_templates(get_template_path(dataset_type, args)) for dataset_type in DATASET_TYPES}
    for dataset_type, template_list in templates.items():
        print(f"Loaded {len(template_list)} templates for {dataset_type}")

    print("Indexing captions, tokens, and renders...")
    caption_map = load_caption_map(args.caption_csv)
    token_map = load_token_map(args.token_dir)
    render_map = load_render_map(args.render_dir)

    token_ids = set(token_map)
    render_ids = set(render_map)
    caption_ids = set(caption_map)
    common_asset_ids = sorted(token_ids & render_ids & caption_ids)

    print(f"Caption assets: {len(caption_ids)}")
    print(f"Token assets: {len(token_ids)}")
    print(f"Render assets: {len(render_ids)}")
    print(f"Common paired assets: {len(common_asset_ids)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        dataset_type: DatasetWriter(args.output_dir, dataset_type, args.max_file_size) for dataset_type in DATASET_TYPES
    }
    stats = {
        dataset_type: {
            "processed_assets": 0,
            "skipped_assets": 0,
        }
        for dataset_type in DATASET_TYPES
    }

    for asset_index, asset_id in enumerate(common_asset_ids, start=1):
        token_sequence = read_token_sequence(token_map[asset_id])
        caption = caption_map[asset_id].strip()

        if not token_sequence or not caption:
            for dataset_type in DATASET_TYPES:
                stats[dataset_type]["skipped_assets"] += 1
            continue

        if writers["description"].total_conversations < args.limit:
            template = random.choice(templates["description"])
            writers["description"].add(
                generate_conversation(template, caption, token_sequence, dataset_type="description")
            )
            stats["description"]["processed_assets"] += 1

        if writers["understanding"].total_conversations < args.limit:
            template = random.choice(templates["understanding"])
            writers["understanding"].add(
                generate_conversation(template, caption, token_sequence, dataset_type="understanding")
            )
            stats["understanding"]["processed_assets"] += 1

        if writers["image"].total_conversations < args.limit:
            image_candidates = render_map.get(asset_id, [])
            if image_candidates:
                selected_image = random.choice(image_candidates)
                template = random.choice(templates["image"])
                writers["image"].add(
                    generate_conversation(
                        template,
                        caption,
                        token_sequence,
                        dataset_type="image",
                        image_path=os.fspath(selected_image),
                    )
                )
                stats["image"]["processed_assets"] += 1
            else:
                stats["image"]["skipped_assets"] += 1

        if asset_index % 100 == 0:
            print(
                "Processed "
                f"{asset_index}/{len(common_asset_ids)} assets | "
                f"description={writers['description'].total_conversations}, "
                f"understanding={writers['understanding'].total_conversations}, "
                f"image={writers['image'].total_conversations}"
            )

        if all(writer.total_conversations >= args.limit for writer in writers.values()):
            print("All dataset types reached the requested limit. Stop processing.")
            break

    for writer in writers.values():
        writer.finalize()

    print("\n=== Finished ===")
    print(f"output dir: {args.output_dir}")
    print(f"caption csv: {args.caption_csv}")
    print(f"token dir: {args.token_dir}")
    print(f"render dir: {args.render_dir}")
    print(f"common paired assets: {len(common_asset_ids)}")
    for dataset_type in DATASET_TYPES:
        writer = writers[dataset_type]
        print(
            f"{dataset_type}: total_conversations={writer.total_conversations}, "
            f"processed_assets={stats[dataset_type]['processed_assets']}, "
            f"skipped_assets={stats[dataset_type]['skipped_assets']}, "
            f"total_files={writer.file_counter - 1}"
        )


if __name__ == "__main__":
    main()
