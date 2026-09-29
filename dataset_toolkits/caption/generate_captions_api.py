#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import csv
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:
    from openai import OpenAI
else:
    OpenAI = Any


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-plus"
DEFAULT_SUBSETS = ("ObjaverseXL_sketchfab")
DEFAULT_OUTPUT_FILENAME = "qwen_vl_plus_geometry_captions.csv"
DEFAULT_PROMPT = (
    "Describe this 3D object in English using only its geometry and shape.\n"
    "Ignore color, material, texture, lighting, and background.\n"
    "Focus on overall structure, major parts, proportions, and spatial arrangement.\n"
    "Keep the description moderately detailed and concise.\n"
    'Directly output the description without any other text. For example: "A spider with multiple legs and a segmented body."\n'
    "The description should not longer than 100 words."
)
CSV_FIELDNAMES = ("sha256", "text")
TARGET_NUM_VIEWS = 8
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3
MAX_DESCRIPTION_WORDS = 100


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate geometry-only captions for Objaverse renders with qwen-vl-plus."
    )
    parser.add_argument(
        "--objaverse-root",
        type=Path,
        default=Path("datasets/meshes"),
        help="Root directory that contains Objaverse subset folders.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        help="Subset names to scan. Defaults to the rendered subsets used in this workflow.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("datasets/trellis/geometry_captions_qwen35_plus.csv"),
        help="Output CSV filename or path. Defaults to writing geometry_captions_qwen35_plus.csv inside each subset folder.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="DashScope model name.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt sent together with the multiview images.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="Optional max number of new samples to process in this run. Use 0 for all samples.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows that are already present in the output CSV.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Number of API retries before writing an api_error row.",
    )
    args = parser.parse_args(argv)
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1.")

    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0.")

    return args


def init_client() -> tuple[OpenAI | None, str | None]:
    try:
        from openai import OpenAI as OpenAIClient

        client = OpenAIClient(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=DASHSCOPE_BASE_URL,
        )
    except Exception as exc:  # pragma: no cover - depends on local env/openai behavior
        return None, f"Failed to initialize OpenAI client: {exc}"

    return client, None


def load_processed_keys(output_csv: Path) -> set[str]:
    if not output_csv.is_file():
        return set()

    processed_keys: set[str] = set()
    with output_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            sha256 = (row.get("sha256") or "").strip()
            if sha256:
                processed_keys.add(sha256)

    return processed_keys


def ensure_output_csv(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.is_file() and output_csv.stat().st_size > 0:
        return

    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()


def iter_metadata_rows(metadata_csv: Path) -> Iterable[dict[str, str]]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            yield row


def collect_render_images(render_dir: Path) -> list[Path]:
    return sorted(path.resolve() for path in render_dir.glob("*.png") if path.is_file())


def select_view_indices(num_views: int, target_num_views: int = TARGET_NUM_VIEWS) -> list[int]:
    if num_views < target_num_views:
        raise ValueError(f"Need at least {target_num_views} views, got {num_views}.")

    if num_views == target_num_views:
        return list(range(target_num_views))

    return [(index * (num_views - 1)) // (target_num_views - 1) for index in range(target_num_views)]


def select_view_paths(image_paths: Sequence[Path], target_num_views: int = TARGET_NUM_VIEWS) -> list[Path]:
    indices = select_view_indices(len(image_paths), target_num_views=target_num_views)
    return [image_paths[index] for index in indices]


def encode_image_as_data_url(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_messages(image_paths: Sequence[Path], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_as_data_url(image_path)},
            }
        )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return " ".join(part for part in parts if part).strip()

    return str(content or "").strip()


def clean_generated_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    for prefix in ("assistant:", "assistant", "description:", "caption:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    cleaned = cleaned.strip().strip("\"'").strip()
    return " ".join(cleaned.split())


def truncate_english_words(text: str, max_words: int = MAX_DESCRIPTION_WORDS) -> str:
    tokens = text.split()
    if not tokens:
        return ""

    kept_tokens: list[str] = []
    english_word_count = 0
    for token in tokens:
        kept_tokens.append(token)
        if re.search(r"[A-Za-z]", token):
            english_word_count += 1
        if english_word_count >= max_words:
            break

    return " ".join(kept_tokens)


def generate_geometry_description(
    client: OpenAI,
    model: str,
    image_paths: Sequence[Path],
    prompt: str,
    request_timeout: float,
    max_retries: int,
) -> str:
    messages = build_messages(image_paths=image_paths, prompt=prompt)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=200,
                temperature=0.2,
                extra_body={"enable_thinking": False},
                timeout=request_timeout,
            )
            message = completion.choices[0].message
            text = extract_message_text(message.content)
            return truncate_english_words(clean_generated_text(text))
        except Exception as exc:  # pragma: no cover - exercised by mocked failures in tests
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(2**attempt)

    raise RuntimeError(str(last_error) if last_error is not None else "Unknown API error.")


def format_image_paths(image_paths: Sequence[Path]) -> str:
    return "|".join(str(path.resolve()) for path in image_paths)


def resolve_output_csv_path(output_arg: Path | None, subset_dir: Path, num_subsets: int) -> Path:
    if output_arg is None:
        return subset_dir / DEFAULT_OUTPUT_FILENAME

    output_arg = output_arg.expanduser()
    if output_arg.parent == Path("."):
        return subset_dir / output_arg.name

    if num_subsets > 1:
        raise ValueError(
            "--output-csv must be a filename without parent directories when processing multiple subsets."
        )

    return output_arg.resolve()


def process_metadata_row(
    subset: str,
    subset_dir: Path,
    metadata_row: dict[str, str],
    client: OpenAI | None,
    client_error: str | None,
    model: str,
    prompt: str,
    request_timeout: float,
    max_retries: int,
) -> tuple[dict[str, str] | None, str, str]:
    sha256 = (metadata_row.get("sha256") or "").strip()
    render_dir = subset_dir / "renders_cond" / sha256
    if not render_dir.is_dir():
        return None, "missing_render_dir", f"Render directory not found: {render_dir}"

    all_image_paths = collect_render_images(render_dir)
    if len(all_image_paths) < TARGET_NUM_VIEWS:
        return (
            None,
            "missing_views",
            f"Need at least {TARGET_NUM_VIEWS} png views, found {len(all_image_paths)} in {render_dir}.",
        )

    selected_image_paths = select_view_paths(all_image_paths)
    if client is None:
        return None, "api_error", client_error or "OpenAI client is unavailable."

    try:
        description = generate_geometry_description(
            client=client,
            model=model,
            image_paths=selected_image_paths,
            prompt=prompt,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
        return {"sha256": sha256, "text": description}, "ok", ""
    except Exception as exc:
        return (
            None,
            "api_error",
            f"{exc}. Selected views: {format_image_paths(selected_image_paths)}",
        )


def append_output_row(output_csv: Path, row: dict[str, Any]) -> None:
    with output_csv.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)


def validate_subset_dir(subset_dir: Path) -> None:
    metadata_csv = subset_dir / "metadata.csv"
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"Subset directory does not exist: {subset_dir}")
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"metadata.csv does not exist: {metadata_csv}")


def run(args: argparse.Namespace) -> dict[str, int]:
    objaverse_root = args.objaverse_root.expanduser().resolve()
    subsets = ["ObjaverseXL_sketchfab"]
    client, client_error = init_client()

    processed_count = 0
    status_counts = {
        "ok": 0,
        "missing_render_dir": 0,
        "missing_views": 0,
        "api_error": 0,
        "skipped_resume": 0,
    }
    print(f"Processing {len(subsets)} subsets: {subsets}")
    for subset in subsets:
        subset_dir = objaverse_root / subset
        validate_subset_dir(subset_dir)
        metadata_csv = subset_dir / "metadata.csv"
        output_csv = resolve_output_csv_path(args.output_csv, subset_dir, len(subsets))
        processed_keys = load_processed_keys(output_csv) if args.resume else set()
        ensure_output_csv(output_csv)
        print(f"Scanning subset: {subset}")

        for metadata_row in iter_metadata_rows(metadata_csv):
            sha256 = (metadata_row.get("sha256") or "").strip()
            if not sha256:
                continue

            if sha256 in processed_keys:
                status_counts["skipped_resume"] += 1
                continue

            row, status, error = process_metadata_row(
                subset=subset,
                subset_dir=subset_dir,
                metadata_row=metadata_row,
                client=client,
                client_error=client_error,
                model=args.model,
                prompt=args.prompt,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            processed_count += 1
            status_counts[status] += 1
            if row is not None:
                append_output_row(output_csv, row)
                processed_keys.add(sha256)
            else:
                print(f"[{subset}] {sha256} -> {status}: {error}")

            if args.max_samples > 0 and processed_count >= args.max_samples:
                print("Reached --max-samples limit, stopping early.")
                return status_counts

    return status_counts


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    status_counts = run(args)
    print("Finished.")
    for status, count in status_counts.items():
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
