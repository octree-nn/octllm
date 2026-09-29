import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


_SELECTED_VIEWS = ("014", "015", "016", "017")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_TEXT_KEYWORDS = ("text",)
_IMAGE_KEYWORDS = ("image", "img")


def _ensure_repo_root_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_path()


ConditionType = Literal["text", "image"]


@dataclass(frozen=True)
class EvalEntry:
    asset_id: str
    text_description: str
    render_image_path: Path


@dataclass(frozen=True)
class TextConditionItem:
    asset_id: str
    text: str
    generated_image_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ImageConditionItem:
    asset_id: str
    condition_image_path: Path
    condition_view: str
    generated_image_paths: tuple[Path, ...]
    generated_view_names: tuple[str, ...]
    same_view_generated_image_path: Optional[Path]


def _mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return mean, std


def _clip_score_from_cosine(value: float) -> float:
    return float(max(value, 0.0) * 100.0)


def _load_eval_entries(split_json: Path, max_assets: Optional[int] = None) -> list[EvalEntry]:
    raw_items = json.loads(split_json.read_text(encoding="utf-8"))
    entries_by_asset: dict[str, EvalEntry] = {}

    for item in raw_items:
        asset_id = str(item["asset_id"])
        if asset_id in entries_by_asset:
            continue
        entries_by_asset[asset_id] = EvalEntry(
            asset_id=asset_id,
            text_description=str(item.get("text_description") or "").strip(),
            render_image_path=Path(item.get("render_image_path") or "").expanduser(),
        )

    asset_ids = sorted(entries_by_asset)
    if max_assets is not None:
        asset_ids = asset_ids[:max_assets]

    return [entries_by_asset[asset_id] for asset_id in asset_ids]


def _parse_generated_render_filename(path: Path) -> Optional[tuple[str, str]]:
    if path.suffix.lower() not in _IMAGE_EXTS:
        return None
    parts = path.stem.split("__")
    if len(parts) < 3:
        return None
    asset_id = parts[0]
    view_name = parts[-1]
    if not asset_id or not view_name:
        return None
    return asset_id, view_name


def _iter_generated_render_files(rendered_dir: Path) -> Iterable[Path]:
    for path in sorted(rendered_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
            yield path


def _index_generated_renders(rendered_dir: Path) -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {}
    for path in _iter_generated_render_files(rendered_dir):
        parsed = _parse_generated_render_filename(path)
        if parsed is None:
            continue
        asset_id, view_name = parsed
        index.setdefault(asset_id, {})[view_name] = path
    return index


def infer_condition_type_from_dir_name(dir_name: str) -> ConditionType:
    lowered = dir_name.lower()
    if any(keyword in lowered for keyword in _IMAGE_KEYWORDS):
        return "image"
    if any(keyword in lowered for keyword in _TEXT_KEYWORDS):
        return "text"
    raise ValueError(
        f"Unable to infer condition type from directory name: {dir_name}. "
        "Use --condition_type_overrides dir_name=text|image."
    )


def _parse_condition_type_overrides(values: Optional[Sequence[str]]) -> dict[str, ConditionType]:
    overrides: dict[str, ConditionType] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid override '{value}'. Expected format: <dir_name>=text|image")
        dir_name, condition_type = value.split("=", 1)
        normalized = condition_type.strip().lower()
        if normalized not in {"text", "image"}:
            raise ValueError(f"Invalid condition type in override '{value}'. Expected text or image.")
        overrides[dir_name.strip()] = "text" if normalized == "text" else "image"
    return overrides


def _resolve_condition_dirs(method_dir: Path, requested: Optional[Sequence[str]]) -> list[Path]:
    if requested:
        resolved: list[Path] = []
        for value in requested:
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                resolved.append(candidate.resolve())
            else:
                resolved.append((method_dir / value).resolve())
        return resolved

    discovered = sorted(path.resolve() for path in method_dir.iterdir() if path.is_dir() and path.name.endswith("rendered"))
    if discovered:
        return discovered
    if any(path.is_file() and path.suffix.lower() in _IMAGE_EXTS for path in method_dir.iterdir()):
        return [method_dir.resolve()]
    return []


def _select_generated_views(
    generated_by_view: dict[str, Path],
    selected_views: Sequence[str],
    use_all_views: bool,
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    if use_all_views:
        ordered_views = tuple(sorted(generated_by_view))
    else:
        ordered_views = tuple(view for view in selected_views if view in generated_by_view)
    ordered_paths = tuple(generated_by_view[view] for view in ordered_views)
    return ordered_views, ordered_paths


def collect_text_condition_items(
    split_json: Path,
    generated_render_dir: Path,
    selected_views: Sequence[str],
    max_assets: Optional[int] = None,
) -> tuple[list[TextConditionItem], dict[str, object]]:
    entries = _load_eval_entries(split_json, max_assets=max_assets)
    generated_index = _index_generated_renders(generated_render_dir)

    items: list[TextConditionItem] = []
    missing_text: list[str] = []
    missing_generated: list[str] = []
    partial_generated: list[str] = []

    for entry in entries:
        if not entry.text_description:
            missing_text.append(entry.asset_id)
            continue

        generated_by_view = generated_index.get(entry.asset_id, {})
        ordered_views, ordered_paths = _select_generated_views(
            generated_by_view,
            selected_views=selected_views,
            use_all_views=False,
        )
        if not ordered_paths:
            missing_generated.append(entry.asset_id)
            continue
        if len(ordered_views) < len(selected_views):
            partial_generated.append(entry.asset_id)

        items.append(
            TextConditionItem(
                asset_id=entry.asset_id,
                text=entry.text_description,
                generated_image_paths=ordered_paths,
            )
        )

    summary: dict[str, object] = {
        "condition_type": "text",
        "requested_asset_count": len(entries),
        "paired_asset_count": len(items),
        "selected_views": list(selected_views),
        "missing_text_count": len(missing_text),
        "missing_generated_count": len(missing_generated),
        "partial_generated_count": len(partial_generated),
        "missing_text_examples": missing_text[:10],
        "missing_generated_examples": missing_generated[:10],
        "partial_generated_examples": partial_generated[:10],
    }
    return items, summary


def collect_image_condition_items(
    split_json: Path,
    generated_render_dir: Path,
    selected_views: Sequence[str],
    use_all_views: bool,
    max_assets: Optional[int] = None,
) -> tuple[list[ImageConditionItem], dict[str, object]]:
    entries = _load_eval_entries(split_json, max_assets=max_assets)
    generated_index = _index_generated_renders(generated_render_dir)

    items: list[ImageConditionItem] = []
    missing_condition_image: list[str] = []
    missing_generated: list[str] = []
    partial_generated: list[str] = []
    same_view_available = 0

    for entry in entries:
        if not entry.render_image_path.is_file():
            missing_condition_image.append(entry.asset_id)
            continue

        generated_by_view = generated_index.get(entry.asset_id, {})
        ordered_views, ordered_paths = _select_generated_views(
            generated_by_view,
            selected_views=selected_views,
            use_all_views=use_all_views,
        )
        if not ordered_paths:
            missing_generated.append(entry.asset_id)
            continue
        if not use_all_views and len(ordered_views) < len(selected_views):
            partial_generated.append(entry.asset_id)

        condition_view = entry.render_image_path.stem
        same_view_generated_image_path = generated_by_view.get(condition_view)
        if same_view_generated_image_path is not None:
            same_view_available += 1

        items.append(
            ImageConditionItem(
                asset_id=entry.asset_id,
                condition_image_path=entry.render_image_path,
                condition_view=condition_view,
                generated_image_paths=ordered_paths,
                generated_view_names=ordered_views,
                same_view_generated_image_path=same_view_generated_image_path,
            )
        )

    summary: dict[str, object] = {
        "condition_type": "image",
        "requested_asset_count": len(entries),
        "paired_asset_count": len(items),
        "selected_views": list(selected_views),
        "generated_view_mode": "all" if use_all_views else "selected",
        "missing_condition_image_count": len(missing_condition_image),
        "missing_generated_count": len(missing_generated),
        "partial_generated_count": len(partial_generated),
        "same_view_available_count": same_view_available,
        "missing_condition_image_examples": missing_condition_image[:10],
        "missing_generated_examples": missing_generated[:10],
        "partial_generated_examples": partial_generated[:10],
    }
    return items, summary


class ClipFeatureExtractor:
    def __init__(self, model_name_or_path: str, device: str):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Missing dependency: torch") from exc

        try:
            from transformers import AutoProcessor, CLIPModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Missing dependency: transformers") from exc

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(device)
        self.model.eval()
        self.backend = "transformers"

    def encode_texts(self, texts: Sequence[str], batch_size: int, progress_desc: str) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size), desc=progress_desc):
            batch_texts = list(texts[start : start + batch_size])
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with self.torch.no_grad():
                features = self.model.get_text_features(**inputs)
            features = self.torch.nn.functional.normalize(features, dim=-1)
            batches.append(features.detach().cpu().numpy().astype(np.float64, copy=False))
        return np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float64)

    def encode_images(self, image_paths: Sequence[Path], batch_size: int, progress_desc: str) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in tqdm(range(0, len(image_paths), batch_size), desc=progress_desc):
            batch_paths = image_paths[start : start + batch_size]
            batch_images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    batch_images.append(image.convert("RGB"))
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                features = self.model.get_image_features(**inputs)
            features = self.torch.nn.functional.normalize(features, dim=-1)
            batches.append(features.detach().cpu().numpy().astype(np.float64, copy=False))
        return np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float64)


def summarize_text_condition_scores(per_asset_view_cosines: Sequence[Sequence[float]]) -> dict[str, float]:
    asset_mean_cosines = [float(np.mean(view_cosines)) for view_cosines in per_asset_view_cosines if len(view_cosines) > 0]
    asset_mean_scores = [_clip_score_from_cosine(value) for value in asset_mean_cosines]
    mean_cosine, std_cosine = _mean_and_std(asset_mean_cosines)
    mean_score, std_score = _mean_and_std(asset_mean_scores)
    mean_view_count, _ = _mean_and_std([float(len(view_cosines)) for view_cosines in per_asset_view_cosines if len(view_cosines) > 0])
    return {
        "asset_clip_cosine_mean": mean_cosine,
        "asset_clip_cosine_std": std_cosine,
        "asset_clip_score_mean": mean_score,
        "asset_clip_score_std": std_score,
        "asset_view_count_mean": mean_view_count,
    }


def summarize_image_condition_scores(
    per_asset_max_cosines: Sequence[float],
    per_asset_mean_view_cosines: Sequence[float],
    per_asset_same_view_cosines: Sequence[float],
) -> dict[str, float]:
    max_mean, max_std = _mean_and_std(per_asset_max_cosines)
    mean_views_mean, mean_views_std = _mean_and_std(per_asset_mean_view_cosines)
    same_view_mean, same_view_std = _mean_and_std(per_asset_same_view_cosines)
    return {
        "asset_clip_cosine_max_mean": max_mean,
        "asset_clip_cosine_max_std": max_std,
        "asset_clip_score_max_mean": _mean_and_std([_clip_score_from_cosine(value) for value in per_asset_max_cosines])[0],
        "asset_clip_score_max_std": _mean_and_std([_clip_score_from_cosine(value) for value in per_asset_max_cosines])[1],
        "asset_clip_cosine_mean_views_mean": mean_views_mean,
        "asset_clip_cosine_mean_views_std": mean_views_std,
        "asset_clip_score_mean_views_mean": _mean_and_std([_clip_score_from_cosine(value) for value in per_asset_mean_view_cosines])[0],
        "asset_clip_score_mean_views_std": _mean_and_std([_clip_score_from_cosine(value) for value in per_asset_mean_view_cosines])[1],
        "asset_clip_cosine_same_view_mean": same_view_mean,
        "asset_clip_cosine_same_view_std": same_view_std,
        "asset_clip_score_same_view_mean": _mean_and_std([_clip_score_from_cosine(value) for value in per_asset_same_view_cosines])[0],
        "asset_clip_score_same_view_std": _mean_and_std([_clip_score_from_cosine(value) for value in per_asset_same_view_cosines])[1],
        "same_view_asset_count": int(len(per_asset_same_view_cosines)),
    }


def _compute_text_item_view_cosines(
    text_features: np.ndarray,
    image_features: np.ndarray,
    items: Sequence[TextConditionItem],
) -> list[list[float]]:
    per_asset_view_cosines: list[list[float]] = []
    offset = 0
    for item_index, item in enumerate(items):
        view_count = len(item.generated_image_paths)
        batch_features = image_features[offset : offset + view_count]
        offset += view_count
        cosines = batch_features @ text_features[item_index]
        per_asset_view_cosines.append(cosines.astype(np.float64, copy=False).tolist())
    return per_asset_view_cosines


def compute_image_asset_metrics(
    reference_feature: np.ndarray,
    generated_features: np.ndarray,
    generated_view_names: Sequence[str],
    condition_view: str,
) -> dict[str, float]:
    cosines = generated_features @ reference_feature
    result: dict[str, float] = {
        "max_cosine": float(np.max(cosines)),
        "mean_views_cosine": float(np.mean(cosines)),
    }
    for index, view_name in enumerate(generated_view_names):
        if view_name == condition_view:
            result["same_view_cosine"] = float(cosines[index])
            break
    return result


def compute_text_clip_metrics(
    items: Sequence[TextConditionItem],
    extractor: ClipFeatureExtractor,
    batch_size: int,
    progress_prefix: str,
) -> dict[str, float]:
    all_texts = [item.text for item in items]
    all_generated_images = [path for item in items for path in item.generated_image_paths]
    text_features = extractor.encode_texts(all_texts, batch_size=batch_size, progress_desc=f"CLIP text [{progress_prefix}]")
    image_features = extractor.encode_images(
        all_generated_images,
        batch_size=batch_size,
        progress_desc=f"CLIP image [{progress_prefix}] Generated",
    )
    per_asset_view_cosines = _compute_text_item_view_cosines(text_features, image_features, items)
    metrics = summarize_text_condition_scores(per_asset_view_cosines)
    metrics["pair_count"] = int(sum(len(values) for values in per_asset_view_cosines))
    metrics["clip_embedding_dim"] = int(text_features.shape[1])
    return metrics


def compute_image_clip_metrics(
    items: Sequence[ImageConditionItem],
    extractor: ClipFeatureExtractor,
    batch_size: int,
    progress_prefix: str,
) -> dict[str, float]:
    all_condition_images = [item.condition_image_path for item in items]
    all_generated_images = [path for item in items for path in item.generated_image_paths]
    condition_features = extractor.encode_images(
        all_condition_images,
        batch_size=batch_size,
        progress_desc=f"CLIP image [{progress_prefix}] Condition",
    )
    generated_features = extractor.encode_images(
        all_generated_images,
        batch_size=batch_size,
        progress_desc=f"CLIP image [{progress_prefix}] Generated",
    )

    per_asset_max_cosines: list[float] = []
    per_asset_mean_view_cosines: list[float] = []
    per_asset_same_view_cosines: list[float] = []
    offset = 0
    for item_index, item in enumerate(items):
        view_count = len(item.generated_image_paths)
        batch_features = generated_features[offset : offset + view_count]
        offset += view_count
        asset_metrics = compute_image_asset_metrics(
            reference_feature=condition_features[item_index],
            generated_features=batch_features,
            generated_view_names=item.generated_view_names,
            condition_view=item.condition_view,
        )
        per_asset_max_cosines.append(asset_metrics["max_cosine"])
        per_asset_mean_view_cosines.append(asset_metrics["mean_views_cosine"])
        if "same_view_cosine" in asset_metrics:
            per_asset_same_view_cosines.append(asset_metrics["same_view_cosine"])

    metrics = summarize_image_condition_scores(
        per_asset_max_cosines=per_asset_max_cosines,
        per_asset_mean_view_cosines=per_asset_mean_view_cosines,
        per_asset_same_view_cosines=per_asset_same_view_cosines,
    )
    metrics["pair_count"] = int(sum(len(item.generated_image_paths) for item in items))
    metrics["clip_embedding_dim"] = int(condition_features.shape[1])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Toys4k CLIP alignment metrics for text-conditioned and image-conditioned generations."
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default="datasets/toys4k/test.json",
        help="Path to the Toys4k evaluation split JSON.",
    )
    parser.add_argument(
        "--method_dir",
        type=str,
        default="outputs/toys4k",
        help="Method directory containing rendered generation results.",
    )
    parser.add_argument(
        "--condition_dirs",
        nargs="*",
        default=None,
        help="Optional rendered subdirectories to evaluate.",
    )
    parser.add_argument(
        "--condition_type_overrides",
        nargs="*",
        default=None,
        help="Optional per-directory overrides like image_conditionrendered=image.",
    )
    parser.add_argument(
        "--selected_views",
        nargs="+",
        default=list(_SELECTED_VIEWS),
        help="View indices used for text-conditioned evaluation and optional image-conditioned filtering.",
    )
    parser.add_argument(
        "--image_condition_use_all_views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all generated views for image-conditioned max/mean aggregation.",
    )
    parser.add_argument(
        "--clip_model_name_or_path",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model name or path.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for CLIP feature extraction, e.g. cpu/cuda/cuda:0.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for CLIP feature extraction.")
    parser.add_argument("--max_assets", type=int, default=None, help="Optional cap for quick debugging runs.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save metrics as JSON.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency: torch") from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    split_json = Path(args.split_json).expanduser().resolve()
    method_dir = Path(args.method_dir).expanduser().resolve()
    condition_dirs = _resolve_condition_dirs(method_dir, args.condition_dirs)
    if not condition_dirs:
        raise ValueError(f"No rendered condition directories found under: {method_dir}")

    overrides = _parse_condition_type_overrides(args.condition_type_overrides)
    extractor = ClipFeatureExtractor(model_name_or_path=args.clip_model_name_or_path, device=device)

    results: dict[str, object] = {
        "split_json": split_json.as_posix(),
        "method_dir": method_dir.as_posix(),
        "device": device,
        "clip_model_name_or_path": args.clip_model_name_or_path,
        "clip_backend": extractor.backend,
        "selected_views": list(args.selected_views),
        "image_condition_use_all_views": bool(args.image_condition_use_all_views),
        "results": {},
    }

    for condition_dir in condition_dirs:
        condition_type = overrides.get(condition_dir.name) or infer_condition_type_from_dir_name(condition_dir.name)
        if condition_type == "text":
            items, summary = collect_text_condition_items(
                split_json=split_json,
                generated_render_dir=condition_dir,
                selected_views=args.selected_views,
                max_assets=args.max_assets,
            )
            if not items:
                raise ValueError(f"No valid text-conditioned evaluation pairs found for: {condition_dir}")
            metrics = compute_text_clip_metrics(
                items=items,
                extractor=extractor,
                batch_size=args.batch_size,
                progress_prefix=condition_dir.name,
            )
        else:
            items, summary = collect_image_condition_items(
                split_json=split_json,
                generated_render_dir=condition_dir,
                selected_views=args.selected_views,
                use_all_views=bool(args.image_condition_use_all_views),
                max_assets=args.max_assets,
            )
            if not items:
                raise ValueError(f"No valid image-conditioned evaluation pairs found for: {condition_dir}")
            metrics = compute_image_clip_metrics(
                items=items,
                extractor=extractor,
                batch_size=args.batch_size,
                progress_prefix=condition_dir.name,
            )

        condition_result = dict(summary)
        condition_result.update(metrics)
        results["results"][condition_dir.name] = condition_result

    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output_json:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
