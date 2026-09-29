from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from compute_language_metrics import (
    ALL_METRIC_KEYS,
    EMBEDDING_METRIC_KEYS,
    LEXICAL_METRIC_KEYS,
    compute_embedding_metrics,
    compute_lexical_metrics,
)


_DEFAULT_GT_JSON = Path("datasets/toys4k/test.json")
_DEFAULT_PRED_ROOT = Path("outputs/toys4k_understanding")


@dataclass(frozen=True)
class EvalEntry:
    asset_id: str
    text_description: str


def _pointllm_reference(item: Mapping[str, object], index: int) -> str:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        raise TypeError(f"PointLLM item {index} has no conversations list.")
    for turn in conversations:
        if not isinstance(turn, Mapping):
            continue
        role = str(turn.get("from", "")).strip().lower()
        if role in {"gpt", "assistant"}:
            return str(turn.get("value") or "").strip()
    raise ValueError(f"PointLLM item {index} has no GPT/assistant reference turn.")


def _load_eval_entries(gt_json: Path, max_assets: int | None = None) -> list[EvalEntry]:
    raw_items = json.loads(gt_json.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raise ValueError(f"Expected {gt_json} to contain a JSON list.")

    entries_by_asset: dict[str, EvalEntry] = {}
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            raise TypeError(f"Expected item {index} in {gt_json} to be a JSON object.")
        if "asset_id" in item:
            asset_id = str(item["asset_id"]).strip()
            reference = str(item.get("text_description") or "").strip()
        elif "object_id" in item:
            asset_id = str(item["object_id"]).strip()
            reference = _pointllm_reference(item, index)
        else:
            raise ValueError(f"Item {index} has neither asset_id nor object_id.")
        if not asset_id or asset_id in entries_by_asset:
            continue
        entries_by_asset[asset_id] = EvalEntry(
            asset_id=asset_id,
            text_description=reference,
        )

    asset_ids = sorted(entries_by_asset)
    if max_assets is not None:
        asset_ids = asset_ids[:max_assets]
    return [entries_by_asset[asset_id] for asset_id in asset_ids]


def _has_txt_files(path: Path) -> bool:
    return path.is_dir() and any(child.is_file() and child.suffix == ".txt" for child in path.iterdir())


def _resolve_method_dirs(pred_dir: Path, methods: Sequence[str] | None) -> list[Path]:
    if methods:
        resolved = []
        for method in methods:
            method_path = Path(method).expanduser()
            if not method_path.is_absolute():
                method_path = pred_dir / method
            resolved.append(method_path)
        return resolved

    if _has_txt_files(pred_dir):
        return [pred_dir]

    return sorted(path for path in pred_dir.iterdir() if path.is_dir() and _has_txt_files(path))


def evaluate_method(
    pred_dir: Path,
    entries: Sequence[EvalEntry],
    missing_policy: str,
    *,
    include_embedding: bool = True,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, object]:
    gt_asset_ids = {entry.asset_id for entry in entries}
    pred_files = {path.stem: path for path in pred_dir.iterdir() if path.is_file() and path.suffix == ".txt"}
    extra_pred_ids = sorted(set(pred_files) - gt_asset_ids)

    active_metric_keys: tuple[str, ...] = (
        ALL_METRIC_KEYS if include_embedding else LEXICAL_METRIC_KEYS
    )
    total = {key: 0.0 for key in active_metric_keys}
    matched_pred_texts: list[str] = []
    matched_ref_texts: list[str] = []
    missing = 0
    empty_prediction = 0
    empty_reference = 0
    missing_asset_ids: list[str] = []

    for entry in entries:
        pred_path = pred_files.get(entry.asset_id)
        if pred_path is None:
            missing += 1
            missing_asset_ids.append(entry.asset_id)
            continue

        pred_text = pred_path.read_text(encoding="utf-8").strip()
        ref_text = entry.text_description
        if not pred_text:
            empty_prediction += 1
        if not ref_text:
            empty_reference += 1

        metrics = compute_lexical_metrics(pred_text, ref_text)
        for key in LEXICAL_METRIC_KEYS:
            total[key] += metrics[key]
        matched_pred_texts.append(pred_text)
        matched_ref_texts.append(ref_text)

    matched = len(matched_pred_texts)

    if include_embedding and matched:
        emb = compute_embedding_metrics(
            matched_pred_texts,
            matched_ref_texts,
            batch_size=batch_size,
            device=device,
        )
        for key in EMBEDDING_METRIC_KEYS:
            if key in emb:
                total[key] = sum(emb[key])

    denominator = len(entries) if missing_policy == "zero" else matched
    averages = {key: (value / denominator if denominator else 0.0) for key, value in total.items()}

    return {
        "method": pred_dir.name,
        "pred_dir": pred_dir.as_posix(),
        "requested_asset_count": len(entries),
        "matched_asset_count": matched,
        "scored_asset_count": denominator,
        "missing_prediction_count": missing,
        "extra_prediction_count": len(extra_pred_ids),
        "empty_prediction_count": empty_prediction,
        "empty_reference_count": empty_reference,
        "missing_prediction_examples": missing_asset_ids[:10],
        "extra_prediction_examples": extra_pred_ids[:10],
        "metric_keys": list(active_metric_keys),
        **averages,
    }


def _print_result(result: dict[str, object]) -> None:
    print(f"Method: {result['method']}")
    print(f"Prediction dir: {result['pred_dir']}")
    print(f"Requested assets: {result['requested_asset_count']}")
    print(f"Matched assets: {result['matched_asset_count']}")
    print(f"Scored assets: {result['scored_asset_count']}")
    print(f"Missing predictions: {result['missing_prediction_count']}")
    print(f"Extra predictions: {result['extra_prediction_count']}")
    print(f"Empty predictions: {result['empty_prediction_count']}")
    for key in result.get("metric_keys", []):
        if key in result:
            print(f"{key}: {float(result[key]):.6f}")


def _write_json(path: Path, results: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(results), indent=2), encoding="utf-8")


def _write_csv(path: Path, results: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "method",
        "pred_dir",
        "requested_asset_count",
        "matched_asset_count",
        "scored_asset_count",
        "missing_prediction_count",
        "extra_prediction_count",
        "empty_prediction_count",
        "empty_reference_count",
    ]
    metric_fields: list[str] = []
    for result in results:
        for key in result.get("metric_keys", []):
            if key not in metric_fields:
                metric_fields.append(key)
    fieldnames = base_fields + metric_fields

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key, "") for key in fieldnames}
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute BLEU-1..4, ROUGE-1/2/L, METEOR, and SBERT/SimCSE similarities "
            "for Toys4k or PointLLM 3D understanding predictions."
        )
    )
    parser.add_argument(
        "--gt_json",
        type=Path,
        default=_DEFAULT_GT_JSON,
        help=(
            "Toys4k JSON with asset_id/text_description, or PointLLM JSON with "
            "object_id/conversations."
        ),
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        default=_DEFAULT_PRED_ROOT,
        help=(
            "Prediction directory. If it contains txt files, it is evaluated as one method; "
            "otherwise each child directory with txt files is evaluated as a method."
        ),
    )
    parser.add_argument(
        "--method",
        action="append",
        help=(
            "Method subdirectory name under --pred_dir, or an absolute prediction directory. "
            "Can be passed multiple times. By default all method directories are evaluated."
        ),
    )
    parser.add_argument(
        "--missing_policy",
        choices=("zero", "skip"),
        default="skip",
        help="How to handle assets from the eval JSON without a prediction txt file.",
    )
    parser.add_argument("--max_assets", type=int, default=None, help="Evaluate only the first N sorted assets.")
    parser.add_argument(
        "--no_embedding",
        action="store_true",
        help="Skip SBERT/SimCSE similarity computation (lexical metrics only).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size used when encoding texts with SBERT/SimCSE.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for embedding models (e.g. 'cuda', 'cuda:0', 'cpu'). Auto-detected if unset.",
    )
    parser.add_argument("--output_json", type=Path, default=None, help="Optional path to save detailed JSON results.")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional path to save summary CSV results.")
    args = parser.parse_args()

    entries = _load_eval_entries(args.gt_json.expanduser(), max_assets=args.max_assets)
    method_dirs = _resolve_method_dirs(args.pred_dir.expanduser(), args.method)
    if not method_dirs:
        raise FileNotFoundError(f"No prediction txt files found under {args.pred_dir}")

    results = []
    for idx, method_dir in enumerate(method_dirs):
        if not method_dir.is_dir():
            raise FileNotFoundError(f"Prediction directory does not exist: {method_dir}")
        result = evaluate_method(
            method_dir,
            entries,
            missing_policy=args.missing_policy,
            include_embedding=not args.no_embedding,
            batch_size=args.batch_size,
            device=args.device,
        )
        results.append(result)
        if idx:
            print()
        _print_result(result)

    if args.output_json is not None:
        _write_json(args.output_json.expanduser(), results)
    if args.output_csv is not None:
        _write_csv(args.output_csv.expanduser(), results)


if __name__ == "__main__":
    main()
