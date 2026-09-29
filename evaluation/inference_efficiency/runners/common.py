"""Shared manifest and result I/O for inference-efficiency runners.

The benchmark writes one JSON file per asset.  Keeping samples independent makes
an interrupted run resumable and avoids losing completed measurements when a
later model invocation fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "inference-efficiency.v1"
_SAFE_ASSET_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ManifestSample:
    """A single normalized line from the benchmark JSONL manifest."""

    sample_order: int
    dataset_index: int
    asset_id: str
    text_description: str
    mesh_path: str | None = None
    render_image_path: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_asset_id(asset_id: str, *, max_length: int = 120) -> str:
    """Return a stable filename-safe rendering of an asset identifier."""

    safe = _SAFE_ASSET_RE.sub("_", asset_id.strip()).strip("._-")
    if not safe:
        safe = "asset"
    return safe[:max_length]


def sample_output_path(output_dir: Path, sample: ManifestSample) -> Path:
    return output_dir / f"{sample.sample_order:03d}_{safe_asset_id(sample.asset_id)}.json"


def _required_int(row: Mapping[str, Any], key: str, line_number: int) -> int:
    if key not in row:
        raise ValueError(f"manifest line {line_number}: missing {key!r}")
    value = row[key]
    if isinstance(value, bool):
        raise ValueError(f"manifest line {line_number}: {key!r} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"manifest line {line_number}: {key!r} must be an integer"
        ) from exc


def _required_string(row: Mapping[str, Any], key: str, line_number: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"manifest line {line_number}: {key!r} must be a non-empty string"
        )
    return value.strip()


def _optional_string(row: Mapping[str, Any], key: str, line_number: int) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"manifest line {line_number}: {key!r} must be a string or null")
    return value


def load_manifest(path: Path) -> list[ManifestSample]:
    """Load and strictly validate the standardized JSONL manifest."""

    samples: list[ManifestSample] = []
    seen_orders: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"manifest line {line_number}: expected a JSON object")

            sample_order = _required_int(row, "sample_order", line_number)
            if sample_order < 0:
                raise ValueError(f"manifest line {line_number}: sample_order must be >= 0")
            if sample_order in seen_orders:
                raise ValueError(
                    f"manifest line {line_number}: duplicate sample_order {sample_order}"
                )
            seen_orders.add(sample_order)

            samples.append(
                ManifestSample(
                    sample_order=sample_order,
                    dataset_index=_required_int(row, "dataset_index", line_number),
                    asset_id=_required_string(row, "asset_id", line_number),
                    text_description=_required_string(
                        row, "text_description", line_number
                    ),
                    mesh_path=_optional_string(row, "mesh_path", line_number),
                    render_image_path=_optional_string(
                        row, "render_image_path", line_number
                    ),
                )
            )

    if not samples:
        raise ValueError(f"manifest contains no samples: {path}")
    return sorted(samples, key=lambda sample: sample.sample_order)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace ``path`` with a complete JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_existing_result(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"existing result is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"existing result is not a JSON object: {path}")
    return payload


def result_matches(
    payload: Mapping[str, Any],
    *,
    config_hash: str,
    method: str,
    sample: ManifestSample,
) -> bool:
    """Check whether an existing per-asset result is safe to resume from."""

    return (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("config_hash") == config_hash
        and payload.get("method") == method
        and payload.get("sample_order") == sample.sample_order
        and payload.get("dataset_index") == sample.dataset_index
        and payload.get("asset_id") == sample.asset_id
        and payload.get("text_description_sha256")
        == sha256_text(sample.text_description)
    )


def pending_samples(
    samples: Iterable[ManifestSample],
    *,
    output_dir: Path,
    config_hash: str,
    method: str,
    overwrite: bool,
) -> tuple[list[ManifestSample], int]:
    """Return samples to run and the number safely resumed from disk."""

    pending: list[ManifestSample] = []
    skipped = 0
    for sample in samples:
        path = sample_output_path(output_dir, sample)
        existing = read_existing_result(path)
        if existing is None or overwrite:
            pending.append(sample)
            continue
        if result_matches(
            existing,
            config_hash=config_hash,
            method=method,
            sample=sample,
        ):
            if existing.get("status") == "ok":
                skipped += 1
            else:
                pending.append(sample)
            continue
        raise RuntimeError(
            "refusing to overwrite a result whose identity or config hash differs: "
            f"{path}; pass --overwrite to replace it"
        )
    return pending, skipped


def base_result(
    *,
    method: str,
    config_hash: str,
    sample: ManifestSample,
) -> dict[str, Any]:
    """Fields shared by successful and failed per-asset results."""

    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "status": "pending",
        "config_hash": config_hash,
        "sample_order": sample.sample_order,
        "dataset_index": sample.dataset_index,
        "asset_id": sample.asset_id,
        "text_description_sha256": sha256_text(sample.text_description),
        "latency_ms": None,
        "cuda_latency_ms": None,
        "wall_latency_ms": None,
        "prompt_token_count": None,
        "output_token_count": None,
        "structure_token_count": None,
        "autoregressive_steps": None,
        "token_definition": None,
        "termination": None,
        "truncated": None,
        "error": None,
        "model": None,
        "source": None,
        "runtime": None,
        "created_at": utc_now_iso(),
    }
