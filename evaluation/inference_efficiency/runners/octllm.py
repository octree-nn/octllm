#!/usr/bin/env python3
"""Benchmark only OctLLM's custom autoregressive token-generation loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_octree import _load_inference_runtime, _predict_next_token  # noqa: E402


METHOD = "OctLLM"
SCHEMA_VERSION = "inference-efficiency.v1"
DEFAULT_PROMPT_PREFIX = "Generate a 3D asset based on the following description:"
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant specialized in 3D asset generation and understanding, image understanding and chatting. When you are asked to generate a 3D asset from an image, you should reply in formats like: "I've produced a 3D model based on the image: ", "Based on the image, here's the 3D mesh asset I've created: ", etc. And if you are asked to generate a 3D asset from a text description, you should reply in formats like: "I've produced a 3D mesh asset based on your description: ", "Of course! I've generated a 3D mesh based on your text prompt: ", etc. If you are asked to describe a 3D asset, just describe it in detail. The text can be varied, but the colon ":" must be included."""


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "text_description" not in row or "asset_id" not in row:
                raise ValueError(f"{path}:{line_number}: missing asset_id or text_description")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def _safe_name(row: dict[str, Any]) -> str:
    asset = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["asset_id"])).strip("._") or "asset"
    return f"{int(row['sample_order']):03d}_{asset}.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_matching_result(path: Path, config_hash: str, row: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    description_hash = hashlib.sha256(str(row["text_description"]).encode("utf-8")).hexdigest()
    matches = (
        existing.get("schema_version") == SCHEMA_VERSION
        and existing.get("method") == METHOD
        and existing.get("config_hash") == config_hash
        and existing.get("sample_order") == int(row["sample_order"])
        and existing.get("dataset_index") == int(row["dataset_index"])
        and existing.get("asset_id") == str(row["asset_id"])
        and existing.get("text_description_sha256") == description_hash
    )
    return existing if matches else None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _runtime_metadata(config_path: Path, runtime: Any) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(runtime.model.device) if runtime.model.device.type == "cuda" else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "gpu": gpu_name,
        "device": str(runtime.model.device),
        "dtype": str(runtime.model.dtype),
        "config_path": str(config_path.resolve()),
    }


def _termination(result: dict[str, Any], runtime: Any, num_new_tokens: int) -> tuple[str, bool]:
    ids = result["generated_ids"]
    mesh_eos = runtime.tokenizer.convert_tokens_to_ids("<mesh_eos>")
    if ids and ids[-1] == mesh_eos:
        return "mesh_eos", False
    if ids and ids[-1] == runtime.tokenizer.eos_token_id:
        return "eos", False
    return ("max_new_tokens", True) if result["autoregressive_forward_steps"] >= num_new_tokens else ("stopped", False)


def _octree_generation_trace(result: dict[str, Any], runtime: Any) -> dict[str, Any]:
    """Summarize whether the serialized output actually entered the octree loop."""
    generated_ids = [int(token_id) for token_id in result["generated_ids"]]
    mesh_bos_id = int(runtime.tokenizer.convert_tokens_to_ids("<mesh_bos>"))
    mesh_eos_id = int(runtime.tokenizer.convert_tokens_to_ids("<mesh_eos>"))
    mesh_bos_indices = [index for index, token_id in enumerate(generated_ids) if token_id == mesh_bos_id]
    mesh_eos_indices = [index for index, token_id in enumerate(generated_ids) if token_id == mesh_eos_id]

    first_mesh_bos_index = mesh_bos_indices[0] if mesh_bos_indices else None
    first_mesh_eos_after_bos_index = None
    if first_mesh_bos_index is not None:
        first_mesh_eos_after_bos_index = next(
            (index for index in mesh_eos_indices if index > first_mesh_bos_index),
            None,
        )

    entered_octree = first_mesh_bos_index is not None
    completed_octree = (
        first_mesh_eos_after_bos_index is not None
        and first_mesh_eos_after_bos_index == len(generated_ids) - 1
    )
    if not entered_octree:
        phase = "text_only"
    elif completed_octree:
        phase = "octree_complete"
    else:
        phase = "octree_incomplete"

    return {
        "phase": phase,
        "entered_octree_generation": entered_octree,
        "completed_octree_generation": completed_octree,
        "mesh_bos_token_id": mesh_bos_id,
        "mesh_eos_token_id": mesh_eos_id,
        "mesh_bos_token_indices": mesh_bos_indices,
        "mesh_eos_token_indices": mesh_eos_indices,
        "first_mesh_bos_token_index": first_mesh_bos_index,
        "first_mesh_eos_after_bos_token_index": first_mesh_eos_after_bos_index,
        "non_octree_prefix_token_count": (
            first_mesh_bos_index if first_mesh_bos_index is not None else len(generated_ids)
        ),
        "mesh_byte_token_count": int(result["structure_token_count"]),
        "serialized_output_token_count": len(generated_ids),
        "token_index_definition": "zero-based index within generated_token_ids",
    }


def _generate(
    row: dict[str, Any],
    *,
    runtime: Any,
    infer_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    description = str(row["text_description"])
    message = {"role": "user", "content": f"{args.prompt_prefix}{description}"}
    result = _predict_next_token(
        message,
        infer_cfg,
        image_paths=[],
        num_new_tokens=args.max_new_tokens,
        max_layer=args.max_layer,
        full_depth=args.full_depth,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        bos_top_k=args.bos_top_k,
        system_prompt=args.system_prompt,
        keep_mask_in_cache=args.keep_mask_in_cache,
        verbose=False,
        runtime=runtime,
        measure_latency=True,
    )
    termination, truncated = _termination(result, runtime, args.max_new_tokens)
    generated_text = str(result["generated_text"])
    generated_token_ids = [int(token_id) for token_id in result["generated_ids"]]
    return {
        "latency_ms": result["autoregressive_latency_ms"],
        "cuda_latency_ms": result["autoregressive_cuda_latency_ms"],
        "wall_latency_ms": result["autoregressive_wall_latency_ms"],
        "output_token_count": result["autoregressive_output_token_count"],
        "structure_token_count": result["structure_token_count"],
        "autoregressive_steps": result["autoregressive_forward_steps"],
        "token_definition": {
            "output_token_count": (
                "tokens selected from model logits in the autoregressive loop; the programmatically appended "
                "<mesh_eos> is excluded"
            ),
            "structure_token_count": "emitted <mesh0>...<mesh255> byte tokens",
        },
        "termination": termination,
        "truncated": truncated,
        "valid_structure": termination == "mesh_eos" and result["structure_token_count"] > 0,
        "generated_sha256": hashlib.sha256(generated_text.encode("utf-8")).hexdigest(),
        "generated_preview": generated_text[:240],
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "diagnostics": {
            "transient_mask_count": result["transient_mask_count"],
            "forced_mesh_bos_count": result["forced_mesh_bos_count"],
            "final_kv_cache_length": result["final_kv_cache_length"],
            "serialized_output_token_count": result["output_token_count"],
            "octree_generation": _octree_generation_trace(result, runtime),
        },
    }


def _base_record(row: dict[str, Any], args: argparse.Namespace, runtime_meta: dict[str, Any]) -> dict[str, Any]:
    description = str(row["text_description"])
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "status": "error",
        "config_hash": args.config_hash,
        "sample_order": int(row["sample_order"]),
        "dataset_index": int(row["dataset_index"]),
        "asset_id": str(row["asset_id"]),
        "text_description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {"path": str(Path(args.model_path).resolve())},
        "source": {"repository": str(REPO_ROOT), "commit": args.source_commit},
        "runtime": runtime_meta,
        "generation": {
            "prompt_prefix": args.prompt_prefix,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "bos_top_k": args.bos_top_k,
            "max_layer": args.max_layer,
            "full_depth": args.full_depth,
            "keep_mask_in_cache": args.keep_mask_in_cache,
            "sample_seed": args.seed + int(row["dataset_index"]),
            "system_prompt_sha256": hashlib.sha256(args.system_prompt.encode("utf-8")).hexdigest(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", default="Plurato123/OctLLM")
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--source-commit", default="working-tree")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--max-layer", type=int, default=6)
    parser.add_argument("--full-depth", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--bos-top-k", type=int, default=2)
    parser.add_argument("--prompt-prefix", default=DEFAULT_PROMPT_PREFIX)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--keep-mask-in-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("OctLLM benchmark requires a CUDA GPU")
    from scripts.model_sources import OCTLLM_MODEL_FILES, load_inference_config, resolve_model_directory

    infer_cfg = load_inference_config(args.config)
    configured_model_path = resolve_model_directory(
        infer_cfg["model_name_or_path"], local_files_only=True, allow_patterns=OCTLLM_MODEL_FILES,
        cache_dir=infer_cfg.get("cache_dir"), revision=infer_cfg.get("model_revision"),
    )
    model_path = resolve_model_directory(args.model_path, local_files_only=True, allow_patterns=OCTLLM_MODEL_FILES)
    if model_path != configured_model_path:
        raise ValueError("--model-path and the YAML model_name_or_path disagree")
    infer_cfg["model_name_or_path"] = str(model_path)

    manifest = _load_manifest(args.manifest)
    pending: list[dict[str, Any]] = []
    for row in manifest:
        output_path = args.output_dir / _safe_name(row)
        if not output_path.exists() or args.overwrite:
            pending.append(row)
        else:
            existing = _load_matching_result(output_path, args.config_hash, row)
            if existing is None:
                raise RuntimeError(
                    "refusing to overwrite a result whose identity or config hash differs: "
                    f"{output_path}; pass --overwrite to replace it"
                )
            if existing.get("status") == "ok":
                continue
            pending.append(row)
    if not pending:
        print("[octllm] all matching results already exist", flush=True)
        return 0

    runtime = _load_inference_runtime(infer_cfg, max_layer=args.max_layer, full_depth=args.full_depth)
    runtime_meta = _runtime_metadata(args.config, runtime)

    for warmup_index in range(max(0, args.warmup)):
        _seed_everything(args.seed + 10_000_000 + warmup_index)
        _generate(pending[warmup_index % len(pending)], runtime=runtime, infer_cfg=infer_cfg, args=args)

    failures = 0
    for row in pending:
        output_path = args.output_dir / _safe_name(row)
        _seed_everything(args.seed + int(row["dataset_index"]))
        record = _base_record(row, args, runtime_meta)
        try:
            record.update(_generate(row, runtime=runtime, infer_cfg=infer_cfg, args=args))
            record["status"] = "ok"
            record["error"] = None
            print(
                f"[octllm] {row['asset_id']}: {record['output_token_count']} tokens, "
                f"{record['latency_ms'] / 1000.0:.3f} s",
                flush=True,
            )
        except Exception as exc:  # keep other assets resumable
            failures += 1
            record["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            print(f"[octllm] ERROR {row['asset_id']}: {exc}", file=sys.stderr, flush=True)
        _atomic_json(output_path, record)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
