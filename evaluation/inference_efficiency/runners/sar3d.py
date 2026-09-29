#!/usr/bin/env python3
"""Benchmark only SAR3D's text-conditioned next-scale autoregressive loop.

The official inference method also decodes the generated VQ codes into a
triplane.  Timing is therefore attached to the first transformer's
``kv_caching(True/False)`` calls: the official method enables the cache
immediately before its next-scale loop and disables it immediately after.
CLIP encoding and the subsequent VAE/triplane decoder are not timed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evaluation.inference_efficiency.model_artifacts import resolve_method_artifacts


METHOD = "SAR3D"
SCHEMA_VERSION = "inference-efficiency.v1"
PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
STRUCTURE_TOKEN_COUNT = 3 * sum(patch_num**2 for patch_num in PATCH_NUMS)
AUTOREGRESSIVE_STEPS = len(PATCH_NUMS)
OFFICIAL_REPOSITORY = "https://github.com/cyw-3d/SAR3D"
DEFAULT_SOURCE_DIR = Path(os.environ.get("SAR3D_SOURCE_DIR", "/tmp/octllm-benchmark-sources/SAR3D"))
DEFAULT_AR_CHECKPOINT = "cyw-3d/sar3d"
DEFAULT_VQVAE_CHECKPOINT = "cyw-3d/sar3d"
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"sample_order", "dataset_index", "asset_id", "text_description"}
            missing = sorted(required.difference(row))
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields: {', '.join(missing)}")
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
            handle.flush()
            os.fsync(handle.fileno())
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
    torch.cuda.manual_seed_all(seed)


def _git_commit(source_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _validate_inputs(args: argparse.Namespace) -> None:
    required_paths = {
        "SAR3D source": args.source_dir,
        "AR checkpoint": args.ar_checkpoint,
        "VQVAE checkpoint": args.vqvae_checkpoint,
        "CLIP model": args.clip_model,
    }
    for label, path in required_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    expected_source_files = (
        args.source_dir / "models" / "var.py",
        args.source_dir / "utils" / "arg_util.py",
        args.source_dir / "files" / "empty_text_embedding.npy",
        args.source_dir / "files" / "empty_text_pooler_output.npy",
    )
    for path in expected_source_files:
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete SAR3D checkout; missing {path}")


def _load_official_ln3diff_config(source_dir: Path) -> SimpleNamespace:
    """Read Args.LN3Diff_kwargs without importing the official CLI parser."""
    config_source = source_dir / "utils" / "arg_util.py"
    syntax_tree = ast.parse(config_source.read_text(encoding="utf-8"), filename=str(config_source))
    for node in syntax_tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Args":
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "LN3Diff_kwargs" for target in statement.targets):
                value = ast.literal_eval(statement.value)
                if not isinstance(value, dict):
                    break
                return SimpleNamespace(**value)
    raise RuntimeError(f"Could not locate literal Args.LN3Diff_kwargs in {config_source}")


def _load_runtime(args: argparse.Namespace) -> SimpleNamespace:
    """Load the official SAR3D model and local CLIP text encoder."""
    source_dir = args.source_dir.resolve()
    os.chdir(source_dir)  # Official inference reads ./files/*.npy at runtime.
    sys.path.insert(0, str(source_dir))

    from transformers import CLIPTextModel, CLIPTokenizer
    from utils import dist
    from models import build_vae_var_3D_VAR

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("SAR3D efficiency evaluation requires a CUDA device")
    if device.index is not None:
        dist.set_gpu_id(device.index)
    else:
        dist.set_gpu_id(torch.cuda.current_device())
    device = torch.device(dist.get_device())

    # Match Args.tf32=True in the official test entrypoint.  This is also the
    # intended fast FP32 path on an RTX 5090.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Use a fresh namespace so the class-level LN3DiffConfig in the official
    # argument parser cannot leak mutations across runner invocations.
    model_args = SimpleNamespace(
        LN3DiffConfig=_load_official_ln3diff_config(source_dir),
        flexicubes=False,
        text_conditioned=True,
    )
    vae, var = build_vae_var_3D_VAR(
        device=device,
        patch_nums=PATCH_NUMS,
        num_classes=1,
        depth=16,
        shared_aln=False,
        attn_l2_norm=True,
        flash_if_available=True,
        fused_if_available=True,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=-1,
        args=model_args,
    )

    vae_state = torch.load(args.vqvae_checkpoint, map_location="cpu", weights_only=False)
    vae.load_state_dict(vae_state, strict=True)
    del vae_state

    ar_payload = torch.load(args.ar_checkpoint, map_location="cpu", weights_only=False)
    try:
        ar_state = ar_payload["trainer"]["var_wo_ddp"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unexpected SAR3D AR checkpoint layout; expected trainer.var_wo_ddp") from exc
    incompatible = var.load_state_dict(ar_state, strict=False)
    del ar_payload, ar_state
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "SAR3D AR checkpoint did not load exactly: "
            f"missing={list(incompatible.missing_keys)}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )

    vae.eval()
    var.eval()

    # Match the official test.py behavior: CLIP text encoding runs on CPU and
    # only its hidden states are copied to the GPU.  It is outside all timers.
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model, local_files_only=True)
    clip_encoder = CLIPTextModel.from_pretrained(args.clip_model, local_files_only=True).eval()

    return SimpleNamespace(
        device=device,
        vae=vae,
        var=var,
        clip_tokenizer=clip_tokenizer,
        clip_encoder=clip_encoder,
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
    )


@torch.inference_mode()
def _encode_text(runtime: SimpleNamespace, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = runtime.clip_tokenizer
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # Deliberately omit attention_mask to reproduce official test.py.
    output = runtime.clip_encoder(encoded.input_ids)
    embeddings = output[0].to(runtime.device, non_blocking=True)
    pooled = output[1].to(runtime.device, non_blocking=True)
    return embeddings, pooled


@torch.inference_mode()
def _generate(
    row: dict[str, Any],
    *,
    runtime: SimpleNamespace,
    sample_seed: int,
) -> dict[str, Any]:
    embeddings, pooled = _encode_text(runtime, str(row["text_description"]))

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    first_attention = runtime.var.blocks[0].attn
    last_attention = runtime.var.blocks[-1].attn
    if first_attention is last_attention:
        raise RuntimeError("SAR3D timing wrapper requires at least two transformer blocks")
    original_first_kv_caching = first_attention.kv_caching
    original_last_kv_caching = last_attention.kv_caching
    timing: dict[str, Any] = {"start_calls": 0, "end_calls": 0}

    def timed_first_kv_caching(enable: bool) -> Any:
        # The first cache-disable call occurs immediately after the last AR
        # scale.  Close the interval before any cache teardown or VAE decode.
        if not enable:
            timing["end_calls"] += 1
            end_event.record(torch.cuda.current_stream(runtime.device))
            end_event.synchronize()
            timing["wall_end"] = time.perf_counter()
        return original_first_kv_caching(enable)

    def timed_last_kv_caching(enable: bool) -> Any:
        result = original_last_kv_caching(enable)
        # The last cache-enable call is immediately before the first AR scale.
        # Open the interval after all blocks have completed cache setup.
        if enable:
            timing["start_calls"] += 1
            # Drain CLIP's non-blocking H2D copies and any decoder work from
            # the previous sample before opening the wall-clock interval.
            torch.cuda.synchronize(runtime.device)
            timing["wall_start"] = time.perf_counter()
            start_event.record(torch.cuda.current_stream(runtime.device))
        return result

    first_attention.kv_caching = timed_first_kv_caching
    last_attention.kv_caching = timed_last_kv_caching
    try:
        triplane, generated_indices = runtime.var.autoregressive_infer_cfg_3D_VAR_text_l2norm(
            B=1,
            dino_image_embeddings=embeddings,
            pooler_output=pooled,
            cfg=4,
            top_k=10,
            top_p=0.5,
            g_seed=sample_seed,
            more_smooth=False,
        )
    finally:
        first_attention.kv_caching = original_first_kv_caching
        last_attention.kv_caching = original_last_kv_caching

    if timing["start_calls"] != 1 or timing["end_calls"] != 1:
        raise RuntimeError(
            "SAR3D timing boundary was not observed exactly once "
            f"(start={timing['start_calls']}, end={timing['end_calls']})"
        )
    end_event.synchronize()
    cuda_latency_ms = float(start_event.elapsed_time(end_event))
    wall_latency_ms = float((timing["wall_end"] - timing["wall_start"]) * 1000.0)
    output_tokens = int(generated_indices.shape[-1])
    if output_tokens != STRUCTURE_TOKEN_COUNT:
        raise RuntimeError(f"Expected {STRUCTURE_TOKEN_COUNT} VQ indices, got {output_tokens}")

    del embeddings, pooled, triplane, generated_indices
    return {
        "latency_ms": cuda_latency_ms,
        "cuda_latency_ms": cuda_latency_ms,
        "wall_latency_ms": wall_latency_ms,
        "output_token_count": output_tokens,
        "structure_token_count": output_tokens,
        "autoregressive_steps": AUTOREGRESSIVE_STEPS,
        "token_definition": {
            "output_token_count": "native SAR3D discrete VQ indices across all three triplanes",
            "structure_token_count": "same native VQ indices; 3 * sum(patch_num^2)",
            "autoregressive_steps": "next-scale transformer forwards, not token-by-token forwards",
            "patch_nums": list(PATCH_NUMS),
        },
        "termination": "fixed_multiscale_schedule",
        "truncated": False,
        "valid_structure": True,
        "diagnostics": {
            "cfg": 4.0,
            "top_k": 10,
            "top_p": 0.5,
            "ar_autocast": False,
            "official_cli_fp16_setting": 2,
            "model_eval_mode": True,
            "timing_boundary": "after last cache-enable to before first cache-disable",
            "clip_and_vae_decoder_excluded": True,
        },
    }


def _runtime_metadata(runtime: SimpleNamespace) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "xformers": _package_version("xformers"),
        "gpu": torch.cuda.get_device_name(runtime.device),
        "device": str(runtime.device),
        "dtype": str(next(runtime.var.parameters()).dtype),
        "ar_autocast": False,
        "official_cli_fp16_setting": 2,
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _base_record(
    row: dict[str, Any],
    args: argparse.Namespace,
    runtime: SimpleNamespace,
    runtime_meta: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
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
        "latency_ms": None,
        "cuda_latency_ms": None,
        "wall_latency_ms": None,
        "prompt_token_count": None,
        "output_token_count": None,
        "structure_token_count": None,
        "autoregressive_steps": None,
        "token_definition": None,
        "termination": "error",
        "truncated": False,
        "error": None,
        "model": {
            "ar_checkpoint": str(args.ar_checkpoint.resolve()),
            "vqvae_checkpoint": str(args.vqvae_checkpoint.resolve()),
            "clip_model": str(args.clip_model.resolve()),
            "ar_missing_keys": runtime.missing_keys,
            "ar_unexpected_keys": runtime.unexpected_keys,
        },
        "source": {
            "repository": OFFICIAL_REPOSITORY,
            "path": str(args.source_dir.resolve()),
            "commit": source_commit,
            "entrypoint": "models.var.VAR_text.autoregressive_infer_cfg_3D_VAR_text_l2norm",
        },
        "runtime": runtime_meta,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "cfg": 4.0,
            "top_k": 10,
            "top_p": 0.5,
            "patch_nums": list(PATCH_NUMS),
            "seed": args.seed + int(row["dataset_index"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-dir", "--source-root", dest="source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--ar-checkpoint", default=DEFAULT_AR_CHECKPOINT)
    parser.add_argument(
        "--vqvae-checkpoint", "--vae-checkpoint", dest="vqvae_checkpoint", default=DEFAULT_VQVAE_CHECKPOINT
    )
    parser.add_argument("--clip-model", "--clip-path", dest="clip_model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SAR3D benchmark requires a CUDA GPU")
    artifacts = resolve_method_artifacts("sar3d", {
        "ar_checkpoint": args.ar_checkpoint,
        "vae_checkpoint": args.vqvae_checkpoint,
        "clip_path": args.clip_model,
    })
    args.ar_checkpoint = Path(artifacts["ar_checkpoint"])
    args.vqvae_checkpoint = Path(artifacts["vae_checkpoint"])
    args.clip_model = Path(artifacts["clip_path"])
    # The official source changes the working directory because it loads
    # ./files/*.npy.  Resolve every user-facing path before that happens.
    for attribute in (
        "manifest",
        "output_dir",
        "source_dir",
        "ar_checkpoint",
        "vqvae_checkpoint",
        "clip_model",
    ):
        setattr(args, attribute, Path(getattr(args, attribute)).expanduser().resolve())
    _validate_inputs(args)
    manifest = _load_manifest(args.manifest)

    pending: list[dict[str, Any]] = []
    for row in manifest:
        output_path = args.output_dir / _safe_name(row)
        if args.overwrite or not output_path.exists():
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
        print("[sar3d] all matching results already exist", flush=True)
        return 0

    runtime = _load_runtime(args)
    runtime_meta = _runtime_metadata(runtime)
    source_commit = args.source_commit or _git_commit(args.source_dir)

    for warmup_index in range(max(0, args.warmup)):
        warmup_row = manifest[warmup_index % len(manifest)]
        warmup_seed = args.seed + 10_000_000 + warmup_index
        _seed_everything(warmup_seed)
        _generate(warmup_row, runtime=runtime, sample_seed=warmup_seed)

    failures = 0
    for row in manifest:
        output_path = args.output_dir / _safe_name(row)
        existing = _load_matching_result(output_path, args.config_hash, row)
        if not args.overwrite and existing is not None and existing.get("status") == "ok":
            print(f"[sar3d] skip {row['asset_id']}: matching result exists", flush=True)
            continue
        sample_seed = args.seed + int(row["dataset_index"])
        _seed_everything(sample_seed)
        record = _base_record(row, args, runtime, runtime_meta, source_commit)
        try:
            record.update(_generate(row, runtime=runtime, sample_seed=sample_seed))
            record["status"] = "ok"
            print(
                f"[sar3d] {row['asset_id']}: {record['output_token_count']} native VQ indices, "
                f"{record['latency_ms'] / 1000.0:.3f} s",
                flush=True,
            )
        except Exception as exc:  # Keep the remaining assets resumable.
            failures += 1
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"[sar3d] ERROR {row['asset_id']}: {exc}", file=sys.stderr, flush=True)
        _atomic_json(output_path, record)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
