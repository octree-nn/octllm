#!/usr/bin/env python3
"""Run multi-GPU OctLLM 3D-understanding inference on PointLLM-200 and Toys4k.

Each worker is a separate process which sees exactly one physical GPU as
``cuda:0``.  The worker loads OctLLM once, converts each source GLB to the
same pruned sparse-octree byte sequence used by
``dataset_toolkits/tokenize_trellis.py``, and writes one atomic text file
per asset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shlex
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = {
    "pointllm": ("datasets/pointllm/PointLLM_brief_description_val_200_GT.json", "datasets/pointllm/glbs", "outputs/pointllm", "object_id"),
    "toys4k": ("datasets/toys4k/test.json", "datasets/toys4k/glbs", "outputs/toys4k_understanding", "asset_id"),
}
DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference" / "pipeline.yaml"
DEFAULT_METHOD_NAME = "OctLLM"
DEFAULT_PROMPT = "Describe this 3D mesh in detail: "
ASSIGNED_GPU_ENV = "OCTLLM_UNDERSTANDING_PHYSICAL_GPU"
_DISTRIBUTED_ENV_VARS = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@dataclass(frozen=True)
class Asset:
    dataset_index: int
    asset_id: str
    glb_path: Path


@dataclass
class Stats:
    selected: int
    completed: int = 0
    skipped: int = 0
    failed: int = 0

    def report(self) -> None:
        print(
            f"selected={self.selected} completed={self.completed} "
            f"skipped={self.skipped} failed={self.failed}",
            flush=True,
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _is_nonempty_text(path: Path) -> bool:
    try:
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())
    except OSError:
        return False


def _safe_method_name(value: str) -> str:
    name = value.strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("--method-name must be one safe directory name.")
    return name


def load_assets(dataset_path: Path, glb_dir: Path, *, id_field: str) -> list[Asset]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_path}")
    if not glb_dir.is_dir():
        raise FileNotFoundError(f"GLB directory not found: {glb_dir}")

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {dataset_path}, got {type(payload).__name__}.")

    assets: list[Asset] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} is not a JSON object.")
        asset_id = str(item.get(id_field, "")).strip()
        if not asset_id or Path(asset_id).name != asset_id or asset_id in {".", ".."}:
            raise ValueError(f"Dataset item {index} has an unsafe {id_field}: {asset_id!r}")
        if asset_id in seen:
            raise ValueError(f"Duplicate {id_field} in dataset: {asset_id}")
        seen.add(asset_id)
        assets.append(Asset(index, asset_id, glb_dir / f"{asset_id}.glb"))
    return assets


def select_assets(
    assets: Sequence[Asset],
    *,
    requested_ids: Sequence[str] | None,
    limit: int | None,
    num_shards: int,
    shard_id: int,
) -> list[Asset]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1.")
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards).")

    selected = list(assets)
    requested = set(requested_ids or [])
    if requested:
        available = {asset.asset_id for asset in assets}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Unknown --asset-id value(s): {', '.join(missing)}")
        selected = [asset for asset in selected if asset.asset_id in requested]
    if limit is not None:
        selected = selected[:limit]
    return [asset for position, asset in enumerate(selected) if position % num_shards == shard_id]


def validate_glbs(assets: Sequence[Asset]) -> None:
    missing = [asset.glb_path for asset in assets if not asset.glb_path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        suffix = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
        raise FileNotFoundError(f"Missing {len(missing)} source GLB(s): {preview}{suffix}")


def parse_gpu_ids(value: str) -> tuple[str, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--gpus must be a comma-separated list such as 0,1,2,3.")
    if any(not part.isdigit() for part in parts):
        raise ValueError("--gpus accepts non-negative physical GPU indices only.")
    gpu_ids = tuple(str(int(part)) for part in parts)
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus contains a duplicate GPU index.")
    return gpu_ids


def _sample_seed(base_seed: int, asset_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:understanding:{asset_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_trellis_preprocessor() -> ModuleType:
    """Load the shared TRELLIS tokenizer without duplicating its geometry logic."""
    source = REPO_ROOT / "dataset_toolkits" / "tokenize_trellis.py"
    process_dir = str(source.parent)
    if process_dir not in sys.path:
        sys.path.insert(0, process_dir)
    spec = importlib.util.spec_from_file_location("octllm_process_trellis_500k", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load preprocessing module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preprocess_octree(
    module: ModuleType,
    asset: Asset,
    args: argparse.Namespace,
) -> tuple[list[int], str]:
    byte_sequence = module.process_single_model(
        glb_path=str(asset.glb_path),
        num_samples=args.num_samples,
        depth=args.depth,
        full_depth=args.full_depth,
        mesh_scale=args.mesh_scale,
        shift=args.shift,
        threshold=args.threshold,
        device=args.octree_device,
        target_depth=args.target_depth,
        drop_prob=args.drop_prob,
        prune=True,
    )
    if byte_sequence is None:
        raise RuntimeError("GLB-to-octree preprocessing returned no sequence.")

    values = [int(value) for value in byte_sequence]
    if any(value < 0 or value > 255 for value in values):
        raise ValueError("Preprocessing produced a byte outside [0, 255].")
    if len(values) < args.min_sequence_length or len(values) > args.max_sequence_length:
        raise ValueError(
            f"Octree byte length {len(values)} is outside the allowed range "
            f"[{args.min_sequence_length}, {args.max_sequence_length}]."
        )

    mesh_sequence = "<mesh_bos>" + "".join(f"<mesh{value}>" for value in values) + "<mesh_eos>"
    return values, mesh_sequence


def _config_value(override: Any, mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    return override if override is not None else mapping.get(key, default)


def _load_runtime(args: argparse.Namespace) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from inference import _load_settings
    from scripts.generate_octree import _load_inference_runtime
    from scripts.model_sources import load_inference_config

    settings = _load_settings(str(args.config))
    mllm = settings.get("mllm", {})
    model_config = mllm.get("config")
    if not model_config:
        raise ValueError("Set mllm.config in the unified inference YAML.")
    infer_cfg = load_inference_config(model_config)
    if not isinstance(infer_cfg, dict):
        raise ValueError(f"Expected a mapping in OctLLM model config: {model_config}")
    if args.model_path is not None:
        model_path = str(args.model_path)
        infer_cfg["model_name_or_path"] = model_path

    print(f"Loading OctLLM once from {infer_cfg.get('model_name_or_path')}...", flush=True)
    runtime = _load_inference_runtime(infer_cfg, max_layer=args.depth, full_depth=args.full_depth)
    return runtime, infer_cfg, settings


def _clean_understanding_text(generated_text: str) -> str:
    from inference import MESH_TOKEN_PATTERN, clean_assistant_text

    if "<mesh_bos>" in generated_text or MESH_TOKEN_PATTERN.search(generated_text):
        raise RuntimeError("OctLLM generated mesh tokens for an understanding request instead of text.")
    answer = clean_assistant_text(generated_text)
    if not answer:
        raise RuntimeError("OctLLM returned an empty understanding response.")
    return answer


def _configure_assigned_gpu() -> None:
    physical_gpu = os.environ.get(ASSIGNED_GPU_ENV)
    if physical_gpu is None:
        return
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != physical_gpu:
        raise RuntimeError(
            f"Worker GPU binding changed: {ASSIGNED_GPU_ENV}={physical_gpu!r}, "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in worker assigned to physical GPU {physical_gpu}.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Worker assigned to physical GPU {physical_gpu} sees {torch.cuda.device_count()} devices; expected 1."
        )
    torch.cuda.set_device(0)
    print(
        f"GPU binding verified: physical GPU {physical_gpu} -> local cuda:0 "
        f"({torch.cuda.get_device_name(0)}).",
        flush=True,
    )


def run_worker(args: argparse.Namespace, assets: Sequence[Asset]) -> Stats:
    from scripts.generate_octree import _predict_next_token

    stats = Stats(selected=len(assets))
    pending: list[Asset] = []
    for asset in assets:
        output_path = args.method_dir / f"{asset.asset_id}.txt"
        if args.resume and _is_nonempty_text(output_path):
            stats.skipped += 1
        else:
            pending.append(asset)

    if not pending:
        stats.report()
        return stats

    runtime, infer_cfg, settings = _load_runtime(args)
    preprocessor = _load_trellis_preprocessor()
    generation = settings.get("mllm", {}).get("generation", {})
    system_prompt = settings.get("mllm", {}).get("system_prompt")
    max_new_tokens = max(1, int(_config_value(args.max_new_tokens, generation, "max_new_tokens", 512)))
    temperature = float(_config_value(args.temperature, generation, "temperature", 0.5))
    top_p = float(_config_value(args.top_p, generation, "top_p", 0.9))
    top_k = int(_config_value(args.top_k, generation, "top_k", 40))
    keep_mask_in_cache = bool(generation.get("keep_mask_in_cache", False))

    for position, asset in enumerate(pending, start=1):
        output_path = args.method_dir / f"{asset.asset_id}.txt"
        seed = _sample_seed(args.seed, asset.asset_id)
        print(
            f"[understanding {position}/{len(pending)}] index={asset.dataset_index} "
            f"asset_id={asset.asset_id} seed={seed}",
            flush=True,
        )
        try:
            _seed_everything(seed)
            byte_sequence, mesh_sequence = _preprocess_octree(preprocessor, asset, args)
            user_content = f"{args.prompt}{mesh_sequence}"
            result = _predict_next_token(
                {"role": "user", "content": user_content},
                infer_cfg,
                image_paths=[],
                num_new_tokens=max_new_tokens,
                max_layer=args.depth,
                full_depth=args.full_depth,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                bos_top_k=0,
                system_prompt=system_prompt,
                keep_mask_in_cache=keep_mask_in_cache,
                verbose=bool(settings.get("mllm", {}).get("verbose", False)),
                runtime=runtime,
            )
            answer = _clean_understanding_text(str(result["generated_text"]))
            _atomic_write_text(output_path, answer + "\n")
            stats.completed += 1
            print(
                f"[understanding] wrote {output_path} "
                f"(octree_bytes={len(byte_sequence)}, response_chars={len(answer)})",
                flush=True,
            )
        except Exception as exc:
            stats.failed += 1
            print(
                f"[understanding] FAILED {asset.asset_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            if args.fail_fast:
                break

    stats.report()
    return stats


def _worker_environment(gpu_id: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    for name in _DISTRIBUTED_ENV_VARS:
        environment.pop(name, None)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment[ASSIGNED_GPU_ENV] = gpu_id
    return environment


def _child_command(args: argparse.Namespace, num_shards: int, shard_id: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset-format",
        args.dataset_format,
        "--dataset",
        str(args.dataset),
        "--glb-dir",
        str(args.glb_dir),
        "--output-dir",
        str(args.output_dir),
        "--method-name",
        args.method_name,
        "--config",
        str(args.config),
        "--prompt",
        args.prompt,
        "--num-shards",
        str(num_shards),
        "--shard-id",
        str(shard_id),
        "--seed",
        str(args.seed),
        "--num-samples",
        str(args.num_samples),
        "--depth",
        str(args.depth),
        "--full-depth",
        str(args.full_depth),
        "--mesh-scale",
        str(args.mesh_scale),
        "--threshold",
        str(args.threshold),
        "--target-depth",
        str(args.target_depth),
        "--drop-prob",
        str(args.drop_prob),
        "--octree-device",
        args.octree_device,
        "--min-sequence-length",
        str(args.min_sequence_length),
        "--max-sequence-length",
        str(args.max_sequence_length),
    ]
    for asset_id in args.asset_id or []:
        command.extend(["--asset-id", asset_id])
    for flag, value in (
        ("--limit", args.limit),
        ("--model-path", args.model_path),
        ("--max-new-tokens", args.max_new_tokens),
        ("--temperature", args.temperature),
        ("--top-p", args.top_p),
        ("--top-k", args.top_k),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    for enabled, flag in (
        (args.resume, "--resume"),
        (args.shift, "--shift"),
        (args.fail_fast, "--fail-fast"),
    ):
        if enabled:
            command.append(flag)
    return command


def run_multi_gpu(args: argparse.Namespace, gpu_ids: Sequence[str]) -> int:
    if args.num_shards != 1 or args.shard_id != 0:
        raise ValueError("Do not combine --gpus with --num-shards/--shard-id.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_root = (
        args.log_dir
        if args.log_dir is not None
        else args.method_dir / ".logs" / timestamp
    )
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"Launching {len(gpu_ids)} isolated GPU worker(s). Logs: {log_root}", flush=True)

    processes: list[tuple[int, str, Path, subprocess.Popen[bytes], Any]] = []
    try:
        for worker_index, gpu_id in enumerate(gpu_ids):
            command = _child_command(args, len(gpu_ids), worker_index)
            log_path = log_root / f"worker_{worker_index:02d}_gpu{gpu_id}.log"
            log_handle = log_path.open("wb")
            print(
                f"[worker {worker_index}/{len(gpu_ids)}] physical_gpu={gpu_id} "
                f"shard={worker_index}/{len(gpu_ids)} log={log_path}",
                flush=True,
            )
            print(f"  {shlex.join(command)}", flush=True)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=_worker_environment(gpu_id),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                log_handle.close()
                raise
            processes.append((worker_index, gpu_id, log_path, process, log_handle))

        failed = False
        for worker_index, gpu_id, log_path, process, log_handle in processes:
            return_code = process.wait()
            log_handle.close()
            status = "completed" if return_code == 0 else f"failed (exit {return_code})"
            print(f"[worker {worker_index}] GPU {gpu_id} {status}; log={log_path}", flush=True)
            failed = failed or return_code != 0
        return 1 if failed else 0
    except KeyboardInterrupt:
        print("Interrupted; terminating GPU workers...", file=sys.stderr, flush=True)
        for _, _, _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, _, process, log_handle in processes:
            process.wait()
            if not log_handle.closed:
                log_handle.close()
        return 130
    except Exception:
        for _, _, _, process, log_handle in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if not log_handle.closed:
                log_handle.close()
        raise


def build_parser(dataset_format: str = "pointllm") -> argparse.ArgumentParser:
    dataset, glb_dir, output_dir, _ = DATASETS[dataset_format]
    parser = argparse.ArgumentParser(
        description="Convert PointLLM-200 or Toys4k GLBs to pruned octrees and run multi-GPU OctLLM 3D understanding.",
    )
    parser.add_argument("--dataset-format", choices=tuple(DATASETS), default=dataset_format)
    parser.add_argument("--dataset", type=Path, default=Path(dataset))
    parser.add_argument("--glb-dir", type=Path, default=Path(glb_dir))
    parser.add_argument("--output-dir", type=Path, default=Path(output_dir))
    parser.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path, help="Override the complete OctLLM checkpoint.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--gpus", help="Physical GPU IDs, for example 0,1,2,3.")
    parser.add_argument("--log-dir", type=Path, help="Override the multi-GPU worker log directory.")
    parser.add_argument("--asset-id", action="append", help="Run only this asset_id; repeat for several assets.")
    parser.add_argument("--limit", type=int, help="Use only the first N selected assets.")
    parser.add_argument("--resume", action="store_true", help="Skip an existing, non-empty <asset_id>.txt.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop a worker after its first failed asset.")
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--full-depth", type=int, default=3)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-depth", type=int, default=5)
    parser.add_argument("--drop-prob", type=float, default=0.5)
    parser.add_argument("--octree-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--min-sequence-length", type=int, default=100)
    parser.add_argument("--max-sequence-length", type=int, default=7000)

    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, help="Default: value from the unified YAML.")
    parser.add_argument("--top-p", type=float, help="Default: value from the unified YAML.")
    parser.add_argument("--top-k", type=int, help="Default: value from the unified YAML.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1.")
    if args.full_depth < 0 or args.depth <= args.full_depth:
        raise ValueError("Require 0 <= --full-depth < --depth.")
    if args.target_depth < args.full_depth or args.target_depth >= args.depth:
        raise ValueError("--target-depth must be in [--full-depth, --depth).")
    if not 0.0 <= args.drop_prob <= 1.0:
        raise ValueError("--drop-prob must be in [0, 1].")
    if args.min_sequence_length < 1 or args.max_sequence_length < args.min_sequence_length:
        raise ValueError("Invalid sequence-length range.")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    if args.temperature is not None and args.temperature < 0:
        raise ValueError("--temperature must be non-negative.")
    if args.top_p is not None and not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1].")
    if args.top_k is not None and args.top_k < 0:
        raise ValueError("--top-k must be non-negative.")


def main(dataset_format: str = "pointllm") -> None:
    args = build_parser(dataset_format).parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    args.glb_dir = args.glb_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.method_name = _safe_method_name(args.method_name)
    args.method_dir = args.output_dir / args.method_name
    if args.model_path is not None:
        args.model_path = args.model_path.expanduser().resolve()
    if args.log_dir is not None:
        args.log_dir = args.log_dir.expanduser().resolve()
    _validate_args(args)
    if not args.config.is_file():
        raise FileNotFoundError(f"Unified inference config not found: {args.config}")

    assets = load_assets(args.dataset, args.glb_dir, id_field=DATASETS[args.dataset_format][3])
    selected = select_assets(
        assets,
        requested_ids=args.asset_id,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    validate_glbs(selected)
    print(
        f"Selected {len(selected)} asset(s), shard {args.shard_id}/{args.num_shards}; "
        f"output={args.method_dir}; prompt={args.prompt!r}; prune_drop_prob={args.drop_prob}.",
        flush=True,
    )

    if args.gpus is not None:
        raise SystemExit(run_multi_gpu(args, parse_gpu_ids(args.gpus)))

    _configure_assigned_gpu()
    stats = run_worker(args, selected)
    raise SystemExit(1 if stats.failed else 0)


if __name__ == "__main__":
    main()
