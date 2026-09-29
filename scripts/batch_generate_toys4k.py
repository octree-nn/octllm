#!/usr/bin/env python3
"""Batch OctLLM generation for the Toys4k image/text evaluation split.

The workload is intentionally split into two process-isolated stages:

1. ``tokens`` loads OctLLM once and writes one octree sequence per asset.
2. ``meshes`` loads the VAE/TRELLIS stack and converts sequences to GLB.

``--stage all`` runs those stages in fresh child processes so their GPU models
never need to coexist. Outputs are resumable and are invalidated when the
corresponding octree sequence changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import (  # noqa: E402
    DEFAULT_CONFIG,
    MESH_TOKEN_PATTERN,
    OctreeLayerProgress,
    _load_settings,
    clean_assistant_text,
    extract_mesh_sequence,
)


DEFAULT_DATASET = Path("datasets/toys4k/test.json")
DEFAULT_OUTPUT_ROOT = Path("outputs/toys4k")
TEXT_PROMPT_PREFIX = "Generate a 3D mesh based on the following text description: "
IMAGE_PROMPT = "Generate a 3D mesh based on this image:"
REQUIRED_FIELDS = ("asset_id", "mesh_path", "render_image_path", "text_description")
CONDITION_DIRS = {
    "image": "image_to_3d",
    "text": "text_to_3d",
}
ASSIGNED_GPU_ENV = "OCTLLM_ASSIGNED_PHYSICAL_GPU"
_DISTRIBUTED_ENV_VARS = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@dataclass(frozen=True)
class Job:
    dataset_index: int
    item: dict[str, Any]
    condition: str

    @property
    def asset_id(self) -> str:
        return str(self.item["asset_id"])


@dataclass
class StageStats:
    stage: str
    selected: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0

    def report(self) -> None:
        print(
            f"[{self.stage}] selected={self.selected} completed={self.completed} "
            f"skipped={self.skipped} failed={self.failed}",
            flush=True,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _remove_stale_error(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Toys4k dataset JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(payload).__name__}.")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} is not a JSON object.")
        missing = [field for field in REQUIRED_FIELDS if field not in item]
        if missing:
            raise ValueError(f"Dataset item {index} is missing fields: {', '.join(missing)}")
        asset_id = str(item["asset_id"]).strip()
        if not asset_id or Path(asset_id).name != asset_id or asset_id in {".", ".."}:
            raise ValueError(f"Dataset item {index} has an unsafe asset_id: {asset_id!r}")
        if asset_id in seen:
            raise ValueError(f"Duplicate asset_id in dataset: {asset_id}")
        seen.add(asset_id)
        normalized.append(item)
    return normalized


def resolve_conditions(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("image", "text")
    if value not in CONDITION_DIRS:
        raise ValueError(f"Unsupported condition: {value}")
    return (value,)


def parse_gpu_ids(value: str) -> tuple[str, ...]:
    """Parse physical GPU indices used to create isolated worker processes."""
    raw_ids = [part.strip() for part in value.split(",")]
    if not raw_ids or any(not part for part in raw_ids):
        raise ValueError("--gpus must be a comma-separated list such as 0,1,2,3.")
    if any(not part.isdigit() for part in raw_ids):
        raise ValueError("--gpus accepts non-negative physical GPU indices only.")
    gpu_ids = tuple(str(int(part)) for part in raw_ids)
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus contains a duplicate GPU index.")
    return gpu_ids


def select_items(
    dataset: Sequence[dict[str, Any]],
    *,
    asset_ids: Sequence[str] | None,
    limit: int | None,
    num_shards: int,
    shard_id: int,
) -> list[tuple[int, dict[str, Any]]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards).")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1.")

    requested = set(asset_ids or [])
    indexed = list(enumerate(dataset))
    if requested:
        available = {str(item["asset_id"]) for item in dataset}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Unknown --asset-id value(s): {', '.join(missing)}")
        indexed = [(index, item) for index, item in indexed if str(item["asset_id"]) in requested]
    if limit is not None:
        indexed = indexed[:limit]
    return [entry for position, entry in enumerate(indexed) if position % num_shards == shard_id]


def make_jobs(items: Sequence[tuple[int, dict[str, Any]]], conditions: Sequence[str]) -> list[Job]:
    return [Job(index, item, condition) for condition in conditions for index, item in items]


def validate_job_inputs(jobs: Iterable[Job]) -> None:
    for job in jobs:
        if job.condition == "image":
            image_path = Path(str(job.item["render_image_path"])).expanduser()
            if not image_path.is_file():
                raise FileNotFoundError(f"Condition image not found for {job.asset_id}: {image_path}")
        elif not str(job.item["text_description"]).strip():
            raise ValueError(f"Empty text_description for {job.asset_id}")


def build_prompt_and_images(job: Job) -> tuple[str, list[str]]:
    if job.condition == "image":
        return IMAGE_PROMPT, [str(Path(str(job.item["render_image_path"])).expanduser().resolve())]
    description = str(job.item["text_description"]).strip()
    return f"{TEXT_PROMPT_PREFIX}{description}", []


def asset_output_dir(output_root: Path, job: Job) -> Path:
    return output_root / CONDITION_DIRS[job.condition] / job.asset_id


def token_path(output_root: Path, job: Job) -> Path:
    return asset_output_dir(output_root, job) / "octree.tokens.txt"


def token_metadata_path(output_root: Path, job: Job) -> Path:
    return asset_output_dir(output_root, job) / "octllm.json"


def glb_path(output_root: Path, job: Job) -> Path:
    return asset_output_dir(output_root, job) / f"{job.asset_id}.glb"


def mesh_metadata_path(output_root: Path, job: Job) -> Path:
    return asset_output_dir(output_root, job) / "mesh.json"


def _read_mesh_sequence(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Octree token file not found: {path}")
    sequence = extract_mesh_sequence(path.read_text(encoding="utf-8"))
    if sequence is None:
        raise ValueError(f"No complete octree sequence in {path}")
    return sequence


def has_valid_tokens(path: Path) -> bool:
    try:
        _read_mesh_sequence(path)
        return True
    except (OSError, ValueError):
        return False


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def is_valid_glb(path: Path) -> bool:
    """Check the GLB header without loading a potentially large mesh."""
    try:
        size = path.stat().st_size
        if size < 12:
            return False
        with path.open("rb") as handle:
            header = handle.read(12)
        if header[:4] != b"glTF":
            return False
        version = int.from_bytes(header[4:8], "little")
        declared_size = int.from_bytes(header[8:12], "little")
        return version in {1, 2} and declared_size == size
    except OSError:
        return False


def has_current_mesh(output_root: Path, job: Job, sequence_sha256: str) -> bool:
    output_glb = glb_path(output_root, job)
    metadata_path = mesh_metadata_path(output_root, job)
    if not is_valid_glb(output_glb) or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("status") == "success" and metadata.get("octree_sha256") == sequence_sha256


def _sample_seed(base_seed: int, job: Job, stage: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{stage}:{job.condition}:{job.asset_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _error_payload(stage: str, job: Job, exc: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "stage": stage,
        "condition": CONDITION_DIRS[job.condition],
        "asset_id": job.asset_id,
        "dataset_index": job.dataset_index,
        "timestamp_utc": _utc_now(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _value(override: Any, mapping: dict[str, Any], key: str, default: Any) -> Any:
    return override if override is not None else mapping.get(key, default)


def run_token_stage(args: argparse.Namespace, jobs: Sequence[Job]) -> StageStats:
    stats = StageStats(stage="tokens", selected=len(jobs))
    pending: list[Job] = []
    for job in jobs:
        if has_valid_tokens(token_path(args.output_root, job)) and not args.overwrite:
            stats.skipped += 1
        else:
            pending.append(job)

    if not pending:
        stats.report()
        return stats

    settings = _load_settings(str(args.config))
    mllm = settings.get("mllm", {})
    generation = mllm.get("generation", {})
    image_config = mllm.get("image", {})
    model_config_path = mllm.get("config")
    if not model_config_path:
        raise ValueError("Set mllm.config in the unified inference YAML.")

    from scripts.generate_octree import (
        _load_inference_runtime,
        _predict_next_token,
        _preprocess_image_paths,
    )
    from scripts.model_sources import load_inference_config

    infer_cfg = load_inference_config(model_config_path)
    if not isinstance(infer_cfg, dict):
        raise ValueError(f"Expected a mapping in OctLLM model config: {model_config_path}")

    max_layer = int(_value(args.max_layer, generation, "max_layer", 6))
    full_depth = int(_value(args.full_depth, generation, "full_depth", 3))
    print(f"Loading OctLLM once for {len(pending)} pending token job(s)...", flush=True)
    runtime = _load_inference_runtime(infer_cfg, max_layer=max_layer, full_depth=full_depth)

    for position, job in enumerate(pending, start=1):
        output_dir = asset_output_dir(args.output_root, job)
        error_path = output_dir / "tokens.error.json"
        prompt, source_images = build_prompt_and_images(job)
        seed = _sample_seed(args.seed, job, "tokens")
        print(
            f"[tokens {position}/{len(pending)}] {CONDITION_DIRS[job.condition]}/{job.asset_id}",
            flush=True,
        )
        progress = OctreeLayerProgress(enabled=not args.no_progress)
        try:
            _seed_everything(seed)
            processed_images = _preprocess_image_paths(
                source_images,
                enabled=bool(image_config.get("preprocess", True)),
                size=max(1, int(image_config.get("size", 1024))),
                object_fill=float(image_config.get("object_fill", 0.75)),
                save_dir=image_config.get("save_dir"),
            )
            model_result = _predict_next_token(
                {"role": "user", "content": prompt},
                infer_cfg,
                image_paths=processed_images,
                num_new_tokens=max(1, int(_value(args.max_new_tokens, generation, "max_new_tokens", 10000))),
                max_layer=max_layer,
                full_depth=full_depth,
                temperature=float(_value(args.temperature, generation, "temperature", 0.5)),
                top_p=float(_value(args.top_p, generation, "top_p", 0.9)),
                top_k=int(_value(args.top_k, generation, "top_k", 40)),
                bos_top_k=int(_value(args.bos_top_k, generation, "bos_top_k", 1)),
                system_prompt=mllm.get("system_prompt"),
                keep_mask_in_cache=bool(generation.get("keep_mask_in_cache", False)),
                progress_callback=progress,
                verbose=bool(mllm.get("verbose", False)),
                runtime=runtime,
            )
            generated_text = str(model_result["generated_text"])
            sequence = extract_mesh_sequence(generated_text)
            if sequence is None:
                if "<mesh_bos>" in generated_text:
                    raise RuntimeError("Octree generation stopped before a complete <mesh_eos> token.")
                raise RuntimeError("OctLLM response did not contain an octree sequence.")

            sequence_hash = _sequence_sha256(sequence)
            _atomic_write_text(token_path(args.output_root, job), sequence + "\n")
            generation_summary = {
                key: value
                for key, value in model_result.items()
                if key not in {"generated_ids", "generated_probs", "generated_text"}
            }
            _atomic_write_json(
                token_metadata_path(args.output_root, job),
                {
                    "status": "success",
                    "stage": "tokens",
                    "condition": CONDITION_DIRS[job.condition],
                    "asset_id": job.asset_id,
                    "dataset_index": job.dataset_index,
                    "timestamp_utc": _utc_now(),
                    "seed": seed,
                    "prompt": prompt,
                    "condition_image": source_images[0] if source_images else None,
                    "text_description": str(job.item["text_description"]),
                    "reference_mesh": str(job.item["mesh_path"]),
                    "assistant_text": clean_assistant_text(generated_text, sequence),
                    "mesh_token_count": len(MESH_TOKEN_PATTERN.findall(sequence)),
                    "octree_sha256": sequence_hash,
                    "generation": generation_summary,
                },
            )
            _remove_stale_error(error_path)
            stats.completed += 1
        except Exception as exc:
            stats.failed += 1
            _atomic_write_json(error_path, _error_payload("tokens", job, exc))
            print(f"[tokens] FAILED {job.asset_id}: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                break
        finally:
            progress.close()

    stats.report()
    return stats


def _release_trellis_pipeline() -> None:
    try:
        from scripts.decode_octree import clear_trellis_pipeline_cache

        clear_trellis_pipeline_cache()
    except ImportError:
        pass


def run_mesh_stage(args: argparse.Namespace, jobs: Sequence[Job]) -> StageStats:
    stats = StageStats(stage="meshes", selected=len(jobs))
    runnable: list[tuple[Job, str, str]] = []
    for job in jobs:
        try:
            sequence = _read_mesh_sequence(token_path(args.output_root, job))
            sequence_hash = _sequence_sha256(sequence)
        except (OSError, ValueError) as exc:
            stats.failed += 1
            output_dir = asset_output_dir(args.output_root, job)
            _atomic_write_json(output_dir / "mesh.error.json", _error_payload("meshes", job, exc))
            continue
        if has_current_mesh(args.output_root, job, sequence_hash) and not args.overwrite:
            stats.skipped += 1
        else:
            runnable.append((job, sequence, sequence_hash))

    if not runnable:
        stats.report()
        return stats

    settings = _load_settings(str(args.config))
    generation = settings.get("mllm", {}).get("generation", {})
    max_layer = int(_value(args.max_layer, generation, "max_layer", 6))
    full_depth = int(_value(args.full_depth, generation, "full_depth", 3))
    mesh_settings = settings.get("mesh", {})
    if args.vae_checkpoint:
        mesh_settings.setdefault("vae", {})["checkpoint"] = str(args.vae_checkpoint)
    if args.image_model:
        mesh_settings.setdefault("trellis", {})["image_model_path"] = str(args.image_model)
    if args.text_model:
        mesh_settings.setdefault("trellis", {})["text_model_path"] = str(args.text_model)
    if args.text_condition_model:
        mesh_settings.setdefault("trellis", {})["text_condition_model_path"] = str(args.text_condition_model)

    from inference import _complete_voxel, _generate_glb, _octree_tokens_to_voxel

    processed = 0
    for condition in resolve_conditions(args.condition):
        condition_jobs = [entry for entry in runnable if entry[0].condition == condition]
        if not condition_jobs:
            continue
        print(
            f"Preparing TRELLIS {CONDITION_DIRS[condition]} for {len(condition_jobs)} pending mesh job(s)...",
            flush=True,
        )
        for job, sequence, sequence_hash in condition_jobs:
            processed += 1
            output_dir = asset_output_dir(args.output_root, job)
            error_path = output_dir / "mesh.error.json"
            prompt, source_images = build_prompt_and_images(job)
            seed = _sample_seed(args.seed, job, "meshes")
            print(
                f"[meshes {processed}/{len(runnable)}] {CONDITION_DIRS[job.condition]}/{job.asset_id}",
                flush=True,
            )
            try:
                _seed_everything(seed)
                voxel = _octree_tokens_to_voxel(
                    sequence,
                    depth=max_layer,
                    full_depth=full_depth,
                    device=str(mesh_settings.get("octree_device", "cpu")),
                    threshold=float(mesh_settings.get("octree_threshold", 0.0)),
                )
                voxel = _complete_voxel(voxel, mesh_settings)
                generated_path = _generate_glb(
                    voxel,
                    prompt,
                    source_images,
                    mesh_settings,
                    output_dir,
                    job.asset_id,
                    condition,
                    str(job.item["text_description"]).strip() if condition == "text" else None,
                    None,
                )
                expected_path = glb_path(args.output_root, job).resolve()
                if generated_path != expected_path:
                    raise RuntimeError(f"Unexpected GLB output path: {generated_path} (expected {expected_path})")
                if not is_valid_glb(expected_path):
                    raise RuntimeError(f"TRELLIS wrote an invalid or incomplete GLB: {expected_path}")
                _atomic_write_json(
                    mesh_metadata_path(args.output_root, job),
                    {
                        "status": "success",
                        "stage": "meshes",
                        "condition": CONDITION_DIRS[job.condition],
                        "asset_id": job.asset_id,
                        "dataset_index": job.dataset_index,
                        "timestamp_utc": _utc_now(),
                        "seed": seed,
                        "prompt": prompt,
                        "condition_image": source_images[0] if source_images else None,
                        "text_description": str(job.item["text_description"]),
                        "octree_sha256": sequence_hash,
                        "glb_path": str(expected_path),
                        "glb_size_bytes": expected_path.stat().st_size,
                    },
                )
                _remove_stale_error(error_path)
                stats.completed += 1
            except Exception as exc:
                stats.failed += 1
                _atomic_write_json(error_path, _error_payload("meshes", job, exc))
                print(f"[meshes] FAILED {job.asset_id}: {exc}", file=sys.stderr, flush=True)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                if args.fail_fast:
                    _release_trellis_pipeline()
                    stats.report()
                    return stats
        _release_trellis_pipeline()

    stats.report()
    return stats


def _append_child_option(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _child_command(
    args: argparse.Namespace,
    stage: str,
    condition: str,
    *,
    num_shards: int | None = None,
    shard_id: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--condition",
        condition,
        "--dataset",
        str(args.dataset),
        "--output-root",
        str(args.output_root),
        "--config",
        str(args.config),
        "--num-shards",
        str(args.num_shards if num_shards is None else num_shards),
        "--shard-id",
        str(args.shard_id if shard_id is None else shard_id),
        "--seed",
        str(args.seed),
    ]
    for asset_id in args.asset_id or []:
        command.extend(["--asset-id", asset_id])
    for flag, value in (
        ("--limit", args.limit),
        ("--max-new-tokens", args.max_new_tokens),
        ("--temperature", args.temperature),
        ("--top-p", args.top_p),
        ("--top-k", args.top_k),
        ("--bos-top-k", args.bos_top_k),
        ("--max-layer", args.max_layer),
        ("--full-depth", args.full_depth),
        ("--vae-checkpoint", args.vae_checkpoint),
        ("--image-model", args.image_model),
        ("--text-model", args.text_model),
        ("--text-condition-model", args.text_condition_model),
    ):
        _append_child_option(command, flag, value)
    for enabled, flag in (
        (args.overwrite, "--overwrite"),
        (args.fail_fast, "--fail-fast"),
        (args.no_progress, "--no-progress"),
    ):
        if enabled:
            command.append(flag)
    return command


def _worker_environment(gpu_id: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Create a clean single-GPU environment before Python/Torch is imported."""
    environment = dict(os.environ if base is None else base)
    for name in _DISTRIBUTED_ENV_VARS:
        environment.pop(name, None)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    environment[ASSIGNED_GPU_ENV] = gpu_id
    return environment


def _configure_assigned_gpu() -> None:
    """Verify that a worker sees only its assigned physical GPU as local cuda:0."""
    physical_gpu = os.environ.get(ASSIGNED_GPU_ENV)
    if physical_gpu is None:
        return
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != physical_gpu:
        raise RuntimeError(
            f"Worker GPU binding changed unexpectedly: {ASSIGNED_GPU_ENV}={physical_gpu!r}, "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in worker assigned to physical GPU {physical_gpu}.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Worker assigned to physical GPU {physical_gpu} sees {torch.cuda.device_count()} CUDA devices; expected 1."
        )
    torch.cuda.set_device(0)
    print(
        f"GPU binding verified: physical GPU {physical_gpu} -> local cuda:0 "
        f"({torch.cuda.get_device_name(0)}).",
        flush=True,
    )


def run_multi_gpu(args: argparse.Namespace, gpu_ids: Sequence[str]) -> int:
    """Launch one isolated, evenly sharded inference process per physical GPU."""
    if args.num_shards != 1 or args.shard_id != 0:
        raise ValueError("Do not combine --gpus with --num-shards/--shard-id; GPU workers create shards automatically.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_root = (
        args.log_dir.expanduser().resolve()
        if args.log_dir is not None
        else args.output_root / "logs" / f"{timestamp}_{args.stage}_{args.condition}"
    )
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"Launching {len(gpu_ids)} isolated GPU worker(s). Logs: {log_root}", flush=True)

    processes: list[tuple[int, str, Path, subprocess.Popen[bytes], Any]] = []
    try:
        for worker_index, gpu_id in enumerate(gpu_ids):
            command = _child_command(
                args,
                args.stage,
                args.condition,
                num_shards=len(gpu_ids),
                shard_id=worker_index,
            )
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


def run_all_stages(args: argparse.Namespace) -> int:
    token_result = subprocess.run(_child_command(args, "tokens", args.condition), check=False)
    return_code = token_result.returncode
    if args.fail_fast and return_code != 0:
        return return_code

    for condition in resolve_conditions(args.condition):
        mesh_result = subprocess.run(_child_command(args, "meshes", condition), check=False)
        if mesh_result.returncode != 0:
            return_code = mesh_result.returncode
        if args.fail_fast and mesh_result.returncode != 0:
            break
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Toys4k OctLLM image/text octrees and TRELLIS GLBs with resumable two-stage batching.",
    )
    parser.add_argument("--stage", choices=("all", "tokens", "meshes"), default="all")
    parser.add_argument("--condition", choices=("both", "image", "text"), default="both")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--asset-id", action="append", help="Only run this asset_id; repeat for several assets.")
    parser.add_argument("--limit", type=int, help="Use only the first N selected dataset items.")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of deterministic dataset shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="This shard in [0, num-shards).")
    parser.add_argument(
        "--gpus",
        help=(
            "Comma-separated physical GPU indices, for example 0,1,2,3. Launches one isolated, evenly sharded "
            "worker per GPU; do not combine with --num-shards/--shard-id."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Multi-GPU worker log directory (default: <output-root>/logs/<timestamp>_<stage>_<condition>).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed; each asset receives a stable derived seed.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate valid current outputs too.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop the current stage after its first failure.")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-octree layer progress bars.")

    parser.add_argument("--max-new-tokens", type=int, help="Override mllm.generation.max_new_tokens.")
    parser.add_argument("--temperature", type=float, help="Override mllm.generation.temperature.")
    parser.add_argument("--top-p", type=float, help="Override mllm.generation.top_p.")
    parser.add_argument("--top-k", type=int, help="Override mllm.generation.top_k.")
    parser.add_argument("--bos-top-k", type=int, help="Override mllm.generation.bos_top_k.")
    parser.add_argument("--max-layer", type=int, help="Override mllm.generation.max_layer.")
    parser.add_argument("--full-depth", type=int, help="Override mllm.generation.full_depth.")
    parser.add_argument("--vae-checkpoint", type=Path, help="Override mesh.vae.checkpoint.")
    parser.add_argument("--image-model", help="TRELLIS image model: Hugging Face repository ID or local directory.")
    parser.add_argument("--text-model", help="TRELLIS text model: Hugging Face repository ID or local directory.")
    parser.add_argument(
        "--text-condition-model",
        help="CLIP text conditioning model: Hugging Face repository ID or local directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    if not args.config.is_file():
        raise FileNotFoundError(f"Unified inference config not found: {args.config}")

    dataset = load_dataset(args.dataset)
    selected_items = select_items(
        dataset,
        asset_ids=args.asset_id,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    jobs = make_jobs(selected_items, resolve_conditions(args.condition))
    validate_job_inputs(jobs)
    print(
        f"Selected {len(selected_items)} asset(s), {len(jobs)} job(s), "
        f"shard {args.shard_id}/{args.num_shards}.",
        flush=True,
    )

    if args.gpus is not None:
        raise SystemExit(run_multi_gpu(args, parse_gpu_ids(args.gpus)))
    if args.stage == "all":
        raise SystemExit(run_all_stages(args))
    _configure_assigned_gpu()
    if args.stage == "tokens":
        stats = run_token_stage(args, jobs)
    else:
        stats = run_mesh_stage(args, jobs)
    raise SystemExit(1 if stats.failed else 0)


if __name__ == "__main__":
    main()
