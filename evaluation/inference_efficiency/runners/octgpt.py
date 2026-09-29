#!/usr/bin/env python3
"""Measure the autoregressive stage of the official OctGPT implementation.

This runner intentionally stops at ``OctGPT.generate``.  Text conditioning,
octree initialization, VQVAE decoding, marching cubes, and mesh export are not
part of the measured interval.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evaluation.inference_efficiency.model_artifacts import resolve_method_artifacts


SCHEMA_VERSION = "inference-efficiency.v1"
METHOD = "OctGPT"
OFFICIAL_REPOSITORY = "https://github.com/octree-nn/octgpt"
DEFAULT_CHECKPOINT = "wst2001/OctGPT"
DEFAULT_VAE_CHECKPOINT = "wst2001/OctGPT"
DEFAULT_CLIP_PATH = "openai/clip-vit-large-patch14"
DEFAULT_OFFICIAL_CONFIG = Path("configs/Objaverse/objaverse_octar_text.yaml")
TOKEN_DEFINITION = {
    "output_token_count": (
        "native OctGPT transformer sequence positions: sum of octree nnum at "
        "depths full_depth through depth_stop"
    ),
    "structure_token_count": (
        "octree split-node sequence positions: sum of octree nnum at depths "
        "full_depth through depth_stop-1"
    ),
    "binary_decisions": (
        "structure_token_count + leaf_token_count * vq_groups; this expands each "
        "depth_stop BSQ vector into its autoregressively generated binary groups"
    ),
    "autoregressive_steps": "sum of configured OctGPT num_iters over all generated depths",
}


@dataclass
class OctGPTRuntime:
    torch: Any
    ocnn: Any
    solver: Any
    model: Any
    vqvae: Any
    text_encoder: Any
    flags: Any
    device_index: int
    source_root: Path
    official_config: Path
    source_revision: str | None


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_asset_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe[:160] or "asset"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_manifest(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen_orders: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(sample, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}.")
            missing = [key for key in ("sample_order", "dataset_index", "asset_id", "text_description") if key not in sample]
            if missing:
                raise ValueError(f"Missing {', '.join(missing)} in {path}:{line_number}.")
            sample_order = int(sample["sample_order"])
            if sample_order in seen_orders:
                raise ValueError(f"Duplicate sample_order={sample_order} in {path}:{line_number}.")
            seen_orders.add(sample_order)
            sample["sample_order"] = sample_order
            sample["dataset_index"] = int(sample["dataset_index"])
            sample["asset_id"] = str(sample["asset_id"])
            sample["text_description"] = str(sample["text_description"]).strip()
            if not sample["text_description"]:
                raise ValueError(f"Empty text_description in {path}:{line_number}.")
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"Manifest contains no samples: {path}")
    return samples


def _result_path(output_dir: Path, sample: Mapping[str, Any]) -> Path:
    return output_dir / f"{int(sample['sample_order']):03d}_{_safe_asset_id(str(sample['asset_id']))}.json"


def _load_matching_result(
    path: Path, config_hash: str, sample: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    matches = (
        isinstance(payload, dict)
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("method") == METHOD
        and payload.get("config_hash") == config_hash
        and payload.get("sample_order") == int(sample["sample_order"])
        and payload.get("dataset_index") == int(sample["dataset_index"])
        and payload.get("asset_id") == str(sample["asset_id"])
        and payload.get("text_description_sha256")
        == _sha256_text(str(sample["text_description"]))
    )
    return payload if matches else None


def _pending_samples(
    samples: Sequence[Mapping[str, Any]], output_dir: Path, config_hash: str, overwrite: bool
) -> tuple[list[Mapping[str, Any]], int]:
    pending: list[Mapping[str, Any]] = []
    skipped = 0
    for sample in samples:
        path = _result_path(output_dir, sample)
        if not path.exists() or overwrite:
            pending.append(sample)
        else:
            existing = _load_matching_result(path, config_hash, sample)
            if existing is None:
                raise RuntimeError(
                    "Refusing to overwrite a result whose identity or config hash differs: "
                    f"{path}; pass --overwrite to replace it."
                )
            if existing.get("status") == "ok":
                skipped += 1
            else:
                pending.append(sample)
    return pending, skipped


def _resolve_source_root(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_path = os.environ.get("OCTGPT_SOURCE_ROOT")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("evaluation/inference_efficiency/third_party/octgpt"),
        ]
    )
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "main_octgpt.py").is_file() and (root / "models/octgpt.py").is_file():
            return root
    rendered = ", ".join(str(path.expanduser()) for path in candidates)
    raise FileNotFoundError(
        "Official OctGPT source was not found. Pass --source-root or set OCTGPT_SOURCE_ROOT. "
        f"Checked: {rendered}"
    )


def _resolve_official_config(source_root: Path, config: Path) -> Path:
    candidate = config.expanduser()
    if not candidate.is_absolute():
        candidate = source_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Official OctGPT config not found: {candidate}")
    return candidate


def _validate_local_artifacts(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required local artifact not found: {path}")


def _git_revision(source_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        revision = completed.stdout.strip()
        return revision or None
    except (OSError, subprocess.SubprocessError):
        return None


def _import_official_main(source_root: Path) -> Any:
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module_path = source_root / "main_octgpt.py"
    spec = importlib.util.spec_from_file_location("_octgpt_official_main", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import official OctGPT entry point: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with _working_directory(source_root):
        spec.loader.exec_module(module)
    return module


def _load_official_flags(source_root: Path, config_path: Path, checkpoint: Path, vae_checkpoint: Path, device: int) -> Any:
    from thsolver import config as solver_config

    flags = solver_config.get_config().clone()
    flags.defrost()
    with _working_directory(source_root):
        configs = solver_config._load_from_file(str(config_path))  # noqa: SLF001 - official loader handles BASE.
    for config in configs:
        flags.merge_from_other_cfg(config)
    flags.SOLVER.gpu = (int(device),)
    flags.SOLVER.ckpt = str(checkpoint)
    flags.SOLVER.run = "generate"
    flags.SOLVER.progress_bar = False
    flags.MODEL.vqvae_ckpt = str(vae_checkpoint)
    flags.DATA.test.batch_size = 1
    flags.freeze()
    return flags


def _make_local_text_encoder_class(torch: Any, clip_path: Path) -> type:
    class LocalTextEncoder(torch.nn.Module):
        def __init__(self, encoder_type: str):
            super().__init__()
            if encoder_type != "clip":
                raise ValueError(f"OctGPT text evaluation requires encoder_type='clip', got {encoder_type!r}.")
            from transformers import CLIPProcessor, CLIPTextModel

            self.processor = CLIPProcessor.from_pretrained(
                str(clip_path), trust_remote_code=False, local_files_only=True
            )
            self.encoder = CLIPTextModel.from_pretrained(
                str(clip_path), trust_remote_code=False, local_files_only=True
            )

        def forward(self, text: list[str], device: int | str | Any) -> Any:
            # Preserve the official TextEncoder call exactly; only the model
            # location is replaced so no Hugging Face download is attempted.
            inputs = self.processor(text=text, return_tensors="pt", max_length=77)
            if isinstance(device, int):
                device = torch.device(f"cuda:{device}")
            inputs = inputs.to(device)
            return self.encoder(**inputs).last_hidden_state

    return LocalTextEncoder


def _load_runtime(args: argparse.Namespace) -> OctGPTRuntime:
    import torch
    import ocnn

    if not torch.cuda.is_available():
        raise RuntimeError("OctGPT efficiency evaluation requires CUDA.")
    torch.cuda.set_device(args.device)

    source_root = _resolve_source_root(args.source_root)
    official_config = _resolve_official_config(source_root, args.official_config)
    artifacts = resolve_method_artifacts("octgpt", vars(args))
    checkpoint = args.checkpoint = Path(artifacts["checkpoint"])
    vae_checkpoint = args.vae_checkpoint = Path(artifacts["vae_checkpoint"])
    clip_path = args.clip_path = Path(artifacts["clip_path"])
    _validate_local_artifacts([checkpoint, vae_checkpoint, clip_path])

    official_main = _import_official_main(source_root)
    official_main.TextEncoder = _make_local_text_encoder_class(torch, clip_path)

    # The official module binds tqdm as a function. Replacing it only suppresses
    # nested generation progress bars; it does not change the generated sequence.
    official_model_module = sys.modules.get("models.octgpt")
    if official_model_module is not None:
        official_model_module.tqdm = lambda iterable, *unused_args, **unused_kwargs: iterable

    flags = _load_official_flags(source_root, official_config, checkpoint, vae_checkpoint, args.device)
    solver = official_main.OctGPTSolver(flags, is_master=False)
    with _working_directory(source_root):
        solver.config_model()
        solver.load_checkpoint()
    solver.model.eval()
    solver.model_module.eval()
    solver.vqvae_module.eval()
    solver.cond_enc.eval()

    return OctGPTRuntime(
        torch=torch,
        ocnn=ocnn,
        solver=solver,
        model=solver.model_module,
        vqvae=solver.vqvae_module,
        text_encoder=solver.cond_enc,
        flags=flags,
        device_index=int(args.device),
        source_root=source_root,
        official_config=official_config,
        source_revision=_git_revision(source_root),
    )


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sample_seed(base_seed: int, sample: Mapping[str, Any]) -> int:
    return int(base_seed) + int(sample["dataset_index"])


def _encode_condition(runtime: OctGPTRuntime, text_description: str) -> Any:
    torch = runtime.torch
    with torch.inference_mode():
        condition = runtime.text_encoder([text_description], device=runtime.device_index)
    # Do not allow condition-encoder kernels to spill into the generation interval.
    torch.cuda.synchronize(runtime.device_index)
    return condition


def _num_iterations(model: Any, full_depth: int, depth_stop: int) -> int:
    depths = depth_stop - full_depth + 1
    num_iters = model.num_iters
    if isinstance(num_iters, (list, tuple)):
        if len(num_iters) < depths:
            raise ValueError(f"OctGPT num_iters has {len(num_iters)} values for {depths} generated depths.")
        return sum(int(num_iters[index]) for index in range(depths))
    return depths * int(num_iters)


def _nnum_by_depth(octree: Any, full_depth: int, depth_stop: int) -> dict[int, int]:
    return {depth: int(octree.nnum[depth]) for depth in range(full_depth, depth_stop + 1)}


def _generate_one(runtime: OctGPTRuntime, sample: Mapping[str, Any], base_seed: int) -> dict[str, Any]:
    torch = runtime.torch
    flags = runtime.flags
    full_depth = int(flags.MODEL.full_depth)
    depth_stop = int(flags.MODEL.depth_stop)
    depth = int(flags.MODEL.depth)
    sample_seed = _sample_seed(base_seed, sample)

    # Both condition encoding and the initial full octree are explicitly outside
    # the timed region requested by the benchmark protocol.
    condition = _encode_condition(runtime, str(sample["text_description"]))
    _seed_everything(torch, sample_seed)
    octree = runtime.ocnn.octree.init_octree(depth, full_depth, 1, runtime.device_index)
    torch.cuda.synchronize(runtime.device_index)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", enabled=bool(runtime.solver.use_amp)):
        octree_out, _vq_code = runtime.model.generate(
            octree=octree,
            depth_low=full_depth,
            depth_high=depth_stop,
            vqvae=runtime.vqvae,
            condition=condition,
        )
    end_event.record()
    end_event.synchronize()
    wall_latency_ms = (time.perf_counter() - wall_start) * 1000.0
    cuda_latency_ms = float(start_event.elapsed_time(end_event))

    nnum = _nnum_by_depth(octree_out, full_depth, depth_stop)
    structure_positions = sum(nnum[d] for d in range(full_depth, depth_stop))
    leaf_positions = nnum[depth_stop]
    sequence_positions = structure_positions + leaf_positions
    vq_groups = int(runtime.model.vq_groups)
    binary_decisions = structure_positions + leaf_positions * vq_groups

    return {
        "sample_seed": sample_seed,
        "latency_ms": cuda_latency_ms,
        "cuda_latency_ms": cuda_latency_ms,
        "wall_latency_ms": wall_latency_ms,
        "output_token_count": sequence_positions,
        "structure_token_count": structure_positions,
        "sequence_positions": sequence_positions,
        "binary_decisions": binary_decisions,
        "leaf_token_count": leaf_positions,
        "vq_groups": vq_groups,
        "nodes_by_depth": {str(key): value for key, value in nnum.items()},
        "autoregressive_steps": _num_iterations(runtime.model, full_depth, depth_stop),
    }


def _runtime_metadata(runtime: OctGPTRuntime, args: argparse.Namespace) -> dict[str, Any]:
    torch = runtime.torch
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "device_index": runtime.device_index,
        "device_name": torch.cuda.get_device_name(runtime.device_index),
        "amp": bool(runtime.solver.use_amp),
        "offline": True,
        "warmup": int(args.warmup),
    }


def _base_result(
    sample: Mapping[str, Any], args: argparse.Namespace, runtime: OctGPTRuntime
) -> dict[str, Any]:
    description = str(sample["text_description"])
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "config_hash": args.config_hash,
        "sample_order": int(sample["sample_order"]),
        "dataset_index": int(sample["dataset_index"]),
        "asset_id": str(sample["asset_id"]),
        "text_description_sha256": _sha256_text(description),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "token_definition": TOKEN_DEFINITION,
        "model": {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "vae_checkpoint": str(args.vae_checkpoint.expanduser().resolve()),
            "clip_path": str(args.clip_path.expanduser().resolve()),
        },
        "source": {
            "repository": OFFICIAL_REPOSITORY,
            "source_root": str(runtime.source_root),
            "commit": args.source_commit or runtime.source_revision,
            "detected_revision": runtime.source_revision,
            "config": str(runtime.official_config),
        },
        "runtime": _runtime_metadata(runtime, args),
    }


def _success_result(
    sample: Mapping[str, Any], args: argparse.Namespace, runtime: OctGPTRuntime, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    result = _base_result(sample, args, runtime)
    result.update(metrics)
    result.update(
        {
            "status": "ok",
            "termination": {
                "reason": "configured_depth_stop_reached",
                "depth_stop": int(runtime.flags.MODEL.depth_stop),
            },
            "truncated": False,
            "valid_structure": True,
            "error": None,
        }
    )
    return result


def _failure_result(
    sample: Mapping[str, Any], args: argparse.Namespace, runtime: OctGPTRuntime, exc: BaseException
) -> dict[str, Any]:
    result = _base_result(sample, args, runtime)
    result.update(
        {
            "status": "error",
            "latency_ms": None,
            "cuda_latency_ms": None,
            "wall_latency_ms": None,
            "output_token_count": None,
            "structure_token_count": None,
            "sequence_positions": None,
            "binary_decisions": None,
            "autoregressive_steps": None,
            "termination": {"reason": "error"},
            "truncated": False,
            "valid_structure": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    )
    return result


def _run_warmup(runtime: OctGPTRuntime, samples: Sequence[Mapping[str, Any]], count: int, seed: int) -> None:
    if count <= 0:
        return
    print(f"Running {count} OctGPT warmup generation(s)...", flush=True)
    for index in range(count):
        sample = samples[index % len(samples)]
        _generate_one(runtime, sample, seed - count + index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark only the official OctGPT autoregressive generation stage."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Unified JSONL sample manifest.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for one atomic JSON per asset.")
    parser.add_argument("--config-hash", required=True, help="Hash of the resolved unified benchmark config.")
    parser.add_argument(
        "--source-commit",
        help="Pinned official OctGPT commit recorded by the benchmark orchestrator.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N manifest rows.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-asset results, including mismatched identities or config hashes.",
    )
    parser.add_argument("--device", type=int, default=0, help="Local CUDA device index after CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--source-root", type=Path, help="Checkout of the official octree-nn/octgpt repository.")
    parser.add_argument("--official-config", type=Path, default=DEFAULT_OFFICIAL_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vae-checkpoint", default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--clip-path", default=DEFAULT_CLIP_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    samples = _load_manifest(args.manifest, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending, skipped = _pending_samples(
        samples, args.output_dir, args.config_hash, args.overwrite
    )
    if not pending:
        print(f"OctGPT: all {len(samples)} result(s) already match config_hash={args.config_hash}.")
        return 0

    print(f"OctGPT: selected={len(samples)} pending={len(pending)} skipped={skipped}", flush=True)
    runtime = _load_runtime(args)
    _run_warmup(runtime, pending, args.warmup, args.seed)

    failed = 0
    for position, sample in enumerate(pending, start=1):
        output_path = _result_path(args.output_dir, sample)
        print(
            f"[OctGPT {position}/{len(pending)}] order={sample['sample_order']} asset={sample['asset_id']}",
            flush=True,
        )
        try:
            metrics = _generate_one(runtime, sample, args.seed)
            result = _success_result(sample, args, runtime, metrics)
        except Exception as exc:  # Keep the remaining paper benchmark resumable.
            failed += 1
            result = _failure_result(sample, args, runtime, exc)
            print(f"OctGPT failed for {sample['asset_id']}: {exc}", file=sys.stderr, flush=True)
        _atomic_write_json(output_path, result)
        if failed and args.fail_fast:
            break

    print(f"OctGPT complete: succeeded={len(pending) - failed} failed={failed} skipped={skipped}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
