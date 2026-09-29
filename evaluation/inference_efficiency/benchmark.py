#!/usr/bin/env python3
"""One-command, resumable autoregressive efficiency benchmark for text-to-3D."""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_ROOT.parents[1]
DEFAULT_CONFIG = BENCHMARK_ROOT / "benchmark_config.json"
PROTOCOL_VERSION = "ar-efficiency-v1"
RESULT_SCHEMA_VERSION = "inference-efficiency.v1"
METHOD_ORDER = ["shapellm_omni", "3dgen_r1", "sar3d", "llama_mesh", "octgpt", "octllm"]
RAW_METHOD_NAMES = {
    "shapellm_omni": "ShapeLLM-Omni",
    "3dgen_r1": "3DGen-R1",
    "sar3d": "SAR3D",
    "llama_mesh": "LLaMA-Mesh",
    "octgpt": "OctGPT",
    "octllm": "OctLLM",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampled_sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a large file's size plus first/last chunks for cheap change detection."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        if size > chunk_size:
            handle.seek(max(chunk_size, size - chunk_size))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def path_fingerprint(path: Path) -> dict[str, Any]:
    """Cheap local-artifact identity without hashing multi-GB checkpoint shards."""
    resolved = path.resolve()
    if not resolved.exists():
        return {"path": str(resolved), "missing": True}
    if resolved.is_file():
        stat = resolved.stat()
        result: dict[str, Any] = {
            "path": str(resolved),
            "kind": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= 16 * 1024 * 1024:
            result["sha256"] = sha256_file(resolved)
        else:
            result["sampled_sha256_first_last_1mib"] = sampled_sha256_file(resolved)
        return result

    entries: list[dict[str, Any]] = []
    relevant_suffixes = {
        ".json",
        ".jinja",
        ".model",
        ".txt",
        ".yaml",
        ".yml",
        ".safetensors",
        ".bin",
        ".pth",
        ".pt",
    }
    for child in sorted(resolved.iterdir(), key=lambda item: item.name):
        if not child.is_file() or (child.suffix not in relevant_suffixes and child.name != "latest"):
            continue
        stat = child.stat()
        entry: dict[str, Any] = {"name": child.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if stat.st_size <= 16 * 1024 * 1024:
            entry["sha256"] = sha256_file(child)
        else:
            entry["sampled_sha256_first_last_1mib"] = sampled_sha256_file(child)
        entries.append(entry)
    return {"path": str(resolved), "kind": "directory", "entries": entries}


@functools.cache
def environment_fingerprint(python_path_string: str) -> dict[str, Any]:
    python_path = Path(python_path_string).resolve()
    result = path_fingerprint(python_path)
    conda_history = python_path.parents[1] / "conda-meta" / "history"
    if conda_history.is_file():
        result["conda_history_sha256"] = sha256_file(conda_history)
    probe = (
        "import importlib.metadata as m,json,platform; "
        "names=['torch','torchvision','transformers','accelerate','xformers','ocnn','ognn','thsolver',"
        "'timm','typed-argument-parser','numpy','omegaconf','pillow','beartype','blobfile']; "
        "versions={n:(m.version(n) if any((d.metadata.get('Name') or '').lower()==n.lower() for d in m.distributions()) "
        "else None) for n in names}; "
        "print(json.dumps({'python':platform.python_version(),'packages':versions},sort_keys=True))"
    )
    try:
        probed = subprocess.check_output(
            [str(python_path), "-c", probe], text=True, stderr=subprocess.STDOUT, timeout=30
        ).strip()
        result["versions"] = json.loads(probed)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        result["version_probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


@functools.cache
def python_tree_hash(root_string: str) -> str:
    root = Path(root_string)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_methods(raw: str, config: dict[str, Any]) -> list[str]:
    if raw.strip().lower() == "all":
        selected = [name for name in METHOD_ORDER if name in config["methods"]]
    else:
        selected = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in selected if name not in config["methods"]]
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    if not selected:
        raise ValueError("No methods selected")
    return selected


def safe_result_name(row: dict[str, Any]) -> str:
    asset = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["asset_id"])).strip("._") or "asset"
    return f"{int(row['sample_order']):03d}_{asset}.json"


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"sample_order", "dataset_index", "asset_id", "text_description"}
            missing = required.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            rows.append(row)
    return rows


def sample_dataset(
    dataset_path: Path,
    output_dir: Path,
    *,
    num_assets: int,
    seed: int,
    overwrite_selection: bool,
) -> tuple[Path, dict[str, Any]]:
    dataset_sha = sha256_file(dataset_path)
    selection_path = output_dir / "selection.json"
    manifest_path = output_dir / "assets.jsonl"
    expected_settings = {
        "task": "text_to_3d",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha,
        "num_assets": num_assets,
        "seed": seed,
        "sampling": "random.Random(seed).sample(range(dataset_size), num_assets)",
    }

    if selection_path.exists() and manifest_path.exists() and not overwrite_selection:
        existing = load_json(selection_path)
        mismatches = {key: (existing.get(key), value) for key, value in expected_settings.items() if existing.get(key) != value}
        if mismatches:
            raise RuntimeError(
                f"Existing selection at {selection_path} has different settings: {mismatches}. "
                "Use a new --output-dir or explicitly pass --overwrite-selection."
            )
        rows = read_manifest(manifest_path)
        if len(rows) != num_assets or sha256_file(manifest_path) != existing.get("manifest_sha256"):
            raise RuntimeError(f"Selection manifest is incomplete or modified: {manifest_path}")
        return manifest_path, existing

    dataset = load_json(dataset_path)
    if not isinstance(dataset, list):
        raise TypeError(f"Expected a JSON list in {dataset_path}")
    if num_assets < 1 or num_assets > len(dataset):
        raise ValueError(f"--num-assets must be in [1, {len(dataset)}]")

    sampled_indices = random.Random(seed).sample(range(len(dataset)), num_assets)
    rows: list[dict[str, Any]] = []
    for sample_order, dataset_index in enumerate(sampled_indices):
        item = dataset[dataset_index]
        missing = {"asset_id", "text_description"}.difference(item)
        if missing:
            raise ValueError(f"Dataset row {dataset_index} is missing {sorted(missing)}")
        rows.append(
            {
                "sample_order": sample_order,
                "dataset_index": dataset_index,
                "asset_id": str(item["asset_id"]),
                "text_description": str(item["text_description"]),
                "mesh_path": item.get("mesh_path"),
                "render_image_path": item.get("render_image_path"),
            }
        )

    manifest_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_text(manifest_path, manifest_text)
    selection = {
        "schema_version": 1,
        **expected_settings,
        "dataset_size": len(dataset),
        "sampled_indices": sampled_indices,
        "asset_ids": [row["asset_id"] for row in rows],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_json(selection_path, selection)
    return manifest_path, selection


def run_checked(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(command)
    print(f"+ {printable}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def git_output(arguments: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def prepare_repository(name: str, repo: dict[str, Any], *, proxy: str) -> Path:
    destination = BENCHMARK_ROOT / "third_party" / repo["directory"]
    commit = repo["commit"]

    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run_checked(["git", "init", str(destination)])
        run_checked(["git", "-C", str(destination), "remote", "add", "origin", repo["url"]])
    elif not (destination / ".git").exists():
        raise RuntimeError(f"Source destination exists but is not a git repository: {destination}")

    try:
        current = git_output(["rev-parse", "HEAD"], destination)
    except subprocess.CalledProcessError:
        current = ""
    tracked_changes = bool(
        current and git_output(["status", "--porcelain", "--untracked-files=no"], destination)
    )
    if current == commit:
        if tracked_changes:
            raise RuntimeError(f"Pinned third-party checkout has tracked modifications: {destination}")
        return destination
    if tracked_changes:
        raise RuntimeError(f"Refusing to change a dirty third-party checkout: {destination}")

    # The proxy is scoped to these small, source-only Git operations. Model
    # checkpoints are never fetched by this benchmark.
    fetch = ["git", "-C", str(destination)]
    if proxy:
        fetch.extend(["-c", f"http.proxy={proxy}", "-c", f"https.proxy={proxy}"])
    fetch.extend(["fetch", "--depth", "1", "origin", commit])
    run_checked(fetch)
    run_checked(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"])
    actual = git_output(["rev-parse", "HEAD"], destination)
    if actual != commit:
        raise RuntimeError(f"Expected {commit} for {name}, got {actual}")
    return destination


def prepare_sources(
    methods: list[str], config: dict[str, Any], *, skip: bool
) -> dict[str, Path]:
    source_paths: dict[str, Path] = {}
    repo_names = {config["methods"][method].get("repository") for method in methods}
    for name in sorted(repo_name for repo_name in repo_names if repo_name):
        repo = config["repositories"][name]
        destination = BENCHMARK_ROOT / "third_party" / repo["directory"]
        if skip:
            if not destination.exists():
                raise FileNotFoundError(
                    f"Missing {destination}; remove --skip-source-prepare so the small source repository can be cloned"
                )
            if not (destination / ".git").exists():
                raise RuntimeError(f"Expected a pinned git checkout at {destination}")
            actual = git_output(["rev-parse", "HEAD"], destination)
            if actual != repo["commit"]:
                raise RuntimeError(
                    f"Wrong commit for {name}: expected {repo['commit']}, found {actual}"
                )
            if git_output(["status", "--porcelain", "--untracked-files=no"], destination):
                raise RuntimeError(f"Pinned third-party checkout has tracked modifications: {destination}")
            source_paths[name] = destination
        else:
            proxy = os.environ.get("BENCHMARK_GIT_PROXY", config["git_proxy"])
            source_paths[name] = prepare_repository(name, repo, proxy=proxy)
    return source_paths


def required_method_paths(method: str, method_cfg: dict[str, Any], source_paths: dict[str, Path]) -> list[Path]:
    paths = [Path(method_cfg["python"]), BENCHMARK_ROOT / method_cfg["runner"]]
    for key in ("model_path", "ar_checkpoint", "vae_checkpoint", "checkpoint", "clip_path"):
        if method_cfg.get(key):
            paths.append(Path(method_cfg[key]))
    if method_cfg.get("inference_config"):
        paths.append(BENCHMARK_ROOT / method_cfg["inference_config"])
    if method_cfg.get("repository") and method_cfg["repository"] in source_paths:
        paths.append(source_paths[method_cfg["repository"]])
    return paths


def preflight(methods: list[str], config: dict[str, Any], source_paths: dict[str, Path]) -> None:
    failures: list[str] = []
    source_markers = {
        "shapellm_omni": ("app.py",),
        "3dgen_r1": ("inference.py",),
        "sar3d": ("models/var.py", "files/empty_text_embedding.npy"),
        "llama_mesh": ("app.py",),
        "octgpt": ("main_octgpt.py", "models/octgpt.py", "configs/Objaverse/objaverse_octar_text.yaml"),
    }
    for method in methods:
        cfg = config["methods"][method]
        for path in required_method_paths(method, cfg, source_paths):
            if not path.exists():
                failures.append(f"{method}: missing {path}")
        python_path = Path(cfg["python"])
        if python_path.exists() and not os.access(python_path, os.X_OK):
            failures.append(f"{method}: Python is not executable: {python_path}")
        if cfg.get("model_path"):
            model_dir = Path(cfg["model_path"])
            if model_dir.is_dir():
                if not (model_dir / "config.json").is_file():
                    failures.append(f"{method}: model directory has no config.json: {model_dir}")
                weight_candidates = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("pytorch_model*.bin"))
                if not weight_candidates:
                    failures.append(f"{method}: no local model weight shards found in {model_dir}")
        repo_name = cfg.get("repository")
        if repo_name in source_paths:
            for marker in source_markers.get(repo_name, ()):
                marker_path = source_paths[repo_name] / marker
                if not marker_path.is_file():
                    failures.append(f"{method}: incomplete official source; missing {marker_path}")
    if failures:
        details = "\n  - ".join(failures)
        raise RuntimeError(f"Preflight failed:\n  - {details}")
    print(f"Preflight passed for: {', '.join(methods)}", flush=True)


def validate_single_gpu(gpu: str, expected_name_substring: str | None) -> str:
    if not gpu.strip() or "," in gpu or " " in gpu.strip():
        raise ValueError(f"--gpu must identify exactly one physical GPU, got {gpu!r}")
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        print(
            f"Warning: nvidia-smi could not resolve physical GPU {gpu!r}; "
            "the CUDA workers will still record the actual visible device.",
            file=sys.stderr,
        )
        return "unavailable"
    if expected_name_substring and expected_name_substring.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"Paper protocol requires a GPU containing {expected_name_substring!r}; GPU {gpu} is {gpu_name!r}"
        )
    print(f"GPU preflight passed: physical GPU {gpu} = {gpu_name}", flush=True)
    return gpu_name


def local_source_commit() -> str:
    try:
        commit = git_output(["rev-parse", "HEAD"], REPO_ROOT)
        dirty = bool(git_output(["status", "--porcelain", "--", "scripts/generate_octree.py", str(BENCHMARK_ROOT)], REPO_ROOT))
        return f"{commit}{'+dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def method_config_hash(
    method: str,
    method_cfg: dict[str, Any],
    config: dict[str, Any],
    selection: dict[str, Any],
    *,
    gpu: str,
    gpu_identity: str | None = None,
    seed: int,
    warmup: int,
) -> str:
    runner = BENCHMARK_ROOT / method_cfg["runner"]
    source = None
    if method_cfg.get("repository"):
        source = config["repositories"][method_cfg["repository"]]
    fingerprint: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "method": method,
        "method_config": method_cfg,
        "official_source": source,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "gpu_selection": gpu,
        "gpu_identity": gpu_identity or "not-recorded",
        "expected_gpu_name_substring": config.get("expected_gpu_name_substring"),
        "generation_seed": seed,
        "warmup_generations": warmup,
        "orchestrator_sha256": sha256_file(Path(__file__)),
        "runner_sha256": sha256_file(runner),
        "python_environment": environment_fingerprint(method_cfg["python"]),
        "artifacts": {
            key: path_fingerprint(Path(method_cfg[key]))
            for key in ("model_path", "ar_checkpoint", "vae_checkpoint", "checkpoint", "clip_path")
            if method_cfg.get(key)
        },
    }
    if method in {"shapellm_omni", "3dgen_r1", "llama_mesh"}:
        fingerprint["runner_support_sha256"] = sha256_file(BENCHMARK_ROOT / "runners" / "common.py")
    fingerprint["model_artifacts_sha256"] = sha256_file(BENCHMARK_ROOT / "model_artifacts.py")
    fingerprint["model_sources_sha256"] = sha256_file(REPO_ROOT / "scripts" / "model_sources.py")
    if method == "octllm":
        fingerprint["custom_loop_sha256"] = sha256_file(REPO_ROOT / "scripts" / "generate_octree.py")
        inference_config = BENCHMARK_ROOT / method_cfg["inference_config"]
        fingerprint["inference_config"] = path_fingerprint(inference_config)
        fingerprint["local_source_commit"] = local_source_commit()
        fingerprint["llamafactory_python_tree_sha256"] = python_tree_hash(
            str(REPO_ROOT / "src" / "llamafactory")
        )
    return stable_hash(fingerprint)


def build_method_command(
    method: str,
    method_cfg: dict[str, Any],
    config: dict[str, Any],
    source_paths: dict[str, Path],
    *,
    manifest: Path,
    output_dir: Path,
    config_hash: str,
    seed: int,
    warmup: int,
    overwrite: bool,
) -> list[str]:
    runner = BENCHMARK_ROOT / method_cfg["runner"]
    command = [
        method_cfg["python"],
        str(runner),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir / "raw" / method),
        "--config-hash",
        config_hash,
        "--seed",
        str(seed),
        "--warmup",
        str(warmup),
    ]
    if overwrite:
        command.append("--overwrite")

    repo_name = method_cfg.get("repository")
    source_commit = config["repositories"][repo_name]["commit"] if repo_name else local_source_commit()
    if method in {"shapellm_omni", "3dgen_r1", "llama_mesh"}:
        command.extend(
            [
                "--method",
                method,
                "--model-path",
                method_cfg["model_path"],
                "--source-commit",
                source_commit,
            ]
        )
    elif method == "octllm":
        command.extend(
            [
                "--config",
                str(BENCHMARK_ROOT / method_cfg["inference_config"]),
                "--model-path",
                method_cfg["model_path"],
                "--source-commit",
                source_commit,
            ]
        )
    elif method == "sar3d":
        command.extend(
            [
                "--source-dir",
                str(source_paths[repo_name]),
                "--ar-checkpoint",
                method_cfg["ar_checkpoint"],
                "--vae-checkpoint",
                method_cfg["vae_checkpoint"],
                "--clip-path",
                method_cfg["clip_path"],
                "--source-commit",
                source_commit,
            ]
        )
    elif method == "octgpt":
        command.extend(
            [
                "--source-root",
                str(source_paths[repo_name]),
                "--checkpoint",
                method_cfg["checkpoint"],
                "--vae-checkpoint",
                method_cfg["vae_checkpoint"],
                "--clip-path",
                method_cfg["clip_path"],
                "--source-commit",
                source_commit,
            ]
        )
    else:
        raise AssertionError(method)
    return command


def run_methods(
    methods: list[str],
    config: dict[str, Any],
    source_paths: dict[str, Path],
    *,
    manifest: Path,
    selection: dict[str, Any],
    output_dir: Path,
    gpu: str,
    gpu_identity: str | None,
    seed: int,
    warmup: int,
    overwrite: bool,
    fail_fast: bool,
) -> dict[str, Any]:
    failures: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    environment = os.environ.copy()
    matplotlib_cache = output_dir / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": str(seed),
            "MPLCONFIGDIR": str(matplotlib_cache),
        }
    )

    for method in methods:
        method_cfg = config["methods"][method]
        config_hash = method_config_hash(
            method,
            method_cfg,
            config,
            selection,
            gpu=gpu,
            gpu_identity=gpu_identity,
            seed=seed,
            warmup=warmup,
        )
        hashes[method] = config_hash
        command = build_method_command(
            method,
            method_cfg,
            config,
            source_paths,
            manifest=manifest,
            output_dir=output_dir,
            config_hash=config_hash,
            seed=seed,
            warmup=warmup,
            overwrite=overwrite,
        )
        print(f"\n===== {method_cfg['label']} ({method}) =====", flush=True)
        try:
            run_checked(command, cwd=REPO_ROOT, env=environment)
        except subprocess.CalledProcessError as exc:
            failures[method] = {"returncode": exc.returncode, "command": command}
            print(f"Method {method} failed with exit code {exc.returncode}; partial results are retained.", file=sys.stderr)
            if fail_fast:
                break

    run_record = {
        "protocol": PROTOCOL_VERSION,
        "methods": methods,
        "config_hashes": hashes,
        "gpu": gpu,
        "gpu_identity": gpu_identity,
        "seed": seed,
        "warmup": warmup,
        "failures": failures,
    }
    atomic_json(output_dir / "run.json", run_record)
    return run_record


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_summary(records: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [float(record[key]) for record in records if isinstance(record.get(key), (int, float))]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 95.0),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def valid_result_number(payload: dict[str, Any], key: str, *, positive: bool) -> bool:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and (numeric > 0.0 if positive else numeric >= 0.0)


def fmt(value: float | int | None, digits: int = 2) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def latex_escape(value: str) -> str:
    replacements = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}
    return "".join(replacements.get(char, char) for char in value)


def aggregate(
    methods: list[str],
    config: dict[str, Any],
    selection: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
    *,
    gpu: str,
    gpu_identity: str | None = None,
    seed: int,
    warmup: int,
) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    expected = len(manifest)
    per_asset_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for method in methods:
        records: list[dict[str, Any]] = []
        missing = 0
        metadata_fingerprint: str | None = None
        expected_config_hash = method_config_hash(
            method,
            config["methods"][method],
            config,
            selection,
            gpu=gpu,
            gpu_identity=gpu_identity,
            seed=seed,
            warmup=warmup,
        )
        for row in manifest:
            result_path = output_dir / "raw" / method / safe_result_name(row)
            if result_path.exists():
                try:
                    result = load_json(result_path)
                except (OSError, json.JSONDecodeError) as exc:
                    result = {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}
                if not isinstance(result, dict):
                    result = {
                        "status": "error",
                        "error": {
                            "type": "ResultSchemaError",
                            "message": "raw result must be a JSON object",
                        },
                    }
            else:
                result = {"status": "missing", "error": None}
                missing += 1
            if result.get("status") != "missing" and result.get("config_hash") != expected_config_hash:
                result = {
                    "status": "stale_config",
                    "error": {
                        "type": "ConfigHashMismatch",
                        "message": (
                            f"expected {expected_config_hash}, found {result.get('config_hash')}"
                        ),
                    },
                }
            elif result.get("status") != "missing":
                expected_description_hash = hashlib.sha256(
                    str(row["text_description"]).encode("utf-8")
                ).hexdigest()
                identity_matches = (
                    result.get("sample_order") == int(row["sample_order"])
                    and result.get("dataset_index") == int(row["dataset_index"])
                    and result.get("asset_id") == str(row["asset_id"])
                    and result.get("text_description_sha256") == expected_description_hash
                )
                if not identity_matches:
                    result = {
                        "status": "invalid_identity",
                        "error": {
                            "type": "SampleIdentityMismatch",
                            "message": "raw result does not match the selected manifest row",
                        },
                    }
            if result.get("status") == "ok":
                invalid_fields: list[str] = []
                if result.get("schema_version") != RESULT_SCHEMA_VERSION:
                    invalid_fields.append("schema_version")
                if result.get("method") != RAW_METHOD_NAMES[method]:
                    invalid_fields.append("method")

                for key in ("latency_ms", "cuda_latency_ms", "wall_latency_ms"):
                    if not valid_result_number(result, key, positive=True):
                        invalid_fields.append(key)
                if (
                    valid_result_number(result, "latency_ms", positive=True)
                    and valid_result_number(result, "cuda_latency_ms", positive=True)
                    and not math.isclose(
                        float(result["latency_ms"]),
                        float(result["cuda_latency_ms"]),
                        rel_tol=1e-9,
                        abs_tol=1e-6,
                    )
                ):
                    invalid_fields.append("latency_ms!=cuda_latency_ms")
                for key, positive in (
                    ("output_token_count", True),
                    ("structure_token_count", False),
                    ("autoregressive_steps", True),
                ):
                    if not valid_result_number(result, key, positive=positive) or not float(result[key]).is_integer():
                        invalid_fields.append(key)
                for key in ("truncated", "valid_structure"):
                    if not isinstance(result.get(key), bool):
                        invalid_fields.append(key)
                if result.get("termination") is None:
                    invalid_fields.append("termination")
                for key in ("token_definition", "runtime", "model", "source"):
                    if not isinstance(result.get(key), dict) or not result[key]:
                        invalid_fields.append(key)
                expected_gpu_name = config.get("expected_gpu_name_substring")
                runtime_info = result.get("runtime")
                if isinstance(runtime_info, dict) and expected_gpu_name:
                    runtime_gpu_name = (
                        runtime_info.get("gpu_name")
                        or runtime_info.get("gpu")
                        or runtime_info.get("device_name")
                    )
                    if (
                        not isinstance(runtime_gpu_name, str)
                        or expected_gpu_name.lower() not in runtime_gpu_name.lower()
                    ):
                        invalid_fields.append("runtime.gpu_name")

                if invalid_fields:
                    result = {
                        "status": "invalid_record",
                        "error": {
                            "type": "ResultSchemaError",
                            "message": "invalid or missing fields: " + ", ".join(sorted(set(invalid_fields))),
                        },
                    }
                else:
                    current_metadata = stable_hash(
                        {
                            "runtime": result["runtime"],
                            "model": result["model"],
                            "source": result["source"],
                            "token_definition": result["token_definition"],
                        }
                    )
                    if metadata_fingerprint is None:
                        metadata_fingerprint = current_metadata
                        records.append(result)
                    elif current_metadata != metadata_fingerprint:
                        result = {
                            "status": "inconsistent_metadata",
                            "error": {
                                "type": "RuntimeModelSourceMismatch",
                                "message": "runtime/model/source/token metadata differs within this method",
                            },
                        }
                    else:
                        records.append(result)
            error = result.get("error")
            if isinstance(error, dict):
                error = f"{error.get('type', 'Error')}: {error.get('message', '')}"
            per_asset_rows.append(
                {
                    "method": method,
                    "method_label": config["methods"][method]["label"],
                    "sample_order": row["sample_order"],
                    "dataset_index": row["dataset_index"],
                    "asset_id": row["asset_id"],
                    "status": result.get("status", "error"),
                    "output_token_count": result.get("output_token_count"),
                    "structure_token_count": result.get("structure_token_count"),
                    "sequence_positions": result.get("sequence_positions"),
                    "binary_decisions": result.get("binary_decisions"),
                    "leaf_token_count": result.get("leaf_token_count"),
                    "autoregressive_steps": result.get("autoregressive_steps"),
                    "latency_ms": result.get("latency_ms"),
                    "latency_s": result.get("latency_ms") / 1000.0 if isinstance(result.get("latency_ms"), (int, float)) else None,
                    "cuda_latency_ms": result.get("cuda_latency_ms"),
                    "wall_latency_ms": result.get("wall_latency_ms"),
                    "termination": result.get("termination"),
                    "truncated": result.get("truncated"),
                    "valid_structure": result.get("valid_structure"),
                    "error": error,
                    "result_path": str(result_path.resolve()),
                }
            )

        output_tokens = metric_summary(records, "output_token_count")
        structure_tokens = metric_summary(records, "structure_token_count")
        binary_decisions = metric_summary(records, "binary_decisions")
        autoregressive_steps = metric_summary(records, "autoregressive_steps")
        latency = metric_summary(records, "latency_ms")
        cuda_latency = metric_summary(records, "cuda_latency_ms")
        wall_latency = metric_summary(records, "wall_latency_ms")
        total_tokens = sum(float(record["output_token_count"]) for record in records if isinstance(record.get("output_token_count"), (int, float)))
        total_latency_ms = sum(float(record["latency_ms"]) for record in records if isinstance(record.get("latency_ms"), (int, float)))
        failed = expected - len(records) - missing
        summary = {
            "method": method,
            "label": config["methods"][method]["label"],
            "expected": expected,
            "completed": len(records),
            "failed": failed,
            "missing": missing,
            "complete": len(records) == expected and failed == 0 and missing == 0,
            "truncated": sum(record.get("truncated") is True for record in records),
            "valid_structure": sum(record.get("valid_structure") is True for record in records),
            "output_token_count": output_tokens,
            "structure_token_count": structure_tokens,
            "binary_decisions": binary_decisions,
            "autoregressive_steps": autoregressive_steps,
            "latency_ms": latency,
            "cuda_latency_ms": cuda_latency,
            "wall_latency_ms": wall_latency,
            "aggregate_output_tokens_per_second": (total_tokens * 1000.0 / total_latency_ms) if total_latency_ms else None,
            "token_definition": records[0].get("token_definition") if records else None,
            "runtime": records[0].get("runtime") if records else None,
            "model": records[0].get("model") if records else None,
            "source": records[0].get("source") if records else None,
        }
        summaries[method] = summary

    output_dir.mkdir(parents=True, exist_ok=True)
    per_asset_fields = list(per_asset_rows[0]) if per_asset_rows else []
    per_asset_temp = output_dir / ".per_asset.csv.tmp"
    with per_asset_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_asset_fields)
        writer.writeheader()
        writer.writerows(per_asset_rows)
    os.replace(per_asset_temp, output_dir / "per_asset.csv")

    summary_rows = []
    for method in methods:
        item = summaries[method]
        summary_rows.append(
            {
                "method": method,
                "label": item["label"],
                "completed": item["completed"],
                "expected": item["expected"],
                "failed": item["failed"],
                "missing": item["missing"],
                "truncated": item["truncated"],
                "mean_output_tokens": item["output_token_count"]["mean"],
                "std_output_tokens": item["output_token_count"]["std"],
                "mean_structure_tokens": item["structure_token_count"]["mean"],
                "mean_binary_decisions": item["binary_decisions"]["mean"],
                "mean_autoregressive_steps": item["autoregressive_steps"]["mean"],
                "mean_latency_ms": item["latency_ms"]["mean"],
                "std_latency_ms": item["latency_ms"]["std"],
                "median_latency_ms": item["latency_ms"]["median"],
                "p95_latency_ms": item["latency_ms"]["p95"],
                "mean_cuda_latency_ms": item["cuda_latency_ms"]["mean"],
                "mean_wall_latency_ms": item["wall_latency_ms"]["mean"],
                "aggregate_output_tokens_per_second": item["aggregate_output_tokens_per_second"],
            }
        )
    summary_temp = output_dir / ".summary.csv.tmp"
    with summary_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)
    os.replace(summary_temp, output_dir / "summary.csv")

    summary_blob = {
        "schema_version": 1,
        "protocol": PROTOCOL_VERSION,
        "latency_definition": "CUDA event elapsed time around autoregressive generation only; synchronized before/after",
        "expected_assets": expected,
        "gpu_selection": gpu,
        "gpu_identity": gpu_identity,
        "selection": {
            "dataset_path": selection.get("dataset_path"),
            "dataset_sha256": selection.get("dataset_sha256"),
            "seed": selection.get("seed"),
            "sampled_indices": selection.get("sampled_indices"),
            "manifest_sha256": selection.get("manifest_sha256"),
        },
        "methods": summaries,
    }
    atomic_json(output_dir / "summary.json", summary_blob)

    md_lines = [
        "| Method | Completed | Avg. output tokens | Avg. native structure tokens | AR latency (s) | Output tokens/s | Truncated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    tex_rows: list[str] = []
    for method in methods:
        item = summaries[method]
        latency_s = item["latency_ms"]["mean"] / 1000.0 if item["latency_ms"]["mean"] is not None else None
        md_lines.append(
            f"| {item['label']} | {item['completed']}/{item['expected']} | "
            f"{fmt(item['output_token_count']['mean'])} | {fmt(item['structure_token_count']['mean'])} | "
            f"{fmt(latency_s, 3)} | {fmt(item['aggregate_output_tokens_per_second'])} | {item['truncated']} |"
        )
        tex_rows.append(
            f"{latex_escape(item['label'])} & {item['completed']}/{item['expected']} & "
            f"{fmt(item['output_token_count']['mean'])} & {fmt(item['structure_token_count']['mean'])} & "
            f"{fmt(latency_s, 3)} & {fmt(item['aggregate_output_tokens_per_second'])} \\\\"
        )
    md_lines.extend(
        [
            "",
            "> Latency is CUDA-event time for the autoregressive region only. Token units are method-native and are not "
            "semantically identical (LLM BPE IDs, VQ indices, or octree positions); use the token-definition metadata when reporting.",
            "",
        ]
    )
    atomic_text(output_dir / "paper_table.md", "\n".join(md_lines))
    tex = "\n".join(
        [
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Method & $N$ & Output tokens & Structure tokens & AR latency (s) & Tokens/s \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "% CUDA-event AR-only latency. Token units are method-native and differ across representations.",
            "",
        ]
    )
    atomic_text(output_dir / "paper_table.tex", tex)
    print(f"Wrote aggregate results to {output_dir}", flush=True)
    return summary_blob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["all", "sample", "prepare", "preflight", "run", "aggregate"], nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--methods", default="all", help="Comma-separated names, or all")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-assets", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--overwrite", action="store_true", help="Rerun matching per-asset results")
    parser.add_argument("--overwrite-selection", action="store_true")
    parser.add_argument("--skip-source-prepare", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true", help="Return success even if some raw records failed or are missing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    methods = parse_methods(args.methods, config)
    dataset = (args.dataset or Path(config["dataset"])).resolve()
    num_assets = args.num_assets if args.num_assets is not None else int(config["num_assets"])
    seed = args.seed if args.seed is not None else int(config["seed"])
    gpu = args.gpu if args.gpu is not None else str(config["gpu"])
    warmup = args.warmup if args.warmup is not None else int(config["warmup"])
    output_dir = (args.output_dir or BENCHMARK_ROOT / "results" / f"toys4k_seed{seed}_n{num_assets}").resolve()

    if not dataset.exists():
        raise FileNotFoundError(dataset)
    manifest, selection = sample_dataset(
        dataset,
        output_dir,
        num_assets=num_assets,
        seed=seed,
        overwrite_selection=args.overwrite_selection,
    )
    if args.action == "sample":
        print(f"Selection: {manifest}")
        return

    sys.path.insert(0, str(REPO_ROOT))
    from evaluation.inference_efficiency.model_artifacts import resolve_method_artifacts

    # Workers receive cached paths and run offline; downloads are never timed.
    for method in methods:
        config["methods"][method] = resolve_method_artifacts(
            method, config["methods"][method], local_files_only=args.action == "aggregate",
        )

    source_paths: dict[str, Path] = {}
    if args.action in {"all", "prepare", "preflight", "run"}:
        source_paths = prepare_sources(
            methods,
            config,
            skip=args.skip_source_prepare,
        )
    if args.action == "prepare":
        return
    gpu_identity: str | None = None
    if args.action in {"all", "preflight", "run"}:
        preflight(methods, config, source_paths)
        gpu_identity = validate_single_gpu(gpu, config.get("expected_gpu_name_substring"))
    elif args.action == "aggregate":
        run_path = output_dir / "run.json"
        if run_path.is_file():
            prior_run = load_json(run_path)
            if str(prior_run.get("gpu")) != gpu:
                raise RuntimeError(
                    f"Aggregate --gpu={gpu} does not match recorded physical GPU {prior_run.get('gpu')!r}"
                )
            gpu_identity = prior_run.get("gpu_identity")
        else:
            gpu_identity = validate_single_gpu(gpu, config.get("expected_gpu_name_substring"))
    if args.action == "preflight":
        return

    run_record: dict[str, Any] | None = None
    if args.action in {"all", "run"}:
        run_record = run_methods(
            methods,
            config,
            source_paths,
            manifest=manifest,
            selection=selection,
            output_dir=output_dir,
            gpu=gpu,
            gpu_identity=gpu_identity,
            seed=seed,
            warmup=warmup,
            overwrite=args.overwrite,
            fail_fast=args.fail_fast,
        )
        if args.action == "run" and run_record["failures"] and not args.allow_incomplete:
            raise SystemExit(2)
    if args.action in {"all", "aggregate"}:
        summary = aggregate(
            methods,
            config,
            selection,
            manifest,
            output_dir,
            gpu=gpu,
            gpu_identity=gpu_identity,
            seed=seed,
            warmup=warmup,
        )
        incomplete = [name for name, item in summary["methods"].items() if not item["complete"]]
        if incomplete and not args.allow_incomplete:
            print(
                "Incomplete benchmark methods: " + ", ".join(incomplete) + ". "
                "Partial outputs were retained; rerun to retry errors, or use --overwrite for stale records.",
                file=sys.stderr,
            )
            raise SystemExit(2)


if __name__ == "__main__":
    main()
