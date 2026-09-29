"""Resolve public model repositories and explicit local checkpoint overrides."""

from pathlib import Path
from typing import Optional, Sequence, Union


ModelSource = Union[str, Path]

COMPLETION_FILENAME = "completion/model.safetensors"
OCTLLM_MODEL_FILES = (
    "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja", "model*.safetensors", "pytorch_model*.bin",
)


def load_inference_config(path: ModelSource) -> dict:
    """Load OctLLM options with explicit local paths relative to the YAML file."""
    from omegaconf import OmegaConf

    config_path = Path(path).expanduser().resolve()
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in OctLLM model config: {config_path}")
    source = config.get("model_name_or_path")
    if source:
        local = Path(str(source)).expanduser()
        if local.is_absolute() or str(source).startswith(("./", "../", "~")) or (config_path.parent / local).exists():
            config["model_name_or_path"] = str((config_path.parent / local).resolve())
    return config


def _local_path(source: ModelSource) -> Optional[Path]:
    path = Path(source).expanduser()
    if path.exists():
        return path.resolve()
    if isinstance(source, Path) or str(source).startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"Local model path does not exist: {source}")
    return None


def resolve_model_directory(
    source: ModelSource,
    *,
    local_files_only: bool = False,
    allow_patterns: Optional[Sequence[str]] = None,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Return a local directory, downloading a Hub snapshot when given a repo ID."""
    local = _local_path(source)
    if local is not None:
        if not local.is_dir():
            raise NotADirectoryError(local)
        return local
    from huggingface_hub import snapshot_download

    options = {
        key: value
        for key, value in {"cache_dir": cache_dir, "revision": revision, "token": token}.items()
        if value is not None
    }
    return Path(snapshot_download(
        repo_id=str(source),
        local_files_only=local_files_only,
        allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
        **options,
    )).resolve()


def resolve_model_file(source: ModelSource, filename: str, *, local_files_only: bool = False) -> Path:
    """Resolve a checkpoint file, a directory containing it, or a Hub repo ID."""
    local = _local_path(source)
    if local is not None:
        checkpoint = local / filename if local.is_dir() else local
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=str(source), filename=filename, local_files_only=local_files_only))


def resolve_trellis_vae_directory(source: ModelSource) -> Path:
    """Resolve TRELLIS VAE assets from a Hub repo, snapshot root, or ckpts directory."""
    filenames = [
        f"ckpts/ss_{component}_conv3d_16l8_fp16.{extension}"
        for component in ("enc", "dec")
        for extension in ("json", "safetensors")
    ]
    root = resolve_model_directory(source, allow_patterns=filenames)
    return root / "ckpts" if (root / "ckpts").is_dir() else root
