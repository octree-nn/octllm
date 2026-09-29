"""Cache benchmark weights before validation, fingerprinting, and GPU timing."""

import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.model_sources import OCTLLM_MODEL_FILES, resolve_model_directory, resolve_model_file


CHECKPOINT_FILES = {
    "sar3d": {
        "ar_checkpoint": "text-condition-ckpt.pth",
        "vae_checkpoint": "vqvae-ckpt.pt",
    },
    "octgpt": {
        "checkpoint": "octgpt_objv_text.pth",
        "vae_checkpoint": "vqvae_large_objv_bsq32.pth",
    },
}
TRANSFORMERS_FILES = (
    "*.json", "*.safetensors", "pytorch_model*.bin", "*.txt", "*.model", "*.tiktoken", "*.jinja",
)


def resolve_method_artifacts(
    method: str, method_cfg: dict[str, Any], *, local_files_only: bool = False,
) -> dict[str, Any]:
    """Resolve model repositories into the Hugging Face cache before timing."""
    resolved = dict(method_cfg)
    sources = dict(method_cfg.get("model_sources", {}))
    for key, filename in CHECKPOINT_FILES.get(method, {}).items():
        sources.setdefault(key, str(method_cfg[key]))
        resolved[key] = str(resolve_model_file(method_cfg[key], filename, local_files_only=local_files_only))
    for key in ("model_path", "clip_path"):
        if method_cfg.get(key):
            sources.setdefault(key, str(method_cfg[key]))
            resolved[key] = str(resolve_model_directory(
                method_cfg[key], local_files_only=local_files_only,
                allow_patterns=OCTLLM_MODEL_FILES if method == "octllm" else TRANSFORMERS_FILES,
            ))
    resolved["model_sources"] = sources
    return resolved
