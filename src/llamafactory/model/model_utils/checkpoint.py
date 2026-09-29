"""Locate tensors stored in a complete OctLLM safetensors checkpoint."""

import json
from pathlib import Path

from safetensors import safe_open


def load_safetensors_weight_map(checkpoint_dir: str) -> dict[str, str]:
    checkpoint = Path(checkpoint_dir)
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
        return weight_map if isinstance(weight_map, dict) else {}

    single_path = checkpoint / "model.safetensors"
    if single_path.is_file():
        with safe_open(single_path, framework="pt", device="cpu") as shard:
            return dict.fromkeys(shard.keys(), single_path.name)

    return {}
