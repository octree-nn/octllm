import json

import pytest


@pytest.fixture(params=[False, True], ids=["single-file", "sharded"])
def model_checkpoint(request, tmp_path):
    """Write the same model state in either supported checkpoint layout."""
    safetensors = pytest.importorskip("safetensors.torch")

    def save(state_dict):
        if not request.param:
            safetensors.save_file(state_dict, tmp_path / "model.safetensors")
        else:
            weight_map = {}
            items = list(state_dict.items())
            for index in range(2):
                shard = dict(items[index::2])
                if not shard:
                    continue
                filename = f"model-{index + 1:05d}-of-00002.safetensors"
                safetensors.save_file(shard, tmp_path / filename)
                weight_map.update(dict.fromkeys(shard, filename))
            (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        return str(tmp_path)

    return save
