from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts import batch_understand as understanding
from scripts import batch_understand_pointllm as pointllm


def _annotation(object_id: str, reference: str) -> dict[str, object]:
    return {
        "object_id": object_id,
        "conversations": [
            {"from": "human", "value": "<point>\nDescribe this object."},
            {"from": "gpt", "value": reference},
        ],
    }


def _write_dataset(tmp_path: Path, count: int = 4) -> tuple[Path, Path]:
    glb_dir = tmp_path / "glbs"
    glb_dir.mkdir()
    annotations = []
    for index in range(count):
        object_id = f"object-{index}"
        annotations.append(_annotation(object_id, f"reference {index}"))
        (glb_dir / f"{object_id}.glb").touch()
    annotation_path = tmp_path / "PointLLM_brief_description_val_200_GT.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    return annotation_path, glb_dir


def test_load_assets_maps_pointllm_object_ids_to_glbs(tmp_path: Path):
    annotation_path, glb_dir = _write_dataset(tmp_path, count=2)

    assets = pointllm.load_assets(annotation_path, glb_dir)

    assert [asset.asset_id for asset in assets] == ["object-0", "object-1"]
    assert assets[0].glb_path == glb_dir / "object-0.glb"


def test_load_assets_rejects_duplicate_and_unsafe_object_ids(tmp_path: Path):
    glb_dir = tmp_path / "glbs"
    glb_dir.mkdir()
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps([_annotation("../unsafe", "a"), _annotation("../unsafe", "b")]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe object_id"):
        pointllm.load_assets(annotation_path, glb_dir)


def test_selection_filters_before_deterministic_sharding(tmp_path: Path):
    annotation_path, glb_dir = _write_dataset(tmp_path, count=5)
    assets = pointllm.load_assets(annotation_path, glb_dir)

    selected = understanding.select_assets(
        assets,
        requested_ids=None,
        limit=5,
        num_shards=2,
        shard_id=1,
    )

    assert [asset.asset_id for asset in selected] == ["object-1", "object-3"]
    with pytest.raises(ValueError, match="Unknown --asset-id"):
        understanding.select_assets(
            assets,
            requested_ids=["missing"],
            limit=None,
            num_shards=1,
            shard_id=0,
        )


def test_glb_preprocessing_uses_the_same_pruned_octree_contract(tmp_path: Path):
    annotation_path, glb_dir = _write_dataset(tmp_path, count=1)
    asset = pointllm.load_assets(annotation_path, glb_dir)[0]
    args = pointllm.build_parser().parse_args([])
    args.min_sequence_length = 1
    calls: list[dict[str, object]] = []

    fake_preprocessor = SimpleNamespace(
        process_single_model=lambda **kwargs: calls.append(kwargs) or [0, 17, 255],
    )
    values, mesh_sequence = understanding._preprocess_octree(fake_preprocessor, asset, args)

    assert values == [0, 17, 255]
    assert mesh_sequence == "<mesh_bos><mesh0><mesh17><mesh255><mesh_eos>"
    assert calls == [
        {
            "glb_path": str(asset.glb_path),
            "num_samples": 100000,
            "depth": 6,
            "full_depth": 3,
            "mesh_scale": 1.0,
            "shift": False,
            "threshold": 0.5,
            "device": "cuda",
            "target_depth": 5,
            "drop_prob": 0.5,
            "prune": True,
        }
    ]


def test_worker_keeps_toys4k_prompt_and_per_asset_txt_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    next_token_module = ModuleType("scripts.generate_octree")
    calls: list[dict[str, object]] = []

    def fake_predict(message, infer_cfg, **kwargs):
        calls.append({"message": message, "infer_cfg": infer_cfg, **kwargs})
        return {"generated_text": "A compact wooden chair."}

    next_token_module._predict_next_token = fake_predict
    monkeypatch.setitem(sys.modules, "scripts.generate_octree", next_token_module)
    monkeypatch.setattr(
        understanding,
        "_load_runtime",
        lambda args: (object(), {"model": "fake"}, {"mllm": {"generation": {}}}),
    )
    monkeypatch.setattr(understanding, "_load_trellis_preprocessor", lambda: object())
    monkeypatch.setattr(
        understanding,
        "_preprocess_octree",
        lambda module, asset, args: ([1, 2], "<mesh_bos><mesh1><mesh2><mesh_eos>"),
    )
    monkeypatch.setattr(understanding, "_clean_understanding_text", lambda text: text)
    monkeypatch.setattr(understanding, "_seed_everything", lambda seed: None)

    method_dir = tmp_path / "understanding" / understanding.DEFAULT_METHOD_NAME
    args = pointllm.build_parser().parse_args([])
    args.method_dir = method_dir
    asset = understanding.Asset(0, "object-0", tmp_path / "object-0.glb")

    stats = understanding.run_worker(args, [asset])

    assert stats.completed == 1
    assert (method_dir / "object-0.txt").read_text(encoding="utf-8") == ("A compact wooden chair.\n")
    assert not (method_dir / "predictions").exists()
    assert calls[0]["message"] == {
        "role": "user",
        "content": ("Describe this 3D mesh in detail: <mesh_bos><mesh1><mesh2><mesh_eos>"),
    }
    assert calls[0]["temperature"] == 0.5
    assert calls[0]["top_p"] == 0.9
    assert calls[0]["top_k"] == 40
    assert calls[0]["bos_top_k"] == 0


def test_language_metric_loader_accepts_toys4k_and_pointllm_gt(tmp_path: Path):
    evaluation_dir = Path(__file__).resolve().parents[2] / "evaluation"
    sys.path.insert(0, str(evaluation_dir))
    try:
        from compute_toys4k_language_metrics import _load_eval_entries
    finally:
        sys.path.remove(str(evaluation_dir))

    toys_path = tmp_path / "toys.json"
    toys_path.write_text(
        json.dumps([{"asset_id": "toy-0", "text_description": "a toy"}]),
        encoding="utf-8",
    )
    point_path = tmp_path / "pointllm.json"
    point_path.write_text(
        json.dumps([_annotation("object-0", "a pointllm reference")]),
        encoding="utf-8",
    )

    assert _load_eval_entries(toys_path)[0].text_description == "a toy"
    point_entry = _load_eval_entries(point_path)[0]
    assert point_entry.asset_id == "object-0"
    assert point_entry.text_description == "a pointllm reference"
