from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.batch_generate_toys4k import (
    ASSIGNED_GPU_ENV,
    IMAGE_PROMPT,
    TEXT_PROMPT_PREFIX,
    Job,
    _child_command,
    _worker_environment,
    asset_output_dir,
    build_parser,
    build_prompt_and_images,
    glb_path,
    has_current_mesh,
    has_valid_tokens,
    is_valid_glb,
    load_dataset,
    make_jobs,
    mesh_metadata_path,
    parse_gpu_ids,
    select_items,
    token_path,
)


def _item(asset_id: str, image_path: Path) -> dict[str, str]:
    return {
        "asset_id": asset_id,
        "mesh_path": f"/reference/{asset_id}.pickle",
        "render_image_path": str(image_path),
        "text_description": f"description for {asset_id}",
    }


def _write_glb(path: Path, payload: bytes = b"") -> None:
    size = 12 + len(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"glTF" + (2).to_bytes(4, "little") + size.to_bytes(4, "little") + payload)


def test_prompts_match_the_toys4k_generation_contract(tmp_path: Path):
    image = tmp_path / "condition.png"
    image.write_bytes(b"image")
    item = _item("asset", image)

    image_prompt, image_paths = build_prompt_and_images(Job(0, item, "image"))
    text_prompt, text_paths = build_prompt_and_images(Job(0, item, "text"))

    assert image_prompt == IMAGE_PROMPT == "Generate a 3D mesh based on this image:"
    assert image_paths == [str(image.resolve())]
    assert text_prompt == f"{TEXT_PROMPT_PREFIX}description for asset"
    assert text_paths == []


def test_standard_output_layout_uses_condition_and_asset_directories(tmp_path: Path):
    image = tmp_path / "condition.png"
    item = _item("abc123", image)
    image_job = Job(0, item, "image")
    text_job = Job(0, item, "text")

    assert asset_output_dir(tmp_path, image_job) == tmp_path / "image_to_3d" / "abc123"
    assert token_path(tmp_path, text_job) == tmp_path / "text_to_3d" / "abc123" / "octree.tokens.txt"
    assert glb_path(tmp_path, text_job) == tmp_path / "text_to_3d" / "abc123" / "abc123.glb"


def test_dataset_selection_filters_before_deterministic_sharding(tmp_path: Path):
    image = tmp_path / "condition.png"
    dataset = [_item(f"asset-{index}", image) for index in range(6)]

    selected = select_items(dataset, asset_ids=None, limit=5, num_shards=2, shard_id=1)
    assert [index for index, _ in selected] == [1, 3]
    assert [job.condition for job in make_jobs(selected, ("image", "text"))] == [
        "image",
        "image",
        "text",
        "text",
    ]

    with pytest.raises(ValueError, match="Unknown --asset-id"):
        select_items(dataset, asset_ids=["missing"], limit=None, num_shards=1, shard_id=0)


def test_filtered_selection_is_evenly_sharded_by_selected_position(tmp_path: Path):
    image = tmp_path / "condition.png"
    dataset = [_item(f"asset-{index}", image) for index in range(8)]
    asset_ids = ["asset-1", "asset-3", "asset-5", "asset-7"]

    shard_zero = select_items(dataset, asset_ids=asset_ids, limit=None, num_shards=2, shard_id=0)
    shard_one = select_items(dataset, asset_ids=asset_ids, limit=None, num_shards=2, shard_id=1)

    assert [index for index, _ in shard_zero] == [1, 5]
    assert [index for index, _ in shard_one] == [3, 7]


def test_multi_gpu_ids_and_worker_environment_are_isolated():
    assert parse_gpu_ids("0, 2,07") == ("0", "2", "7")
    with pytest.raises(ValueError, match="duplicate"):
        parse_gpu_ids("1,01")
    with pytest.raises(ValueError, match="non-negative"):
        parse_gpu_ids("0,cuda:1")

    environment = _worker_environment(
        "6",
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "LOCAL_RANK": "3",
            "RANK": "3",
            "WORLD_SIZE": "8",
            "KEEP_ME": "yes",
        },
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "6"
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment[ASSIGNED_GPU_ENV] == "6"
    assert environment["KEEP_ME"] == "yes"
    assert "LOCAL_RANK" not in environment
    assert "RANK" not in environment
    assert "WORLD_SIZE" not in environment


def test_multi_gpu_child_command_uses_automatic_shard_and_does_not_recurse():
    args = build_parser().parse_args(["--stage", "all", "--condition", "both", "--gpus", "4,5"])
    command = _child_command(args, "all", "both", num_shards=2, shard_id=1)

    assert command[command.index("--num-shards") + 1] == "2"
    assert command[command.index("--shard-id") + 1] == "1"
    assert "--gpus" not in command


def test_load_dataset_rejects_duplicate_and_unsafe_asset_ids(tmp_path: Path):
    image = tmp_path / "condition.png"
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([_item("same", image), _item("same", image)]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate asset_id"):
        load_dataset(dataset_path)

    dataset_path.write_text(json.dumps([_item("../escape", image)]), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe asset_id"):
        load_dataset(dataset_path)


def test_token_and_glb_completion_checks_detect_partial_outputs(tmp_path: Path):
    tokens = tmp_path / "octree.tokens.txt"
    tokens.write_text("<mesh_bos><mesh0><mesh255><mesh_eos>\n", encoding="utf-8")
    assert has_valid_tokens(tokens)

    tokens.write_text("<mesh_bos><mesh0>", encoding="utf-8")
    assert not has_valid_tokens(tokens)

    mesh = tmp_path / "mesh.glb"
    _write_glb(mesh)
    assert is_valid_glb(mesh)
    mesh.write_bytes(b"glTF\x02")
    assert not is_valid_glb(mesh)


def test_mesh_is_current_only_when_metadata_matches_token_hash(tmp_path: Path):
    image = tmp_path / "condition.png"
    job = Job(0, _item("asset", image), "text")
    _write_glb(glb_path(tmp_path, job))
    metadata = mesh_metadata_path(tmp_path, job)
    metadata.write_text(
        json.dumps({"status": "success", "octree_sha256": "current"}),
        encoding="utf-8",
    )

    assert has_current_mesh(tmp_path, job, "current")
    assert not has_current_mesh(tmp_path, job, "changed")
