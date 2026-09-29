"""Check that mesh input preserves the dataset's S-Octree representation."""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_toolkits.tokenize_mesh import mesh_to_octree_sequence


def test_missing_mesh_reports_the_input_path(tmp_path):
    path = tmp_path / "missing.glb"
    with pytest.raises(FileNotFoundError, match="missing.glb"):
        mesh_to_octree_sequence(path)


def test_unsupported_mesh_format_is_rejected(tmp_path):
    path = tmp_path / "asset.txt"
    path.touch()
    with pytest.raises(ValueError, match=".glb or .obj"):
        mesh_to_octree_sequence(path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_samples": 0}, "num_samples"),
        ({"full_depth": 7}, "full_depth"),
        ({"target_depth": 6}, "target_depth"),
        ({"target_depth": 2}, "target_depth"),
        ({"drop_prob": 1.1}, "drop_prob"),
    ],
)
def test_invalid_sampling_settings_fail_before_geometry_imports(tmp_path, kwargs, message):
    path = tmp_path / "asset.glb"
    path.touch()
    with pytest.raises(ValueError, match=message):
        mesh_to_octree_sequence(path, **kwargs)


@pytest.fixture
def geometry_runtime():
    torch = pytest.importorskip("torch")
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("ocnn")
    from dataset_toolkits import tokenize_shapenet, tokenize_trellis

    return torch, trimesh, tokenize_shapenet, tokenize_trellis


@pytest.mark.parametrize("suffix", [".glb", ".obj"])
@pytest.mark.parametrize("prune", [True, False])
def test_tokens_match_dataset_preprocessing_for_real_meshes(tmp_path, monkeypatch, geometry_runtime, suffix, prune):
    torch, trimesh, shapenet, trellis = geometry_runtime
    mesh = trimesh.creation.box(extents=[1.0, 0.5, 0.25])
    mesh.apply_translation([2.0, 3.0, 4.0])
    path = tmp_path / f"asset{suffix}"
    mesh.export(path)
    sample_surface = trimesh.sample.sample_surface

    # Trimesh uses its own default_rng; make the surface samples identical for
    # both calls without changing the production tokenizer or global NumPy RNG.
    monkeypatch.setattr(trimesh.sample, "sample_surface", lambda mesh, count: sample_surface(mesh, count, seed=42))
    options = {"num_samples": 2048, "depth": 6, "full_depth": 3, "target_depth": 5, "drop_prob": 0.5, "prune": prune}
    process = trellis.process_single_model if suffix == ".glb" else shapenet.process_single_model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expected_bytes = process(str(path), **options)
        torch.manual_seed(42)
        sequence = mesh_to_octree_sequence(path, **options)

    assert expected_bytes
    assert sequence == "<mesh_bos>" + "".join(f"<mesh{value}>" for value in expected_bytes) + "<mesh_eos>"


@pytest.mark.parametrize("suffix", [".glb", ".obj"])
def test_invalid_geometry_reports_the_source_mesh(tmp_path, geometry_runtime, suffix):
    path = tmp_path / f"empty{suffix}"
    path.touch()
    with pytest.raises(RuntimeError, match="empty"):
        mesh_to_octree_sequence(path)
