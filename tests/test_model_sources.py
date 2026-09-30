"""Check Hub resolution without downloading model weights."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.model_sources import OCTLLM_MODEL_FILES, resolve_model_directory, resolve_model_file, resolve_trellis_vae_directory
from evaluation.inference_efficiency.model_artifacts import resolve_method_artifacts


@pytest.fixture
def hub(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    fake = SimpleNamespace(
        snapshot_download=Mock(return_value=str(snapshot)),
        hf_hub_download=Mock(side_effect=lambda **kw: str(snapshot / kw["filename"])),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return fake, snapshot


def test_local_overrides_do_not_contact_hub(hub, tmp_path):
    fake, _ = hub
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"local fixture")
    assert resolve_model_directory(str(tmp_path)) == tmp_path
    assert resolve_model_file(str(checkpoint), "default.pt") == checkpoint
    assert resolve_model_file(str(tmp_path), "weights.pt") == checkpoint
    with pytest.raises(FileNotFoundError):
        resolve_model_file(str(tmp_path), "missing.pt")
    with pytest.raises(NotADirectoryError):
        resolve_model_directory(str(checkpoint))
    fake.snapshot_download.assert_not_called()
    fake.hf_hub_download.assert_not_called()


@pytest.mark.parametrize("source", ["./missing", "../missing", "/missing/model", Path("missing/model")])
def test_missing_explicit_local_path_is_not_a_hub_id(hub, source):
    fake, _ = hub
    with pytest.raises(FileNotFoundError):
        resolve_model_directory(source)
    fake.snapshot_download.assert_not_called()


def test_alignment_downloads_only_vae_assets_and_supports_local_layouts(hub):
    fake, snapshot = hub
    ckpts = snapshot / "ckpts"
    ckpts.mkdir()
    assert resolve_trellis_vae_directory("microsoft/TRELLIS-image-large") == ckpts
    kwargs = fake.snapshot_download.call_args.kwargs
    assert kwargs["repo_id"] == "microsoft/TRELLIS-image-large"
    assert set(kwargs["allow_patterns"]) == {
        "ckpts/ss_enc_conv3d_16l8_fp16.json", "ckpts/ss_enc_conv3d_16l8_fp16.safetensors",
        "ckpts/ss_dec_conv3d_16l8_fp16.json", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    }
    assert resolve_trellis_vae_directory(snapshot) == ckpts
    assert resolve_trellis_vae_directory(ckpts) == ckpts
    assert fake.snapshot_download.call_count == 1


@pytest.mark.parametrize("method, filenames", [
    ("sar3d", ["text-condition-ckpt.pth", "vqvae-ckpt.pt"]),
    ("octgpt", ["octgpt_objv_text.pth", "vqvae_large_objv_bsq32.pth"]),
])
def test_baseline_repo_ids_resolve_to_named_files_before_workers(hub, method, filenames):
    fake, snapshot = hub
    config = json.loads((ROOT / "evaluation/inference_efficiency/benchmark_config.json").read_text())
    original = config["methods"][method]
    resolved = resolve_method_artifacts(method, original, local_files_only=True)
    calls = [call.kwargs for call in fake.hf_hub_download.call_args_list]
    assert [call["filename"] for call in calls] == filenames
    assert all(call["local_files_only"] for call in calls)
    assert all(call["repo_id"] == original["vae_checkpoint"] for call in calls)
    assert resolved["vae_checkpoint"] == str(snapshot / filenames[1])
    assert resolved["clip_path"] == str(snapshot)
    assert resolved["model_sources"]["clip_path"] == "openai/clip-vit-large-patch14"
    assert "model_sources" not in original
    assert fake.snapshot_download.call_args.kwargs["local_files_only"] is True


def test_octllm_local_override_does_not_contact_hub(hub, tmp_path):
    fake, _ = hub
    config = {"model_path": str(tmp_path)}
    resolved = resolve_method_artifacts("octllm", config)
    assert resolved["model_path"] == str(tmp_path)
    assert resolved["model_sources"] == config
    fake.snapshot_download.assert_not_called()
    fake.hf_hub_download.assert_not_called()


def test_octllm_benchmark_stages_hub_model_without_completion(hub):
    import fnmatch

    fake, snapshot = hub
    config = {"model_path": "Plurato123/OctLLM"}
    resolved = resolve_method_artifacts("octllm", config, local_files_only=True)
    assert resolved["model_path"] == str(snapshot)
    assert resolved["model_sources"] == config
    options = fake.snapshot_download.call_args.kwargs
    assert options["repo_id"] == config["model_path"]
    assert options["local_files_only"] is True
    assert any(fnmatch.fnmatch("model-00001-of-00005.safetensors", p) for p in OCTLLM_MODEL_FILES)
    assert not any(fnmatch.fnmatch("completion/model.safetensors", p) for p in OCTLLM_MODEL_FILES)


def test_cached_transformers_directory_preserves_source_identity(hub):
    fake, snapshot = hub
    cfg = {"model_path": "Zhengyi/LLaMA-Mesh"}
    resolved = resolve_method_artifacts("llama_mesh", cfg)
    assert resolved["model_path"] == str(snapshot)
    assert resolved["model_sources"] == cfg
    assert resolve_method_artifacts("llama_mesh", resolved) == resolved
    assert fake.snapshot_download.call_count == 1


def test_benchmark_stages_weights_before_offline_workers_and_aggregates_from_cache(hub, tmp_path, monkeypatch):
    from evaluation.inference_efficiency import benchmark

    fake, snapshot = hub
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"fingerprint fixture")
    config = json.loads(benchmark.DEFAULT_CONFIG.read_text())
    config["methods"]["llama_mesh"]["python"] = sys.executable
    config["methods"]["llama_mesh"].pop("repository")
    config_path = tmp_path / "benchmark.json"
    config_path.write_text(json.dumps(config))
    dataset = tmp_path / "assets.json"
    dataset.write_text(json.dumps([{"asset_id": "chair", "text_description": "A wooden chair."}]))
    output = tmp_path / "results"
    monkeypatch.setattr(benchmark, "prepare_sources", lambda *args, **kwargs: {})
    monkeypatch.setattr(benchmark, "validate_single_gpu", lambda *args: "fixture GPU")

    def worker(command, *, cwd, env):
        assert fake.snapshot_download.call_count == 1
        assert fake.snapshot_download.call_args.kwargs["local_files_only"] is False
        assert command[command.index("--model-path") + 1] == str(snapshot)
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"

    run_worker = Mock(side_effect=worker)
    monkeypatch.setattr(benchmark, "run_checked", run_worker)
    argv = ["benchmark.py", "run", "--methods", "llama_mesh", "--config", str(config_path),
            "--dataset", str(dataset), "--output-dir", str(output), "--num-assets", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    benchmark.main()
    run_worker.assert_called_once()
    assert json.loads((output / "run.json").read_text())["config_hashes"]["llama_mesh"]

    monkeypatch.setattr(benchmark, "aggregate", Mock(return_value={"methods": {"llama_mesh": {"complete": True}}}))
    monkeypatch.setattr(sys, "argv", [argv[0], "aggregate", *argv[2:]])
    benchmark.main()
    assert fake.snapshot_download.call_args.kwargs["local_files_only"] is True
    assert fake.snapshot_download.call_count == 2
    run_worker.assert_called_once()
