"""Regression checks for public pretrained model loading and feature extraction."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.mark.parametrize("local", [False, True])
def test_octllm_runtime_loads_all_components_from_one_snapshot(tmp_path, monkeypatch, local):
    import huggingface_hub

    from scripts import generate_octree

    source = str(tmp_path) if local else "anonymous-octllm/OctLLM"
    config = {"model_name_or_path": source}
    model_args = SimpleNamespace(
        model_name_or_path=source, cache_dir=str(tmp_path / "cache"),
        model_revision="test-revision", hf_hub_token="test-token",
    )
    data_args = SimpleNamespace(template="qwen2_vl")
    finetuning_args = SimpleNamespace()
    monkeypatch.setattr(generate_octree, "get_infer_args", lambda _: (model_args, data_args, finetuning_args, None))
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    seen = []

    def check_source(stage, args):
        assert args.model_name_or_path == str(tmp_path)
        seen.append(stage)

    def load_tokenizer(args):
        check_source("tokenizer", args)
        return {"tokenizer": object(), "processor": object()}

    def load_model(**kwargs):
        check_source("model", kwargs["model_args"])
        return torch.nn.Identity()

    monkeypatch.setattr(generate_octree, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(generate_octree, "load_model", load_model)
    monkeypatch.setattr(generate_octree, "get_template_and_fix_tokenizer", lambda *_: object())
    monkeypatch.setattr(generate_octree, "replace_qwen25_vl_for_conditional_generation_forward_with_mesh_mask_loss", Mock())
    monkeypatch.setattr(generate_octree, "_install_and_load_qwen25_3d_router", lambda _, args, __: check_source("router", args))
    monkeypatch.setattr(generate_octree, "_load_position_embedding_weights", lambda args, _: check_source("positions", args))

    runtime = generate_octree._load_inference_runtime(config, max_layer=6, full_depth=3)

    assert seen == ["tokenizer", "model", "router", "positions"]
    assert runtime.max_layer == 6
    assert config["model_name_or_path"] == source
    assert download.call_count == (0 if local else 1)
    if not local:
        options = download.call_args.kwargs
        assert options["repo_id"] == source
        assert options["cache_dir"] == model_args.cache_dir
        assert options["revision"] == "test-revision"
        assert options["token"] == "test-token"


@pytest.mark.parametrize("local", [False, True])
def test_trellis_manifest_resolves_relative_and_cross_repository_models(tmp_path, monkeypatch, local):
    from trellis import models
    import huggingface_hub

    # Load the base module without importing GPU decoders or background removal.
    path = Path(models.__file__).parents[1] / "pipelines" / "base.py"
    spec = importlib.util.spec_from_file_location("trellis.pipelines._base_source_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    shared = "JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
    config = tmp_path / "pipeline.json"
    config.write_text(json.dumps({"args": {"models": {
        "flow": "ckpts/ss_flow_txt_dit_XL_16l8_fp16", "decoder": shared,
    }}}))
    download = Mock(return_value=str(config))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    load_model = Mock(side_effect=lambda source: torch.nn.Identity())
    monkeypatch.setattr(models, "from_pretrained", load_model)
    source = str(tmp_path) if local else "microsoft/TRELLIS-text-xlarge"

    pipeline = module.Pipeline.from_pretrained(source)

    assert [call.args[0] for call in load_model.call_args_list] == [
        f"{source}/ckpts/ss_flow_txt_dit_XL_16l8_fp16", shared,
    ]
    assert set(pipeline.models) == {"flow", "decoder"}
    assert download.call_count == (0 if local else 1)


@pytest.mark.parametrize("previous", [None, "existing/clip"])
@pytest.mark.parametrize("fail", [False, True])
def test_text_pipeline_accepts_clip_repo_id_and_restores_environment(monkeypatch, previous, fail):
    from scripts import decode_octree

    key = "TRELLIS_TEXT_COND_MODEL_PATH"
    if previous is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, previous)
    fake_pipeline = SimpleNamespace(cuda=Mock())

    def load(source):
        assert source == "microsoft/TRELLIS-text-xlarge"
        assert os.environ[key] == "openai/clip-vit-large-patch14"
        if fail:
            raise RuntimeError("model load failed")
        return fake_pipeline

    text_loader = Mock(side_effect=load)
    fake_module = SimpleNamespace(
        TrellisImageTo3DPipeline=SimpleNamespace(from_pretrained=Mock()),
        TrellisTextTo3DPipeline=SimpleNamespace(from_pretrained=text_loader),
    )
    monkeypatch.setitem(sys.modules, "trellis.pipelines", fake_module)
    monkeypatch.setattr(decode_octree, "_trellis_pipeline_cache", {"pipeline": None})
    args = ("microsoft/TRELLIS-text-xlarge", "text", "openai/clip-vit-large-patch14")
    if fail:
        with pytest.raises(RuntimeError, match="model load failed"):
            decode_octree.get_trellis_pipeline(*args)
        fake_pipeline.cuda.assert_not_called()
    else:
        assert decode_octree.get_trellis_pipeline(*args) is fake_pipeline
        assert decode_octree.get_trellis_pipeline(*args) is fake_pipeline
        text_loader.assert_called_once()
        fake_pipeline.cuda.assert_called_once()
    assert os.environ.get(key) == previous


def test_dino_hub_and_explicit_local_backends_share_preprocessing(tmp_path, monkeypatch):
    from transformers import AutoModel
    from evaluation.compute_toys4k_fid_kid import DinoFeatureExtractor

    class Features(torch.nn.Module):
        def forward_features(self, inputs):
            self.inputs = inputs
            return {"x_norm_clstoken": inputs.mean(dim=(2, 3))}

        def forward(self, pixel_values):
            cls_token = self.forward_features(pixel_values)["x_norm_clstoken"]
            return SimpleNamespace(last_hidden_state=torch.stack([cls_token, cls_token * 2], dim=1))

    hf_model, local_model = Features(), Features()
    hf_loader = Mock(return_value=hf_model)
    monkeypatch.setattr(AutoModel, "from_pretrained", hf_loader)
    monkeypatch.setattr(torch.hub, "load", Mock(return_value=local_model))
    weights = tmp_path / "dino.pth"
    weights.write_bytes(b"weights are supplied by the fixture")
    image_path = tmp_path / "view.png"
    pixels = np.arange(73 * 121 * 3, dtype=np.uint8).reshape(73, 121, 3)
    Image.fromarray(pixels).save(image_path)
    hf_extractor = DinoFeatureExtractor("facebook/dinov2-with-registers-large", "cpu", None, None, 518)
    local_extractor = DinoFeatureExtractor("unused", "cpu", str(tmp_path), str(weights), 518)

    hf_features = hf_extractor.extract([image_path], 1, "HF fixture")
    local_features = local_extractor.extract([image_path], 1, "local fixture")

    hf_loader.assert_called_once_with("facebook/dinov2-with-registers-large")
    assert hf_extractor.model_mode == "transformers"
    assert local_extractor.model_mode == "torchhub"
    assert hf_model.inputs.shape == (1, 3, 518, 518)
    torch.testing.assert_close(hf_model.inputs, local_model.inputs)
    np.testing.assert_array_equal(hf_features, local_features)
    with pytest.raises(ValueError, match="Set both"):
        DinoFeatureExtractor("unused", "cpu", str(tmp_path), None, 518)
