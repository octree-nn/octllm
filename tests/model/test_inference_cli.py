from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import inference
from dataset_toolkits import tokenize_mesh
from inference import (
    OctreeLayerProgress,
    choose_trellis_condition,
    clean_assistant_text,
    extract_mesh_sequence,
    extract_text_condition,
)


@pytest.mark.parametrize("local", [False, True])
def test_default_and_local_sources_share_lazy_completion(tmp_path, monkeypatch, local):
    from scripts.model_sources import load_inference_config

    if local:
        monkeypatch.setenv("OCTLLM_WEIGHTS_DIR", str(tmp_path))
        completion = tmp_path / "completion/model.safetensors"
        completion.parent.mkdir()
        completion.touch()
    else:
        monkeypatch.delenv("OCTLLM_WEIGHTS_DIR", raising=False)
        completion = tmp_path / "downloaded-completion.safetensors"
    hub = SimpleNamespace(hf_hub_download=Mock(return_value=str(completion)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.chdir(tmp_path)

    settings = inference._load_settings(str(inference.DEFAULT_CONFIG))
    source = str(tmp_path) if local else "anonymous-octllm/OctLLM"
    assert load_inference_config(settings["mllm"]["config"])["model_name_or_path"] == source
    assert settings["mesh"]["vae"]["model_source"] == source
    hub.hf_hub_download.assert_not_called()

    decoder = SimpleNamespace(get_vae_model=Mock(), vae_complete_voxel=Mock(), vae_complete_voxel_trellis_align=Mock())
    monkeypatch.setitem(sys.modules, "scripts.decode_octree", decoder)
    inference._complete_voxel("voxel", settings["mesh"])
    assert decoder.get_vae_model.call_args.args == (str(completion),)
    assert hub.hf_hub_download.call_count == (0 if local else 1)
    if not local:
        assert hub.hf_hub_download.call_args.kwargs["repo_id"] == source
        assert hub.hf_hub_download.call_args.kwargs["filename"] == "completion/model.safetensors"


def test_extract_mesh_sequence_and_clean_visible_text():
    raw = "I've generated it: <mesh_bos><mesh0> <mesh255><mesh_eos><|im_end|>"
    sequence = extract_mesh_sequence(raw)

    assert sequence == "<mesh_bos><mesh0><mesh255><mesh_eos>"
    assert clean_assistant_text(raw, sequence) == "I've generated it:"


def test_extract_mesh_sequence_rejects_out_of_range_byte():
    with pytest.raises(ValueError, match="outside"):
        extract_mesh_sequence("<mesh_bos><mesh256><mesh_eos>")


def test_text_condition_uses_description_after_generation_prefix():
    prompt = "Generate a 3D mesh based on the following text description: a red chair."

    assert extract_text_condition(prompt) == "a red chair."


def test_auto_trellis_condition_reuses_original_image():
    mode, image, text = choose_trellis_condition("auto", "make a chair", ["/tmp/chair.png"])

    assert (mode, image, text) == ("image", "/tmp/chair.png", None)


def test_auto_trellis_condition_falls_back_to_text():
    prompt = "Generate a 3D mesh based on the following text description: a dolphin."

    assert choose_trellis_condition("auto", prompt, []) == ("text", None, "a dolphin.")


def test_image_condition_requires_an_image():
    with pytest.raises(ValueError, match="requires an input image"):
        choose_trellis_condition("image", "make a chair", [])


def test_octree_progress_tracks_each_depth_independently(monkeypatch):
    class FakeTqdm:
        bars = []

        def __init__(self, total, initial, **_kwargs):
            self.total = total
            self.n = initial
            self.closed = False
            self.bars.append(self)

        @staticmethod
        def write(_message):
            pass

        def update(self, amount):
            self.n += amount

        def refresh(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=FakeTqdm))
    progress = OctreeLayerProgress()

    progress({"event": "octree_layer", "layer": 3, "remaining": 504, "split_length": 8, "complete": False})
    progress({"event": "octree_layer", "layer": 4, "remaining": 128, "split_length": 512, "complete": False})
    progress({"event": "octree_layer", "layer": 4, "remaining": 0, "split_length": 640, "complete": True})

    assert [(bar.total, bar.n, bar.closed) for bar in FakeTqdm.bars] == [
        (512, 512, True),
        (128, 128, True),
    ]


@pytest.mark.parametrize("suffix", [".glb", ".obj"])
@pytest.mark.parametrize("prompt_source", ["default", "prompt", "file"])
def test_mesh_understanding_saves_input_tokens_and_passes_question_to_model(
    tmp_path, monkeypatch, suffix, prompt_source
):
    mesh_path = tmp_path / f"asset{suffix}"
    mesh_path.touch()
    sequence = "<mesh_bos><mesh0><mesh17><mesh255><mesh_eos>"
    question = "What shape is this object?"
    argv = ["--mesh", str(mesh_path), "--no-mesh", "--output-dir", str(tmp_path), "--output-name", "understanding"]
    if prompt_source == "prompt":
        argv += ["--prompt", question]
    elif prompt_source == "file":
        prompt_file = tmp_path / "question.txt"
        prompt_file.write_text(question, encoding="utf-8")
        argv += ["--prompt-file", str(prompt_file)]
    else:
        question = inference.DEFAULT_UNDERSTANDING_PROMPT

    monkeypatch.setattr(inference, "_load_settings", lambda _path: {})
    monkeypatch.setattr(tokenize_mesh, "mesh_to_octree_sequence", lambda path, **kwargs: sequence)
    prompts = []

    def predict(prompt, image_paths, settings, progress):
        prompts.append(prompt)
        return {"generated_text": "A small chair.<|im_end|>"}

    monkeypatch.setattr(inference, "_model_inference", predict)
    result = inference.run(inference.build_parser().parse_args(argv))

    assert prompts == [f"{question} {sequence}"]
    assert result["type"] == "text"
    assert result["assistant_text"] == "A small chair."
    assert result["glb_path"] is None
    assert result["octree_tokens_path"] is None
    assert result["input"]["text"] == question
    assert result["input"]["model_prompt"] == prompts[0]
    assert result["input"]["mesh"] == str(mesh_path)
    assert Path(result["input"]["octree_tokens_path"]).read_text(encoding="utf-8") == sequence + "\n"
    assert json.loads(Path(result["result_path"]).read_text(encoding="utf-8")) == result


@pytest.mark.parametrize(
    ("prompt", "with_image"),
    [
        ("Explain what an octree is.", False),
        ("What is shown in this image?", True),
        ("Describe this asset: <mesh_bos><mesh0><mesh255><mesh_eos>", False),
    ],
)
def test_text_image_and_existing_octree_prompts_do_not_invoke_mesh_preprocessing(
    tmp_path, monkeypatch, prompt, with_image
):
    argv = ["--prompt", prompt, "--no-mesh", "--output-dir", str(tmp_path), "--output-name", "chat"]
    image_paths = []
    if with_image:
        image_path = tmp_path / "image.png"
        image_path.touch()
        image_paths = [str(image_path)]
        argv += ["--image", str(image_path)]
    monkeypatch.setattr(inference, "_load_settings", lambda _path: {})

    def unexpected_mesh_input(*args, **kwargs):
        pytest.fail("Text/image chat and pre-tokenized prompts must not preprocess a mesh.")

    def predict(actual_prompt, actual_images, settings, progress):
        assert actual_prompt == prompt
        assert actual_images == image_paths
        return {"generated_text": "An ordinary reply.<|im_end|>"}

    monkeypatch.setattr(tokenize_mesh, "mesh_to_octree_sequence", unexpected_mesh_input)
    monkeypatch.setattr(inference, "_model_inference", predict)
    result = inference.run(inference.build_parser().parse_args(argv))

    assert result["assistant_text"] == "An ordinary reply."
    assert result["input"] == {"text": prompt, "images": image_paths}
    assert not list(tmp_path.glob("*.tokens.txt"))


def test_mesh_input_rejects_an_existing_sequence_before_loading_the_model(tmp_path, monkeypatch):
    def unexpected_model_call(*args, **kwargs):
        pytest.fail("Invalid mesh input must fail before loading the model.")

    monkeypatch.setattr(inference, "_load_settings", lambda _path: {})
    monkeypatch.setattr(inference, "_model_inference", unexpected_model_call)
    args = inference.build_parser().parse_args(
        ["--mesh", "asset.glb", "--prompt", "Describe: <mesh_bos><mesh0><mesh_eos>", "--output-dir", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="text-only question"):
        inference.run(args)
