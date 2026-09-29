from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.generate_octree import OctLLMInferenceRuntime, _predict_next_token


class _DummyTokenizer:
    def __init__(self) -> None:
        self.unk_token_id = -1
        self.eos_token_id = 2
        self.mesh_bos_id = 300
        self.mesh_eos_id = 301
        self.mask_id = 302
        self.decode_calls: list[list[int]] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<mesh_bos>":
            return self.mesh_bos_id
        if token == "<mesh_eos>":
            return self.mesh_eos_id
        if token == "<MASK>":
            return self.mask_id
        if token.startswith("<mesh") and token.endswith(">"):
            return 1000 + int(token[5:-1])
        return self.unk_token_id

    def encode(self, _text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [42]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        assert not skip_special_tokens
        values = [int(token_id) for token_id in token_ids]
        self.decode_calls.append(values)
        return "".join(self._token_text(token_id) for token_id in values)

    def _token_text(self, token_id: int) -> str:
        if token_id == self.mesh_bos_id:
            return "<mesh_bos>"
        if token_id == self.mesh_eos_id:
            return "<mesh_eos>"
        if 1000 <= token_id < 1256:
            return f"<mesh{token_id - 1000}>"
        return f"<{token_id}>"


class _DummyCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, target_length: int) -> None:
        self.length = target_length


class _DummyModel:
    def __init__(self, output_token_ids: list[int]) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.output_token_ids = output_token_ids
        self.forward_kwargs: list[dict] = []

    def __call__(self, *, input_ids: torch.Tensor, past_key_values=None, **kwargs):
        self.forward_kwargs.append(kwargs)
        token_id = self.output_token_ids[len(self.forward_kwargs) - 1]
        logits = torch.full((1, input_ids.shape[1], 1300), -100.0)
        logits[0, -1, token_id] = 100.0
        previous_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        return SimpleNamespace(
            logits=logits,
            past_key_values=_DummyCache(previous_length + input_ids.shape[1]),
            rope_deltas=torch.tensor([[0]], dtype=torch.long),
        )


class _DummyMMPlugin:
    def process_messages(self, messages, *_args):
        return messages

    def process_token_ids(self, prompt_ids, labels, *_args):
        return prompt_ids, labels

    def get_mm_inputs(self, **_kwargs):
        return {}


class _DummyTemplate:
    def __init__(self) -> None:
        self.mm_plugin = _DummyMMPlugin()

    def encode_oneturn(self, *_args):
        return [10, 11], None


def test_generation_loop_uses_incremental_metadata_without_history_decoding():
    tokenizer = _DummyTokenizer()
    model = _DummyModel(
        [
            tokenizer.mesh_bos_id,
            tokenizer.convert_tokens_to_ids("<mesh128>"),
            tokenizer.convert_tokens_to_ids("<mesh0>"),
        ]
    )
    runtime = OctLLMInferenceRuntime(
        model=model,
        tokenizer=tokenizer,
        processor=None,
        template=_DummyTemplate(),
        max_layer=2,
        full_depth=1,
    )

    result = _predict_next_token(
        {"role": "user", "content": "build a mesh"},
        {},
        num_new_tokens=10,
        max_layer=2,
        full_depth=1,
        temperature=0.0,
        verbose=False,
        runtime=runtime,
    )

    assert result["generated_ids"] == [
        tokenizer.mesh_bos_id,
        tokenizer.convert_tokens_to_ids("<mesh128>"),
        tokenizer.convert_tokens_to_ids("<mesh0>"),
        tokenizer.mesh_eos_id,
    ]
    assert result["autoregressive_forward_steps"] == 3
    assert len(tokenizer.decode_calls) == 1

    assert model.forward_kwargs[0]["octree_decode_metadata"] is None
    first_mesh_metadata = model.forward_kwargs[1]["octree_decode_metadata"]
    assert first_mesh_metadata.previous_position is None
    assert first_mesh_metadata.next_position.depth_index == 0

    second_mesh_metadata = model.forward_kwargs[2]["octree_decode_metadata"]
    assert second_mesh_metadata.previous_position.depth_index == 0
    assert second_mesh_metadata.next_position.depth_index == 1
    assert all("current_binary_sequence" not in kwargs for kwargs in model.forward_kwargs)


def test_generation_loop_preserves_legacy_pre_bos_mesh_byte_parsing():
    tokenizer = _DummyTokenizer()
    model = _DummyModel(
        [
            tokenizer.convert_tokens_to_ids("<mesh5>"),
            tokenizer.mesh_bos_id,
            tokenizer.convert_tokens_to_ids("<mesh7>"),
        ]
    )
    runtime = OctLLMInferenceRuntime(
        model=model,
        tokenizer=tokenizer,
        processor=None,
        template=_DummyTemplate(),
        max_layer=1,
        full_depth=1,
    )
    progress_events: list[dict] = []

    result = _predict_next_token(
        {"role": "user", "content": "build a mesh"},
        {},
        num_new_tokens=10,
        max_layer=1,
        full_depth=1,
        temperature=0.0,
        verbose=False,
        runtime=runtime,
        progress_callback=progress_events.append,
    )

    assert result["generated_ids"] == [
        tokenizer.convert_tokens_to_ids("<mesh5>"),
        tokenizer.mesh_bos_id,
        tokenizer.convert_tokens_to_ids("<mesh7>"),
        tokenizer.mesh_eos_id,
    ]
    layer_event = next(event for event in progress_events if event["event"] == "octree_layer")
    assert layer_event == {
        "event": "octree_layer",
        "layer": 1,
        "complete": True,
        "remaining": 0,
        "split_length": 16,
    }
