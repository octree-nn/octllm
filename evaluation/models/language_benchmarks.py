# ruff: noqa: E402

import argparse
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepeval.benchmarks import GSM8K, MMLU, HellaSwag, IFEval
from deepeval.models.base_model import DeepEvalBaseLLM

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import CHOICES, IMAGE_PLACEHOLDER
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.model.bytes_to_split import binary_to_split_tensor, bytes_to_binary_sequence
from llamafactory.model.qwen25_vl_3d_replace import (
    replace_qwen25_vl_for_conditional_generation_forward_with_mesh_mask_loss,
)
from llamafactory.model.split_to_position_embedding import create_template_octree, determine_current_layer_status
from scripts.generate_octree import (
    _ensure_mesh_token_config,
    _ensure_qwen2_vl_template,
    _install_and_load_qwen25_3d_router,
    _load_position_embedding_weights,
    _parse_mesh_tokens_to_bytes,
)


DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs/inference/octllm.yaml"
DEFAULT_OUTPUT_DIR = Path("outputs/language")
DEFAULT_BENCHMARKS = ("mmlu", "gsm8k", "hellaswag", "ifeval")
GENERATIVE_BENCHMARKS = frozenset({"gsm8k", "ifeval"})
GSM8K_CONFINEMENT_INSTRUCTIONS = (
    "Think step by step carefully before answering, and output the numerical answer at last "
    "in format ### Answer: <numerical answer>."
)
ANSWER_MODES = {
    "mmlu": {"multiple_choice": True, "numeric_answer": False},
    "gsm8k": {"multiple_choice": False, "numeric_answer": True},
    "hellaswag": {"multiple_choice": True, "numeric_answer": False},
    "ifeval": {"multiple_choice": False, "numeric_answer": False},
}
DEFAULT_EVAL_TEMPERATURE = 0.0
DEFAULT_EVAL_TOP_P = 1.0
DEFAULT_EVAL_TOP_K = 0


def _extract_last_number(text: str) -> str:
    """Extract the final GSM8K-style number while preserving its displayed form."""
    matches = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    return matches[-1] if matches else text


def _decode_generated_text(tokenizer: Any, generated_ids: list[int]) -> str:
    """Decode output without dropping a final token when generation hits its length cap."""
    token_ids = generated_ids
    if token_ids and token_ids[-1] == tokenizer.eos_token_id:
        token_ids = token_ids[:-1]
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _set_answer_mode(model: Any, benchmark_name: str) -> None:
    model.set_answer_mode(**ANSWER_MODES[benchmark_name])


class _GenerationCaptureMixin:
    """Keep raw generative outputs so post-processing does not hide model mistakes."""

    def _initialize_generation_capture(self) -> None:
        self._generation_capture_enabled = False
        self._generation_records: list[dict[str, str]] = []

    def start_generation_capture(self, enabled: bool = True) -> None:
        self._generation_records = []
        self._generation_capture_enabled = enabled

    def stop_generation_capture(self) -> list[dict[str, str]]:
        records = list(getattr(self, "_generation_records", []))
        self._generation_capture_enabled = False
        return records

    def _capture_generation(self, raw_output: str, returned_output: str) -> str:
        if getattr(self, "_generation_capture_enabled", False):
            self._generation_records.append(
                {
                    "raw_output": raw_output,
                    "returned_output": returned_output,
                }
            )
        return returned_output


def _constrain_mesh_token_logits(
    logits: torch.Tensor,
    mesh_byte_token_ids: set[int],
    mesh_bos_id: int,
    mesh_eos_id: int,
    mask_token_id: int,
    start_mesh_generation: bool,
) -> torch.Tensor:
    """Require <mesh_bos> before mesh bytes and keep mesh generation byte-only."""
    vocab_size = logits.shape[-1]
    valid_mesh_byte_ids = [token_id for token_id in mesh_byte_token_ids if 0 <= token_id < vocab_size]
    if start_mesh_generation:
        constrained = torch.full_like(logits, float("-inf"))
        if valid_mesh_byte_ids:
            constrained[:, valid_mesh_byte_ids] = logits[:, valid_mesh_byte_ids]
        return constrained

    blocked_ids = valid_mesh_byte_ids
    blocked_ids.extend(
        token_id
        for token_id in (mesh_eos_id, mask_token_id)
        if isinstance(token_id, int) and 0 <= token_id < vocab_size and token_id != mesh_bos_id
    )
    constrained = logits.clone()
    if blocked_ids:
        constrained[:, blocked_ids] = float("-inf")
    return constrained


class OctLLM(_GenerationCaptureMixin, DeepEvalBaseLLM):
    def __init__(
        self,
        args: Optional[Union[dict[str, Any], list[str]]] = None,
        config_path: Optional[str] = None,
        max_new_tokens: int = 10000,
        max_layer: int = 6,
        full_depth: int = 3,
        temperature: float = DEFAULT_EVAL_TEMPERATURE,
        top_p: float = DEFAULT_EVAL_TOP_P,
        top_k: int = DEFAULT_EVAL_TOP_K,
    ):
        self.max_new_tokens = max_new_tokens
        self.max_layer = max_layer
        self.full_depth = full_depth
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.default_multiple_choice = False
        self.default_numeric_answer = False
        self._initialize_generation_capture()
        if args is None:
            if config_path is None:
                config_path = "configs/inference/octllm.yaml"
            config_file = Path(config_path)
            if not config_file.is_absolute():
                config_file = PROJECT_ROOT / config_file
            args = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)

        if isinstance(args, dict):
            _ensure_mesh_token_config(args)
        self.model_args, self.data_args, self.finetuning_args, self.generating_args = get_infer_args(args)
        if self.data_args.template == "qwen2_vl":
            _ensure_qwen2_vl_template(self.data_args)
        tokenizer_module = load_tokenizer(self.model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module.get("processor", None)
        self.tokenizer.padding_side = "right"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, self.finetuning_args)
        if self.data_args.template == "qwen2_vl":
            replace_qwen25_vl_for_conditional_generation_forward_with_mesh_mask_loss(
                self.tokenizer,
                is_train=False,
                full_depth=self.full_depth,
                max_depth=self.max_layer,
                add_mask_token=False,
            )
            if hasattr(self.model, "_old_forward"):
                self.model._old_forward = type(self.model).forward.__get__(self.model, type(self.model))
            base_model = getattr(self.model, "model", None)
            if base_model is not None and hasattr(base_model, "_old_forward"):
                base_model._old_forward = type(base_model).forward.__get__(base_model, type(base_model))

            # Keep the language benchmark on exactly the same dual-stream
            # architecture and checkpoint-loading path as batch_generate_shapenet.py.
            _install_and_load_qwen25_3d_router(self.model, self.model_args, self.finetuning_args)
            _load_position_embedding_weights(self.model_args, self.data_args)

    def set_answer_mode(self, multiple_choice: bool = False, numeric_answer: bool = False) -> None:
        if multiple_choice and numeric_answer:
            raise ValueError("multiple_choice and numeric_answer cannot both be enabled")
        self.default_multiple_choice = multiple_choice
        self.default_numeric_answer = numeric_answer

    def load_model(self):
        return self.model

    def generate(
        self,
        prompt: str,
        image_paths: Optional[list[str]] = None,
        max_layer: int = 6,
        full_depth: int = 3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = True,
        multiple_choice: Optional[bool] = None,
        numeric_answer: Optional[bool] = None,
    ) -> str:
        model = self.load_model()
        tokenizer = self.tokenizer
        template = self.template
        processor = self.processor
        if multiple_choice is None:
            multiple_choice = self.default_multiple_choice
        if numeric_answer is None:
            numeric_answer = self.default_numeric_answer
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if top_k is None:
            top_k = self.top_k

        message = {"role": "user", "content": prompt}
        assert message.get("role") == "user", "message.role must be 'user'"
        mm_forward_kwargs: dict[str, Any] = {}

        if self.data_args.template == "qwen2_vl":
            images = image_paths or []
            images = []

            if images and (IMAGE_PLACEHOLDER not in message.get("content", "")):
                message = {
                    "role": "user",
                    "content": (IMAGE_PLACEHOLDER * len(images)) + message["content"],
                }

            messages_mm = [message]
            messages_mm = template.mm_plugin.process_messages(messages_mm, images, [], [], processor)
            paired_messages = messages_mm + [{"role": "assistant", "content": ""}]

            prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, system_prompt, None)
            prompt_ids, _ = template.mm_plugin.process_token_ids(
                prompt_ids,
                None,
                images,
                [],
                [],
                tokenizer,
                processor,
            )

            input_ids = torch.tensor([prompt_ids], device=model.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            mm_inputs = template.mm_plugin.get_mm_inputs(
                images=images,
                videos=[],
                audios=[],
                imglens=[len(images)],
                vidlens=[0],
                audlens=[0],
                batch_ids=[prompt_ids],
                processor=processor,
            )

            processed_mm_inputs = {}
            for key, value in mm_inputs.items():
                if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                    value = torch.stack(value)
                elif (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], list)
                    and value[0]
                    and isinstance(value[0][0], torch.Tensor)
                ):
                    value = torch.stack([torch.stack(v) for v in value])
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)

                if torch.is_floating_point(value):
                    value = value.to(model.dtype)

                if key == "second_per_grid_ts":
                    processed_mm_inputs[key] = value.tolist()
                else:
                    processed_mm_inputs[key] = value.to(model.device)

            mm_forward_kwargs = processed_mm_inputs
        else:
            messages = [message, {"role": "assistant", "content": ""}]
            prompt_ids, _ = template.encode_oneturn(
                tokenizer=tokenizer,
                messages=messages,
                system=system_prompt,
                tools=None,
            )
            input_ids = torch.tensor([prompt_ids], device=model.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if multiple_choice:
            choice_token_ids = [tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in CHOICES]
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    **mm_forward_kwargs,
                )
                logits = outputs.logits[:, -1, :]
                choice_logits = torch.stack([logits[0, idx] for idx in choice_token_ids], dim=0)
                choice_probs = torch.nn.functional.softmax(choice_logits, dim=-1)
                best_idx = torch.argmax(choice_probs, dim=-1).item()
                return CHOICES[best_idx]

        mesh_byte_token_ids = []
        for i in range(256):
            token_id = tokenizer.convert_tokens_to_ids(f"<mesh{i}>")
            if token_id != tokenizer.unk_token_id:
                mesh_byte_token_ids.append(token_id)

        mesh_bos_id = tokenizer.convert_tokens_to_ids("<mesh_bos>")
        mesh_byte_token_ids = set(mesh_byte_token_ids)
        mask_token_id = tokenizer.convert_tokens_to_ids("<MASK>")
        mesh_eos_id = tokenizer.convert_tokens_to_ids("<mesh_eos>")
        mesh_eos_token_id = torch.tensor([[mesh_eos_id]], device=model.device)

        if verbose:
            print(f"Detected {len(mesh_byte_token_ids)} mesh byte tokens.")
            print(f"MASK token ID: {mask_token_id}")

        full_sequence_ids = input_ids.clone()
        generated_ids = []
        generated_probs = []
        past_key_values = None
        current_attention_mask = attention_mask.clone()
        current_binary_sequence = None
        rope_deltas = None
        previous_layer = None
        start_mesh_generation = False
        last_token_id: Optional[torch.Tensor] = None

        with torch.inference_mode():
            for step in range(self.max_new_tokens):
                if step == 0:
                    model_input_ids = input_ids
                    model_attention_mask = current_attention_mask
                    extra_kwargs = mm_forward_kwargs if self.data_args.template == "qwen2_vl" else {}
                else:
                    assert last_token_id is not None
                    last_token_value = last_token_id[0, 0].item()
                    should_add_mask_token = last_token_value == mesh_bos_id or (
                        start_mesh_generation and last_token_value in mesh_byte_token_ids
                    )
                    if should_add_mask_token and self.data_args.template == "qwen2_vl":
                        mask_token_tensor = torch.tensor([[mask_token_id]], device=model.device)
                        model_input_ids = torch.cat([last_token_id, mask_token_tensor], dim=1)
                    else:
                        model_input_ids = last_token_id
                    model_attention_mask = current_attention_mask
                    extra_kwargs = {}

                extra_kwargs["current_binary_sequence"] = current_binary_sequence
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + model_input_ids.shape[1], device=model.device
                )

                outputs = model(
                    input_ids=model_input_ids,
                    attention_mask=model_attention_mask,
                    past_key_values=past_key_values,
                    rope_deltas=rope_deltas,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True,
                    **extra_kwargs,
                )

                past_key_values = outputs.past_key_values
                rope_deltas = outputs.rope_deltas

                next_token_logits = outputs.logits[:, -1, :]
                constrained_logits = _constrain_mesh_token_logits(
                    next_token_logits,
                    mesh_byte_token_ids=mesh_byte_token_ids,
                    mesh_bos_id=mesh_bos_id,
                    mesh_eos_id=mesh_eos_id,
                    mask_token_id=mask_token_id,
                    start_mesh_generation=start_mesh_generation,
                )
                use_sampling = temperature is not None and float(temperature) > 0.0
                if use_sampling:
                    logits = constrained_logits / float(temperature)

                    if top_k is not None and int(top_k) > 0:
                        topk_vals, topk_idx = torch.topk(logits, k=int(top_k), dim=-1)
                        logits_filtered = torch.full_like(logits, float("-inf"))
                        logits_filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                    else:
                        logits_filtered = logits

                    if top_p is not None and 0.0 < float(top_p) < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits_filtered, descending=True, dim=-1)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > float(top_p)
                        sorted_indices_to_remove[..., 0] = False
                        indices_to_remove = torch.zeros_like(logits_filtered, dtype=torch.bool)
                        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                        logits_filtered = logits_filtered.masked_fill(indices_to_remove, float("-inf"))

                    next_token_probs = torch.softmax(logits_filtered, dim=-1)
                    next_token_id = torch.multinomial(next_token_probs, num_samples=1)
                else:
                    next_token_probs = torch.softmax(constrained_logits, dim=-1)
                    next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(-1)
                last_token_id = next_token_id

                token_prob = next_token_probs[0, next_token_id[0, 0]].item()
                next_token_value = next_token_id[0, 0].item()

                generated_ids.append(next_token_value)
                generated_probs.append(token_prob)

                full_sequence_ids = torch.cat([full_sequence_ids, next_token_id], dim=1)
                current_attention_mask = torch.cat(
                    [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=1
                )

                if next_token_value == mesh_bos_id or (
                    start_mesh_generation and next_token_value in mesh_byte_token_ids
                ):
                    if next_token_value == mesh_bos_id:
                        start_mesh_generation = True
                    current_attention_mask = torch.cat(
                        [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=1
                    )

                if next_token_value == mesh_bos_id or (
                    start_mesh_generation and next_token_value in mesh_byte_token_ids
                ):
                    if next_token_value == mesh_bos_id:
                        current_binary_sequence = torch.tensor([], device=model.device)
                        continue

                    current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                    byte_sequence = _parse_mesh_tokens_to_bytes(current_text)
                    if byte_sequence is not None:
                        binary_sequence = bytes_to_binary_sequence(byte_sequence)
                        current_binary_sequence = binary_sequence
                        split_sequence = binary_to_split_tensor(binary_sequence)
                        octree_template = create_template_octree(max_layer, full_depth, model.device)
                        current_layer, layer_complete, remaining_in_layer = determine_current_layer_status(
                            split_sequence.to(model.device), max_layer, full_depth, octree_template, 0.0
                        )
                        if verbose:
                            print(
                                f"Layer status: layer={current_layer}, complete={layer_complete}, remaining={remaining_in_layer}"
                            )

                        if previous_layer is not None and previous_layer == current_layer - 1 and verbose:
                            print(f"previous layer: {previous_layer}, current layer: {current_layer}")
                            print(f"current generated text: {current_text}")
                        previous_layer = current_layer

                        if current_layer == max_layer and layer_complete:
                            mesh_eos_value = mesh_eos_token_id[0, 0].item()
                            generated_ids.append(mesh_eos_value)
                            generated_probs.append(1.0)

                            full_sequence_ids = torch.cat([full_sequence_ids, mesh_eos_token_id], dim=1)
                            current_attention_mask = torch.cat(
                                [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)],
                                dim=1,
                            )
                            current_binary_sequence = None
                            break
                if next_token_value == tokenizer.eos_token_id:
                    break

        generated_text = _decode_generated_text(tokenizer, generated_ids)
        if verbose:
            print(f"generated text: {generated_text}")
        returned_text = _extract_last_number(generated_text) if numeric_answer else generated_text
        return self._capture_generation(generated_text, returned_text)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "OctLLM"


class Qwen25VL(_GenerationCaptureMixin, DeepEvalBaseLLM):
    def __init__(
        self,
        args: Optional[Union[dict[str, Any], list[str]]] = None,
        config_path: Optional[str] = None,
        max_new_tokens: int = 10000,
        temperature: float = DEFAULT_EVAL_TEMPERATURE,
        top_p: float = DEFAULT_EVAL_TOP_P,
        top_k: int = DEFAULT_EVAL_TOP_K,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.default_multiple_choice = False
        self.default_numeric_answer = False
        self._initialize_generation_capture()
        if args is None:
            if config_path is None:
                raise ValueError(
                    "Provide config_path for the Qwen25VL baseline, including its checkpoint and template."
                )
            config_file = Path(config_path)
            if not config_file.is_absolute():
                config_file = PROJECT_ROOT / config_file
            args = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)

        self.model_args, self.data_args, self.finetuning_args, self.generating_args = get_infer_args(args)
        tokenizer_module = load_tokenizer(self.model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module.get("processor", None)
        self.tokenizer.padding_side = "right"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, self.finetuning_args)

    def set_answer_mode(self, multiple_choice: bool = False, numeric_answer: bool = False) -> None:
        if multiple_choice and numeric_answer:
            raise ValueError("multiple_choice and numeric_answer cannot both be enabled")
        self.default_multiple_choice = multiple_choice
        self.default_numeric_answer = numeric_answer

    def load_model(self):
        return self.model

    def generate(
        self,
        prompt: str,
        image_paths: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        multiple_choice: Optional[bool] = None,
        numeric_answer: Optional[bool] = None,
    ) -> str:
        model = self.load_model()
        tokenizer = self.tokenizer
        template = self.template
        processor = self.processor
        if multiple_choice is None:
            multiple_choice = self.default_multiple_choice
        if numeric_answer is None:
            numeric_answer = self.default_numeric_answer
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if top_k is None:
            top_k = self.top_k

        message = {"role": "user", "content": prompt}
        assert message.get("role") == "user", "message.role must be 'user'"
        mm_forward_kwargs: dict[str, Any] = {}

        images = image_paths or []
        images = []

        if images and (IMAGE_PLACEHOLDER not in message.get("content", "")):
            message = {
                "role": "user",
                "content": (IMAGE_PLACEHOLDER * len(images)) + message["content"],
            }

        messages_mm = [message]
        messages_mm = template.mm_plugin.process_messages(messages_mm, images, [], [], processor)
        paired_messages = messages_mm + [{"role": "assistant", "content": ""}]

        prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, system_prompt, None)
        prompt_ids, _ = template.mm_plugin.process_token_ids(
            prompt_ids,
            None,
            images,
            [],
            [],
            tokenizer,
            processor,
        )

        input_ids = torch.tensor([prompt_ids], device=model.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        mm_inputs = template.mm_plugin.get_mm_inputs(
            images=images,
            videos=[],
            audios=[],
            imglens=[len(images)],
            vidlens=[0],
            audlens=[0],
            batch_ids=[prompt_ids],
            processor=processor,
        )

        processed_mm_inputs = {}
        for key, value in mm_inputs.items():
            if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                value = torch.stack(value)
            elif (
                isinstance(value, list)
                and value
                and isinstance(value[0], list)
                and value[0]
                and isinstance(value[0][0], torch.Tensor)
            ):
                value = torch.stack([torch.stack(v) for v in value])
            elif not isinstance(value, torch.Tensor):
                value = torch.tensor(value)

            if torch.is_floating_point(value):
                value = value.to(model.dtype)

            if key == "second_per_grid_ts":
                processed_mm_inputs[key] = value.tolist()
            else:
                processed_mm_inputs[key] = value.to(model.device)

        mm_forward_kwargs = processed_mm_inputs
        if multiple_choice:
            choice_token_ids = [tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in CHOICES]
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    **mm_forward_kwargs,
                )
                logits = outputs.logits[:, -1, :]
                choice_logits = torch.stack([logits[0, idx] for idx in choice_token_ids], dim=0)
                choice_probs = torch.nn.functional.softmax(choice_logits, dim=-1)
                best_idx = torch.argmax(choice_probs, dim=-1).item()
                return CHOICES[best_idx]

        full_sequence_ids = input_ids.clone()
        generated_ids = []
        generated_probs = []
        past_key_values = None
        current_attention_mask = attention_mask.clone()
        rope_deltas = None
        last_token_id: Optional[torch.Tensor] = None

        with torch.inference_mode():
            for step in range(self.max_new_tokens):
                if step == 0:
                    model_input_ids = input_ids
                    model_attention_mask = current_attention_mask
                    extra_kwargs = mm_forward_kwargs if self.data_args.template == "qwen2_vl" else {}
                else:
                    assert last_token_id is not None
                    model_input_ids = last_token_id
                    model_attention_mask = current_attention_mask
                    extra_kwargs = {}

                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + model_input_ids.shape[1], device=model.device
                )

                outputs = model(
                    input_ids=model_input_ids,
                    attention_mask=model_attention_mask,
                    past_key_values=past_key_values,
                    rope_deltas=rope_deltas,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True,
                    **extra_kwargs,
                )

                past_key_values = outputs.past_key_values
                rope_deltas = outputs.rope_deltas

                next_token_logits = outputs.logits[:, -1, :]
                use_sampling = temperature is not None and float(temperature) > 0.0
                if use_sampling:
                    logits = next_token_logits / float(temperature)

                    if top_k is not None and int(top_k) > 0:
                        topk_vals, topk_idx = torch.topk(logits, k=int(top_k), dim=-1)
                        logits_filtered = torch.full_like(logits, float("-inf"))
                        logits_filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                    else:
                        logits_filtered = logits

                    if top_p is not None and 0.0 < float(top_p) < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits_filtered, descending=True, dim=-1)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > float(top_p)
                        sorted_indices_to_remove[..., 0] = False
                        indices_to_remove = torch.zeros_like(logits_filtered, dtype=torch.bool)
                        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                        logits_filtered = logits_filtered.masked_fill(indices_to_remove, float("-inf"))

                    next_token_probs = torch.softmax(logits_filtered, dim=-1)
                    next_token_id = torch.multinomial(next_token_probs, num_samples=1)
                else:
                    next_token_probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(-1)
                last_token_id = next_token_id

                token_prob = next_token_probs[0, next_token_id[0, 0]].item()
                next_token_value = next_token_id[0, 0].item()

                generated_ids.append(next_token_value)
                generated_probs.append(token_prob)

                full_sequence_ids = torch.cat([full_sequence_ids, next_token_id], dim=1)
                current_attention_mask = torch.cat(
                    [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=1
                )

                current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                if verbose:
                    print(f"current generated text: {current_text}")

                if next_token_value == tokenizer.eos_token_id:
                    break

        generated_text = _decode_generated_text(tokenizer, generated_ids)
        if verbose:
            print(f"generated text: {generated_text}")
        returned_text = _extract_last_number(generated_text) if numeric_answer else generated_text
        return self._capture_generation(generated_text, returned_text)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)

    def get_model_name(self):
        return "Qwen2.5 VL"


def _resolve_project_path(path: Union[str, Path]) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _validate_eval_config(config_path: Union[str, Path]) -> tuple[Path, dict[str, Any]]:
    resolved_path = _resolve_project_path(config_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {resolved_path}")

    config = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Evaluation config must contain a YAML mapping: {resolved_path}")
    for key in ("model_name_or_path", "template"):
        if not config.get(key):
            raise ValueError(f"Missing required evaluation config field: {key}")

    if config["template"] != "qwen2_vl":
        raise ValueError("The dual-stream language evaluation requires `template: qwen2_vl`")
    if not bool(config.get("use_3d_token_router", False)):
        raise ValueError("The dual-stream language evaluation requires `use_3d_token_router: true`")

    model_path = Path(str(config["model_name_or_path"])).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    router_config = model_config.get("llamafactory_3d_router_config") or model_config.get("text_config", {}).get(
        "llamafactory_3d_router_config"
    )
    if not isinstance(router_config, dict) or not router_config:
        raise ValueError(f"No 3D router architecture found in checkpoint config: {model_path}")
    model_weight_markers = (
        "model.safetensors",
        "model.safetensors.index.json",
    )
    if not any((model_path / marker).is_file() for marker in model_weight_markers):
        raise FileNotFoundError(f"No model weights found in checkpoint: {model_path}")

    index_path = model_path / "model.safetensors.index.json"
    weight_names: tuple[str, ...] = ()
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid weight_map in checkpoint index: {index_path}")
        weight_names = tuple(weight_map)
        raw_layer_ids = router_config.get("layer_ids", [])
        if isinstance(raw_layer_ids, str):
            layer_ids = [int(value.strip()) for value in raw_layer_ids.split(",") if value.strip()]
        else:
            layer_ids = [int(value) for value in (raw_layer_ids or [])]
        for layer_id in layer_ids:
            layer_marker = f".layers.{layer_id}."
            if bool(router_config.get("replace_ffn", False)) and not any(
                layer_marker in name and ".mesh_mlp." in name for name in weight_names
            ):
                raise FileNotFoundError(f"Missing dual-stream FFN weights for router layer {layer_id}: {index_path}")
            if bool(router_config.get("replace_attn_proj", False)) and not any(
                layer_marker in name and ".self_attn.mesh_" in name for name in weight_names
            ):
                raise FileNotFoundError(
                    f"Missing dual-stream attention weights for router layer {layer_id}: {index_path}"
                )
        router_shards = {
            str(filename)
            for name, filename in weight_map.items()
            if ".mesh_mlp." in name or ".self_attn.mesh_" in name
        }
        for shard in router_shards:
            if not (model_path / shard).is_file():
                raise FileNotFoundError(f"Missing dual-stream checkpoint shard: {model_path / shard}")

    if not index_path.is_file():
        from safetensors import safe_open

        with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as shard:
            weight_names = tuple(shard.keys())
    if not any(".mesh_mlp." in name or ".self_attn.mesh_" in name for name in weight_names):
        raise FileNotFoundError(f"No dual-stream router weights found in checkpoint: {model_path}")
    if bool(config.get("use_separate_new_token_embeddings", False)):
        if not all(
            any(name.endswith(key) for name in weight_names)
            for key in ("separate_new_token_embeddings.weight", "separate_new_token_lm_head.weight")
        ):
            raise FileNotFoundError(f"No separate new-token weights found in checkpoint: {model_path}")

    return resolved_path, config


def _build_benchmark(name: str, args: argparse.Namespace) -> Any:
    limit = args.limit
    if name == "mmlu":
        return MMLU(n_shots=args.mmlu_shots, n_problems_per_task=limit, verbose_mode=args.verbose)
    if name == "gsm8k":
        return GSM8K(
            n_shots=args.gsm8k_shots,
            enable_cot=args.gsm8k_cot,
            n_problems=limit if limit is not None else 1319,
            verbose_mode=args.verbose,
            confinement_instructions=GSM8K_CONFINEMENT_INSTRUCTIONS,
        )
    if name == "hellaswag":
        return HellaSwag(n_shots=args.hellaswag_shots, n_problems_per_task=limit, verbose_mode=args.verbose)
    if name == "ifeval":
        return IFEval(n_problems=limit, verbose_mode=args.verbose)
    raise ValueError(f"Unsupported benchmark: {name}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _find_dataframe_column(dataframe: Any, candidates: tuple[str, ...]) -> Optional[str]:
    normalized_columns = {str(column).strip().lower().replace(" ", "_"): str(column) for column in dataframe.columns}
    for candidate in candidates:
        column = normalized_columns.get(candidate.strip().lower().replace(" ", "_"))
        if column is not None:
            return column
    return None


def _score_is_correct(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "passed", "correct"}
    return bool(value)


def _json_safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _save_generation_errors(
    name: str,
    predictions: Any,
    generation_records: list[dict[str, str]],
    output_dir: Path,
) -> Optional[Path]:
    if name not in GENERATIVE_BENCHMARKS:
        return None

    correct_candidates = ("Correct", "Score") if name == "gsm8k" else ("All_Instructions_Correct", "Correct", "Score")
    correct_column = _find_dataframe_column(predictions, correct_candidates)
    if correct_column is None:
        print(
            f"Warning: unable to save {name} errors because no correctness column was found in predictions.",
            file=sys.stderr,
        )
        return None

    input_column = _find_dataframe_column(predictions, ("Input", "Question", "Prompt"))
    prediction_column = _find_dataframe_column(predictions, ("Prediction", "Generated_Output", "Output"))
    expected_column = _find_dataframe_column(predictions, ("Expected Output", "Expected_Output", "Answer"))
    error_samples = []

    for position, (_, row) in enumerate(predictions.iterrows()):
        if _score_is_correct(row[correct_column]):
            continue

        prediction = _json_safe_scalar(row[prediction_column]) if prediction_column else None
        capture = generation_records[position] if position < len(generation_records) else None
        generated_output = capture["raw_output"] if capture is not None else prediction
        sample = {
            "problem_index": position + 1,
            "input": _json_safe_scalar(row[input_column]) if input_column else None,
            "generated_output": generated_output,
            "evaluated_prediction": prediction,
            "correct": False,
        }
        if expected_column is not None:
            sample["expected_output"] = _json_safe_scalar(row[expected_column])
        error_samples.append(sample)

    error_path = output_dir / f"{name}_errors.json"
    _write_json(
        error_path,
        {
            "benchmark": name,
            "total_predictions": len(predictions),
            "error_count": len(error_samples),
            "raw_generation_records": len(generation_records),
            "samples": error_samples,
        },
    )
    return error_path


def _save_benchmark_artifacts(
    name: str,
    benchmark: Any,
    output_dir: Path,
    generation_records: Optional[list[dict[str, str]]] = None,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    predictions = getattr(benchmark, "predictions", None)
    if predictions is not None:
        prediction_path = output_dir / f"{name}_predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        artifacts["predictions"] = prediction_path.name
        error_path = _save_generation_errors(name, predictions, generation_records or [], output_dir)
        if error_path is not None:
            artifacts["errors"] = error_path.name

    task_scores = getattr(benchmark, "task_scores", None)
    if task_scores is not None:
        task_scores_path = output_dir / f"{name}_task_scores.csv"
        task_scores.to_csv(task_scores_path, index=False)
        artifacts["task_scores"] = task_scores_path.name

    instruction_breakdown = getattr(benchmark, "instruction_breakdown", None)
    if instruction_breakdown is not None:
        breakdown_path = output_dir / f"{name}_instruction_breakdown.json"
        _write_json(breakdown_path, instruction_breakdown)
        artifacts["instruction_breakdown"] = breakdown_path.name
    return artifacts


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the OctLLM dual-stream checkpoint on MMLU, GSM8K, HellaSwag, and IFEval."
    )
    parser.add_argument("--config", default=str(DEFAULT_EVAL_CONFIG), help="Configured dual-stream inference YAML.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON and CSV results.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=DEFAULT_BENCHMARKS,
        default=list(DEFAULT_BENCHMARKS),
        help="Benchmarks to run in order (default: all four).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=10000, help="Generation cap for GSM8K and IFEval.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_EVAL_TEMPERATURE,
        help="Shared decoding temperature for every generative benchmark (default: deterministic greedy decoding).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_EVAL_TOP_P,
        help="Shared nucleus-sampling threshold; inactive when temperature is zero.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_EVAL_TOP_K,
        help="Shared top-k sampling cutoff; inactive when temperature is zero.",
    )
    parser.add_argument("--max-layer", type=int, default=6, help="Maximum octree layer used by the inference patch.")
    parser.add_argument("--full-depth", type=int, default=3, help="Full octree depth used by the inference patch.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit (per task for MMLU/HellaSwag, total for GSM8K/IFEval).",
    )
    parser.add_argument("--mmlu-shots", type=int, default=5)
    parser.add_argument("--gsm8k-shots", type=int, default=3)
    parser.add_argument("--hellaswag-shots", type=int, default=10)
    parser.add_argument(
        "--no-gsm8k-cot", dest="gsm8k_cot", action="store_false", help="Disable GSM8K chain of thought."
    )
    parser.set_defaults(gsm8k_cot=True)
    parser.add_argument("--verbose", action="store_true", help="Print DeepEval per-example logs.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed benchmark.")
    return parser


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the interval (0, 1]")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.max_layer <= 0:
        parser.error("--max-layer must be greater than zero")
    if args.full_depth < 0 or args.full_depth > args.max_layer:
        parser.error("--full-depth must be between zero and --max-layer")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if not 0 <= args.mmlu_shots <= 5:
        parser.error("--mmlu-shots must be between 0 and 5")
    if not 0 <= args.gsm8k_shots <= 15:
        parser.error("--gsm8k-shots must be between 0 and 15")
    if not 0 <= args.hellaswag_shots <= 15:
        parser.error("--hellaswag-shots must be between 0 and 15")


def main(argv: Optional[list[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    config_path, config = _validate_eval_config(args.config)
    output_dir = _resolve_project_path(args.output_dir)

    plan = {
        "config": str(config_path),
        "checkpoint": str(config["model_name_or_path"]),
        "output_dir": str(output_dir),
        "benchmarks": [{"name": name, **ANSWER_MODES[name]} for name in args.benchmarks],
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_layer": args.max_layer,
        "full_depth": args.full_depth,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        **plan,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": {},
    }
    _write_json(output_dir / "summary.json", summary)

    print(f"Loading the OctLLM dual-stream checkpoint once from {config_path}")
    model = OctLLM(
        config_path=str(config_path),
        max_new_tokens=args.max_new_tokens,
        max_layer=args.max_layer,
        full_depth=args.full_depth,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    any_failed = False

    for name in args.benchmarks:
        mode = ANSWER_MODES[name]
        print(f"\nRunning {name}: multiple_choice={mode['multiple_choice']}, numeric_answer={mode['numeric_answer']}")
        _set_answer_mode(model, name)
        benchmark = _build_benchmark(name, args)
        model.start_generation_capture(enabled=name in GENERATIVE_BENCHMARKS)
        started_at = datetime.now(timezone.utc)
        try:
            result = benchmark.evaluate(model)
            generation_records = model.stop_generation_capture()
            record = {
                "status": "completed",
                "overall_accuracy": float(result.overall_accuracy),
                "answer_mode": mode,
                "artifacts": _save_benchmark_artifacts(
                    name,
                    benchmark,
                    output_dir,
                    generation_records=generation_records,
                ),
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"{name} overall accuracy: {record['overall_accuracy']:.6f}")
        except Exception as error:
            model.stop_generation_capture()
            any_failed = True
            record = {
                "status": "failed",
                "answer_mode": mode,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"{name} failed: {record['error']}", file=sys.stderr)

        summary["results"][name] = record
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(output_dir / "summary.json", summary)
        if any_failed and args.fail_fast:
            break

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
