import argparse
import json
import os
from typing import Any, Optional

import torch
from generate_octree import (
    _crop_cache_to_length,
    _ensure_mesh_token_config,
    _ensure_qwen2_vl_template,
    _get_cache_seq_length,
    _install_and_load_qwen25_3d_router,
    _load_position_embedding_weights,
    _mask_logits_to_token_ids,
    _maybe_force_mesh_bos,
    _parse_mesh_tokens_to_bytes,
)
from omegaconf import OmegaConf
from tqdm import tqdm

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IMAGE_PLACEHOLDER
from llamafactory.hparams import get_infer_args
from llamafactory.model.bytes_to_split import binary_to_split_tensor, bytes_to_binary_sequence
from llamafactory.model.loader import load_model, load_tokenizer
from llamafactory.model.qwen25_vl_3d_replace import (
    replace_qwen25_vl_for_conditional_generation_forward_with_mesh_mask_loss,
)
from llamafactory.model.split_to_position_embedding import create_template_octree, determine_current_layer_status


def _parse_str_filter(values: Optional[list[str]]) -> set[str]:
    if not values:
        return set()

    parsed: set[str] = set()
    for value in values:
        parsed.update(part.strip() for part in value.split(",") if part.strip())

    return parsed


def _parse_index_filter(values: Optional[list[str]]) -> set[int]:
    parsed_values = _parse_str_filter(values)
    if not parsed_values:
        return set()

    indices: set[int] = set()
    for value in parsed_values:
        try:
            indices.add(int(value))
        except ValueError as exc:
            raise ValueError(f"Invalid sample index: {value}") from exc

    return indices


def generate_single_sample(
    model: torch.nn.Module,
    tokenizer: Any,
    processor: Any,
    template: Any,
    message: dict[str, str],
    image_paths: Optional[list[str]] = None,
    num_new_tokens: int = 10000,
    max_layer: int = 6,
    full_depth: int = 3,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    bos_top_k: int = 0,
    data_args: Any = None,
    system_prompt: str = None,
    keep_mask_in_cache: bool = False,
) -> str:
    # Build single-turn messages and tokenize.
    assert message.get("role") == "user", "message.role must be 'user'"
    images = image_paths or []
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

    mm_forward_kwargs = {}
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
            mm_forward_kwargs[key] = value.tolist()
        else:
            mm_forward_kwargs[key] = value.to(model.device)

    mesh_byte_token_ids = []
    for i in range(256):
        token_id = tokenizer.convert_tokens_to_ids(f"<mesh{i}>")
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            mesh_byte_token_ids.append(token_id)

    mesh_bos_id = tokenizer.convert_tokens_to_ids("<mesh_bos>")
    mesh_byte_token_ids = set(mesh_byte_token_ids)
    mask_token_id = tokenizer.convert_tokens_to_ids("<MASK>")
    mesh_eos_token_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<mesh_eos>")]], device=model.device)
    _octree_start_ids = tokenizer.encode(" octree-start", add_special_tokens=False)

    full_sequence_ids = input_ids.clone()
    generated_ids = []
    past_key_values = None
    current_attention_mask = attention_mask.clone()
    current_binary_sequence = None
    rope_deltas = None
    start_mesh_generation = False
    last_token_id = None

    with torch.inference_mode():
        for step in range(num_new_tokens):
            uses_transient_mask = False
            if step == 0:
                model_input_ids = input_ids
                model_attention_mask = current_attention_mask
                extra_kwargs = mm_forward_kwargs
            else:
                if last_token_id is None:
                    raise RuntimeError("last_token_id must be initialized after the first generation step.")
                last_token_value = last_token_id[0, 0].item()
                should_add_mask_token = last_token_value == mesh_bos_id or (
                    start_mesh_generation and last_token_value in mesh_byte_token_ids
                )
                if should_add_mask_token:
                    mask_token_tensor = torch.tensor([[mask_token_id]], device=model.device)
                    model_input_ids = torch.cat([last_token_id, mask_token_tensor], dim=1)
                    model_attention_mask = torch.cat(
                        [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=1
                    )
                    if not keep_mask_in_cache:
                        uses_transient_mask = True
                else:
                    model_input_ids = last_token_id
                    model_attention_mask = current_attention_mask
                extra_kwargs = {}

            extra_kwargs["current_binary_sequence"] = current_binary_sequence
            past_seen_tokens = _get_cache_seq_length(past_key_values)
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
            if uses_transient_mask:
                past_key_values = _crop_cache_to_length(past_key_values, past_seen_tokens + 1)
            rope_deltas = outputs.rope_deltas
            next_token_logits = outputs.logits[:, -1, :]

            use_sampling = temperature is not None and float(temperature) > 0.0
            bos_force_logits = next_token_logits / float(temperature) if use_sampling else next_token_logits
            forced_bos_id, _, _ = (
                _maybe_force_mesh_bos(bos_force_logits, mesh_bos_id, bos_top_k)
                if not start_mesh_generation
                else (None, None, None)
            )

            if forced_bos_id is not None:
                next_token_probs = torch.softmax(bos_force_logits, dim=-1)
                next_token_id = forced_bos_id
            elif use_sampling:
                logits = next_token_logits / float(temperature)

                if start_mesh_generation:
                    logits = _mask_logits_to_token_ids(logits, mesh_byte_token_ids)

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
                greedy_logits = (
                    _mask_logits_to_token_ids(next_token_logits, mesh_byte_token_ids)
                    if start_mesh_generation
                    else next_token_logits
                )
                next_token_probs = torch.softmax(greedy_logits, dim=-1)
                next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(-1)

            last_token_id = next_token_id
            next_token_value = next_token_id[0, 0].item()

            generated_ids.append(next_token_value)

            full_sequence_ids = torch.cat([full_sequence_ids, next_token_id], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=1
            )

            if next_token_value == mesh_bos_id or (start_mesh_generation and next_token_value in mesh_byte_token_ids):
                if not start_mesh_generation:
                    start_mesh_generation = True
                if keep_mask_in_cache:
                    current_attention_mask = torch.cat(
                        [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=1
                    )

            if next_token_value == mesh_bos_id or (start_mesh_generation and next_token_value in mesh_byte_token_ids):
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
                    current_layer, layer_complete, _ = determine_current_layer_status(
                        split_sequence.to(model.device), max_layer, full_depth, octree_template, 0.0
                    )
                    if current_layer == max_layer and layer_complete:
                        mesh_eos_value = mesh_eos_token_id[0, 0].item()
                        generated_ids.append(mesh_eos_value)
                        break

            if next_token_value == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate 3D assets.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_new_tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument(
        "--bos_top_k",
        type=int,
        default=0,
        help="Force <mesh_bos> when it ranks in the top k of the full vocabulary; 0 disables this behavior.",
    )
    parser.add_argument("--max_layer", type=int, default=6)
    parser.add_argument("--full_depth", type=int, default=3)
    parser.add_argument("--prompt", type=str, default="Generate a 3D asset from the following image:<image>")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--input_mode", type=str, default="image", choices=["image", "text", "understanding"])
    parser.add_argument(
        "--sample_ids",
        "--sample_names",
        nargs="+",
        default=None,
        help="Process only the specified sample names, separated by spaces or commas.",
    )
    parser.add_argument(
        "--sample_indices",
        nargs="+",
        default=None,
        help="Process only these zero-based test indices, separated by spaces or commas. Filter before sharding.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate selected samples even when output files already exist.",
    )
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of dataset shards.")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id in [0, num_shards).")
    parser.add_argument(
        "--keep_mask_in_cache",
        action="store_true",
        help="Keep temporary <MASK> tokens in the KV cache. By default, discard them after each prediction.",
    )

    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    _ensure_mesh_token_config(cfg)
    model_args, data_args, finetuning_args, _ = get_infer_args(cfg)
    _ensure_qwen2_vl_template(data_args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor", None)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=False,
        add_valuehead=False,
    )

    replace_qwen25_vl_for_conditional_generation_forward_with_mesh_mask_loss(
        tokenizer=tokenizer,
        is_train=False,
        full_depth=args.full_depth,
        max_depth=args.max_layer,
        add_mask_token=False,
    )
    if hasattr(model, "_old_forward"):
        model._old_forward = type(model).forward.__get__(model, type(model))

    base_model = getattr(model, "model", None)
    if base_model is not None and hasattr(base_model, "_old_forward"):
        base_model._old_forward = type(base_model).forward.__get__(base_model, type(base_model))

    _install_and_load_qwen25_3d_router(model, model_args, finetuning_args)

    _load_position_embedding_weights(model_args, data_args)

    with open(args.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    sample_ids = _parse_str_filter(args.sample_ids)
    sample_indices = _parse_index_filter(args.sample_indices)
    indexed_dataset = list(enumerate(dataset))
    if sample_ids or sample_indices:
        indexed_dataset = [
            (idx, item)
            for idx, item in indexed_dataset
            if str(item.get("name", "")) in sample_ids or idx in sample_indices
        ]

    if args.num_shards > 1:
        indexed_dataset = [(idx, item) for idx, item in indexed_dataset if (idx % args.num_shards) == args.shard_id]

    for _, item in tqdm(indexed_dataset, desc=f"Generating (shard {args.shard_id}/{args.num_shards})"):
        name = item["name"]
        output_file = os.path.join(args.output_dir, f"{name}.txt")

        if os.path.exists(output_file) and not args.overwrite:
            continue

        if args.input_mode == "text":
            description = item["description"]
            message = {"role": "user", "content": f"{args.prompt}{description}"}
            images = []
        elif args.input_mode == "image":
            images = item.get("images", [])
            message = {"role": "user", "content": args.prompt}
        else:
            description = item["token_sequence"]
            message = {"role": "user", "content": f"{args.prompt}{description}"}
            images = []
        generated_text = generate_single_sample(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            template=template,
            message=message,
            image_paths=images,
            num_new_tokens=args.num_new_tokens,
            max_layer=args.max_layer,
            full_depth=args.full_depth,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            bos_top_k=args.bos_top_k,
            data_args=data_args,
            system_prompt=args.system_prompt,
            keep_mask_in_cache=args.keep_mask_in_cache,
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)


if __name__ == "__main__":
    main()
