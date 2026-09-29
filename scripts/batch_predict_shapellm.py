import argparse
import json
import os
from typing import Any, Optional

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IMAGE_PLACEHOLDER
from llamafactory.hparams import get_infer_args
from llamafactory.model.loader import load_model, load_tokenizer


def generate_single_sample(
    model: torch.nn.Module,
    tokenizer: Any,
    processor: Any,
    template: Any,
    message: dict[str, str],
    image_paths: Optional[list[str]] = None,
    num_new_tokens: int = 10000,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    data_args: Any = None,
    system_prompt: str = None,
) -> str:
    # Build single-turn messages and tokenize. For qwen2_vl, enable multimodal processing.
    assert message.get("role") == "user", "message.role must be 'user'"
    mm_forward_kwargs = {}

    if data_args.template == "qwen2_vl":
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
            tokenizer=tokenizer, messages=messages, system=system_prompt, tools=None
        )
        input_ids = torch.tensor([prompt_ids], device=model.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    mesh_byte_token_ids = []
    for i in range(8192):
        token_id = tokenizer.convert_tokens_to_ids(f"<mesh{i}>")
        if token_id != tokenizer.unk_token_id:
            mesh_byte_token_ids.append(token_id)
    mesh_bos_id = tokenizer.convert_tokens_to_ids("<mesh_bos>")
    mesh_byte_token_ids = set(mesh_byte_token_ids)
    mesh_eos_token_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<mesh_eos>")]], device=model.device)

    full_sequence_ids = input_ids.clone()
    generated_ids = []
    past_key_values = None
    current_attention_mask = attention_mask.clone()
    rope_deltas = None
    start_mesh_generation = False
    mesh_tokens_generated = 0
    last_token_id = None

    with torch.inference_mode():
        for step in range(num_new_tokens):
            if step == 0:
                model_input_ids = input_ids
                model_attention_mask = current_attention_mask
                extra_kwargs = mm_forward_kwargs if data_args.template == "qwen2_vl" else {}
            else:
                if last_token_id is None:
                    raise RuntimeError("last_token_id must be initialized after the first generation step.")
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

                if start_mesh_generation:
                    mask = torch.full_like(logits, float("-inf"))
                    for tid in mesh_byte_token_ids:
                        if tid < logits.shape[-1]:
                            mask[0, tid] = 0
                    logits = logits + mask

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
                if start_mesh_generation:
                    mask = torch.full_like(next_token_logits, float("-inf"))
                    for tid in mesh_byte_token_ids:
                        if tid < next_token_logits.shape[-1]:
                            mask[0, tid] = 0
                    next_token_logits = next_token_logits + mask
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(-1)

            last_token_id = next_token_id
            next_token_value = next_token_id[0, 0].item()

            generated_ids.append(next_token_value)

            full_sequence_ids = torch.cat([full_sequence_ids, next_token_id], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=1
            )

            activated_mesh_generation = False
            if next_token_value == mesh_bos_id or (start_mesh_generation and next_token_value in mesh_byte_token_ids):
                if not start_mesh_generation:
                    start_mesh_generation = True
                    mesh_tokens_generated = 0
                    activated_mesh_generation = True

            if start_mesh_generation and not activated_mesh_generation:
                mesh_tokens_generated += 1
                if mesh_tokens_generated >= 1024:
                    mesh_eos_value = mesh_eos_token_id[0, 0].item()
                    generated_ids.append(mesh_eos_value)
                    full_sequence_ids = torch.cat([full_sequence_ids, mesh_eos_token_id], dim=1)
                    current_attention_mask = torch.cat(
                        [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=1
                    )
                    break

            if not start_mesh_generation and next_token_value == tokenizer.eos_token_id:
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
    parser.add_argument("--prompt", type=str, default="Generate a 3D asset from the following image:<image>")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--input_mode", type=str, default="image", choices=["image", "text"])
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of dataset shards.")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id in [0, num_shards).")

    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    model_args, data_args, finetuning_args, _ = get_infer_args(cfg)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor", None)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=False)
    with open(args.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_shards > 1:
        dataset = [item for idx, item in enumerate(dataset) if (idx % args.num_shards) == args.shard_id]

    for item in tqdm(dataset, desc=f"Generating (shard {args.shard_id}/{args.num_shards})"):
        name = item["name"]
        output_file = os.path.join(args.output_dir, f"{name}.txt")

        if os.path.exists(output_file):
            continue

        if args.input_mode == "text":
            description = item["token_sequence"]
            message = {"role": "user", "content": f"{args.prompt}{description}"}
            images = []
        else:
            images = item.get("images", [])
            message = {"role": "user", "content": args.prompt}
        generated_text = generate_single_sample(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            template=template,
            message=message,
            image_paths=images,
            num_new_tokens=args.num_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            system_prompt=args.system_prompt,
            data_args=data_args,
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)


if __name__ == "__main__":
    main()
