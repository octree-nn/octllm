#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate geometry-only captions from multiview images.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Optional dataset JSON override.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt override.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Current shard id in [0, num_shards).")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional max number of samples after sharding.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples whose output file already exists.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    return config


def resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")

    return dtype_map[key]


def get_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device") and isinstance(model.device, torch.device):
        return model.device

    return next(model.parameters()).device


def load_model_and_processor(config: dict[str, Any]) -> tuple[Any, torch.nn.Module]:
    model_name_or_path = config.get("model_name_or_path")
    if not model_name_or_path or str(model_name_or_path).startswith("/path/to/"):
        raise ValueError("Please set model_name_or_path in the config before running inference.")

    torch_dtype = resolve_dtype(config.get("torch_dtype", "bfloat16"))
    device_map = config.get("device_map", "auto")
    trust_remote_code = bool(config.get("trust_remote_code", True))

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    attn_implementation = config.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = None
    last_error: Exception | None = None
    candidate_classes = (
        AutoModelForImageTextToText,
        AutoModelForCausalLM,
    )
    for candidate_class in candidate_classes:
        try:
            model = candidate_class.from_pretrained(model_name_or_path, **model_kwargs)
            break
        except Exception as exc:
            last_error = exc

    if model is None:
        raise RuntimeError(
            "Failed to load the multimodal model with AutoModelForImageTextToText, "
            "or AutoModelForCausalLM."
        ) from last_error

    model.eval()
    return processor, model


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError(f"Dataset JSON must be a list: {dataset_path}")

    return dataset


def shard_dataset(dataset: list[dict[str, Any]], num_shards: int, shard_id: int) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")

    return [item for index, item in enumerate(dataset) if index % num_shards == shard_id]


def open_images(image_paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))

    return images


def build_messages(image_paths: list[str], prompt: str) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": image_path} for image_path in image_paths]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved_inputs[key] = value.to(device)
        else:
            moved_inputs[key] = value

    return moved_inputs


def clean_generated_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_+-]*", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    for prefix in ("assistant:", "assistant", "description:", "caption:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()

    text = text.strip().strip("\"'").strip()
    return " ".join(text.split())


def generate_description(
    processor: Any,
    model: torch.nn.Module,
    image_paths: list[str],
    prompt: str,
    generation_config: dict[str, Any],
) -> str:
    messages = build_messages(image_paths, prompt)
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError("The loaded processor does not support apply_chat_template.")

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    images = open_images(image_paths)
    model_inputs = processor(
        text=[chat_text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    input_device = get_input_device(model)
    model_inputs = move_inputs_to_device(model_inputs, input_device)

    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, **generation_config)

    prompt_length = model_inputs["input_ids"].shape[1]
    new_token_ids = generated_ids[0][prompt_length:]
    output_text = processor.decode(new_token_ids, skip_special_tokens=True)
    return clean_generated_text(output_text)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_path = (args.dataset_path or Path(config["dataset_path"])).expanduser().resolve()
    output_dir = (args.output_dir or Path(config["output_dir"])).expanduser().resolve()
    prompt = args.prompt or config.get("prompt")
    generation_config = {
        "max_new_tokens": int(config.get("max_new_tokens", 200)),
        "temperature": float(config.get("temperature", 0.2)),
        "top_p": float(config.get("top_p", 0.9)),
        "repetition_penalty": float(config.get("repetition_penalty", 1.05)),
        "do_sample": bool(config.get("do_sample", True)),
    }

    dataset = load_dataset(dataset_path)
    dataset = shard_dataset(dataset, args.num_shards, args.shard_id)
    if args.max_samples > 0:
        dataset = dataset[: args.max_samples]

    output_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_model_and_processor(config)

    for item in tqdm(dataset, desc=f"Generating shard {args.shard_id}/{args.num_shards}"):
        name = item["name"]
        image_paths = item["images"]
        output_path = output_dir / f"{name}.txt"

        if args.skip_existing and output_path.is_file():
            continue

        description = generate_description(
            processor=processor,
            model=model,
            image_paths=image_paths,
            prompt=prompt,
            generation_config=generation_config,
        )

        with output_path.open("w", encoding="utf-8") as f:
            f.write(description + "\n")


if __name__ == "__main__":
    main()
