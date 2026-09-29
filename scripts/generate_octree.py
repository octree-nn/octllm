import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageOps

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IMAGE_PLACEHOLDER
from llamafactory.hparams import get_infer_args
from llamafactory.model.incremental_octree_state import IncrementalOctreeState
from llamafactory.model.loader import load_model, load_tokenizer
from llamafactory.model.qwen25_vl_3d_replace import (
    load_pos_emb_weights as load_pos_emb_weights_qwen25,
)
from llamafactory.model.qwen25_vl_3d_replace import (
    replace_qwen25_vl_for_conditional_generation_forward_with_mesh_mask_loss,
)
from llamafactory.model.qwen25_vl_3d_router import (
    get_qwen25_vl_3d_router_config,
    install_qwen25_vl_3d_router,
    load_3d_router_weights,
)


# Match ShapeNet-style training renders under imgs/: square RGBA, black under
# transparent pixels, object centered with ~0.75 max-side fill on a 1024 canvas.
_DEFAULT_IMAGE_SIZE = 1024
_DEFAULT_OBJECT_FILL = 0.75
_ALPHA_FG_THRESHOLD = 10


def _mesh_special_tokens() -> list[str]:
    return [f"<mesh{i}>" for i in range(256)] + ["<mesh_bos>", "<mesh_eos>", "<MASK>"]


def _infer_alpha_from_border(rgb: np.ndarray, border: int = 4, tol: int = 12) -> np.ndarray:
    """Remove a near-uniform background only where it connects to the image border."""
    from scipy.ndimage import binary_propagation

    h, w = rgb.shape[:2]
    border = max(1, min(border, h // 8, w // 8))
    samples = np.concatenate(
        [
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(samples.astype(np.float32), axis=0)
    if samples.std(axis=0).mean() > 25:
        return np.full((h, w), 255, dtype=np.uint8)

    dist = np.linalg.norm(rgb.astype(np.float32) - bg[None, None, :], axis=-1)
    background = dist <= tol
    seeds = np.zeros_like(background)
    seeds[0, :] = background[0, :]
    seeds[-1, :] = background[-1, :]
    seeds[:, 0] = background[:, 0]
    seeds[:, -1] = background[:, -1]
    background = binary_propagation(seeds, mask=background)
    alpha = np.where(background, 0, 255).astype(np.uint8)
    if float((alpha > 0).mean()) < 0.005 or float((alpha > 0).mean()) > 0.98:
        return np.full((h, w), 255, dtype=np.uint8)
    return alpha


def preprocess_image_to_train_format(
    image: Image.Image | str | os.PathLike[str],
    size: int = _DEFAULT_IMAGE_SIZE,
    object_fill: float = _DEFAULT_OBJECT_FILL,
    alpha_mode: str = "straight",
) -> Image.Image:
    """Normalize an input image to the ShapeNet/imgs training render format.

    Produces a square RGBA PNG-like image with:
    - resolution ``size x size`` (training default 1024)
    - transparent background whose RGB is pure black
    - RGB composited onto black for the model's RGBA-to-RGB conversion
    - object centered with max(bbox) occupying about ``object_fill`` of the canvas

    Standard PNGs use straight alpha. Set ``alpha_mode="premultiplied"`` only
    when input RGB values have already been multiplied by alpha; dark object
    colors cannot reliably distinguish the two representations automatically.
    """
    if size < 1:
        raise ValueError("Image size must be at least 1.")
    if not 0.0 < object_fill <= 1.0:
        raise ValueError("object_fill must be in (0, 1].")
    if alpha_mode not in {"straight", "premultiplied"}:
        raise ValueError("alpha_mode must be 'straight' or 'premultiplied'.")

    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as source:
            rgba = ImageOps.exif_transpose(source).convert("RGBA")
    else:
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    if not np.any(alpha < 255):
        alpha = _infer_alpha_from_border(rgb)

    fg = alpha > _ALPHA_FG_THRESHOLD
    if not np.any(fg):
        raise ValueError("Input image has no visible foreground; check its alpha channel.")

    ys, xs = np.where(fg)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cropped_rgb = rgb[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    cropped_alpha = alpha[y0 : y1 + 1, x0 : x1 + 1]

    # Composite onto black exactly once. The model drops alpha with convert("RGB").
    if alpha_mode == "premultiplied":
        premul_rgb = np.clip(np.round(cropped_rgb), 0, 255).astype(np.uint8)
    else:
        a = cropped_alpha.astype(np.float32) / 255.0
        premul_rgb = np.clip(np.round(cropped_rgb * a[..., None]), 0, 255).astype(np.uint8)
    premul_rgb[cropped_alpha == 0] = 0

    obj_h, obj_w = cropped_alpha.shape
    target_side = max(1, int(round(size * object_fill)))
    scale = target_side / max(obj_w, obj_h)
    new_w = max(1, int(round(obj_w * scale)))
    new_h = max(1, int(round(obj_h * scale)))

    # Resize the already-composited RGB and alpha independently: Pillow's RGBA
    # resize would internally premultiply RGB a second time.
    resized_rgb = Image.fromarray(premul_rgb, mode="RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_alpha = Image.fromarray(cropped_alpha, mode="L").resize((new_w, new_h), Image.Resampling.LANCZOS)
    obj = Image.merge("RGBA", (*resized_rgb.split(), resized_alpha))

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    # Copy RGBA directly; using obj as the paste mask would square its alpha.
    canvas.paste(obj, (paste_x, paste_y))

    # Ensure fully transparent pixels stay pure black RGB.
    out = np.array(canvas)
    out[out[:, :, 3] == 0, :3] = 0
    return Image.fromarray(out, mode="RGBA")


def _preprocess_image_paths(
    image_paths: list[str] | None,
    *,
    enabled: bool = True,
    size: int = _DEFAULT_IMAGE_SIZE,
    object_fill: float = _DEFAULT_OBJECT_FILL,
    alpha_mode: str = "straight",
    save_dir: str | None = None,
) -> list[Any] | None:
    if not image_paths:
        return image_paths
    if not enabled:
        return image_paths

    processed: list[Any] = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for idx, path in enumerate(image_paths):
        image = preprocess_image_to_train_format(path, size=size, object_fill=object_fill, alpha_mode=alpha_mode)
        if save_dir:
            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(save_dir, f"{stem}_preprocessed.png")
            image.save(out_path)
            print(f"Saved preprocessed image: {out_path} ({image.size[0]}x{image.size[1]}, {image.mode})")
        processed.append(image)
        print(f"Preprocessed image[{idx}]: {path} -> {image.size[0]}x{image.size[1]} RGBA, object_fill={object_fill}")

    return processed


def _parse_mesh_tokens_to_bytes(mesh_content_text: str) -> list[int] | None:
    matches = re.findall(r"<mesh(\d+)>", mesh_content_text or "")
    if not matches:
        return None

    byte_sequence = [int(value) for value in matches]
    if any(value < 0 or value > 255 for value in byte_sequence):
        return None

    return byte_sequence


def _ensure_qwen2_vl_template(data_args: Any) -> None:
    if data_args.template != "qwen2_vl":
        raise ValueError("This inference script only supports `template: qwen2_vl`.")


def _ensure_mesh_token_config(infer_cfg: dict[str, Any]) -> None:
    if not bool(infer_cfg.get("use_separate_new_token_embeddings", False)):
        return

    if not infer_cfg.get("add_tokens") and not infer_cfg.get("add_special_tokens"):
        infer_cfg["add_tokens"] = ", ".join(_mesh_special_tokens())


def _candidate_checkpoint_paths(model_args: Any) -> list[str]:
    paths: list[str] = []
    adapter_paths = getattr(model_args, "adapter_name_or_path", None)
    if adapter_paths:
        paths.extend(adapter_paths)

    model_path = getattr(model_args, "model_name_or_path", None)
    if model_path:
        paths.append(model_path)

    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        path = str(path)
        if path not in seen:
            deduped.append(path)
            seen.add(path)

    return deduped


def _install_and_load_qwen25_3d_router(model: torch.nn.Module, model_args: Any, finetuning_args: Any) -> None:
    router_config = get_qwen25_vl_3d_router_config(model)
    if not getattr(finetuning_args, "use_3d_token_router", False) and not router_config:
        return

    selected_layer_ids = install_qwen25_vl_3d_router(
        model,
        replace_ffn=bool(router_config.get("replace_ffn", finetuning_args.route_3d_replace_ffn)),
        replace_attn_proj=bool(router_config.get("replace_attn_proj", finetuning_args.route_3d_replace_attn_proj)),
        attn_proj_mode=router_config.get("attn_proj_mode", finetuning_args.route_3d_attn_proj_mode),
        layer_scope=router_config.get("layer_scope", finetuning_args.route_3d_layer_scope),
        last_n_layers=int(router_config.get("last_n_layers", finetuning_args.route_3d_last_n_layers)),
        layer_ids=router_config.get("layer_ids", finetuning_args.route_3d_layer_ids),
        mlp_ratio=float(router_config.get("mlp_ratio", finetuning_args.route_3d_mlp_ratio)),
        init_from_base=bool(router_config.get("init_from_base", finetuning_args.route_3d_init_from_base)),
        freeze_base=bool(router_config.get("freeze_base", finetuning_args.route_3d_freeze_base)),
    )
    for checkpoint_path in _candidate_checkpoint_paths(model_args):
        if load_3d_router_weights(model, checkpoint_path):
            print(f"Loaded Qwen2.5-VL 3D token router: layers={selected_layer_ids}, checkpoint={checkpoint_path}")
            return
    raise FileNotFoundError("No 3D router weights found in the OctLLM model checkpoint.")


def _load_position_embedding_weights(model_args: Any, data_args: Any = None) -> None:
    if data_args is not None:
        _ensure_qwen2_vl_template(data_args)

    checkpoint_paths = _candidate_checkpoint_paths(model_args)

    for checkpoint_path in checkpoint_paths:
        if load_pos_emb_weights_qwen25(checkpoint_path, strict=False):
            return


def _get_cache_seq_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        first_layer = past_key_values[0]
        if isinstance(first_layer, (tuple, list)) and len(first_layer) > 0:
            return int(first_layer[0].shape[-2])
    return 0


def _crop_cache_to_length(past_key_values: Any, target_length: int) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(target_length)
        return past_key_values
    if isinstance(past_key_values, tuple):
        return tuple(
            tuple(cache_tensor[..., :target_length, :] for cache_tensor in layer_cache)
            for layer_cache in past_key_values
        )
    if isinstance(past_key_values, list):
        return [
            tuple(cache_tensor[..., :target_length, :] for cache_tensor in layer_cache)
            for layer_cache in past_key_values
        ]
    raise TypeError(f"Unsupported past_key_values type for cache crop: {type(past_key_values)}")


def _get_token_rank(logits: torch.Tensor, token_id: int) -> int | None:
    if not isinstance(token_id, int) or token_id < 0 or token_id >= logits.shape[-1]:
        return None

    token_logit = logits[0, token_id]
    return int((logits[0] > token_logit).sum().item()) + 1


def _maybe_force_mesh_bos(
    logits: torch.Tensor,
    mesh_bos_id: int,
    bos_top_k: int,
) -> tuple[torch.Tensor | None, int | None, float | None]:
    if bos_top_k is None or int(bos_top_k) <= 0:
        return None, None, None

    rank = _get_token_rank(logits, mesh_bos_id)
    if rank is None or rank > int(bos_top_k):
        return None, rank, None

    probs = torch.softmax(logits, dim=-1)
    next_token_id = torch.tensor([[mesh_bos_id]], device=logits.device)
    return next_token_id, rank, float(probs[0, mesh_bos_id].item())


def _mask_logits_to_token_ids(logits: torch.Tensor, token_ids: set[int]) -> torch.Tensor:
    valid_token_ids = [tid for tid in token_ids if isinstance(tid, int) and 0 <= tid < logits.shape[-1]]
    if not valid_token_ids:
        return logits

    mask = torch.full_like(logits, float("-inf"))
    mask[:, valid_token_ids] = 0
    return logits + mask


@dataclass
class OctLLMInferenceRuntime:
    """Reusable model state for running many samples with one checkpoint load."""

    model: Any
    tokenizer: Any
    processor: Any
    template: Any
    max_layer: int
    full_depth: int


def _load_inference_runtime(
    infer_cfg: dict[str, Any],
    *,
    max_layer: int,
    full_depth: int,
) -> OctLLMInferenceRuntime:
    """Load OctLLM once so batch callers can reuse it across samples."""
    _ensure_mesh_token_config(infer_cfg)
    model_args, data_args, finetuning_args, _ = get_infer_args(infer_cfg)
    _ensure_qwen2_vl_template(data_args)

    from scripts.model_sources import OCTLLM_MODEL_FILES, resolve_model_directory

    # The custom 3D router, position embeddings, and token heads read local shards.
    # Resolve one shared Hub snapshot before loading any component.
    model_args.model_name_or_path = str(resolve_model_directory(
        model_args.model_name_or_path,
        allow_patterns=OCTLLM_MODEL_FILES,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.hf_hub_token,
    ))

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
        full_depth=full_depth,
        max_depth=max_layer,
        add_mask_token=False,
    )

    # Update _old_forward on the model instance if it exists (for accelerate compatibility).
    if hasattr(model, "_old_forward"):
        model._old_forward = type(model).forward.__get__(model, type(model))

    base_model = getattr(model, "model", None)
    if base_model is not None and hasattr(base_model, "_old_forward"):
        base_model._old_forward = type(base_model).forward.__get__(base_model, type(base_model))

    _install_and_load_qwen25_3d_router(model, model_args, finetuning_args)
    _load_position_embedding_weights(model_args, data_args)

    return OctLLMInferenceRuntime(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        template=template,
        max_layer=int(max_layer),
        full_depth=int(full_depth),
    )


def _predict_next_token(
    message: dict[str, str],
    infer_cfg: dict[str, Any],
    image_paths: list[Any] | None = None,
    num_new_tokens: int = 1,
    max_layer: int = 5,
    full_depth: int = 4,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    bos_top_k: int = 0,
    system_prompt: str | None = None,
    keep_mask_in_cache: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    verbose: bool = True,
    runtime: OctLLMInferenceRuntime | None = None,
    measure_latency: bool = False,
) -> dict[str, Any]:
    if runtime is None:
        runtime = _load_inference_runtime(infer_cfg, max_layer=max_layer, full_depth=full_depth)
    elif runtime.max_layer != int(max_layer) or runtime.full_depth != int(full_depth):
        raise ValueError(
            "The reusable OctLLM runtime depth does not match this request: "
            f"runtime=({runtime.max_layer}, {runtime.full_depth}), "
            f"request=({max_layer}, {full_depth})."
        )

    model = runtime.model
    tokenizer = runtime.tokenizer
    processor = runtime.processor
    template = runtime.template

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

    mesh_token_id_to_byte: dict[int, int] = {}
    for i in range(256):
        token_id = tokenizer.convert_tokens_to_ids(f"<mesh{i}>")
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            mesh_token_id_to_byte[token_id] = i

    mesh_bos_id = tokenizer.convert_tokens_to_ids("<mesh_bos>")
    mesh_byte_token_ids = set(mesh_token_id_to_byte)
    mask_token_id = tokenizer.convert_tokens_to_ids("<MASK>")
    mesh_eos_token_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<mesh_eos>")]], device=model.device)
    _octree_start_ids = tokenizer.encode(" octree-start", add_special_tokens=False)

    if verbose:
        print(f"Detected {len(mesh_byte_token_ids)} mesh byte tokens")
        print(f"MASK token ID: {mask_token_id}")
        print(f"separate new-token embeddings: {bool(getattr(model, '_lf_use_separate_new_token_embeddings', False))}")

    # Keep the full sequence for output decoding and mesh end detection.
    full_sequence_ids = input_ids.clone()
    generated_ids = []
    generated_probs = []

    past_key_values = None
    current_attention_mask = attention_mask.clone()
    octree_state: IncrementalOctreeState | None = None
    rope_deltas = None
    cache_position = None
    previous_layer = None
    start_mesh_generation = False
    pre_mesh_byte_values: list[int] = []
    parsed_mesh_split_length = 0
    transient_mask_count = 0
    forced_mesh_bos_count = 0
    last_token_id = None
    autoregressive_forward_steps = 0

    # The benchmark times only the custom autoregressive loop. Prompt formatting,
    # tokenization, multimodal preprocessing, model loading, and final result
    # decoding are deliberately outside this boundary. Incremental octree state
    # updates remain inside because they control the next AR step and termination.
    latency_wall_start: float | None = None
    latency_wall_ms: float | None = None
    latency_cuda_ms: float | None = None
    latency_cuda_start: torch.cuda.Event | None = None
    latency_cuda_end: torch.cuda.Event | None = None
    if measure_latency:
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
            latency_cuda_start = torch.cuda.Event(enable_timing=True)
            latency_cuda_end = torch.cuda.Event(enable_timing=True)
        latency_wall_start = time.perf_counter()
        if latency_cuda_start is not None:
            latency_cuda_start.record()

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
                    # Concatenate last_token_id with MASK token
                    mask_token_tensor = torch.tensor([[mask_token_id]], device=model.device)
                    model_input_ids = torch.cat([last_token_id, mask_token_tensor], dim=1)
                    if keep_mask_in_cache:
                        model_attention_mask = torch.cat(
                            [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=1
                        )
                    else:
                        uses_transient_mask = True
                        transient_mask_count += 1
                        model_attention_mask = torch.cat(
                            [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=1
                        )
                else:
                    model_input_ids = last_token_id
                    model_attention_mask = current_attention_mask
                extra_kwargs = {}

            # The model consumes only the previous/next token positions. The
            # incremental state avoids rebuilding and transferring the full
            # split history on every autoregressive step.
            extra_kwargs["octree_decode_metadata"] = (
                octree_state.position_metadata if octree_state is not None else None
            )

            past_seen_tokens = _get_cache_seq_length(past_key_values)
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + model_input_ids.shape[1], device=model.device
            )
            if verbose:
                print(f"rope_deltas: {rope_deltas}")

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
            autoregressive_forward_steps += 1

            past_key_values = outputs.past_key_values
            if uses_transient_mask:
                past_key_values = _crop_cache_to_length(past_key_values, past_seen_tokens + 1)
            rope_deltas = outputs.rope_deltas

            next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            # Use sampling for positive temperatures and greedy decoding otherwise.
            use_sampling = temperature is not None and float(temperature) > 0.0
            bos_rank = None
            bos_force_logits = next_token_logits / float(temperature) if use_sampling else next_token_logits
            forced_bos_id, bos_rank, forced_bos_prob = (
                _maybe_force_mesh_bos(bos_force_logits, mesh_bos_id, bos_top_k)
                if not start_mesh_generation
                else (None, None, None)
            )

            if forced_bos_id is not None:
                next_token_probs = torch.softmax(bos_force_logits, dim=-1)
                next_token_id = forced_bos_id
                forced_mesh_bos_count += 1
                if verbose:
                    print(f"Forcing <mesh_bos>: rank={bos_rank}, prob={forced_bos_prob}, bos_top_k={bos_top_k}")
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

            token_prob = next_token_probs[0, next_token_id[0, 0]].item()
            next_token_value = next_token_id[0, 0].item()

            # Preserve the legacy parser's unusual edge case: once <mesh_bos>
            # appears, it used to collect every <meshN> token from the entire
            # generated prefix, including any emitted before the BOS marker.
            if not start_mesh_generation and next_token_value in mesh_byte_token_ids:
                pre_mesh_byte_values.append(mesh_token_id_to_byte[next_token_value])

            generated_ids.append(next_token_value)
            generated_probs.append(token_prob)

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
                if verbose:
                    print(f"Detected mesh token {next_token_value}, prob: {token_prob}")

                if next_token_value == mesh_bos_id:
                    octree_state = IncrementalOctreeState(max_depth=max_layer, full_depth=full_depth)
                    if progress_callback is not None:
                        progress_callback({"event": "mesh_start"})
                    if verbose:
                        print(f"Appending <mesh_bos> token {next_token_value}")
                    continue

                if octree_state is None:
                    raise RuntimeError("octree state must be initialized before consuming mesh bytes")

                byte_values_to_consume = pre_mesh_byte_values + [mesh_token_id_to_byte[next_token_value]]
                pre_mesh_byte_values = []
                parsed_mesh_split_length += len(byte_values_to_consume) * 8
                layer_status = None
                for byte_value in byte_values_to_consume:
                    if not octree_state.complete:
                        layer_status = octree_state.consume_byte(byte_value)
                if layer_status is None:
                    raise RuntimeError("octree state completed without consuming the current mesh prefix")
                current_layer = layer_status.layer
                layer_complete = layer_status.complete
                remaining_in_layer = layer_status.remaining
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "octree_layer",
                            "layer": int(current_layer),
                            "complete": bool(layer_complete),
                            "remaining": int(remaining_in_layer),
                            "split_length": int(parsed_mesh_split_length),
                        }
                    )
                if verbose:
                    print(
                        f"Octree layer status: layer={current_layer}, complete={layer_complete}, "
                        f"remaining={remaining_in_layer}"
                    )

                if verbose and previous_layer is not None and previous_layer == current_layer - 1:
                    print(f"previous layer: {previous_layer}, current layer: {current_layer}")
                previous_layer = current_layer

                if current_layer == max_layer and layer_complete:
                    mesh_eos_value = mesh_eos_token_id[0, 0].item()
                    generated_ids.append(mesh_eos_value)
                    generated_probs.append(1.0)  # This terminator is inserted deterministically.

                    full_sequence_ids = torch.cat([full_sequence_ids, mesh_eos_token_id], dim=1)
                    current_attention_mask = torch.cat(
                        [current_attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=1
                    )
                    octree_state = None
                    if progress_callback is not None:
                        progress_callback({"event": "mesh_complete", "layer": int(current_layer)})
                    if verbose:
                        print(f"Appending <mesh_eos> token {mesh_eos_value}")
                    break
            else:
                if verbose:
                    print(
                        f"not mesh token {tokenizer.decode([next_token_value], skip_special_tokens=False)}, "
                        f"prob: {token_prob}"
                    )

            if next_token_value == tokenizer.eos_token_id:
                break

    if measure_latency:
        if latency_cuda_end is not None and latency_cuda_start is not None:
            latency_cuda_end.record()
            torch.cuda.synchronize(model.device)
            latency_cuda_ms = float(latency_cuda_start.elapsed_time(latency_cuda_end))
        if latency_wall_start is not None:
            latency_wall_ms = float((time.perf_counter() - latency_wall_start) * 1000.0)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    result: dict[str, Any] = {
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "generated_probs": generated_probs,
        "transient_mask_count": transient_mask_count,
        "forced_mesh_bos_count": forced_mesh_bos_count,
        "final_kv_cache_length": _get_cache_seq_length(past_key_values),
        "output_token_count": len(generated_ids),
        "autoregressive_output_token_count": autoregressive_forward_steps,
        "structure_token_count": sum(token_id in mesh_byte_token_ids for token_id in generated_ids),
        "autoregressive_forward_steps": autoregressive_forward_steps,
    }

    if measure_latency:
        result.update(
            {
                "autoregressive_latency_ms": latency_cuda_ms if latency_cuda_ms is not None else latency_wall_ms,
                "autoregressive_cuda_latency_ms": latency_cuda_ms,
                "autoregressive_wall_latency_ms": latency_wall_ms,
            }
        )

    # Retain the result keys used by single-token callers.
    if len(generated_ids) == 1:
        result.update(
            {
                "next_token_id": generated_ids[0],
                "next_token": tokenizer.decode([generated_ids[0]], skip_special_tokens=False),
                "next_token_prob": generated_probs[0] if generated_probs else None,
            }
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next N tokens for a single user message (greedy).")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model YAML configuration, e.g. configs/inference/octllm.yaml.",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=False,
        default=None,
        help='JSON message such as {"role": "user", "content": "..."}. Omit to read messages interactively.',
    )
    parser.add_argument(
        "--num_new_tokens",
        type=int,
        default=10000,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature; positive values enable sampling, otherwise use greedy decoding.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold in (0, 1]; used only when sampling is enabled.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Number of top-k candidates to retain; positive values apply only when sampling is enabled.",
    )
    parser.add_argument(
        "--bos_top_k",
        type=int,
        default=0,
        help="Force <mesh_bos> when it ranks in the top k of the full vocabulary; 0 disables this behavior.",
    )
    parser.add_argument(
        "--max_layer",
        type=int,
        default=6,
        help="Maximum octree depth.",
    )
    parser.add_argument(
        "--full_depth",
        type=int,
        default=3,
        help="Depth through which all octree nodes are allocated.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to an input image.",
    )
    parser.add_argument(
        "--image_paths",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated image paths. Takes precedence over --image_path.",
    )
    parser.add_argument(
        "--no_preprocess_image",
        action="store_true",
        help="Disable preprocessing that aligns input images with the training render format.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=_DEFAULT_IMAGE_SIZE,
        help="Preprocessed image size in pixels (default: 1024, matching training renders).",
    )
    parser.add_argument(
        "--object_fill",
        type=float,
        default=_DEFAULT_OBJECT_FILL,
        help="Target object size as a fraction of the canvas (default: 0.75, matching training renders).",
    )
    parser.add_argument(
        "--image_alpha_mode",
        choices=("straight", "premultiplied"),
        default="straight",
        help="Input alpha encoding; standard PNGs use straight alpha.",
    )
    parser.add_argument(
        "--save_preprocessed_dir",
        type=str,
        default=None,
        help="Directory for saving preprocessed PNG images.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt to use during inference.",
    )
    parser.add_argument(
        "--keep_mask_in_cache",
        action="store_true",
        help="Keep temporary <MASK> tokens in the KV cache. By default, discard them after each prediction.",
    )

    args = parser.parse_args()

    # Parse model options separately from generation-specific CLI arguments.
    from scripts.model_sources import load_inference_config

    cfg = load_inference_config(args.config)

    if args.input_json is None:
        print('Enter JSON ({"role": "user", "content": "..."}); press Ctrl-D to exit:')
        raw = []
        try:
            while True:
                line = input()
                raw.append(line)
        except EOFError:
            pass
        payload = json.loads("\n".join(raw)) if raw else {"role": "user", "content": "Hello"}
    else:
        payload = json.loads(args.input_json)

    # Prefer --image_paths when both image options are supplied.
    resolved_image_paths = None
    if args.image_paths is not None and len(args.image_paths) > 0:
        resolved_image_paths = args.image_paths
    elif args.image_path is not None:
        resolved_image_paths = [args.image_path]

    resolved_image_paths = _preprocess_image_paths(
        resolved_image_paths,
        enabled=not args.no_preprocess_image,
        size=max(1, int(args.image_size)),
        object_fill=float(args.object_fill),
        alpha_mode=args.image_alpha_mode,
        save_dir=args.save_preprocessed_dir,
    )

    result = _predict_next_token(
        payload,
        cfg,
        image_paths=resolved_image_paths,
        num_new_tokens=max(1, args.num_new_tokens),
        max_layer=args.max_layer,
        full_depth=args.full_depth,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        bos_top_k=args.bos_top_k,
        system_prompt=args.system_prompt,
        keep_mask_in_cache=args.keep_mask_in_cache,
    )
    print(json.dumps(result, ensure_ascii=False))
    print(f"prompt: {payload['content']}")
    print(f"generated text: {result['generated_text']}")


if __name__ == "__main__":
    main()
