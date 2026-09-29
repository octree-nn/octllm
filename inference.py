#!/usr/bin/env python3
"""Unified single-turn CLI for OctLLM text and 3D inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference" / "pipeline.yaml"
MESH_SEQUENCE_PATTERN = re.compile(r"<mesh_bos>(?:\s*<mesh\d+>\s*)+<mesh_eos>", re.DOTALL)
MESH_TOKEN_PATTERN = re.compile(r"<mesh(\d+)>")
TEXT_TO_3D_PREFIX = "Generate a 3D mesh based on the following text description:"
DEFAULT_UNDERSTANDING_PROMPT = "Describe this 3D asset in detail:"
END_TOKENS = ("<|im_end|>", "<|endoftext|>")


def extract_mesh_sequence(text: str) -> str | None:
    """Return the first complete octree token sequence in a model response."""
    match = MESH_SEQUENCE_PATTERN.search(text or "")
    if match is None:
        return None

    sequence = re.sub(r"\s+", "", match.group(0))
    values = [int(value) for value in MESH_TOKEN_PATTERN.findall(sequence)]
    if not values or any(value < 0 or value > 255 for value in values):
        raise ValueError("The generated octree contains a mesh byte outside [0, 255].")
    return sequence


def clean_assistant_text(text: str, mesh_sequence: str | None = None) -> str:
    """Remove octree payload and chat-template end tokens from visible text."""
    visible = text or ""
    if mesh_sequence:
        visible = MESH_SEQUENCE_PATTERN.sub("", visible, count=1)
    for token in END_TOKENS:
        visible = visible.replace(token, "")
    return " ".join(visible.split()).strip()


def extract_text_condition(prompt: str, override: str | None = None) -> str:
    """Extract the TRELLIS text condition from the original OctLLM prompt."""
    if override and override.strip():
        return override.strip()

    prompt = (prompt or "").strip()
    prefix_index = prompt.casefold().find(TEXT_TO_3D_PREFIX.casefold())
    if prefix_index >= 0:
        description = prompt[prefix_index + len(TEXT_TO_3D_PREFIX) :].strip()
        if description:
            return description

    without_octree = MESH_SEQUENCE_PATTERN.sub("", prompt).strip()
    if "<mesh_bos>" in without_octree or MESH_TOKEN_PATTERN.search(without_octree):
        without_octree = re.sub(r"<mesh_bos>|<mesh_eos>|<mesh\d+>", "", without_octree).strip()
    if not without_octree:
        raise ValueError("No text condition is available. Set `trellis.text_prompt` or pass --trellis-text.")
    return without_octree


def choose_trellis_condition(
    mode: str,
    prompt: str,
    image_paths: list[str],
    text_override: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Resolve an image or text condition while preserving the original input."""
    mode = (mode or "auto").lower()
    if mode not in {"auto", "image", "text"}:
        raise ValueError(f"Unsupported TRELLIS condition: {mode!r}")
    if mode == "auto":
        mode = "image" if image_paths else "text"

    if mode == "image":
        if not image_paths:
            raise ValueError("Image-conditioned TRELLIS generation requires an input image.")
        return mode, image_paths[0], None

    return mode, None, extract_text_condition(prompt, text_override)


class OctreeLayerProgress:
    """Render one sequential progress bar for every generated octree depth."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._bar = None
        self._layer: int | None = None
        self._completed_before = 0
        self._current_total = 0

    def __call__(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if event.get("event") == "mesh_start":
            from tqdm import tqdm

            tqdm.write("Octree generation started.")
            return
        if event.get("event") == "mesh_complete":
            self.close(complete=True)
            return
        if event.get("event") != "octree_layer":
            return

        layer = int(event["layer"])
        remaining = max(0, int(event["remaining"]))
        split_length = max(0, int(event["split_length"]))

        if layer != self._layer:
            self.close(complete=True)
            completed_in_layer = max(0, split_length - self._completed_before)
            total = completed_in_layer + remaining
            self._layer = layer
            self._current_total = total
            from tqdm import tqdm

            self._bar = tqdm(
                total=total,
                initial=min(completed_in_layer, total),
                desc=f"Octree depth {layer}",
                unit="node",
                dynamic_ncols=True,
                leave=True,
            )
        else:
            completed_in_layer = max(0, split_length - self._completed_before)
            total = completed_in_layer + remaining
            if total > self._current_total:
                self._current_total = total
                self._bar.total = total
                self._bar.refresh()
            target = min(completed_in_layer, self._current_total)
            self._bar.update(max(0, target - self._bar.n))

        if bool(event["complete"]):
            self.close(complete=True)

    def close(self, complete: bool = False) -> None:
        if self._bar is None:
            return
        if complete:
            self._bar.update(max(0, self._bar.total - self._bar.n))
            self._completed_before += int(self._bar.total)
        self._bar.close()
        self._bar = None
        self._layer = None
        self._current_total = 0


def _load_settings(path: str) -> dict[str, Any]:
    from omegaconf import OmegaConf

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"OctLLM inference config not found: {config_path}")
    settings = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(settings, dict):
        raise ValueError(f"Expected a mapping in {config_path}")

    mllm = settings.get("mllm", {})
    mllm_config = mllm.get("config") if isinstance(mllm, dict) else None
    if mllm_config and not Path(str(mllm_config)).is_absolute():
        mllm["config"] = str((config_path.parent / str(mllm_config)).resolve())
    vae = settings.get("mesh", {}).get("vae", {})
    if mllm_config and str(vae.get("mode", "none")).lower() == "legacy" and not vae.get("checkpoint"):
        from scripts.model_sources import load_inference_config

        # Resolve the completion file lazily so text-only inference needs no decoder weights.
        vae["model_source"] = load_inference_config(mllm["config"])["model_name_or_path"]
    return settings


def _read_prompt(args: argparse.Namespace) -> tuple[str, bool]:
    if args.prompt_file is not None:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip(), False
    if args.prompt is not None:
        return args.prompt.strip(), False
    if args.mesh:
        return DEFAULT_UNDERSTANDING_PROMPT, False
    return input("Text / octree input: ").strip(), True


def _resolve_images(args: argparse.Namespace, interactive: bool) -> list[str]:
    images = [str(Path(path).expanduser().resolve()) for path in (args.image or [])]
    if interactive and not images:
        raw = input("Image path (optional, press Enter to skip): ").strip()
        if raw:
            images.append(str(Path(raw).expanduser().resolve()))
    missing = [path for path in images if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Input image not found: {missing[0]}")
    return images


def _prepare_mesh_input(
    mesh_path: str,
    prompt: str,
    settings: dict[str, Any],
) -> tuple[str, Path, str]:
    from dataset_toolkits.tokenize_mesh import mesh_to_octree_sequence

    if "<mesh_bos>" in prompt or "<mesh_eos>" in prompt or MESH_TOKEN_PATTERN.search(prompt):
        raise ValueError("Use a text-only question with --mesh; remove the embedded octree tokens.")

    path = Path(mesh_path).expanduser().resolve()
    generation = settings.get("mllm", {}).get("generation", {})
    mesh_input = settings.get("mesh_input", {})
    depth = int(generation.get("max_layer", 6))
    sequence = mesh_to_octree_sequence(
        path,
        num_samples=int(mesh_input.get("num_samples", 100000)),
        depth=depth,
        full_depth=int(generation.get("full_depth", 3)),
        device=str(mesh_input.get("device", "cpu")),
        target_depth=int(mesh_input.get("target_depth", depth - 1)),
        drop_prob=float(mesh_input.get("drop_prob", 0.5)),
        prune=bool(mesh_input.get("prune", True)),
    )
    return f"{prompt} {sequence}", path, sequence


def _model_inference(
    prompt: str,
    image_paths: list[str],
    settings: dict[str, Any],
    progress: OctreeLayerProgress,
) -> dict[str, Any]:
    from scripts.generate_octree import _predict_next_token, _preprocess_image_paths
    from scripts.model_sources import load_inference_config

    mllm = settings.get("mllm", {})
    generation = mllm.get("generation", {})
    image_config = mllm.get("image", {})
    model_config_path = mllm.get("config")
    if not model_config_path:
        raise ValueError("Set `mllm.config` in the unified inference config.")

    infer_cfg = load_inference_config(model_config_path)
    processed_images = _preprocess_image_paths(
        image_paths,
        enabled=bool(image_config.get("preprocess", True)),
        size=max(1, int(image_config.get("size", 1024))),
        object_fill=float(image_config.get("object_fill", 0.75)),
        alpha_mode=str(image_config.get("alpha_mode", "straight")),
        save_dir=image_config.get("save_dir"),
    )
    return _predict_next_token(
        {"role": "user", "content": prompt},
        infer_cfg,
        image_paths=processed_images,
        num_new_tokens=max(1, int(generation.get("max_new_tokens", 10000))),
        max_layer=int(generation.get("max_layer", 6)),
        full_depth=int(generation.get("full_depth", 3)),
        temperature=float(generation.get("temperature", 0.5)),
        top_p=float(generation.get("top_p", 0.9)),
        top_k=int(generation.get("top_k", 40)),
        bos_top_k=int(generation.get("bos_top_k", 1)),
        system_prompt=mllm.get("system_prompt"),
        keep_mask_in_cache=bool(generation.get("keep_mask_in_cache", False)),
        progress_callback=progress,
        verbose=bool(mllm.get("verbose", False)),
    )


def _octree_tokens_to_voxel(
    mesh_sequence: str,
    depth: int,
    full_depth: int,
    device: str,
    threshold: float,
):
    from llamafactory.model.utils.utils import seq2octree
    from scripts.decode_octree import (
        binary_to_split_tensor,
        bytes_to_binary_sequence,
        create_template_octree,
        octree_to_voxel,
    )

    byte_values = [int(value) for value in MESH_TOKEN_PATTERN.findall(mesh_sequence)]
    split_tensor = binary_to_split_tensor(bytes_to_binary_sequence(byte_values))
    octree_template = create_template_octree(depth, full_depth, device)
    if octree_template is None:
        raise RuntimeError("Failed to create the octree template.")
    octree = seq2octree(octree_template, split_tensor.to(device), full_depth, depth + 1, threshold)
    if octree is None:
        raise RuntimeError("Failed to reconstruct the generated octree sequence.")
    voxel = octree_to_voxel(octree, depth=depth)
    if voxel is None:
        raise RuntimeError("Failed to convert the generated octree to voxels.")
    return voxel


def _complete_voxel(voxel, mesh_settings: dict[str, Any]):
    vae = mesh_settings.get("vae", {})
    mode = str(vae.get("mode", "none")).lower()
    if mode == "none":
        return voxel

    from scripts.decode_octree import vae_complete_voxel, vae_complete_voxel_trellis_align

    device = str(vae.get("device", "cuda"))
    threshold = vae.get("threshold")
    if mode == "legacy":
        from scripts.decode_octree import get_vae_model

        checkpoint = vae.get("checkpoint")
        if not checkpoint:
            from scripts.model_sources import COMPLETION_FILENAME, resolve_model_file

            source = vae.get("model_source")
            if not source:
                raise ValueError("Set `mllm.config` for bundled completion or `mesh.vae.checkpoint` for an override.")
            checkpoint = resolve_model_file(source, COMPLETION_FILENAME)
        model = get_vae_model(str(checkpoint), device=device)
        return vae_complete_voxel(voxel, model, device=device, threshold=threshold)
    if mode == "trellis_align":
        checkpoint = vae.get("encoder_align_checkpoint")
        trellis_vae_dir = vae.get("trellis_vae_dir")
        if not checkpoint or not trellis_vae_dir:
            raise ValueError(
                "Set `mesh.vae.encoder_align_checkpoint` and `mesh.vae.trellis_vae_dir` "
                "when VAE mode is `trellis_align`."
            )
        return vae_complete_voxel_trellis_align(
            voxel,
            encoder_ckpt=str(checkpoint),
            trellis_dir=str(trellis_vae_dir),
            device=device,
            threshold=threshold,
        )
    raise ValueError(f"Unsupported VAE completion mode: {mode!r}")


def _generate_glb(
    voxel,
    prompt: str,
    image_paths: list[str],
    mesh_settings: dict[str, Any],
    output_dir: Path,
    output_name: str,
    condition_override: str | None,
    text_override: str | None,
    model_override: str | None,
) -> Path:
    from scripts.decode_octree import get_trellis_pipeline, trellis_generate_from_voxel

    trellis = mesh_settings.get("trellis", {})
    condition_mode, image_path, text_prompt = choose_trellis_condition(
        condition_override or trellis.get("condition", "auto"),
        prompt,
        image_paths,
        text_override or trellis.get("text_prompt"),
    )
    model_path = model_override or trellis.get(f"{condition_mode}_model_path") or trellis.get("model_path")
    if not model_path:
        raise ValueError(
            f"Set `mesh.trellis.{condition_mode}_model_path`, `mesh.trellis.model_path`, or --trellis-model."
        )

    pipeline = get_trellis_pipeline(
        str(model_path),
        task_type=condition_mode,
        text_condition_model_path=trellis.get("text_condition_model_path"),
    )
    glb_path = trellis_generate_from_voxel(
        voxel,
        image_path,
        text_prompt,
        pipeline,
        output_dir=str(output_dir),
        output_name=output_name,
        simplify=float(trellis.get("simplify", 0.95)),
        texture_size=int(trellis.get("texture_size", 1024)),
        task_type=condition_mode,
        voxel_rotation_axis=trellis.get("voxel_rotation_axis", "x"),
        voxel_rotation_quarter_turns=int(trellis.get("voxel_rotation_quarter_turns", 1)),
    )
    return Path(glb_path).resolve()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cuda_device is not None:
        if args.cuda_device < 0:
            raise ValueError("--cuda-device must be a non-negative GPU index.")
        # Select before importing Torch or preprocessing meshes. All stages,
        # including completion and TRELLIS, then use this GPU as local cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    settings = _load_settings(args.config)
    prompt, interactive = _read_prompt(args)
    if not prompt:
        raise ValueError("The input text cannot be empty.")
    image_paths = _resolve_images(args, interactive)

    output = settings.get("output", {})
    output_dir = Path(args.output_dir or output.get("dir", "outputs/octllm")).expanduser().resolve()
    output_name = args.output_name or output.get("name") or datetime.now().strftime("octllm_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_metadata = {"text": prompt, "images": image_paths}
    if args.mesh:
        print(f"Converting input mesh to S-Octree tokens: {args.mesh}", flush=True)
        prompt, input_mesh_path, input_sequence = _prepare_mesh_input(args.mesh, prompt, settings)
        input_tokens_path = output_dir / f"{output_name}.input.tokens.txt"
        input_tokens_path.write_text(input_sequence + "\n", encoding="utf-8")
        input_metadata.update(
            mesh=str(input_mesh_path),
            model_prompt=prompt,
            octree_tokens_path=str(input_tokens_path.resolve()),
        )
        print(f"Input octree tokens: {input_tokens_path}", flush=True)

    progress = OctreeLayerProgress(enabled=not args.no_progress)
    try:
        model_result = _model_inference(prompt, image_paths, settings, progress)
    finally:
        progress.close()

    generated_text = str(model_result["generated_text"])
    mesh_sequence = extract_mesh_sequence(generated_text)
    if mesh_sequence is None and "<mesh_bos>" in generated_text:
        raise RuntimeError("Octree generation ended before a complete <mesh_eos> token was produced.")

    assistant_text = clean_assistant_text(generated_text, mesh_sequence)
    glb_path = None
    tokens_path = None
    mesh_settings = settings.get("mesh", {})
    mesh_enabled = bool(mesh_settings.get("enabled", True)) and not args.no_mesh
    if mesh_sequence is not None:
        tokens_path = output_dir / f"{output_name}.tokens.txt"
        tokens_path.write_text(mesh_sequence + "\n", encoding="utf-8")
        if mesh_enabled:
            generation = settings.get("mllm", {}).get("generation", {})
            depth = int(generation.get("max_layer", 6))
            full_depth = int(generation.get("full_depth", 3))
            voxel = _octree_tokens_to_voxel(
                mesh_sequence,
                depth=depth,
                full_depth=full_depth,
                device=str(mesh_settings.get("octree_device", "cpu")),
                threshold=float(mesh_settings.get("octree_threshold", 0.0)),
            )
            voxel = _complete_voxel(voxel, mesh_settings)
            glb_path = _generate_glb(
                voxel,
                prompt,
                image_paths,
                mesh_settings,
                output_dir,
                output_name,
                args.trellis_condition,
                args.trellis_text,
                args.trellis_model,
            )

    result = {
        "type": "mesh" if mesh_sequence is not None else "text",
        "assistant_text": assistant_text,
        "glb_path": str(glb_path) if glb_path else None,
        "octree_tokens_path": str(tokens_path.resolve()) if tokens_path else None,
        "input": input_metadata,
        "generation": model_result,
    }
    result_path = (output_dir / f"{output_name}.json").resolve()
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one OctLLM turn with text, images, or a GLB/OBJ mesh; decode generated octrees to GLB.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Unified OctLLM inference YAML.")
    parser.add_argument(
        "--cuda-device",
        type=int,
        help="Physical CUDA GPU index (e.g. 1). Overrides CUDA_VISIBLE_DEVICES; omit to preserve the environment.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-p", "--prompt", help="Text or octree input. Defaults to describing --mesh, or interactive input otherwise."
    )
    input_group.add_argument("--prompt-file", help="UTF-8 file containing a text or octree input.")
    parser.add_argument("-i", "--image", action="append", help="Optional image path; repeat for multiple images.")
    parser.add_argument("--mesh", help="Input .glb or .obj mesh for 3D understanding; append its S-Octree tokens.")
    parser.add_argument("--output-dir", help="Override output.dir from the YAML.")
    parser.add_argument("--output-name", help="Override output.name from the YAML (without extension).")
    parser.add_argument("--no-mesh", action="store_true", help="Keep generated octree tokens but skip GLB creation.")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-layer octree progress bars.")
    parser.add_argument("--trellis-model", help="Override the selected TRELLIS repository ID or local model directory.")
    parser.add_argument("--trellis-condition", choices=("auto", "image", "text"), help="Override condition mode.")
    parser.add_argument("--trellis-text", help="Override the automatically extracted TRELLIS text condition.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except (EOFError, KeyboardInterrupt):
        print("\nInference cancelled.")
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Assistant: {result['assistant_text'] or '(mesh generated)'}")
    if result["glb_path"]:
        print(f"GLB: {result['glb_path']}")
    elif result["octree_tokens_path"]:
        print(f"Octree tokens: {result['octree_tokens_path']}")
    print(f"Result: {result['result_path']}")


if __name__ == "__main__":
    main()
