#!/usr/bin/env python3
"""Hugging Face autoregressive efficiency runner.

Supported model families:
  * ShapeLLM-Omni and 3DGen-R1 (Qwen2.5-VL with discrete mesh tokens)
  * LLaMA-Mesh (Llama emitting a textual OBJ representation)

Only ``model.generate`` is measured.  Prompt construction, tokenization, model
loading, VQ-VAE decoding, flow models, rendering, and mesh export are excluded.
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import transformers

try:
    from .common import (
        ManifestSample,
        atomic_write_json,
        base_result,
        load_manifest,
        pending_samples,
        sample_output_path,
        sha256_text,
    )
except ImportError:  # Support ``python path/to/hf_generate.py``.
    from common import (  # type: ignore[no-redef]
        ManifestSample,
        atomic_write_json,
        base_result,
        load_manifest,
        pending_samples,
        sample_output_path,
        sha256_text,
    )


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evaluation.inference_efficiency.model_artifacts import resolve_method_artifacts


QWEN_PROMPT = "Please generate a 3D mesh based on the prompt I provided: {caption}"
LLAMA_PROMPT = "Create a 3D model in OBJ format based on this description: {caption}"
MESH_TOKEN_MIN = 151665
MESH_TOKEN_MAX = 159856
MESH_START_TOKEN_ID = 159857
MESH_END_TOKEN_ID = 159858

_METHOD_ALIASES = {
    "shapellm-omni": ("ShapeLLM-Omni", "qwen_mesh"),
    "shapellm_omni": ("ShapeLLM-Omni", "qwen_mesh"),
    "shapellmomni": ("ShapeLLM-Omni", "qwen_mesh"),
    "shape-llm-omni": ("ShapeLLM-Omni", "qwen_mesh"),
    "shape_llm_omni": ("ShapeLLM-Omni", "qwen_mesh"),
    "3dgen-r1": ("3DGen-R1", "qwen_mesh"),
    "3dgen_r1": ("3DGen-R1", "qwen_mesh"),
    "3dgenr1": ("3DGen-R1", "qwen_mesh"),
    "llama-mesh": ("LLaMA-Mesh", "llama_obj"),
    "llama_mesh": ("LLaMA-Mesh", "llama_obj"),
    "llamamesh": ("LLaMA-Mesh", "llama_obj"),
}


@dataclass(frozen=True)
class GenerationResult:
    output_ids: list[int]
    output_text: str
    cuda_latency_ms: float
    wall_latency_ms: float


def canonical_method(value: str) -> tuple[str, str]:
    key = value.strip().lower()
    try:
        return _METHOD_ALIASES[key]
    except KeyError as exc:
        supported = "ShapeLLM-Omni, 3DGen-R1, LLaMA-Mesh"
        raise argparse.ArgumentTypeError(
            f"unsupported method {value!r}; expected one of: {supported}"
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure only the autoregressive Hugging Face generation stage, "
            "using local model files and resumable per-asset JSON results."
        )
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--model-path", required=True, help="Hugging Face repository ID or local model directory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of full, unrecorded generations using the first pending sample.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-asset results, including mismatched hashes.",
    )
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if not args.config_hash.strip():
        parser.error("--config-hash must be non-empty")
    if not args.source_commit.strip():
        parser.error("--source-commit must be non-empty")
    try:
        args.canonical_method, args.model_family = canonical_method(args.method)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def _deduplicate(values: list[int | None]) -> list[int]:
    result: list[int] = []
    for value in values:
        if value is not None and value >= 0 and value not in result:
            result.append(int(value))
    return result


class HFGenerateRunner:
    def __init__(
        self,
        *,
        method: str,
        family: str,
        model_path: str | Path,
        device: str,
        dtype: str,
        attn_implementation: str,
    ) -> None:
        self.method = method
        self.family = family
        self.requested_device = torch.device(device)
        if self.requested_device.type != "cuda":
            raise ValueError("this benchmark requires a CUDA device for CUDA-event timing")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        artifacts = resolve_method_artifacts(method, {"model_path": model_path})
        self.model_path = Path(artifacts["model_path"])

        self.dtype = getattr(torch, dtype)
        self.attn_implementation = attn_implementation
        device_index = (
            self.requested_device.index
            if self.requested_device.index is not None
            else torch.cuda.current_device()
        )
        torch.cuda.set_device(device_index)

        common_load_kwargs = {
            "local_files_only": True,
            "torch_dtype": self.dtype,
            "device_map": {"": device_index},
            "low_cpu_mem_usage": True,
            "attn_implementation": attn_implementation,
        }
        if family == "qwen_mesh":
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path), local_files_only=True, use_fast=False
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.model_path), **common_load_kwargs
            )
            self.tokenizer = self.processor.tokenizer
            self.max_new_tokens = 2048
            self.prompt_template = QWEN_PROMPT
            self.stop_token_ids = _deduplicate(
                [self.tokenizer.eos_token_id, MESH_END_TOKEN_ID]
            )
            self.generation_parameters = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.7,
                "top_k": 8192,
                "repetition_penalty": 1.05,
                "eos_token_id": self.stop_token_ids,
                "use_cache": True,
                "num_beams": 1,
            }
        elif family == "llama_obj":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), local_files_only=True, use_fast=False
            )
            self.processor = None
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), **common_load_kwargs
            )
            self.max_new_tokens = 8192
            self.prompt_template = LLAMA_PROMPT
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            self.stop_token_ids = _deduplicate(
                [self.tokenizer.eos_token_id, eot_id]
            )
            self.generation_parameters = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": 0.95,
                "top_p": 0.9,
                "eos_token_id": self.stop_token_ids,
                "use_cache": True,
                "num_beams": 1,
            }
        else:
            raise AssertionError(f"unknown model family: {family}")

        self.model.eval()
        self.model_device = next(self.model.parameters()).device
        if self.model_device.type != "cuda":
            raise RuntimeError(f"model was not loaded on CUDA: {self.model_device}")
        self.runtime_metadata = self._runtime_metadata()
        self.model_metadata = self._model_metadata()

    def _runtime_metadata(self) -> dict[str, Any]:
        index = self.model_device.index or 0
        props = torch.cuda.get_device_properties(index)
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(self.model_device),
            "gpu_name": props.name,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "gpu_total_memory_bytes": props.total_memory,
            "attention_implementation": self.attn_implementation,
            "timing_scope": (
                "synchronized model.generate only; includes prompt prefill and "
                "autoregressive decode; excludes input processing and all 3D postprocessing"
            ),
        }

    def _model_metadata(self) -> dict[str, Any]:
        config = self.model.config
        return {
            "path": str(self.model_path),
            "class": type(self.model).__name__,
            "model_type": getattr(config, "model_type", None),
            "config_transformers_version": getattr(
                config, "transformers_version", None
            ),
            "dtype": str(self.dtype).removeprefix("torch."),
            "max_new_tokens": self.max_new_tokens,
            "generation_parameters": dict(self.generation_parameters),
        }

    def prompt_for(self, sample: ManifestSample) -> str:
        return self.prompt_template.format(caption=sample.text_description)

    def prepare_inputs(self, prompt: str) -> Mapping[str, torch.Tensor]:
        messages = [{"role": "user", "content": prompt}]
        if self.family == "qwen_mesh":
            qwen_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
            rendered = self.processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[rendered], padding=True, return_tensors="pt"
            )
        else:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                rendered, add_special_tokens=False, return_tensors="pt"
            )
        return {key: value.to(self.model_device) for key, value in inputs.items()}

    def generate(self, prepared: Mapping[str, torch.Tensor], *, seed: int) -> GenerationResult:
        prompt_length = int(prepared["input_ids"].shape[1])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(self.model_device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **prepared,
                **self.generation_parameters,
                return_dict_in_generate=True,
                output_scores=False,
            )
        end_event.record()
        end_event.synchronize()
        wall_latency_ms = (time.perf_counter() - wall_start) * 1000.0
        cuda_latency_ms = start_event.elapsed_time(end_event)

        new_ids_tensor = generated.sequences[0, prompt_length:]
        output_ids = [int(value) for value in new_ids_tensor.detach().cpu().tolist()]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        return GenerationResult(
            output_ids=output_ids,
            output_text=output_text,
            cuda_latency_ms=float(cuda_latency_ms),
            wall_latency_ms=float(wall_latency_ms),
        )

    def structure_token_count(self, output_ids: list[int]) -> int:
        if self.family == "qwen_mesh":
            return sum(MESH_TOKEN_MIN <= token_id <= MESH_TOKEN_MAX for token_id in output_ids)
        # LLaMA-Mesh's native representation is OBJ text, so every emitted token
        # other than the terminator is part of the native structural response.
        return sum(token_id not in self.stop_token_ids for token_id in output_ids)

    def token_definition(self) -> dict[str, Any]:
        if self.family == "qwen_mesh":
            structure = (
                f"generated IDs in [{MESH_TOKEN_MIN}, {MESH_TOKEN_MAX}]; "
                "<mesh-start>/<mesh-end> are excluded"
            )
        else:
            structure = (
                "all generated native OBJ-text tokens except EOS/EOT terminators"
            )
        return {
            "output_token_count": (
                "all token IDs returned after the input prompt, including a generated terminator"
            ),
            "structure_token_count": structure,
            "autoregressive_steps": "equal to output_token_count for decoder-only generation",
        }

    def termination(self, output_ids: list[int]) -> tuple[dict[str, Any], bool]:
        last_id = output_ids[-1] if output_ids else None
        stopped = last_id in self.stop_token_ids if last_id is not None else False
        truncated = len(output_ids) >= self.max_new_tokens and not stopped
        if stopped:
            reason = "eos_token"
        elif truncated:
            reason = "max_new_tokens"
        elif output_ids:
            reason = "generation_stopped_without_configured_eos"
        else:
            reason = "empty_output"
        return (
            {
                "reason": reason,
                "terminated_by_configured_eos": stopped,
                "last_token_id": last_id,
                "last_token": (
                    self.tokenizer.convert_ids_to_tokens(last_id)
                    if last_id is not None
                    else None
                ),
                "configured_eos_token_ids": list(self.stop_token_ids),
            },
            truncated,
        )

    def structure_diagnostics(self, output_ids: list[int], output_text: str) -> dict[str, Any]:
        if self.family == "qwen_mesh":
            return {
                "mesh_start_count": output_ids.count(MESH_START_TOKEN_ID),
                "mesh_end_count": output_ids.count(MESH_END_TOKEN_ID),
                "mesh_code_count": self.structure_token_count(output_ids),
                "expected_mesh_code_count": 1024,
            }
        lines = output_text.splitlines()
        return {
            "obj_vertex_line_count": sum(line.lstrip().startswith("v ") for line in lines),
            "obj_face_line_count": sum(line.lstrip().startswith("f ") for line in lines),
        }


def success_payload(
    *,
    runner: HFGenerateRunner,
    sample: ManifestSample,
    config_hash: str,
    source_commit: str,
    seed: int,
    prompt: str,
    result: GenerationResult,
) -> dict[str, Any]:
    payload = base_result(method=runner.method, config_hash=config_hash, sample=sample)
    termination, truncated = runner.termination(result.output_ids)
    output_count = len(result.output_ids)
    structure = runner.structure_diagnostics(result.output_ids, result.output_text)
    if runner.family == "qwen_mesh":
        valid_structure = (
            structure["mesh_start_count"] == 1
            and structure["mesh_end_count"] == 1
            and structure["mesh_code_count"] == structure["expected_mesh_code_count"]
        )
    else:
        valid_structure = structure["obj_vertex_line_count"] > 0 and structure["obj_face_line_count"] > 0
    payload.update(
        {
            "status": "ok",
            "latency_ms": result.cuda_latency_ms,
            "cuda_latency_ms": result.cuda_latency_ms,
            "wall_latency_ms": result.wall_latency_ms,
            "output_token_count": output_count,
            "structure_token_count": runner.structure_token_count(result.output_ids),
            "autoregressive_steps": output_count,
            "token_definition": runner.token_definition(),
            "termination": termination,
            "truncated": truncated,
            "valid_structure": valid_structure,
            "error": None,
            "model": runner.model_metadata,
            "source": {"commit": source_commit},
            "runtime": runner.runtime_metadata,
            "seed": seed,
            "prompt": prompt,
            "output_text": result.output_text,
            "output_text_sha256": sha256_text(result.output_text),
            "structure": structure,
        }
    )
    return payload


def error_payload(
    *,
    runner: HFGenerateRunner,
    sample: ManifestSample,
    config_hash: str,
    source_commit: str,
    seed: int,
    prompt: str,
    exc: BaseException,
) -> dict[str, Any]:
    payload = base_result(method=runner.method, config_hash=config_hash, sample=sample)
    payload.update(
        {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
            "model": runner.model_metadata,
            "source": {"commit": source_commit},
            "runtime": runner.runtime_metadata,
            "seed": seed,
            "prompt": prompt,
        }
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    method, family = args.canonical_method, args.model_family
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    samples = load_manifest(manifest)
    pending, skipped = pending_samples(
        samples,
        output_dir=output_dir,
        config_hash=args.config_hash,
        method=method,
        overwrite=args.overwrite,
    )
    print(
        f"method={method} samples={len(samples)} pending={len(pending)} "
        f"resumed={skipped} output_dir={output_dir}",
        flush=True,
    )
    if not pending:
        return 0

    runner = HFGenerateRunner(
        method=method,
        family=family,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    if args.warmup:
        warmup_sample = pending[0]
        warmup_prompt = runner.prompt_for(warmup_sample)
        warmup_inputs = runner.prepare_inputs(warmup_prompt)
        print(f"running {args.warmup} warmup generation(s)", flush=True)
        for warmup_index in range(args.warmup):
            runner.generate(warmup_inputs, seed=args.seed + warmup_index)
        del warmup_inputs

    failures = 0
    for index, sample in enumerate(pending, start=1):
        path = sample_output_path(output_dir, sample)
        prompt = runner.prompt_for(sample)
        print(
            f"[{index}/{len(pending)}] sample_order={sample.sample_order} "
            f"asset_id={sample.asset_id}",
            flush=True,
        )
        try:
            prepared = runner.prepare_inputs(prompt)
            prompt_token_count = int(prepared["input_ids"].shape[1])
            sample_seed = args.seed + sample.dataset_index
            result = runner.generate(prepared, seed=sample_seed)
            payload = success_payload(
                runner=runner,
                sample=sample,
                config_hash=args.config_hash,
                source_commit=args.source_commit,
                seed=sample_seed,
                prompt=prompt,
                result=result,
            )
            payload["prompt_token_count"] = prompt_token_count
            del prepared
        except Exception as exc:  # Keep later assets runnable after one bad sample.
            failures += 1
            payload = error_payload(
                runner=runner,
                sample=sample,
                config_hash=args.config_hash,
                source_commit=args.source_commit,
                seed=args.seed + sample.dataset_index,
                prompt=prompt,
                exc=exc,
            )
            torch.cuda.empty_cache()
            print(f"sample failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        atomic_write_json(path, payload)

    print(f"completed={len(pending)} failures={failures} resumed={skipped}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
