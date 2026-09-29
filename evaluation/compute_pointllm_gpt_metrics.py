#!/usr/bin/env python3
"""Evaluate existing PointLLM-200 captions with OpenEVA-style LLM judges.

This is an OctLLM-facing adaptation of the GPT-ref and GPT-img protocols in
SeeleAI/OpenEVA (Apache-2.0), using an Alibaba Cloud Model Studio judge. It reads the
per-object ``.txt`` files produced by ``scripts/batch_understand_pointllm.py``
and never loads OctLLM.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "qwen3.8-max"
DEFAULT_BASE_URL = ""  # Set DASHSCOPE_BASE_URL or --base-url for your workspace.
DEFAULT_SEED = 20260615
PROTOCOL_VERSION = "openeva-eva01-dashscope-2026-08-17"
RGB_VIEWS = ("front", "right", "back", "left")

# Kept verbatim to preserve OpenEVA's published GPT-ref prompt protocol.
GPT_REF_PROMPT = """Evaluate a model-generated caption against a human-generated caption (ground truth) for a 3D model. Identify the aspects mentioned in the human caption and calculate the percentage of these aspects correctly mentioned or partially matched in the model caption. Score from 0 to 100, where each aspect contributes equally to the score. Consider similar concepts for partial score.

Provide your score (0-100) and a short justification (less than 15 words) in the format of 'score#reason'

Example:
Human: A white brown skeleton
Model: This is a 3D model of a small, cartoon-like robot. It has a spherical body and is covered in a layer of white dust.
Output: 50#mention white; skeleton and robot have similar appearence.

Now score the following:
Human: {ground_truth}
Model: {model_output}
Output: """

GPT_IMG_RUBRIC = """Render-based GPT judge for PointLLM-200 mesh captioning.

Inputs:
- Four RGB renders of the same 3D object: front, right, back, and left.
- One model-generated caption.
- The human caption is intentionally not provided.

Goal:
Score how faithfully the caption describes the visible 3D object in the renders.
The score is a scalar integer from 0 to 100, where higher is better.

Rubric:
1. Core object identity and function (0-35): award high credit when the caption names the correct object category or a close synonym.
2. Geometry, structure, and parts (0-25): reward correct major visible components, shape, attachments, symmetry, and distinctive geometry.
3. Color, material, and texture (0-15): reward correct visible colors, material cues, texture patterns, and surface finish.
4. Fine-grained attributes and style (0-15): reward accurate style, decorative motifs, proportions, pose, orientation, and special identifying features.
5. Caption quality and specificity (0-10): reward concise but informative captions and penalize vague or repetitive captions.

Penalties and caps:
- Severe hallucination should reduce the score.
- Empty, non-natural-language, or token-like captions should score 0-10.
- Captions for a completely different object should score 0-20.
- Broadly correct category with many wrong details should usually score 40-70.

Output strict JSON only:
{"score": <integer 0-100>, "reason": "<one short sentence>", "matched": ["..."], "errors": ["..."]}
"""


@dataclass(frozen=True)
class EvalSample:
    object_id: str
    ground_truth: str
    model_output: str
    renders: dict[str, Path] | None = None


def _pointllm_reference(item: Mapping[str, object], index: int) -> str:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        raise TypeError(f"PointLLM item {index} has no conversations list.")
    for turn in conversations:
        if not isinstance(turn, Mapping):
            continue
        role = str(turn.get("from", "")).strip().lower()
        if role in {"gpt", "assistant"}:
            return str(turn.get("value") or "").strip()
    raise ValueError(f"PointLLM item {index} has no GPT/assistant reference turn.")


def _resolve_render_paths(render_dir: Path, object_id: str) -> dict[str, Path]:
    base = render_dir / object_id
    return {view: base / f"{view}.png" for view in RGB_VIEWS}


def load_samples(
    gt_json: Path,
    pred_dir: Path,
    *,
    render_dir: Path | None = None,
    limit: int | None = None,
    allow_missing: bool = False,
) -> tuple[list[EvalSample], dict[str, object]]:
    if not gt_json.is_file():
        raise FileNotFoundError(f"PointLLM annotation JSON not found: {gt_json}")
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if render_dir is not None and not render_dir.is_dir():
        raise FileNotFoundError(f"Render directory not found: {render_dir}")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1.")

    payload = json.loads(gt_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected {gt_json} to contain a JSON list.")

    annotations: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise TypeError(f"Expected item {index} in {gt_json} to be a JSON object.")
        object_id = str(item.get("object_id", "")).strip()
        if not object_id:
            raise ValueError(f"PointLLM item {index} has no object_id.")
        if object_id in seen:
            raise ValueError(f"Duplicate PointLLM object_id: {object_id}")
        seen.add(object_id)
        annotations.append((object_id, _pointllm_reference(item, index)))

    if limit is not None:
        annotations = annotations[:limit]
    selected_ids = {object_id for object_id, _ in annotations}
    pred_files = {path.stem: path for path in pred_dir.iterdir() if path.is_file() and path.suffix == ".txt"}
    missing_predictions = [object_id for object_id, _ in annotations if object_id not in pred_files]
    extra_predictions = sorted(set(pred_files) - seen)

    missing_render_sets: list[str] = []
    samples: list[EvalSample] = []
    for object_id, ground_truth in annotations:
        pred_path = pred_files.get(object_id)
        if pred_path is None:
            continue
        renders = None
        if render_dir is not None:
            renders = _resolve_render_paths(render_dir, object_id)
            if not all(path.is_file() for path in renders.values()):
                missing_render_sets.append(object_id)
                renders = None
        samples.append(
            EvalSample(
                object_id=object_id,
                ground_truth=ground_truth,
                model_output=pred_path.read_text(encoding="utf-8").strip(),
                renders=renders,
            )
        )

    if missing_predictions and not allow_missing:
        preview = ", ".join(missing_predictions[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_predictions)} prediction file(s), including: {preview}. "
            "Pass --allow-missing to score only matched samples."
        )

    report: dict[str, object] = {
        "annotation_count": len(payload),
        "selected_count": len(annotations),
        "matched_prediction_count": len(samples),
        "missing_prediction_count": len(missing_predictions),
        "missing_prediction_examples": missing_predictions[:10],
        "extra_prediction_count": len(extra_predictions),
        "extra_prediction_examples": extra_predictions[:10],
        "empty_prediction_count": sum(not sample.model_output for sample in samples),
        "empty_reference_count": sum(not sample.ground_truth for sample in samples),
        "complete_render_set_count": sum(sample.renders is not None for sample in samples),
        "missing_render_set_count": len(missing_render_sets),
        "missing_render_set_examples": missing_render_sets[:10],
        "selected_object_id_count": len(selected_ids),
    }
    return samples, report


def build_ref_prompt(sample: EvalSample) -> str:
    return GPT_REF_PROMPT.format(
        ground_truth=sample.ground_truth,
        model_output=sample.model_output,
    )


def image_to_data_url(path: Path, *, max_side: int, quality: int) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("GPT-img requires Pillow: pip install Pillow") from exc

    with Image.open(path) as source:
        image = source.convert("RGB")
    if max_side > 0:
        width, height = image.size
        scale = min(1.0, float(max_side) / float(max(width, height)))
        if scale < 1.0:
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_img_content(sample: EvalSample, *, max_image_side: int, jpeg_quality: int) -> list[dict[str, object]]:
    if sample.renders is None:
        raise ValueError(f"No complete render set for object_id={sample.object_id}")
    prompt = (
        f"{GPT_IMG_RUBRIC}\n\n"
        "Now score the following generated caption against the provided RGB renders only.\n"
        "Do not use any external ground-truth caption or dataset knowledge.\n\n"
        f"Generated caption:\n{sample.model_output or '[EMPTY]'}"
    )
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for view in RGB_VIEWS:
        content.append({"type": "text", "text": f"RGB render view: {view}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        sample.renders[view],
                        max_side=max_image_side,
                        quality=jpeg_quality,
                    )
                },
            }
        )
    return content


def parse_chat_completion(
    payload: Mapping[str, object],
) -> tuple[str, dict[str, object], dict[str, object]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("DashScope response has no choices[0].")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("DashScope response has no choices[0].message.")
    raw_content = message.get("content")
    if isinstance(raw_content, str):
        text = raw_content.strip()
    elif isinstance(raw_content, list):
        text = "".join(
            str(part.get("text") or "")
            for part in raw_content
            if isinstance(part, Mapping) and part.get("type") == "text"
        ).strip()
    else:
        text = ""
    if not text:
        raise ValueError("Qwen judge returned an empty message.")

    raw_usage = payload.get("usage")
    usage: dict[str, object] = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
    usage.setdefault("prompt_tokens", 0)
    usage.setdefault("completion_tokens", 0)
    routing = {
        "response_id": payload.get("id"),
        "resolved_model": payload.get("model"),
        "system_fingerprint": payload.get("system_fingerprint"),
    }
    return text, usage, routing


def _request_proxies() -> dict[str, str | None]:
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or https_proxy
    return {"http": http_proxy, "https": https_proxy}


def request_dashscope_response(
    *,
    content: str | list[dict[str, object]],
    model: str,
    base_url: str,
    timeout: float,
    max_output_tokens: int,
    temperature: float,
    seed: int,
    reasoning_effort: str,
    json_response: bool,
) -> tuple[str, dict[str, object], dict[str, object]]:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing.")
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("DashScope evaluation requires requests: pip install requests") from exc

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    request_payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    request_payload["reasoning_effort"] = reasoning_effort
    if json_response:
        request_payload["response_format"] = {"type": "json_object"}

    proxies = _request_proxies()
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers=headers,
        json=request_payload,
        proxies=proxies,
        timeout=timeout,
    )
    if not response.ok:
        detail = (response.text or "").strip()[:2000]
        raise RuntimeError(
            f"DashScope API {response.status_code} {response.reason} for {response.url} "
            f"via {proxies['https']}: {detail}"
        )
    try:
        response_payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"DashScope returned non-JSON data: {(response.text or '')[:500]}") from exc
    if not isinstance(response_payload, Mapping):
        raise RuntimeError("DashScope returned a non-object JSON response.")
    return parse_chat_completion(response_payload)


def parse_ref_score(raw_text: str) -> tuple[float | None, str]:
    match = re.search(r"(?<!\d)(\d{1,3})\s*#\s*(.*)", raw_text.strip(), flags=re.DOTALL)
    if not match:
        return None, raw_text.strip()[:300]
    score = int(match.group(1))
    if score < 0 or score > 100:
        return None, raw_text.strip()[:300]
    return float(score), match.group(2).strip()


def parse_img_score(raw_text: str) -> tuple[float | None, str, list[str], list[str], bool]:
    text = raw_text.strip()
    parsed: Mapping[str, object] | None = None
    try:
        candidate = json.loads(text)
        parsed = candidate if isinstance(candidate, Mapping) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                candidate = json.loads(match.group(0))
                parsed = candidate if isinstance(candidate, Mapping) else None
            except json.JSONDecodeError:
                pass

    if parsed is None:
        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if not match:
            return None, "", [], [], False
        score = float(match.group(1))
        converted = 0.0 <= score <= 1.0
        if converted:
            score *= 100.0
        return max(0.0, min(100.0, score)), text[:300], [], [], converted

    try:
        score = float(parsed.get("score"))
    except (TypeError, ValueError):
        return None, str(parsed.get("reason") or ""), [], [], False
    converted = 0.0 <= score <= 1.0
    if converted:
        score *= 100.0
    matched = parsed.get("matched") or []
    errors = parsed.get("errors") or []
    if not isinstance(matched, list):
        matched = [str(matched)]
    if not isinstance(errors, list):
        errors = [str(errors)]
    return (
        max(0.0, min(100.0, score)),
        str(parsed.get("reason") or ""),
        [str(value) for value in matched],
        [str(value) for value in errors],
        converted,
    )


def _score_with_retries(
    sample: EvalSample,
    *,
    mode: str,
    model: str,
    base_url: str,
    timeout: float,
    max_output_tokens: int,
    temperature: float,
    seed: int,
    reasoning_effort: str,
    max_image_side: int,
    jpeg_quality: int,
    retries: int,
) -> dict[str, object]:
    last_error = ""
    for attempt in range(retries + 1):
        try:
            if mode == "ref":
                content: str | list[dict[str, object]] = build_ref_prompt(sample)
            else:
                content = build_img_content(
                    sample,
                    max_image_side=max_image_side,
                    jpeg_quality=jpeg_quality,
                )
            raw_text, usage, routing = request_dashscope_response(
                content=content,
                model=model,
                base_url=base_url,
                timeout=timeout,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                seed=seed,
                reasoning_effort=reasoning_effort,
                json_response=mode == "img",
            )
            if mode == "ref":
                score, reason = parse_ref_score(raw_text)
                matched: list[str] = []
                errors: list[str] = []
                converted = False
            else:
                score, reason, matched, errors, converted = parse_img_score(raw_text)
            return {
                "object_id": sample.object_id,
                "ground_truth": sample.ground_truth if mode == "ref" else None,
                "model_output": sample.model_output,
                "score": score,
                "valid": score is not None,
                "reason": reason,
                "matched": matched,
                "errors": errors,
                "score_scale_converted_from_0_1": converted,
                "raw_response": raw_text,
                "usage": usage,
                "dashscope": routing,
                "attempts": attempt + 1,
            }
        except Exception as exc:
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    return {
        "object_id": sample.object_id,
        "ground_truth": sample.ground_truth if mode == "ref" else None,
        "model_output": sample.model_output,
        "score": None,
        "valid": False,
        "reason": "",
        "matched": [],
        "errors": [],
        "score_scale_converted_from_0_1": False,
        "raw_response": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "dashscope": {},
        "attempts": retries + 1,
        "request_error": last_error,
    }


def summarize_results(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    scores = [float(row["score"]) for row in results if row.get("score") is not None]

    def usage_value(row: Mapping[str, object], key: str) -> float:
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            return 0.0
        try:
            return float(usage.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "num_samples": len(results),
        "valid_scores": len(scores),
        "invalid_scores": len(results) - len(scores),
        "mean": sum(scores) / len(scores) if scores else None,
        "median": statistics.median(scores) if scores else None,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "prompt_tokens": int(sum(usage_value(row, "prompt_tokens") for row in results)),
        "completion_tokens": int(sum(usage_value(row, "completion_tokens") for row in results)),
        "total_tokens": int(sum(usage_value(row, "total_tokens") for row in results)),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sample_fingerprint(samples: Sequence[EvalSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.object_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.ground_truth.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.model_output.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _new_payload(
    *,
    args: argparse.Namespace,
    samples: Sequence[EvalSample],
    input_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        "metadata": {
            "protocol_version": PROTOCOL_VERSION,
            "backend": "Alibaba Cloud Model Studio OpenAI-compatible Chat Completions",
            "judge_model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "seed": args.seed,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "max_image_side": args.max_image_side,
            "jpeg_quality": args.jpeg_quality,
            "gt_json": str(args.gt_json.resolve()),
            "pred_dir": str(args.pred_dir.resolve()),
            "render_dir": str(args.render_dir.resolve()) if args.render_dir is not None else None,
            "sample_fingerprint": _sample_fingerprint(samples),
            "input_report": dict(input_report),
        }
    }


def _load_resume_payload(
    output_path: Path,
    *,
    args: argparse.Namespace,
    samples: Sequence[EvalSample],
    input_report: Mapping[str, object],
) -> dict[str, object]:
    expected = _new_payload(args=args, samples=samples, input_report=input_report)
    if args.no_resume or not output_path.is_file():
        return expected
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"Cannot resume malformed output JSON: {output_path}")
    old_metadata = payload["metadata"]
    new_metadata = expected["metadata"]
    for key in (
        "protocol_version",
        "judge_model",
        "base_url",
        "temperature",
        "seed",
        "reasoning_effort",
        "max_output_tokens",
        "max_image_side",
        "jpeg_quality",
        "sample_fingerprint",
    ):
        if old_metadata.get(key) != new_metadata.get(key):
            raise ValueError(
                f"Cannot resume {output_path}: metadata field {key!r} changed. "
                "Choose another --output-json or pass --no-resume."
            )
    payload["metadata"] = new_metadata
    return payload


def _section_payload(mode: str, results: Sequence[Mapping[str, object]], model: str) -> dict[str, object]:
    return {
        "judge": "OpenEVA-ref" if mode == "ref" else "OpenEVA-img",
        "judge_model": model,
        "prompt": GPT_REF_PROMPT if mode == "ref" else GPT_IMG_RUBRIC,
        "summary": summarize_results(results),
        "results": list(results),
    }


def run_judge(
    samples: Sequence[EvalSample],
    *,
    mode: str,
    args: argparse.Namespace,
    payload: dict[str, object],
    output_path: Path,
    score_fn: Callable[..., dict[str, object]] = _score_with_retries,
) -> None:
    section_name = f"gpt_{mode}"
    existing_section = payload.get(section_name)
    existing_results = existing_section.get("results", []) if isinstance(existing_section, Mapping) else []
    result_by_id = {
        str(row.get("object_id")): dict(row)
        for row in existing_results
        if isinstance(row, Mapping) and row.get("valid") is True
    }
    pending = [sample for sample in samples if sample.object_id not in result_by_id]
    print(f"{section_name}: {len(result_by_id)} resumed, {len(pending)} pending", flush=True)

    def persist() -> None:
        ordered = [result_by_id[sample.object_id] for sample in samples if sample.object_id in result_by_id]
        payload[section_name] = _section_payload(mode, ordered, args.model)
        _atomic_write_json(output_path, payload)

    if not pending:
        persist()
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                score_fn,
                sample,
                mode=mode,
                model=args.model,
                base_url=args.base_url,
                timeout=args.request_timeout,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
                seed=args.seed,
                reasoning_effort=args.reasoning_effort,
                max_image_side=args.max_image_side,
                jpeg_quality=args.jpeg_quality,
                retries=args.retries,
            ): sample
            for sample in pending
        }
        completed = 0
        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "object_id": sample.object_id,
                    "model_output": sample.model_output,
                    "score": None,
                    "valid": False,
                    "request_error": repr(exc),
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }
            result_by_id[sample.object_id] = result
            completed += 1
            persist()
            status = result.get("score") if result.get("valid") else "FAILED"
            print(f"{section_name}: {completed}/{len(pending)} {sample.object_id} score={status}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute OpenEVA-style reference/image judge metrics through Alibaba Cloud Model Studio."
    )
    parser.add_argument("--gt-json", type=Path, required=True, help="PointLLM-200 ground-truth JSON.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory containing <object_id>.txt files.")
    parser.add_argument(
        "--judge",
        choices=("ref", "img", "both"),
        default="ref",
        help="Run the reference judge, image judge, or both. Image judging also requires --render-dir.",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=Path("datasets/pointllm/renders_blender_pbr"),
        help="Directory containing <object_id>/{front,right,back,left}.png OpenEVA PBR renders.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DASHSCOPE_JUDGE_MODEL") or DEFAULT_MODEL,
        help=f"Alibaba Cloud Model Studio model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL,
        help="Model Studio OpenAI-compatible API base URL.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API calls.")
    parser.add_argument("--retries", type=int, default=2, help="Retries after a failed API call.")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature. qwen3.8-max coerces values below 0.6 to 0.6.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="low",
        help="Qwen reasoning effort. For qwen3.8-max, high maps to xhigh; low limits evaluation cost.",
    )
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N annotation rows.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip GT objects without prediction files.")
    parser.add_argument(
        "--no-resume", action="store_true", help="Ignore any prior output and score every sample again."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Result/checkpoint path. Defaults to <pred-dir>/pointllm_dashscope_metrics_<model>.json.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative.")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive.")
    if args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be at least 1.")
    if args.temperature < 0 or args.temperature >= 2:
        raise ValueError("--temperature must be in [0, 2).")
    if not args.base_url:
        raise ValueError("Set DASHSCOPE_BASE_URL or --base-url to your judge endpoint.")
    placeholder_markers = ("{WorkspaceId}", "{workspace_id}", "YOUR_WORKSPACE_ID", "ws-xxxxxxxx")
    if any(marker in args.base_url for marker in placeholder_markers):
        raise ValueError("Replace {WorkspaceId} in --base-url with the real Alibaba Cloud Model Studio workspace ID.")
    if not args.base_url.startswith("https://"):
        raise ValueError("--base-url must start with https://.")
    if args.max_image_side < 1:
        raise ValueError("--max-image-side must be at least 1.")
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise ValueError("--jpeg-quality must be in [1, 100].")
    if args.judge in {"img", "both"} and args.render_dir is None:
        raise ValueError("--render-dir is required for image judging.")


def main() -> None:
    args = build_parser().parse_args()
    args.gt_json = args.gt_json.expanduser()
    args.pred_dir = args.pred_dir.expanduser()
    args.render_dir = args.render_dir.expanduser() if args.render_dir is not None else None
    _validate_args(args)

    samples, input_report = load_samples(
        args.gt_json,
        args.pred_dir,
        render_dir=args.render_dir,
        limit=args.limit,
        allow_missing=args.allow_missing,
    )
    if args.judge in {"img", "both"}:
        complete_samples = [sample for sample in samples if sample.renders is not None]
        if len(complete_samples) != len(samples):
            raise FileNotFoundError(
                f"Only {len(complete_samples)}/{len(samples)} samples have complete four-view render sets under "
                f"{args.render_dir}."
            )

    print(json.dumps(input_report, ensure_ascii=False, indent=2), flush=True)
    print(
        f"Judge backend: Alibaba Cloud Model Studio | model={args.model} | base_url={args.base_url}",
        flush=True,
    )
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("Set DASHSCOPE_API_KEY before running Alibaba Cloud Model Studio evaluation.")
    print(f"HTTPS proxy configured: {bool(_request_proxies()['https'])}", flush=True)

    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model).strip("-") or "model"
    output_path = (
        args.output_json.expanduser()
        if args.output_json is not None
        else args.pred_dir / f"pointllm_dashscope_metrics_{safe_model}.json"
    )
    payload = _load_resume_payload(
        output_path,
        args=args,
        samples=samples,
        input_report=input_report,
    )
    modes = ("ref", "img") if args.judge == "both" else (args.judge,)
    for mode in modes:
        run_judge(samples, mode=mode, args=args, payload=payload, output_path=output_path)

    invalid = 0
    for mode in modes:
        section = payload[f"gpt_{mode}"]
        summary = section["summary"]
        invalid += int(summary["invalid_scores"])
        print(f"OpenEVA-{mode}: mean={summary['mean']} valid={summary['valid_scores']}/{summary['num_samples']}")
    print(f"Saved results to {output_path}")
    if invalid:
        print(
            f"Warning: {invalid} score(s) are invalid. Re-run the same command to retry only failed samples.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
