from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest


EVALUATION_DIR = Path(__file__).resolve().parents[2] / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))
try:
    import compute_pointllm_gpt_metrics as gpt_metrics
finally:
    sys.path.remove(str(EVALUATION_DIR))


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    gt_json = tmp_path / "gt.json"
    gt_json.write_text(
        json.dumps(
            [
                {
                    "object_id": "object-0",
                    "conversations": [
                        {"from": "human", "value": "Describe it."},
                        {"from": "gpt", "value": "A red wooden chair."},
                    ],
                },
                {
                    "object_id": "object-1",
                    "conversations": [
                        {"from": "human", "value": "Describe it."},
                        {"from": "gpt", "value": "A blue toy plane."},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    (pred_dir / "object-0.txt").write_text("A red chair.\n", encoding="utf-8")
    (pred_dir / "object-1.txt").write_text("A blue airplane.\n", encoding="utf-8")
    return gt_json, pred_dir


def _args(gt_json: Path, pred_dir: Path, output_json: Path) -> argparse.Namespace:
    return argparse.Namespace(
        gt_json=gt_json,
        pred_dir=pred_dir,
        render_dir=None,
        model="qwen3.8-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        workers=2,
        retries=0,
        request_timeout=10.0,
        max_output_tokens=512,
        temperature=0.6,
        seed=20260615,
        reasoning_effort="low",
        max_image_side=1024,
        jpeg_quality=92,
        no_resume=False,
        output_json=output_json,
    )


def test_load_samples_maps_existing_txt_predictions(tmp_path: Path):
    gt_json, pred_dir = _write_inputs(tmp_path)

    samples, report = gpt_metrics.load_samples(gt_json, pred_dir)

    assert [sample.object_id for sample in samples] == ["object-0", "object-1"]
    assert samples[0].ground_truth == "A red wooden chair."
    assert samples[0].model_output == "A red chair."
    assert report["matched_prediction_count"] == 2
    assert report["missing_prediction_count"] == 0


def test_load_samples_rejects_missing_predictions_by_default(tmp_path: Path):
    gt_json, pred_dir = _write_inputs(tmp_path)
    (pred_dir / "object-1.txt").unlink()

    with pytest.raises(FileNotFoundError, match="Missing 1 prediction"):
        gpt_metrics.load_samples(gt_json, pred_dir)

    samples, report = gpt_metrics.load_samples(gt_json, pred_dir, allow_missing=True)
    assert [sample.object_id for sample in samples] == ["object-0"]
    assert report["missing_prediction_count"] == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("85#correct identity and most attributes", (85.0, "correct identity and most attributes")),
        ("Score: 101#outside range", (None, "Score: 101#outside range")),
        ("not parseable", (None, "not parseable")),
    ],
)
def test_parse_ref_score(raw: str, expected: tuple[float | None, str]):
    assert gpt_metrics.parse_ref_score(raw) == expected


def test_parse_img_score_accepts_strict_json_and_openeva_scale_conversion():
    raw = json.dumps({"score": 0.75, "reason": "mostly right", "matched": ["chair"], "errors": ["color"]})

    score, reason, matched, errors, converted = gpt_metrics.parse_img_score(raw)

    assert score == 75.0
    assert reason == "mostly right"
    assert matched == ["chair"]
    assert errors == ["color"]
    assert converted is True


def test_parse_chat_completion_collects_text_usage_and_routing():
    payload = {
        "id": "chatcmpl-1",
        "model": "qwen3.8-max",
        "system_fingerprint": "fp-1",
        "choices": [{"message": {"content": "82#mostly correct"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }

    text, usage, routing = gpt_metrics.parse_chat_completion(payload)

    assert text == "82#mostly correct"
    assert usage == {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
    assert routing == {
        "response_id": "chatcmpl-1",
        "resolved_model": "qwen3.8-max",
        "system_fingerprint": "fp-1",
    }


def test_request_dashscope_uses_qwen_reasoning_and_json_parameters(monkeypatch):
    import requests

    captured: dict[str, object] = {}

    class FakeResponse:
        ok = True
        text = ""

        @staticmethod
        def json():
            return {
                "id": "chatcmpl-1",
                "model": "qwen3.8-max",
                "choices": [{"message": {"content": '{"score": 90}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(requests, "post", fake_post)

    text, usage, _ = gpt_metrics.request_dashscope_response(
        content="score this",
        model="qwen3.8-max",
        base_url="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        timeout=10,
        max_output_tokens=512,
        temperature=0.6,
        seed=20260615,
        reasoning_effort="low",
        json_response=True,
    )

    assert text == '{"score": 90}'
    assert usage["total_tokens"] == 8
    assert captured["url"] == ("https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions")
    assert captured["json"] == {
        "model": "qwen3.8-max",
        "messages": [{"role": "user", "content": "score this"}],
        "max_tokens": 512,
        "temperature": 0.6,
        "seed": 20260615,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }


def test_run_judge_checkpoints_and_resumes_successful_rows(tmp_path: Path):
    gt_json, pred_dir = _write_inputs(tmp_path)
    samples, report = gpt_metrics.load_samples(gt_json, pred_dir)
    output_json = tmp_path / "scores.json"
    args = _args(gt_json, pred_dir, output_json)
    payload = gpt_metrics._new_payload(args=args, samples=samples, input_report=report)
    calls: list[str] = []

    def fake_score(sample, **kwargs):
        calls.append(sample.object_id)
        return {
            "object_id": sample.object_id,
            "model_output": sample.model_output,
            "score": 80.0,
            "valid": True,
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }

    gpt_metrics.run_judge(
        samples,
        mode="ref",
        args=args,
        payload=payload,
        output_path=output_json,
        score_fn=fake_score,
    )
    assert sorted(calls) == ["object-0", "object-1"]
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["gpt_ref"]["summary"]["mean"] == 80.0
    assert saved["gpt_ref"]["summary"]["prompt_tokens"] == 20
    assert saved["gpt_ref"]["summary"]["total_tokens"] == 24

    calls.clear()
    resumed = gpt_metrics._load_resume_payload(
        output_json,
        args=args,
        samples=samples,
        input_report=report,
    )
    gpt_metrics.run_judge(
        samples,
        mode="ref",
        args=args,
        payload=resumed,
        output_path=output_json,
        score_fn=fake_score,
    )
    assert calls == []
