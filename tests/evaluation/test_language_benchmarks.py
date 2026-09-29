# ruff: noqa: PT009

import argparse
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from deepeval.benchmarks import GSM8K, MMLU, HellaSwag, IFEval
from deepeval.benchmarks.hellaswag.task import HellaSwagTask
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.dataset import Golden

from evaluation.models.language_benchmarks import (
    ANSWER_MODES,
    DEFAULT_EVAL_CONFIG,
    DEFAULT_EVAL_TEMPERATURE,
    DEFAULT_EVAL_TOP_K,
    DEFAULT_EVAL_TOP_P,
    GSM8K_CONFINEMENT_INSTRUCTIONS,
    OctLLM,
    Qwen25VL,
    _build_benchmark,
    _constrain_mesh_token_logits,
    _decode_generated_text,
    _extract_last_number,
    _save_benchmark_artifacts,
    _set_answer_mode,
)
from llamafactory.model.qwen25_vl_3d_replace import get_rope_index_with_mesh


class CustomModelsTest(unittest.TestCase):
    def test_main_inference_config_enables_3d_router(self):
        from omegaconf import OmegaConf

        config = OmegaConf.load(DEFAULT_EVAL_CONFIG)
        self.assertTrue(config["use_3d_token_router"])

    def test_extract_last_number_supports_gsm8k_formats(self):
        self.assertEqual(_extract_last_number("work ... answer: 1,234"), "1,234")
        self.assertEqual(_extract_last_number("the result is -42."), "-42")
        self.assertEqual(_extract_last_number("first 2, final 3.5"), "3.5")
        self.assertEqual(_extract_last_number("no number"), "no number")

    def test_decode_only_removes_an_actual_eos(self):
        tokenizer = Mock(eos_token_id=99)
        tokenizer.decode.side_effect = lambda ids, **_: ",".join(map(str, ids))
        self.assertEqual(_decode_generated_text(tokenizer, [1, 2, 99]), "1,2")
        self.assertEqual(_decode_generated_text(tokenizer, [1, 2]), "1,2")

    def test_mesh_bytes_require_mesh_bos_for_sampling_and_greedy(self):
        logits = __import__("torch").arange(8, dtype=__import__("torch").float32).unsqueeze(0)
        before_bos = _constrain_mesh_token_logits(
            logits,
            mesh_byte_token_ids={4, 5},
            mesh_bos_id=3,
            mesh_eos_id=6,
            mask_token_id=7,
            start_mesh_generation=False,
        )
        self.assertTrue(before_bos[0, 3].isfinite())
        self.assertTrue(before_bos[0, 4:8].isneginf().all())

        after_bos = _constrain_mesh_token_logits(
            logits,
            mesh_byte_token_ids={4, 5},
            mesh_bos_id=3,
            mesh_eos_id=6,
            mask_token_id=7,
            start_mesh_generation=True,
        )
        self.assertTrue(after_bos[0, 4:6].isfinite().all())
        self.assertTrue(after_bos[0, :4].isneginf().all())
        self.assertTrue(after_bos[0, 6:].isneginf().all())

    def test_deepeval_fallback_uses_each_answer_mode(self):
        class FakeModel:
            def set_answer_mode(self, multiple_choice=False, numeric_answer=False):
                self.multiple_choice = multiple_choice
                self.numeric_answer = numeric_answer

            def generate(self, prompt):
                if self.multiple_choice:
                    return "A"
                if self.numeric_answer:
                    return "42"
                return "free form"

        model = FakeModel()
        mmlu = MMLU(tasks=[], n_shots=0)
        mmlu.shots_dataset = [{}]
        _set_answer_mode(model, "mmlu")
        self.assertEqual(
            mmlu.predict(model, next(iter(MMLUTask)), Golden(input="question", expected_output="A"))["prediction"],
            "A",
        )

        gsm8k = GSM8K(n_shots=0, n_problems=1)
        gsm8k.shots_dataset = []
        _set_answer_mode(model, "gsm8k")
        self.assertEqual(gsm8k.predict(model, Golden(input="question", expected_output="42"))["prediction"], "42")

        hellaswag = HellaSwag(tasks=[], n_shots=0)
        hellaswag.shots_dataset = [{}]
        _set_answer_mode(model, "hellaswag")
        self.assertEqual(
            hellaswag.predict(
                model,
                next(iter(HellaSwagTask)),
                Golden(input="question", expected_output="A"),
            )["prediction"],
            "A",
        )

        ifeval = IFEval(n_problems=1)
        _set_answer_mode(model, "ifeval")
        prediction, _, _ = ifeval.predict(
            model,
            Golden(input="question", expected_output="", additional_metadata={}),
        )
        self.assertEqual(prediction, "free form")

    def test_base_and_octllm_share_deterministic_generation_defaults(self):
        for model_class in (OctLLM, Qwen25VL):
            parameters = inspect.signature(model_class.__init__).parameters
            self.assertEqual(parameters["temperature"].default, DEFAULT_EVAL_TEMPERATURE)
            self.assertEqual(parameters["top_p"].default, DEFAULT_EVAL_TOP_P)
            self.assertEqual(parameters["top_k"].default, DEFAULT_EVAL_TOP_K)

        baseline = Qwen25VL.__new__(Qwen25VL)
        baseline.default_multiple_choice = False
        baseline.default_numeric_answer = False
        baseline.set_answer_mode(multiple_choice=True)
        self.assertTrue(baseline.default_multiple_choice)
        self.assertFalse(baseline.default_numeric_answer)

    def test_mesh_rope_state_is_absolute_next_position_for_plain_text(self):
        torch = __import__("torch")

        class FakeTokenizer:
            @staticmethod
            def convert_tokens_to_ids(token):
                return {"<mesh_bos>": 900, "<mesh_eos>": 901}[token]

        fake_model = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=2),
                image_token_id=800,
                video_token_id=801,
                vision_start_token_id=802,
            )
        )
        input_ids = torch.tensor([[11, 12, 13, 14]])
        position_ids, next_position = get_rope_index_with_mesh(
            fake_model,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            tokenizer=FakeTokenizer(),
            mesh_xyz_list=[[]],
        )

        self.assertTrue(torch.equal(position_ids[:, 0], torch.arange(4).expand(3, -1)))
        self.assertEqual(next_position.tolist(), [[4]])

    def test_builds_all_four_benchmarks_with_a_limit(self):
        args = argparse.Namespace(
            limit=2,
            mmlu_shots=5,
            gsm8k_shots=3,
            gsm8k_cot=True,
            hellaswag_shots=10,
            verbose=False,
        )
        for benchmark_name in ANSWER_MODES:
            self.assertIsNotNone(_build_benchmark(benchmark_name, args))
        gsm8k = _build_benchmark("gsm8k", args)
        self.assertEqual(gsm8k.confinement_instructions, GSM8K_CONFINEMENT_INSTRUCTIONS)

    def test_saves_result_artifacts(self):
        pandas = __import__("pandas")
        benchmark = SimpleNamespace(
            predictions=pandas.DataFrame([{"prediction": "A"}]),
            task_scores=pandas.DataFrame([{"task": "demo", "score": 1.0}]),
            instruction_breakdown={"demo": 1.0},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = _save_benchmark_artifacts("demo", benchmark, Path(temp_dir))
            self.assertEqual(set(artifacts), {"predictions", "task_scores", "instruction_breakdown"})
            for filename in artifacts.values():
                self.assertTrue((Path(temp_dir) / filename).is_file())

    def test_saves_full_gsm8k_generation_for_incorrect_answers(self):
        pandas = __import__("pandas")
        benchmark = SimpleNamespace(
            predictions=pandas.DataFrame(
                [
                    {
                        "Input": "What is 20 + 22?",
                        "Prediction": "41",
                        "Expected Output": "42",
                        "Correct": False,
                    },
                    {
                        "Input": "What is 1 + 1?",
                        "Prediction": "2",
                        "Expected Output": "2",
                        "Correct": True,
                    },
                ]
            ),
            task_scores=None,
            instruction_breakdown=None,
        )
        generation_records = [
            {"raw_output": "I added incorrectly. ### Answer: 41", "returned_output": "41"},
            {"raw_output": "1 + 1 = 2. ### Answer: 2", "returned_output": "2"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = _save_benchmark_artifacts(
                "gsm8k",
                benchmark,
                Path(temp_dir),
                generation_records=generation_records,
            )
            self.assertEqual(artifacts["errors"], "gsm8k_errors.json")
            payload = json.loads((Path(temp_dir) / artifacts["errors"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["error_count"], 1)
        self.assertEqual(payload["samples"][0]["input"], "What is 20 + 22?")
        self.assertEqual(payload["samples"][0]["generated_output"], generation_records[0]["raw_output"])
        self.assertEqual(payload["samples"][0]["evaluated_prediction"], "41")
        self.assertEqual(payload["samples"][0]["expected_output"], "42")

    def test_saves_only_incorrect_ifeval_outputs(self):
        pandas = __import__("pandas")
        benchmark = SimpleNamespace(
            predictions=pandas.DataFrame(
                [
                    {
                        "Input": "Reply without commas.",
                        "Prediction": "Hello, world",
                        "All_Instructions_Correct": False,
                    },
                    {
                        "Input": "Reply in uppercase.",
                        "Prediction": "HELLO",
                        "All_Instructions_Correct": True,
                    },
                ]
            ),
            task_scores=None,
            instruction_breakdown=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = _save_benchmark_artifacts(
                "ifeval",
                benchmark,
                Path(temp_dir),
                generation_records=[
                    {"raw_output": "Hello, world", "returned_output": "Hello, world"},
                    {"raw_output": "HELLO", "returned_output": "HELLO"},
                ],
            )
            payload = json.loads((Path(temp_dir) / artifacts["errors"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["error_count"], 1)
        self.assertEqual(payload["samples"][0]["input"], "Reply without commas.")
        self.assertEqual(payload["samples"][0]["generated_output"], "Hello, world")


if __name__ == "__main__":
    unittest.main()
