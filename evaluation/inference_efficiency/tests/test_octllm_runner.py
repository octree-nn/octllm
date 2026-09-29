from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

RUNNER_PATH = Path(__file__).resolve().parents[1] / "runners" / "octllm.py"
SPEC = importlib.util.spec_from_file_location("inference_efficiency_octllm_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
octllm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = octllm
SPEC.loader.exec_module(octllm)


class FakeTokenizer:
    eos_token_id = 2

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<mesh_bos>": 1000, "<mesh_eos>": 1001}[token]


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(tokenizer=FakeTokenizer())


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        prompt_prefix="prefix: ",
        max_new_tokens=10000,
        max_layer=6,
        full_depth=3,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        bos_top_k=2,
        system_prompt="system",
        keep_mask_in_cache=False,
    )


class OctLLMRunnerOutputTest(unittest.TestCase):
    def test_generate_records_complete_text_ids_and_octree_trace(self) -> None:
        generated_ids = [11, 12, 1000, 200, 201, 1001]
        generated_text = "complete output: " + "x" * 300 + "<mesh_bos><mesh0><mesh1><mesh_eos>"
        inference_result = {
            "generated_ids": generated_ids,
            "generated_text": generated_text,
            "autoregressive_output_token_count": 5,
            "structure_token_count": 2,
            "autoregressive_forward_steps": 5,
            "autoregressive_latency_ms": 10.0,
            "autoregressive_cuda_latency_ms": 10.0,
            "autoregressive_wall_latency_ms": 12.0,
            "transient_mask_count": 3,
            "forced_mesh_bos_count": 1,
            "final_kv_cache_length": 99,
            "output_token_count": 6,
        }

        with mock.patch.object(octllm, "_predict_next_token", return_value=inference_result):
            record = octllm._generate(
                {"text_description": "a chair"},
                runtime=_runtime(),
                infer_cfg={},
                args=_args(),
            )

        self.assertEqual(record["generated_text"], generated_text)
        self.assertEqual(record["generated_preview"], generated_text[:240])
        self.assertEqual(record["generated_token_ids"], generated_ids)
        self.assertEqual(record["output_token_count"], 5)
        self.assertEqual(record["diagnostics"]["serialized_output_token_count"], 6)
        self.assertEqual(
            record["diagnostics"]["octree_generation"],
            {
                "phase": "octree_complete",
                "entered_octree_generation": True,
                "completed_octree_generation": True,
                "mesh_bos_token_id": 1000,
                "mesh_eos_token_id": 1001,
                "mesh_bos_token_indices": [2],
                "mesh_eos_token_indices": [5],
                "first_mesh_bos_token_index": 2,
                "first_mesh_eos_after_bos_token_index": 5,
                "non_octree_prefix_token_count": 2,
                "mesh_byte_token_count": 2,
                "serialized_output_token_count": 6,
                "token_index_definition": "zero-based index within generated_token_ids",
            },
        )

    def test_octree_trace_identifies_text_only_output(self) -> None:
        trace = octllm._octree_generation_trace(
            {"generated_ids": [31, 32, 2], "structure_token_count": 0},
            _runtime(),
        )

        self.assertEqual(trace["phase"], "text_only")
        self.assertIs(trace["entered_octree_generation"], False)
        self.assertIs(trace["completed_octree_generation"], False)
        self.assertIsNone(trace["first_mesh_bos_token_index"])
        self.assertEqual(trace["non_octree_prefix_token_count"], 3)


if __name__ == "__main__":
    unittest.main()
