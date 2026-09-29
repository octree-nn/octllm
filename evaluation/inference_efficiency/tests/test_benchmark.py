from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("inference_efficiency_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)

COMMON_PATH = BENCHMARK_PATH.parent / "runners" / "common.py"
COMMON_SPEC = importlib.util.spec_from_file_location("inference_efficiency_common", COMMON_PATH)
assert COMMON_SPEC is not None and COMMON_SPEC.loader is not None
common = importlib.util.module_from_spec(COMMON_SPEC)
sys.modules[COMMON_SPEC.name] = common
COMMON_SPEC.loader.exec_module(common)


class SamplingTest(unittest.TestCase):
    def test_selection_is_deterministic_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset.json"
            dataset = [
                {
                    "asset_id": f"asset-{index}",
                    "text_description": f"description {index}",
                    "mesh_path": f"mesh-{index}",
                    "render_image_path": f"render-{index}",
                }
                for index in range(30)
            ]
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

            manifest, selection = benchmark.sample_dataset(
                dataset_path, root / "results", num_assets=5, seed=42, overwrite_selection=False
            )
            rows = benchmark.read_manifest(manifest)
            self.assertEqual(selection["sampled_indices"], [20, 3, 0, 23, 8])
            self.assertEqual([row["asset_id"] for row in rows], ["asset-20", "asset-3", "asset-0", "asset-23", "asset-8"])

            second_manifest, second_selection = benchmark.sample_dataset(
                dataset_path, root / "results", num_assets=5, seed=42, overwrite_selection=False
            )
            self.assertEqual(second_manifest, manifest)
            self.assertEqual(second_selection, selection)

    def test_existing_selection_rejects_changed_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset.json"
            dataset_path.write_text(
                json.dumps(
                    [{"asset_id": str(index), "text_description": str(index)} for index in range(10)]
                ),
                encoding="utf-8",
            )
            benchmark.sample_dataset(dataset_path, root / "out", num_assets=3, seed=1, overwrite_selection=False)
            with self.assertRaises(RuntimeError):
                benchmark.sample_dataset(dataset_path, root / "out", num_assets=3, seed=2, overwrite_selection=False)


class StatisticsTest(unittest.TestCase):
    def test_metric_summary_and_percentile(self) -> None:
        records = [{"x": 1}, {"x": 2}, {"x": 3}, {"x": None}]
        summary = benchmark.metric_summary(records, "x")
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["median"], 2.0)
        self.assertAlmostEqual(summary["p95"], 2.9)


class CommandTest(unittest.TestCase):
    def test_runner_specific_arguments(self) -> None:
        config = json.loads((BENCHMARK_PATH.parent / "benchmark_config.json").read_text(encoding="utf-8"))
        source_paths = {
            name: Path("/tmp") / repo["directory"] for name, repo in config["repositories"].items()
        }
        shared = {
            "manifest": Path("/tmp/assets.jsonl"),
            "output_dir": Path("/tmp/results"),
            "config_hash": "abc",
            "seed": 42,
            "warmup": 1,
            "overwrite": False,
        }

        hf = benchmark.build_method_command(
            "shapellm_omni", config["methods"]["shapellm_omni"], config, source_paths, **shared
        )
        self.assertIn("--method", hf)
        self.assertNotIn("--family", hf)

        octgpt = benchmark.build_method_command(
            "octgpt", config["methods"]["octgpt"], config, source_paths, **shared
        )
        self.assertIn("--source-root", octgpt)

        sar3d = benchmark.build_method_command(
            "sar3d", config["methods"]["sar3d"], config, source_paths, **shared
        )
        self.assertIn("--source-dir", sar3d)
        self.assertIn("--vae-checkpoint", sar3d)


class ResumeTest(unittest.TestCase):
    def test_matching_error_is_retried_but_success_is_skipped(self) -> None:
        sample = common.ManifestSample(
            sample_order=0,
            dataset_index=7,
            asset_id="asset",
            text_description="description",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            path = common.sample_output_path(output_dir, sample)
            payload = common.base_result(method="ShapeLLM-Omni", config_hash="hash", sample=sample)
            payload["status"] = "error"
            common.atomic_write_json(path, payload)
            pending, skipped = common.pending_samples(
                [sample],
                output_dir=output_dir,
                config_hash="hash",
                method="ShapeLLM-Omni",
                overwrite=False,
            )
            self.assertEqual(pending, [sample])
            self.assertEqual(skipped, 0)

            payload["status"] = "ok"
            common.atomic_write_json(path, payload)
            pending, skipped = common.pending_samples(
                [sample],
                output_dir=output_dir,
                config_hash="hash",
                method="ShapeLLM-Omni",
                overwrite=False,
            )
            self.assertEqual(pending, [])
            self.assertEqual(skipped, 1)


class AggregationTest(unittest.TestCase):
    def test_aggregate_writes_paper_outputs_and_keeps_missing_visible(self) -> None:
        config = json.loads((BENCHMARK_PATH.parent / "benchmark_config.json").read_text(encoding="utf-8"))
        methods = ["shapellm_omni", "octllm"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {"asset_id": f"asset-{index}", "text_description": f"description {index}"}
                        for index in range(5)
                    ]
                ),
                encoding="utf-8",
            )
            manifest, selection = benchmark.sample_dataset(
                dataset_path, root / "results", num_assets=2, seed=3, overwrite_selection=False
            )
            rows = benchmark.read_manifest(manifest)
            config_hash = benchmark.method_config_hash(
                "shapellm_omni",
                config["methods"]["shapellm_omni"],
                config,
                selection,
                gpu="0",
                seed=3,
                warmup=1,
            )
            for index, row in enumerate(rows):
                result = {
                    "schema_version": "inference-efficiency.v1",
                    "method": "ShapeLLM-Omni",
                    "status": "ok",
                    "config_hash": config_hash,
                    "sample_order": row["sample_order"],
                    "dataset_index": row["dataset_index"],
                    "asset_id": row["asset_id"],
                    "text_description_sha256": hashlib.sha256(
                        row["text_description"].encode("utf-8")
                    ).hexdigest(),
                    "output_token_count": 10 + index * 10,
                    "structure_token_count": 8 + index * 10,
                    "autoregressive_steps": 10 + index * 10,
                    "latency_ms": 100 + index * 100,
                    "cuda_latency_ms": 100 + index * 100,
                    "wall_latency_ms": 110 + index * 100,
                    "termination": "test_complete",
                    "truncated": False,
                    "valid_structure": True,
                    "token_definition": {"output_token_count": "test"},
                    "runtime": {"gpu": "NVIDIA GeForce RTX 5090", "torch": "test"},
                    "model": {"path": "test-model"},
                    "source": {"commit": "test-commit"},
                }
                path = root / "results" / "raw" / "shapellm_omni" / benchmark.safe_result_name(row)
                benchmark.atomic_json(path, result)

            summary = benchmark.aggregate(
                methods,
                config,
                selection,
                manifest,
                root / "results",
                gpu="0",
                seed=3,
                warmup=1,
            )
            self.assertEqual(summary["methods"]["shapellm_omni"]["output_token_count"]["mean"], 15.0)
            self.assertEqual(summary["methods"]["octllm"]["missing"], 2)
            self.assertTrue((root / "results" / "paper_table.md").is_file())
            self.assertTrue((root / "results" / "paper_table.tex").is_file())

    def test_aggregate_rejects_malformed_ok_record(self) -> None:
        config = json.loads((BENCHMARK_PATH.parent / "benchmark_config.json").read_text(encoding="utf-8"))
        method = "shapellm_omni"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset.json"
            dataset_path.write_text(
                json.dumps([{"asset_id": "asset", "text_description": "description"}]),
                encoding="utf-8",
            )
            manifest, selection = benchmark.sample_dataset(
                dataset_path, root / "results", num_assets=1, seed=1, overwrite_selection=False
            )
            row = benchmark.read_manifest(manifest)[0]
            config_hash = benchmark.method_config_hash(
                method,
                config["methods"][method],
                config,
                selection,
                gpu="0",
                seed=1,
                warmup=1,
            )
            malformed = {
                "schema_version": "inference-efficiency.v1",
                "method": "ShapeLLM-Omni",
                "status": "ok",
                "config_hash": config_hash,
                "sample_order": row["sample_order"],
                "dataset_index": row["dataset_index"],
                "asset_id": row["asset_id"],
                "text_description_sha256": hashlib.sha256(
                    row["text_description"].encode("utf-8")
                ).hexdigest(),
                "output_token_count": 10,
                "structure_token_count": 8,
                "autoregressive_steps": 10,
                "cuda_latency_ms": 100.0,
                "wall_latency_ms": 110.0,
                "truncated": False,
                "token_definition": {"output_token_count": "test"},
                "runtime": {"gpu": "NVIDIA GeForce RTX 5090"},
                "model": {"path": "test-model"},
                "source": {"commit": "test-commit"},
            }
            path = root / "results" / "raw" / method / benchmark.safe_result_name(row)
            benchmark.atomic_json(path, malformed)

            summary = benchmark.aggregate(
                [method],
                config,
                selection,
                manifest,
                root / "results",
                gpu="0",
                seed=1,
                warmup=1,
            )
            self.assertEqual(summary["methods"][method]["completed"], 0)
            self.assertEqual(summary["methods"][method]["failed"], 1)
            per_asset = (root / "results" / "per_asset.csv").read_text(encoding="utf-8")
            self.assertIn("invalid_record", per_asset)
            self.assertIn("latency_ms", per_asset)


if __name__ == "__main__":
    unittest.main()
