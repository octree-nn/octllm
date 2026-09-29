#!/usr/bin/env python3
"""Run OctLLM 3D understanding on pointllm."""

import sys
from pathlib import Path


if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import batch_understand


def build_parser():
    return batch_understand.build_parser("pointllm")


def load_assets(dataset_path, glb_dir):
    return batch_understand.load_assets(dataset_path, glb_dir, id_field=batch_understand.DATASETS["pointllm"][3])


if __name__ == "__main__":
    batch_understand.main("pointllm")
