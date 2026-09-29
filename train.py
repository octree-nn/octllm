#!/usr/bin/env python3
"""Launch OctLLM training with the repository's LLaMA-Factory trainer."""

import sys


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python train.py CONFIG.yaml [key=value ...]")
        print("Main recipe: configs/train/octllm.yaml")
        print("Distributed launch: set FORCE_TORCHRUN=1 and NPROC_PER_NODE.")
        return

    from llamafactory.cli import main as factory_main

    sys.argv.insert(1, "train")
    factory_main()


if __name__ == "__main__":
    main()
