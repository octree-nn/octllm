# Autoregressive efficiency

This suite times ShapeLLM-Omni, 3DGen-R1, SAR3D, LLaMA-Mesh, OctGPT, and
OctLLM on 200 shared Toys4K text prompts. Each method runs sequentially on one
GPU, with one warmup generation and batch size one.

## Setup

Run commands from the repository root. Edit
[benchmark_config.json](benchmark_config.json) to set the evaluation manifest,
each method's Python executable, and any local checkpoint overrides.
All models default to the Hugging Face repositories below. Downloads
finish before preflight and timing; workers load the cached weights offline.
The launcher's Python environment needs `huggingface_hub`, included in the
OctLLM environment. Local model directories and checkpoint files remain
valid overrides. Dataset and local checkpoint paths are relative to the
repository root; runner and inference YAML paths are relative to this directory.

| Model | Default Hugging Face repository | Checkpoint files |
| --- | --- | --- |
| OctLLM | [`Plurato123/OctLLM`](https://huggingface.co/Plurato123/OctLLM) | Transformers model, tokenizer, and 3D branches |
| ShapeLLM-Omni | [`yejunliang23/ShapeLLM-7B-omni`](https://huggingface.co/yejunliang23/ShapeLLM-7B-omni) | Transformers model and tokenizer |
| 3DGen-R1 | [`IvanTang/3DGen-R1`](https://huggingface.co/IvanTang/3DGen-R1) | Transformers model and tokenizer |
| LLaMA-Mesh | [`Zhengyi/LLaMA-Mesh`](https://huggingface.co/Zhengyi/LLaMA-Mesh) | Transformers model and tokenizer |
| SAR3D | [`cyw-3d/sar3d`](https://huggingface.co/cyw-3d/sar3d) | `text-condition-ckpt.pth`, `vqvae-ckpt.pt` |
| OctGPT | [`wst2001/OctGPT`](https://huggingface.co/wst2001/OctGPT) | `octgpt_objv_text.pth`, `vqvae_large_objv_bsq32.pth` |
| SAR3D / OctGPT text encoder | [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | CLIP model and tokenizer |

For SAR3D and OctGPT, both checkpoint entries contain the repository ID;
the runner selects the corresponding filename from this table.

The paper uses an RTX 5090. Prepare compatible environments for the selected
methods; the root CUDA 12.1 installation is a separate inference baseline.
For SAR3D, the optional helper reuses an existing CUDA 12.8+ environment:

```bash
SAR3D_SOURCE_ENV=/path/to/cuda128/environment \
  bash evaluation/inference_efficiency/setup_sar3d_5090.sh
```

Set SAR3D's `python` entry to `.venvs/sar3d-5090/bin/python`, or choose a
different destination with `SAR3D_TARGET_ENV`. The benchmark launcher does
not install environments. Save resolved package versions alongside timings.

Official baseline sources are fetched at the commits pinned in the config.
Use `--skip-source-prepare` for existing checkouts. `BENCHMARK_GIT_PROXY`
optionally sets a source-download proxy. To cache sources and weights ahead
of a run, use `python evaluation/inference_efficiency/benchmark.py prepare`.
`aggregate` only uses cached artifacts and does not download models.

## Run

```bash
# Run the complete suite.
bash evaluation/inference_efficiency/run_benchmark.sh --gpu 0

# Select a method and the number of evaluation assets.
bash evaluation/inference_efficiency/run_benchmark.sh \
  --gpu 0 --methods octllm --num-assets 100

# Recompute summaries from saved records.
python evaluation/inference_efficiency/benchmark.py aggregate --gpu 0
```

`--gpu` selects the physical device. Preflight checks the configured device
name (5090 by default), and records the actual GPU. Collect timings on an idle
card. Multiple methods can be selected with comma-separated names.

## Protocol and outputs

The selection uses `random.Random(42).sample(...)` and is saved with the
manifest hash, asset IDs, and original row indices. Each measured sample uses
seed `42 + dataset_index`. A changed dataset or selection requires a new run.

Timing covers the autoregressive region, including OctLLM's constrained-token
loop and structural termination checks. Model loading, preprocessing,
occupancy completion, mesh decoding, and rendering are excluded. CUDA-event
latency and synchronized wall time are both retained. Token counts use each
method's native representation; see runner diagnostics for structural counts.
Generation parameters are defined in the runners and
[the shared inference config](../../configs/inference/octllm.yaml).

```text
results/toys4k_seed42_n200/
├── assets.jsonl, selection.json   # selected inputs and provenance
├── run.json                      # configuration and process status
├── raw/<method>/*.json           # per-asset records
├── per_asset.csv
├── summary.json, summary.csv
└── paper_table.md, paper_table.tex
```

Matching successful records are reused and failed records are retried.
Pass `--overwrite` after a protocol/configuration change or to regenerate
selected records. Summaries report completion and truncation counts; an
incomplete run exits nonzero unless `--allow-incomplete` is explicitly set.
OctLLM records include generated text, token IDs, and the phase
`text_only`, `octree_incomplete`, or `octree_complete`.
