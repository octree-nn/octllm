# Evaluation

Run commands from the repository root with the OctLLM checkpoint and evaluation
manifests configured. Install evaluation extras as needed:

```bash
python -m pip install nltk rouge-score sentence-transformers requests deepeval
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4
python -m pip check
```

FID/KID uses `clean-fid`, declared in the main requirements. For language
metrics, set `NLTK_DATA` if resources are stored outside the default location.

## Toys4K generation

Use the fixed evaluation manifest with `asset_id`, `mesh_path`,
`render_image_path`, and `text_description` fields:

```json
[
  {
    "asset_id": "example_asset",
    "mesh_path": "/path/to/reference_mesh.pickle",
    "render_image_path": "/path/to/condition.png",
    "text_description": "A chair with four legs and a curved backrest."
  }
]
```

```bash
python scripts/batch_generate_toys4k.py \
  --dataset datasets/toys4k/test.json \
  --output-root outputs/toys4k --condition both --stage all --gpus 0
```

Outputs are stored under `<condition>/<asset_id>/`, with conditions
`image_to_3d` and `text_to_3d`. Completed token and mesh outputs are reused;
pass `--overwrite` to regenerate. Use `--limit 1` for a small run or
`--gpus 0,1,2,3` to distribute assets.

Render generated meshes with Blender 4.0:

```bash
python dataset_toolkits/render/render_mesh_subdirs.py \
  --root_dir outputs/toys4k --layout octllm \
  --num_views 24 --resolution 1024 --blender-path /path/to/blender
```

`--layout octllm` produces `<asset_id>__glb__<view>.png` in
`image_to_3drendered/` and `text_to_3drendered/`, matching the metric inputs.

```bash
python evaluation/compute_toys4k_fid_kid.py \
  --split_json datasets/toys4k/test.json \
  --renders_cond_dir datasets/toys4k/renders_cond \
  --method_dir outputs/toys4k --selected_views 014 015 016 017 --device cuda \
  --output_json outputs/toys4k/fid_kid.json

python evaluation/compute_toys4k_clip.py \
  --split_json datasets/toys4k/test.json \
  --method_dir outputs/toys4k --device cuda \
  --output_json outputs/toys4k/clip.json
```

The DINOv2 ViT-L/14 register-token model defaults to
[`facebook/dinov2-with-registers-large`](https://huggingface.co/facebook/dinov2-with-registers-large),
using 518-pixel bicubic resize, center crop, and ImageNet normalization.
To use an existing torch hub checkout and its original weights, set both
`--dino_repo_dir` and `--dino_weights_path` explicitly.
KID outputs are raw values (multiply by 100 for the paper's scale). Text
CLIP averages views 014–017; image CLIP takes the maximum over 24 views.
Inspect output coverage fields for missing generations or renders, and use
consistent reference/camera settings across methods.

## PointLLM-200 understanding

Use PointLLM annotations (`object_id`, `conversations`) and one
`<object_id>.glb` per asset. This entry tokenizes GLBs before inference:

```bash
python scripts/batch_understand_pointllm.py \
  --dataset datasets/pointllm/PointLLM_brief_description_val_200_GT.json \
  --glb-dir datasets/pointllm/glbs \
  --output-dir outputs/pointllm --method-name OctLLM --gpus 0 --resume
```

Captions are saved to `outputs/pointllm/OctLLM/<object_id>.txt`.

```bash
python evaluation/compute_toys4k_language_metrics.py \
  --gt_json datasets/pointllm/PointLLM_brief_description_val_200_GT.json \
  --pred_dir outputs/pointllm --method OctLLM
```

Despite its filename, the aggregator accepts both PointLLM and Toys4K
annotations. It uses `all-mpnet-base-v2` and
`princeton-nlp/sup-simcse-roberta-large`, with metrics scaled by 100.

For GPT-ref / GPT-img, configure your judge endpoint and credentials:

```bash
export DASHSCOPE_API_KEY=YOUR_API_KEY
export DASHSCOPE_BASE_URL=https://YOUR_WORKSPACE_ENDPOINT/compatible-mode/v1
python evaluation/compute_pointllm_gpt_metrics.py \
  --gt-json datasets/pointllm/PointLLM_brief_description_val_200_GT.json \
  --pred-dir outputs/pointllm/OctLLM \
  --render-dir datasets/pointllm/renders_blender_pbr \
  --judge both --model qwen3.8-max \
  --output-json outputs/pointllm/judge_metrics.json
```

Each render directory contains `front.png`, `right.png`, `back.png`, and
`left.png`. Judging uses the configured API and supports resuming. Preserve the judge version
for comparable scores; GPT-img receives renders without the reference caption.

## General language

```bash
python evaluation/models/language_benchmarks.py \
  --config configs/inference/octllm.yaml --output-dir outputs/language \
  --benchmarks mmlu hellaswag gsm8k ifeval
```

This uses DeepEval and greedy decoding, with 5-shot MMLU, 10-shot HellaSwag,
and 3-shot GSM8K with chain-of-thought.
The shell launcher uses the active Python; `EVAL_PYTHON` selects another
interpreter when needed.

## ShapeNet ablations

Use `scripts/batch_generate_shapenet.py` for responses, then
`scripts/decode_octree.py --input-dir ... --metadata-json ... --output-dir ...`
for mesh decoding. The metadata must supply the original text/image conditions.
Select the matching ablation checkpoint in your inference configuration.

Render generated meshes with `evaluation/render_glb_multiview_normals.py`
and references with `evaluation/render_shapenet_fixed_obj_normals.py`:

```bash
python evaluation/compute_shapenet_fid_kid.py \
  --dir1 /path/to/reference_normals --dir2 /path/to/generated_normals
python evaluation/compute_language_metrics.py \
  --pred_dir /path/to/captions --gt_json /path/to/descriptions.json
```

ShapeNet caption annotations use `name` / `description`. The ShapeNet
language tool prints unscaled similarities. Normal-map FID/KID and the
Toys4K RGB-render metrics use different rendering protocols.

## Autoregressive efficiency

See [inference_efficiency/README.md](inference_efficiency/README.md) for
single-GPU timing on 200 fixed prompts and the separate RTX 5090 environments.
