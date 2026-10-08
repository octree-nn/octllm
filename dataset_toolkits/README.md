# Data preparation

Run commands from the repository root after [installation](../README.md#-installation).
The tools process TRELLIS-500K source subsets (Sketchfab, HSSD, ABO) and five
ShapeNet categories: airplane, car, chair, rifle, and table. Obtain meshes and
metadata from the original providers. Exact training selections and evaluation
split manifests are required to reproduce the paper subsets.

## 1. Tokenize TRELLIS-500K meshes

A metadata CSV must contain `local_path`; each path is joined to the explicit
asset root. The tokenizer produces `<asset_id>_bytes.txt` files containing
space-separated occupancy-byte integers.

```bash
python dataset_toolkits/tokenize_trellis.py \
  --csv_path /path/to/HSSD/metadata.csv \
  --data_root /path/to/HSSD \
  --output_dir datasets/trellis/HSSD/tokenized_bytes \
  --num_samples 100000 --depth 6 --full_depth 3 \
  --target_depth 5 --drop_prob 0.5 --prune
```

Meshes are normalized, sampled, converted to octrees, and pruned. The
TRELLIS tokenizer excludes sequences outside 100–4000 bytes. Pruning uses
random sampling; retaining the processed byte files is necessary to preserve
an exact dataset realization.

For ShapeNet, `--input_dir` is the parent of category IDs, with meshes at
`<category_id>/<asset_id>/model.obj`:

```bash
python dataset_toolkits/tokenize_shapenet.py \
  --input_dir /path/to/shapenet/fixed_mesh \
  --output_dir datasets/shapenet/tokens \
  --depth 6 --full_depth 3 --target_depth 5 --drop_prob 0.5 --prune
```

Outputs retain the category/asset hierarchy. ShapeNet keeps 100–6000
mesh-byte tokens; TRELLIS keeps 100–4000. The ShapeNet implementation's 6000
upper bound is retained and differs from the paper's stated 4000 threshold.

## 2. Render condition images

Use per-asset directories with `000.png` through `023.png` and rendering
metadata. Reuse upstream condition renders when available. The local
condition renderer accepts an OBJ / BLEND mesh tree:

```bash
python dataset_toolkits/render/render_cond.py \
  --input_dir /path/to/meshes \
  --output_dir datasets/trellis/HSSD/renders_cond \
  --blender-path /path/to/blender
```

Install Blender 4.0 and supply `--blender-path` or set `BLENDER_PATH`.
Headless rendering requires an appropriate OpenGL setup.
The renderer uses Blender 4.0 / EEVEE and 1024 × 1024 output. Its output layout
is `<relative_parent>/<mesh_stem>/`; arrange the input tree or output paths
so each caption/SFT asset directory matches the ID in your metadata.
Published condition renders should be reused for exact evaluation conditions.

## 3. Annotate geometry captions

Captioning uses a separate environment for Qwen3.5-9B:

```bash
bash dataset_toolkits/caption/setup.sh
```

Set `model_name_or_path` in [configs/caption/qwen35.yaml](../configs/caption/qwen35.yaml),
then run the annotation pipeline:

```bash
ROOT_DIR=/path/to/HSSD/renders_cond \
OUTPUT_DIR=/path/to/HSSD/captions \
GPU_IDS=0,1,2,3 \
  bash dataset_toolkits/caption/run_caption.sh
```

This pipeline selects views **014–017**, generates geometry-only English
captions with thinking disabled, and writes `<asset_id>.txt` plus a merged
`captions.csv`. The configured prompt excludes color, material, texture,
lighting, and background. It requests at most 100 words; generation is capped
at 200 tokens with temperature 0.2. The word limit is a prompt instruction,
not a post-generation hard truncation in the local model path.

The separate caption environment uses unpinned dependencies; save its resolved
versions with annotation outputs. The CSV conversion can also run independently:

```bash
python dataset_toolkits/caption/export_captions.py \
  --input-dir /path/to/HSSD/captions \
  --output-csv /path/to/HSSD/captions.csv
```

`generate_captions_api.py` is a separate API-based annotation tool. It does
not replace the local Qwen3.5-9B recipe used above.

## 4. Build SFT conversations

```bash
python dataset_toolkits/prepare_trellis_sft.py \
  --token_dir /path/to/HSSD/tokenized_bytes \
  --render_dir /path/to/HSSD/renders_cond \
  --caption_csv /path/to/HSSD/captions.csv \
  --output_dir datasets/octllm/sft/HSSD \
  --limit 200000 --seed 42
```

The tool joins modalities by asset ID and creates all three tasks using the
[conversation templates](../data/templates): text-to-3D (`description`),
3D-to-text (`understanding`), and image-to-3D (`image`). Files are named
`llm_trellis_dataset_<shard>_<task>.json`. Run each source subset into its own
folder. Sharding does not automatically assign a source name to each file.

ShapeNet uses its existing `name` / `description` JSON annotations and one
task per invocation:

```bash
python dataset_toolkits/prepare_shapenet_sft.py \
  --category airplane --task understanding \
  --description-json /path/to/description-02691156.json \
  --token-dir datasets/shapenet/tokens/02691156 \
  --render-dir /path/to/shapenet/renders/02691156 \
  --test-json /path/to/fixed_shapenet_test_ids.json \
  --output-dir datasets/octllm/sft/shapenet --seed 42
```

Repeat for `--task description` and `--task image` and for other categories.
`--test-json` must contain rows with `name` identifiers; when supplied, those
assets are excluded from training and a new test split is not generated.
Without it the tool makes its existing 90/10 random split. Use the finalized
held-out manifest for paper reproduction. 

Image-conversation rows contain `messages` and `images`; text-only rows
contain `messages`. Dataset registration is described in [training](../configs/README.md).

## 5. Prepare occupancy completion pairs

```bash
python dataset_toolkits/prepare_trellis_completion.py \
  --category HSSD \
  --csv_path /path/to/HSSD/metadata.csv --data_root /path/to/HSSD \
  --output_dir_ori datasets/completion/complete \
  --output_dir_pruned datasets/completion/sparse \
  --depth 6 --full_depth 3 --target_depth 5 --drop_prob 0.5

python dataset_toolkits/prepare_shapenet_completion.py \
  --input_dir /path/to/shapenet/fixed_mesh \
  --output_dir_ori datasets/completion/complete \
  --output_dir_pruned datasets/completion/sparse
```

Keep matching filenames across the sparse and complete directories; the
paired voxel dataset uses them to align inputs and targets.
