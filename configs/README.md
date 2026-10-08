# Training

Run commands from the repository root after [installation](../README.md#-installation)
and [data preparation](../dataset_toolkits/README.md).

## Dataset registration

Copy [data/dataset_info.json](../data/dataset_info.json) into the directory
selected by `dataset_dir`. The main configuration expects this layout:

```text
datasets/octllm/sft/
├── dataset_info.json
├── ABO/llm_trellis_dataset_1_{image,description,understanding}.json
├── HSSD/llm_trellis_dataset_1_{image,description,understanding}.json
├── ObjaverseXL_sketchfab/llm_trellis_dataset_{1,2,3}_{image,description,understanding}.json
└── shapenet/{llm,mllm}_shapenet_dataset_<category>_1_<task>.json
```

Braces denote alternative filenames. Update the registry for the shards
produced by your preprocessing. Entries use ShareGPT `messages`; image tasks
also include `images`.

```bash
cp data/dataset_info.json datasets/octllm/sft/dataset_info.json
```

## OctLLM

[train/octllm.yaml](train/octllm.yaml) configures the main model's 12 routed
layers, frozen backbone, and separate mesh embeddings/head. The backbone
loads from `Qwen/Qwen2.5-VL-7B-Instruct` on Hugging Face by default;
`model_name_or_path` also accepts a local model directory. Set `dataset_dir`,
`media_dir`, and `output_dir` to your local paths. All keys in `dataset`
must be present in the registry.

```bash
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 \
  python train.py configs/train/octllm.yaml
```

The configuration uses per-device batch size 2 and accumulation 2. The paper
uses 16 H20 workers for an effective batch size of 64; the eight-worker example
above gives 32. For two eight-GPU nodes, set `NNODES=2`, `NPROC_PER_NODE=8`,
the appropriate `NODE_RANK`, and shared `MASTER_ADDR` / `MASTER_PORT` on
each node.

The recipe runs for 15 epochs; the evaluated checkpoint is step 69,000.
Step counts depend on the dataset and distributed batch size. Set
`resume_from_checkpoint` to resume from a checkpoint containing trainer and
optimizer state. Keep all model and tokenizer files for inference.

## Occupancy completion

Completion is trained separately from the language model:

```bash
NPROC_PER_NODE=8 bash scripts/train_completion.sh \
  --train-input-dir datasets/completion/sparse \
  --train-target-dir datasets/completion/complete \
  --output-dir outputs/completion --batch-size 8
```

The launcher selects the paper's bottleneck model without cross-scale skip
connections (`--model-type vae --recon-loss bce`). The bare Python trainer
defaults to `unet` / `dice`, so use the launcher for the standard recipe.
Eight workers with batch size 8 give an effective batch size of 64.
The launcher retains 50 epochs; the paper checkpoint was trained for 155K
steps. For a newly trained model, set `mesh.vae.checkpoint` in the inference
YAML to the saved `best.pt`. The released inference bundle instead stores
the model tensors in `completion/model.safetensors` alongside OctLLM;
`mesh.vae.checkpoint: null` selects that bundled checkpoint automatically.
Inference defaults to `Plurato123/OctLLM` on Hugging Face; setting
`OCTLLM_WEIGHTS_DIR` to an absolute local directory overrides both model sources.
Explicit relative paths in `model_name_or_path` are resolved from its YAML file.

## Ablations

[ablations/octllm_shapenet.yaml](ablations/octllm_shapenet.yaml) selects the
ShapeNet-airplane subset and eight routed layers. Its data directory is
`datasets/shapenet/sft`; copy the registry there and adjust the airplane paths.
The paper uses four L40 GPUs.

Completion variants are selected with `--model-type vae|unet` and
`--recon-loss bce|dice`. Encoder alignment uses
[scripts/ablations/train_encoder_alignment.py](../scripts/ablations/train_encoder_alignment.py)
with `--pretrained-trellis-dir` (default: `microsoft/TRELLIS-image-large`);
this accepts a Hugging Face repository ID, a local snapshot root, or its
`ckpts` directory. Its matching inference entry is
[infer_encoder_alignment.py](../scripts/ablations/infer_encoder_alignment.py).
