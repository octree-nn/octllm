<h1 align="center">OctLLM</h1>
<h3 align="center">Octrees as an Explicit 3D Language</h3>
<p align="center">
  Ran&nbsp;Dan<sup>1</sup>,
  <a href="https://wst2001.github.io/">Si-Tong&nbsp;Wei</a><sup>1</sup>,
  Pengfei&nbsp;Xiong<sup>2</sup>,
  <a href="https://sites.google.com/site/zhangweinus/?pli=1&amp;authuser=0">Wei&nbsp;Zhang</a><sup>2</sup>,
  <a href="http://www.muyadong.com/">Yadong&nbsp;Mu</a><sup>1</sup>,
  <a href="https://wang-ps.github.io/">Peng-Shuai&nbsp;Wang</a><sup>1,†</sup>
</p>
<p align="center">
  <sup>1</sup>Peking University &nbsp; <sup>2</sup>Independent Researcher<br>
  <sup>†</sup>Corresponding author
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2610.02388"><img src="https://img.shields.io/badge/arXiv-2610.02388-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv: 2610.02388"></a>
  <a href="https://plurato.github.io/OctLLM-page/"><img src="https://img.shields.io/badge/Project%20Page-Website-2e7d32?logo=googlechrome&amp;logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/Plurato123/OctLLM"><img src="https://img.shields.io/badge/Hugging%20Face-Weights-ffd21e?logo=huggingface&amp;logoColor=ffd21e" alt="Hugging Face model weights"></a>
</p>
<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#pretrained-models">Models</a> ·
  <a href="#inference">Inference</a> ·
  <a href="dataset_toolkits/README.md">Data</a> ·
  <a href="configs/README.md">Training</a> ·
  <a href="evaluation/README.md">Evaluation</a>
</p>

<p align="center"><img src="assets/teaser.gif" width="100%" alt="OctLLM generation and understanding"></p>

Official implementation of **Octrees as an Explicit 3D Language**. OctLLM represents 3D shapes as Sparse Octree (S-Octree) token sequences, supporting **text-to-3D, image-to-3D, and 3D understanding** in a single model. Dedicated 3D branches interact with a frozen Qwen2.5-VL backbone; occupancy completion and TRELLIS decoding convert generated octrees into textured meshes.

<!-- Real output turntables: assets/generation.gif (6–10 seconds, ideally under 10 MB).
<p align="center"><img src="assets/generation.gif" width="100%" alt="OctLLM generated meshes"></p>
-->

## Installation

Requires **Linux x86_64**, **conda**, and an **NVIDIA GPU**. Run commands from
the repository root:

```bash
bash setup.sh
conda activate OctLLM
```

The installer creates a Python 3.10 environment with PyTorch 2.4.0 / CUDA 12.1,
Transformers 4.52.4, and xFormers 0.0.27.post2. Dependencies are listed in
[requirements.txt](requirements.txt) and [requirements_inference.txt](requirements_inference.txt).
Use `bash setup.sh --resume` to continue installation in an existing environment.

Data rendering requires Blender 4.0. Caption annotation and evaluation setup
are described in their respective guides below.

## Pretrained Models

| Model | Default source |
| --- | --- |
| OctLLM | [`Plurato123/OctLLM`](https://huggingface.co/Plurato123/OctLLM) |
| Occupancy completion | [`completion/model.safetensors` in the same repository](https://huggingface.co/Plurato123/OctLLM/tree/main/completion) |
| TRELLIS image decoder | [`microsoft/TRELLIS-image-large`](https://huggingface.co/microsoft/TRELLIS-image-large) |
| TRELLIS text decoder | [`microsoft/TRELLIS-text-xlarge`](https://huggingface.co/microsoft/TRELLIS-text-xlarge) |
| CLIP ViT-L/14 for text conditioning | [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) |
| Qwen2.5-VL backbone for training | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |

OctLLM and occupancy completion download automatically from Hugging Face on
first use and reuse the local cache, like TRELLIS. To load a local copy,
download the repository and set `OCTLLM_WEIGHTS_DIR` to its absolute path:

```bash
hf download Plurato123/OctLLM --local-dir checkpoints/OctLLM
export OCTLLM_WEIGHTS_DIR="$(pwd)/checkpoints/OctLLM"
```

Model sources are configured in [configs/inference/octllm.yaml](configs/inference/octllm.yaml)
and [configs/inference/pipeline.yaml](configs/inference/pipeline.yaml).
Keep the complete OctLLM checkpoint, including its tokenizer, `config.json`,
and all model safetensors shards. The shards include the 3D router, position
embeddings, and mesh token embeddings/head; the router architecture is read
from `config.json`. Text-to-3D uses TRELLIS-text and CLIP; image-to-3D uses
TRELLIS-image. 3D understanding and text/image chat only require the OctLLM
checkpoint.

## Inference

### Single-Turn Chat

Use an ordinary text prompt for basic conversation:

```bash
python inference.py \
  --prompt "Explain what an octree is in simple terms." \
  --cuda-device 0 \
  --no-mesh
```

Add `--image` to ask a question about an image:

```bash
python inference.py \
  --prompt "What is shown in this image?" \
  --image ./assets/examples/guitar.png \
  --cuda-device 0 \
  --no-mesh
```

Each command runs one independent turn, prints the assistant's reply, and
saves a `.json` result under `outputs/octllm/`. It uses the existing model's
text and image capabilities with no conversation history. Keep
`mllm.image.preprocess: false` (the default) for general image questions to
preserve the whole image. Omit `--prompt` and `--mesh` to enter a prompt and
an optional image path interactively for one turn.

### Text to 3D

For best results, describe the object's geometry, parts, and proportions in
detail, as in the example below.

```bash
python inference.py \
  --prompt "Generate a 3D mesh based on the following text description: A biplane featuring two sets of wings stacked vertically, connected by struts and wires. The fuselage is elongated with a rounded nose housing a propeller. Tail assembly includes a vertical stabilizer and horizontal stabilizers. Landing gear consists of two wheels positioned under the front section. Overall design emphasizes symmetry and balanced proportions typical of early aviation aircraft." \
  --output-dir outputs/demo --output-name biplane --cuda-device 0
```

### Image to 3D

Try the provided [example images](assets/examples). For your own inputs,
match their rendering style and format: a single centered object in a square
RGBA PNG with a transparent background. Set `mllm.image.preprocess: true` in
[the pipeline YAML](configs/inference/pipeline.yaml) to normalize custom
inputs to a centered 1024 × 1024 image with a consistent object scale.

```bash
python inference.py \
  --prompt "Generate a 3D mesh based on this image:" \
  --image ./assets/examples/deer.png \
  --cuda-device 0 \
  --output-dir outputs/demo --output-name image_asset
```

Each run saves a textured `.glb`, the S-Octree sequence in `.tokens.txt`, and
response metadata in `.json`. Sampling and decoder settings are configured in
[the pipeline YAML](configs/inference/pipeline.yaml). Use `--no-mesh` to skip
mesh reconstruction, or `python inference.py --help` for all options.

### 3D Understanding

Start with the [example meshes](assets/examples/understanding) to test 3D
understanding. Pass a GLB or OBJ file directly with a question:

```bash
python inference.py \
  --mesh ./assets/examples/understanding/airplane.glb \
  --prompt "Describe this 3D asset in detail:" \
  --no-mesh \
  --cuda-device 0 \
  --output-dir outputs/demo --output-name understanding
```

Replace the path with `/path/to/asset.obj` for OBJ input. If `--prompt` is
omitted, the model defaults to describing the asset. `--no-mesh` skips output
mesh reconstruction; it does not disable preprocessing of the input mesh.

The CLI automatically normalizes the mesh, samples 100,000 surface points,
builds an octree, and prunes it using the same
[dataset processing code](dataset_toolkits/tokenize_trellis.py). Defaults
are depth 6, full depth 3, pruning depth 5, and drop probability 0.5. The
resulting S-Octree sequence is appended to the question. Sampling and pruning
settings are under `mesh_input` in
[the pipeline YAML](configs/inference/pipeline.yaml); octree depths follow
`mllm.generation.max_layer` and `full_depth`.

The example saves the input sequence to
`outputs/demo/understanding.input.tokens.txt` and the reply and input metadata
to `outputs/demo/understanding.json`. Sampling and pruning are stochastic;
reuse the saved tokens when you need the exact same input. Geometry is
encoded; mesh textures are not used for 3D understanding.

To convert a mesh separately without loading the OctLLM checkpoint:

```bash
python dataset_toolkits/tokenize_mesh.py \
  --mesh /path/to/asset.obj --output outputs/demo/asset.tokens.txt
```

Existing S-Octree prompts remain supported: put your question followed by
the complete `<mesh_bos>...<mesh_eos>` sequence in a UTF-8 file, then run:

```bash
python inference.py --prompt-file /path/to/prompt.txt --no-mesh --cuda-device 0
```

For batch 3D understanding from GLB assets, see the
[evaluation guide](evaluation/README.md).

## Data and Training

Follow the [data preparation guide](dataset_toolkits/README.md) to tokenize
meshes, generate captions, and build the instruction datasets. Configure the
dataset and output paths in [configs/train/octllm.yaml](configs/train/octllm.yaml):

```bash
python train.py configs/train/octllm.yaml
```

The [training guide](configs/README.md) covers dataset registration,
distributed training, occupancy completion, and ShapeNet ablations.
Evaluation commands are in [evaluation/README.md](evaluation/README.md).

## Acknowledgements

This work builds on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),
[TRELLIS](https://github.com/microsoft/TRELLIS),
[OctGPT](https://github.com/octree-nn/octgpt), and
[O-CNN](https://github.com/octree-nn/ocnn-pytorch).

Original OctLLM contributions use the MIT license. Third-party notices and
terms are included in [LICENSE](LICENSE).

## Citation

```bibtex
@misc{dan2026octreesexplicit3dlanguage,
  title={Octrees as an Explicit 3D Language},
  author={Ran Dan and Si-Tong Wei and Pengfei Xiong
          and Wei Zhang and Yadong Mu and Peng-Shuai Wang},
  year={2026},
  eprint={2610.02388},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.02388},
}
```
