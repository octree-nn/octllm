#!/usr/bin/env python3
"""Decode mesh tokens into octrees, with optional VAE voxel completion."""

import gc
import re
import sys
from pathlib import Path


# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import ocnn
import torch
import trimesh
from ocnn.nn import octree2voxel
from PIL import Image
from safetensors.torch import load_file as load_safetensors


# ============== VAE Completion Functions ==============


def build_vae_model(
    in_channels: int = 1,
    out_channels: int = 1,
    latent_channels: int = 256,
    encoder_channels: List[int] = [64, 128, 256, 256],
    decoder_channels: Optional[Sequence[int]] = None,
    num_res_blocks: int = 2,
    num_res_blocks_middle: int = 2,
    norm_type: str = "layer",
    recon_loss: str = "bce",
    use_fp16: bool = False,
):
    """Build SparseStructureVAE model for voxel completion."""
    from trellis.models.sparse_structure_vae import SparseStructureVAE

    model = SparseStructureVAE(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        num_res_blocks=num_res_blocks,
        num_res_blocks_middle=num_res_blocks_middle,
        norm_type=norm_type,
        recon_loss=recon_loss,
        use_fp16=use_fp16,
    )
    return model


def load_vae_checkpoint(model, path: str) -> None:
    """Load VAE checkpoint weights."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint {path} does not exist.")
    if str(path).endswith(".safetensors"):
        checkpoint = load_safetensors(path, device="cpu")
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load checkpoint. Missing: {missing}, Unexpected: {unexpected}")


def vae_complete_voxel(
    voxel: np.ndarray,
    model,
    device: str = "cuda",
    sample_posterior: bool = False,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Use VAE model to complete the input voxel.

    Args:
        voxel: Input voxel array with shape (64, 64, 64)
        model: Loaded VAE model
        device: Device for inference
        sample_posterior: Whether to sample from posterior
        threshold: Optional threshold for binarization

    Returns:
        Completed voxel array with shape (64, 64, 64)
    """
    # Preprocess: convert to tensor and add batch/channel dimensions
    tensor = torch.from_numpy(voxel).to(torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # Add channel dim: (1, 64, 64, 64)
    tensor = tensor.unsqueeze(0)  # Add batch dim: (1, 1, 64, 64, 64)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = tensor.to(device)

    # Forward pass
    with torch.no_grad():
        with model.autocast_context(enabled=False):
            recon = model.forward(tensor, sample_posterior=sample_posterior)

    # Post-process
    recon = torch.sigmoid(recon)
    if threshold is not None:
        recon = (recon >= threshold).to(recon.dtype)

    # Convert back to numpy
    completed_voxel = recon[0][0].cpu().numpy()
    return completed_voxel


# Global VAE model cache to avoid reloading
_vae_model_cache = {
    "model": None,
    "checkpoint_path": None,
    "device": None,
}


def get_vae_model(checkpoint_path: str, device: str = "cuda"):
    """Get or load VAE model (with caching)."""
    global _vae_model_cache

    if (
        _vae_model_cache["model"] is not None
        and _vae_model_cache["checkpoint_path"] == checkpoint_path
        and _vae_model_cache["device"] == device
    ):
        return _vae_model_cache["model"]

    print(f"Loading VAE model from {checkpoint_path}...")
    model = build_vae_model()
    load_vae_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()

    _vae_model_cache["model"] = model
    _vae_model_cache["checkpoint_path"] = checkpoint_path
    _vae_model_cache["device"] = device

    print("VAE model loaded successfully.")
    return model


def _load_trellis_vae_config(trellis_dir: str) -> Tuple[dict, dict]:
    enc_json = os.path.join(trellis_dir, "ss_enc_conv3d_16l8_fp16.json")
    dec_json = os.path.join(trellis_dir, "ss_dec_conv3d_16l8_fp16.json")
    if not os.path.isfile(enc_json) or not os.path.isfile(dec_json):
        raise FileNotFoundError(f"TRELLIS VAE config json not found in {trellis_dir}")
    with open(enc_json, "r") as f:
        enc_cfg = json.load(f)
    with open(dec_json, "r") as f:
        dec_cfg = json.load(f)
    return enc_cfg, dec_cfg


def _build_trellis_encoder_decoder(enc_cfg: dict, dec_cfg: dict, use_fp16: bool):
    from trellis.models.sparse_structure_vae import SparseStructureDecoder, SparseStructureEncoder

    enc_args = enc_cfg.get("args", {})
    dec_args = dec_cfg.get("args", {})
    encoder = SparseStructureEncoder(
        in_channels=enc_args.get("in_channels", 1),
        latent_channels=enc_args.get("latent_channels", 8),
        num_res_blocks=enc_args.get("num_res_blocks", 2),
        channels=enc_args.get("channels", [32, 128, 512]),
        num_res_blocks_middle=enc_args.get("num_res_blocks_middle", 2),
        norm_type=enc_args.get("norm_type", "layer"),
        use_fp16=use_fp16,
    )
    decoder = SparseStructureDecoder(
        out_channels=dec_args.get("out_channels", 1),
        latent_channels=dec_args.get("latent_channels", 8),
        num_res_blocks=dec_args.get("num_res_blocks", 2),
        channels=dec_args.get("channels", [512, 128, 32]),
        num_res_blocks_middle=dec_args.get("num_res_blocks_middle", 2),
        norm_type=dec_args.get("norm_type", "layer"),
        use_fp16=use_fp16,
    )
    return encoder, decoder


def _load_trellis_weights(
    encoder,
    decoder,
    trellis_dir: str,
    strict: bool = False,
) -> None:
    enc_files = sorted(
        [f for f in os.listdir(trellis_dir) if f.startswith("ss_enc_conv3d_") and f.endswith(".safetensors")]
    )
    dec_files = sorted(
        [f for f in os.listdir(trellis_dir) if f.startswith("ss_dec_conv3d_") and f.endswith(".safetensors")]
    )
    if not enc_files or not dec_files:
        raise FileNotFoundError(f"No TRELLIS safetensors found in {trellis_dir}")
    enc_path = os.path.join(trellis_dir, enc_files[0])
    dec_path = os.path.join(trellis_dir, dec_files[0])
    enc_weights = load_safetensors(enc_path)
    dec_weights = load_safetensors(dec_path)

    encoder_state = encoder.state_dict()
    decoder_state = decoder.state_dict()
    enc_skipped = []
    for key, value in enc_weights.items():
        if key in encoder_state and encoder_state[key].shape == value.shape:
            encoder_state[key] = value
        else:
            enc_skipped.append(key)
    dec_skipped = []
    for key, value in dec_weights.items():
        if key in decoder_state and decoder_state[key].shape == value.shape:
            decoder_state[key] = value
        else:
            dec_skipped.append(key)
    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)
    if strict and (enc_skipped or dec_skipped):
        raise RuntimeError(
            f"TRELLIS weights load mismatch. Encoder skipped: {len(enc_skipped)}, Decoder skipped: {len(dec_skipped)}"
        )


def _load_encoder_align_checkpoint(encoder, path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Encoder checkpoint {path} does not exist.")
    checkpoint = torch.load(path, map_location="cpu")
    if "encoder" not in checkpoint:
        raise KeyError(f"Checkpoint {path} does not contain 'encoder' weights.")
    encoder.load_state_dict(checkpoint["encoder"])


_trellis_vae_cache = {
    "encoder": None,
    "decoder": None,
    "encoder_ckpt": None,
    "trellis_dir": None,
    "device": None,
}


def get_trellis_align_models(encoder_ckpt: str, trellis_dir: str, device: str = "cuda"):
    global _trellis_vae_cache
    if (
        _trellis_vae_cache["encoder"] is not None
        and _trellis_vae_cache["decoder"] is not None
        and _trellis_vae_cache["encoder_ckpt"] == encoder_ckpt
        and _trellis_vae_cache["trellis_dir"] == trellis_dir
        and _trellis_vae_cache["device"] == device
    ):
        return _trellis_vae_cache["encoder"], _trellis_vae_cache["decoder"]

    from scripts.model_sources import resolve_trellis_vae_directory

    weights_dir = str(resolve_trellis_vae_directory(trellis_dir))
    enc_cfg, dec_cfg = _load_trellis_vae_config(weights_dir)
    use_fp16 = bool(enc_cfg.get("args", {}).get("use_fp16", False)) and str(device).startswith("cuda")
    encoder, decoder = _build_trellis_encoder_decoder(enc_cfg, dec_cfg, use_fp16=use_fp16)
    _load_trellis_weights(encoder, decoder, weights_dir, strict=True)
    _load_encoder_align_checkpoint(encoder, encoder_ckpt)
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    _trellis_vae_cache["encoder"] = encoder
    _trellis_vae_cache["decoder"] = decoder
    _trellis_vae_cache["encoder_ckpt"] = encoder_ckpt
    _trellis_vae_cache["trellis_dir"] = trellis_dir
    _trellis_vae_cache["device"] = device
    return encoder, decoder


def vae_complete_voxel_trellis_align(
    voxel: np.ndarray,
    encoder_ckpt: str,
    trellis_dir: str,
    device: str = "cuda",
    threshold: Optional[float] = None,
) -> np.ndarray:
    tensor = torch.from_numpy(voxel).to(torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = tensor.to(device)

    encoder, decoder = get_trellis_align_models(encoder_ckpt, trellis_dir, device=device)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    use_amp = device_type == "cuda"
    amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    with torch.no_grad():
        with torch.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
            _, mean, _ = encoder(tensor, sample_posterior=False, return_raw=True)
            recon = decoder(mean)
    recon = torch.sigmoid(recon)
    if threshold is not None:
        recon = (recon >= threshold).to(recon.dtype)
    completed_voxel = recon[0][0].cpu().numpy()
    return completed_voxel


# ============== Trellis Generation Functions ==============

# Global Trellis pipeline cache
_trellis_pipeline_cache = {
    "pipeline": None,
    "model_path": None,
    "task_type": None,
    "text_condition_model_path": None,
}


def get_trellis_pipeline(
    model_path: str,
    task_type: str = "image",
    text_condition_model_path: Optional[str] = None,
):
    """Get or load a TRELLIS image- or text-conditioned pipeline with caching.

    Args:
        model_path: Local TRELLIS directory or Hugging Face repository ID.
        task_type: Task type, 'image' or 'text'
        text_condition_model_path: Optional local CLIP directory or Hugging Face
            repository ID for text-conditioned generation.
    """
    global _trellis_pipeline_cache

    if (
        _trellis_pipeline_cache["pipeline"] is not None
        and _trellis_pipeline_cache["model_path"] == model_path
        and _trellis_pipeline_cache["task_type"] == task_type
        and _trellis_pipeline_cache["text_condition_model_path"] == text_condition_model_path
    ):
        return _trellis_pipeline_cache["pipeline"]
    if _trellis_pipeline_cache["pipeline"] is not None:
        clear_trellis_pipeline_cache()

    print(f"Loading Trellis pipeline from {model_path}...")
    # TRELLIS reads its attention backend while importing the pipeline modules.
    # Set it before the import so the unified inference environment does not
    # require a flash-attn wheel tied to a specific PyTorch/CUDA ABI.
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline

    if task_type == "image":
        pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    elif task_type == "text":
        previous_text_model = os.environ.get("TRELLIS_TEXT_COND_MODEL_PATH")
        try:
            if text_condition_model_path:
                os.environ["TRELLIS_TEXT_COND_MODEL_PATH"] = os.path.expanduser(str(text_condition_model_path))
            pipeline = TrellisTextTo3DPipeline.from_pretrained(model_path)
        finally:
            if previous_text_model is None:
                os.environ.pop("TRELLIS_TEXT_COND_MODEL_PATH", None)
            else:
                os.environ["TRELLIS_TEXT_COND_MODEL_PATH"] = previous_text_model
    else:
        raise ValueError(f"Invalid task type: {task_type}")
    pipeline.cuda()

    _trellis_pipeline_cache["pipeline"] = pipeline
    _trellis_pipeline_cache["model_path"] = model_path
    _trellis_pipeline_cache["task_type"] = task_type
    _trellis_pipeline_cache["text_condition_model_path"] = text_condition_model_path

    print("Trellis pipeline loaded successfully.")
    return pipeline


def clear_trellis_pipeline_cache() -> None:
    """Release the cached TRELLIS pipeline before switching batch conditions."""
    pipeline = _trellis_pipeline_cache.get("pipeline")
    _trellis_pipeline_cache["pipeline"] = None
    _trellis_pipeline_cache["model_path"] = None
    _trellis_pipeline_cache["task_type"] = None
    _trellis_pipeline_cache["text_condition_model_path"] = None
    if pipeline is not None:
        del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def rotate_voxel_90(
    voxel: np.ndarray,
    axis: str = "x",
    quarter_turns: int = 1,
) -> np.ndarray:
    """Rotate a 3D voxel grid around its center in 90-degree increments.

    The array axes are interpreted as ``(x, y, z)``. Positive rotations follow
    the right-hand rule. In particular, ``axis="x", quarter_turns=1`` applies
    the ShapeNet-to-TRELLIS alignment used by this project:

        x' = x, y' = -z, z' = y

    Args:
        voxel: Voxel array with shape (X, Y, Z).
        axis: Rotation axis: "x", "y", or "z".
        quarter_turns: Number of positive 90-degree turns. Use -1 for -90
            degrees and 0 (or any multiple of 4) to disable rotation.

    Returns:
        A contiguous rotated voxel array.
    """
    voxel = np.asarray(voxel)
    if voxel.ndim != 3:
        raise ValueError(f"Expected a 3D voxel array, got shape {voxel.shape}")

    axis = axis.lower()
    rotation_planes = {
        "x": (1, 2),
        "y": (2, 0),
        "z": (0, 1),
    }
    if axis not in rotation_planes:
        raise ValueError(f"Invalid voxel rotation axis: {axis!r}. Expected 'x', 'y', or 'z'.")
    if not isinstance(quarter_turns, (int, np.integer)):
        raise TypeError(f"quarter_turns must be an integer, got {type(quarter_turns).__name__}")

    rotated = np.rot90(
        voxel,
        k=int(quarter_turns) % 4,
        axes=rotation_planes[axis],
    )
    # np.rot90 may return an array with negative strides, which
    # torch.from_numpy cannot consume.
    return np.ascontiguousarray(rotated)


def voxel_to_coords(voxel: np.ndarray, device: str = "cuda", threshold: float = 0.5) -> torch.Tensor:
    """Convert voxel to sparse coordinates for Trellis pipeline.

    Args:
        voxel: Voxel array with shape (64, 64, 64), values can be float (0-1) or binary
        device: Device for the output tensor
        threshold: Threshold for binarizing float voxel values

    Returns:
        coords: Sparse coordinates tensor with shape (N, 4), format: [batch_id, x, y, z]
    """
    # Convert voxel to tensor and binarize using threshold
    ss = torch.from_numpy(voxel).float()
    ss = (ss > threshold).long()  # Binarize: values > threshold become 1
    ss = ss.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64, 64)

    # Get coordinates where voxel > 0
    coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()  # [batch_id, x, y, z]
    coords = coords.to(device)

    return coords


def trellis_generate_from_voxel(
    voxel: np.ndarray,
    image_path: str,
    text_prompt: str,
    pipeline,
    output_dir: str,
    output_name: str = "generated",
    simplify: float = 0.95,
    texture_size: int = 1024,
    task_type: str = "image",
    voxel_rotation_axis: Optional[str] = "x",
    voxel_rotation_quarter_turns: int = 1,
) -> str:
    """Generate 3D asset from voxel using Trellis pipeline with image condition.

    Args:
        voxel: Completed voxel array with shape (64, 64, 64)
        image_path: Path to the condition image
        text_prompt: Text prompt for Trellis generation
        pipeline: Trellis image-to-3D pipeline
        output_dir: Directory to save the output
        output_name: Name of the output file (without extension)
        simplify: Simplification ratio for mesh
        texture_size: Texture size for the output
        task_type: Task type, 'image' or 'text'
        voxel_rotation_axis: Axis used to align the input voxel to TRELLIS.
            The default "x" matches the ShapeNet-to-TRELLIS conversion used
            in the original ShapeNet VQVAE preprocessing. Set to None to
            disable the orientation correction.
        voxel_rotation_quarter_turns: Number of positive 90-degree turns.
            The default 1 means +90 degrees by the right-hand rule.

    Returns:
        Path to the generated glb file
    """
    from trellis.utils import postprocessing_utils

    # The octree/VAE stays in its original training orientation. Rotate only
    # at the TRELLIS boundary so VAE completion is not fed out-of-distribution
    # inputs. This is an exact integer-grid transform with no interpolation.
    voxel_for_trellis = voxel
    if voxel_rotation_axis is not None:
        voxel_for_trellis = rotate_voxel_90(
            voxel,
            axis=voxel_rotation_axis,
            quarter_turns=voxel_rotation_quarter_turns,
        )
        degrees = (int(voxel_rotation_quarter_turns) % 4) * 90
        print(
            f"Applied TRELLIS voxel orientation correction: {degrees} degrees around +{voxel_rotation_axis.lower()}."
        )

    # Convert the aligned voxel to sparse TRELLIS coordinates.
    coords = voxel_to_coords(voxel_for_trellis, device="cuda")
    print(f"Voxel converted to {coords.shape[0]} sparse coordinates.")

    if task_type == "image":
        with torch.no_grad():
            # Load and preprocess image
            img = pipeline.preprocess_image(Image.open(image_path))

            # Get conditioning from image
            cond = pipeline.get_cond([img])

            # Sample structured latent
            print("Sampling structured latent...")
            slat = pipeline.sample_slat(cond, coords)

            # Decode to mesh and gaussian
            print("Decoding to mesh and gaussian...")
            outputs = pipeline.decode_slat(slat, ["mesh", "gaussian"])
    elif task_type == "text":
        with torch.no_grad():
            # Get conditioning from text
            cond = pipeline.get_cond([text_prompt])

            # Sample structured latent
            print("Sampling structured latent...")
            slat = pipeline.sample_slat(cond, coords)

            # Decode to mesh and gaussian
            print("Decoding to mesh and gaussian...")
            outputs = pipeline.decode_slat(slat, ["mesh", "gaussian"])

    # Generate glb file
    print("Generating glb file...")
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=simplify,
        texture_size=texture_size,
        fill_holes=False,
        verbose=True,
    )

    # Save glb file
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"{output_name}.glb")
    glb.export(glb_path)
    print(f"GLB file saved to: {glb_path}")

    return glb_path


def octree_to_voxel(octree, depth: int):
    """Convert octree to voxel representation."""
    try:
        # Get nodes at specified depth
        batch_id = octree.batch_id(depth=depth, nempty=True)
        data = torch.ones((len(batch_id), 1), device=octree.device)

        # Convert to voxel
        voxel_data = octree2voxel(data=data, octree=octree, depth=depth, nempty=True)
        voxel_data = voxel_data.permute(0, 4, 1, 2, 3).contiguous()

        # Get voxel data from the first batch
        voxel = voxel_data[0].squeeze().cpu().numpy()
        return voxel
    except Exception as e:
        print(f"Failed to convert octree to voxel: {e}")
        return None


def restore_token_sequence(token_sequence: str) -> str:
    """Extract space-separated byte values from a mesh token string.

    For example, "<mesh_bos><mesh125><mesh255><mesh_eos>" becomes "125 255".
    Return an empty string if no mesh byte tokens are present.
    """

    if not token_sequence:
        return ""

    # Remove start and end special tokens
    cleaned_sequence = token_sequence.replace("<mesh_bos>", "").replace("<mesh_eos>", "")

    # Use regex to extract all numbers from mesh tokens
    pattern = r"<mesh(\d+)>"
    matches = re.findall(pattern, cleaned_sequence)

    if not matches:
        print("Warning: no valid mesh token found")
        return ""

    # Connect numbers with spaces
    integer_sequence = " ".join(matches)

    return integer_sequence


def validate_token_sequence(token_sequence: str) -> bool:
    """Validate the format of the token sequence.

    Args:
        token_sequence: Token sequence to validate

    Returns:
        True if valid, False otherwise
    """

    # Check if contains start and end tokens
    if not token_sequence.startswith("<mesh_bos>"):
        idx = token_sequence.find("<mesh_bos>")
        if idx == -1:
            print("Error: token sequence missing <mesh_bos>")
            return False
        token_sequence = token_sequence[idx:]

    if not token_sequence.endswith("<mesh_eos>"):
        print("Error: token sequence must end with <mesh_eos>")
        return False

    # Check if the middle part only contains valid mesh tokens
    middle_part = token_sequence[10:-10]  # Remove <mesh_bos> and <mesh_eos>

    # Validate if the middle part only contains <meshXXX> format tokens
    pattern = r"^(<mesh\d+>)*$"
    if not re.match(pattern, middle_part):
        print("Error: token sequence contains invalid token format")
        return False

    return True


def bytes_to_binary_sequence(byte_sequence: List[int]) -> List[int]:
    """Convert integer sequence of 0-255 to 01 binary sequence.

    Args:
        byte_sequence: Byte sequence

    Returns:
        binary_sequence: 01 binary sequence
    """
    binary_sequence = []

    for byte_val in byte_sequence:
        # Convert byte value to 8-bit binary string (big endian)
        binary_str = format(byte_val, "08b")

        # Convert to integer list
        eight_bits = [int(bit) for bit in binary_str]
        binary_sequence.extend(eight_bits)

    return binary_sequence


def binary_to_split_tensor(
    binary_sequence: List[int], shape: tuple = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert binary split values to a tensor, optionally reshaping it.

    Args:
        binary_sequence: Sequence of binary occupancy values.
        shape: Output shape; None leaves the tensor one-dimensional.
        dtype: Output tensor dtype.

    Returns:
        Tensor containing the split values.

    Raises:
        ValueError: The requested shape does not match the sequence length.
    """
    split_tensor = torch.tensor(binary_sequence, dtype=dtype)

    if shape is not None:
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        if total_elements != len(binary_sequence):
            raise ValueError(f"Shape mismatch: expected {total_elements} elements, got {len(binary_sequence)}")

        split_tensor = split_tensor.reshape(shape)

    return split_tensor


def create_template_octree(depth: int, full_depth: int, device: str = "cpu"):
    """Create template octree for reconstruction."""
    try:
        # Create a simple initial octree
        octree = ocnn.octree.init_octree(depth=depth, full_depth=full_depth, batch_size=1, device=device)
        return octree
    except Exception as e:
        print(f"Failed to create template octree: {e}")
        return None


def voxel2mesh(voxel, threshold: float = 0.4, use_vertex_normal: bool = False):
    """Convert voxel to mesh."""
    try:
        verts, faces, vertex_normals = _voxel2mesh(voxel, threshold)
        if use_vertex_normal:
            return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=vertex_normals)
        else:
            return trimesh.Trimesh(vertices=verts, faces=faces)
    except Exception as e:
        print(f"Failed to convert voxel to mesh: {e}")
        return None


def _voxel2mesh(voxels, threshold: float = 0.5):
    """Internal function: implementation of voxel to mesh conversion."""
    # Define the 6 faces of the cube
    top_verts = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    top_faces = [[0, 1, 3], [1, 2, 3]]
    top_normals = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]

    bottom_verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    bottom_faces = [[1, 0, 3], [2, 1, 3]]
    bottom_normals = [[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]]

    left_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    left_faces = [[0, 1, 3], [2, 0, 3]]
    left_normals = [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]

    right_verts = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    right_faces = [[1, 0, 3], [0, 2, 3]]
    right_normals = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

    front_verts = [[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]
    front_faces = [[1, 0, 3], [0, 2, 3]]
    front_normals = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]

    back_verts = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]
    back_faces = [[0, 1, 3], [2, 0, 3]]
    back_normals = [[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]]

    vert_scale = 1.0
    top_verts = np.array(top_verts) * vert_scale
    top_faces = np.array(top_faces)
    bottom_verts = np.array(bottom_verts) * vert_scale
    bottom_faces = np.array(bottom_faces)
    left_verts = np.array(left_verts) * vert_scale
    left_faces = np.array(left_faces)
    right_verts = np.array(right_verts) * vert_scale
    right_faces = np.array(right_faces)
    front_verts = np.array(front_verts) * vert_scale
    front_faces = np.array(front_faces)
    back_verts = np.array(back_verts) * vert_scale
    back_faces = np.array(back_faces)

    dim = voxels.shape[0]
    new_voxels = np.zeros((dim + 2, dim + 2, dim + 2))
    new_voxels[1 : dim + 1, 1 : dim + 1, 1 : dim + 1] = voxels
    voxels = new_voxels

    scale = 2 / dim
    verts = []
    faces = []
    vertex_normals = []
    curr_vert = 0
    a, b, c = np.where(voxels > threshold)

    for i, j, k in zip(a, b, c):
        if voxels[i, j, k + 1] < threshold:
            verts.extend(scale * (top_verts + np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(top_faces + curr_vert)
            vertex_normals.extend(top_normals)
            curr_vert += len(top_verts)

        if voxels[i, j, k - 1] < threshold:
            verts.extend(scale * (bottom_verts + np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(bottom_faces + curr_vert)
            vertex_normals.extend(bottom_normals)
            curr_vert += len(bottom_verts)

        if voxels[i - 1, j, k] < threshold:
            verts.extend(scale * (left_verts + np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(left_faces + curr_vert)
            vertex_normals.extend(left_normals)
            curr_vert += len(left_verts)

        if voxels[i + 1, j, k] < threshold:
            verts.extend(scale * (right_verts + np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(right_faces + curr_vert)
            vertex_normals.extend(right_normals)
            curr_vert += len(right_verts)

        if voxels[i, j + 1, k] < threshold:
            verts.extend(scale * (front_verts + np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(front_faces + curr_vert)
            vertex_normals.extend(front_normals)
            curr_vert += len(front_verts)

        if voxels[i, j - 1, k] < threshold:
            verts.extend(scale * (back_verts + np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(back_faces + curr_vert)
            vertex_normals.extend(back_normals)
            curr_vert += len(back_verts)

    return np.array(verts) - 1, np.array(faces), np.array(vertex_normals)


def save_mesh(mesh, output_path: str, verbose: bool = True):
    """Save mesh to file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        if verbose:
            print(f"Mesh saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save mesh: {e}")
        return False


def main():
    """Decode saved S-Octrees with the same completion and TRELLIS settings as inference."""
    import argparse

    from inference import (
        DEFAULT_CONFIG,
        _complete_voxel,
        _generate_glb,
        _load_settings,
        _octree_tokens_to_voxel,
        extract_mesh_sequence,
    )

    parser = argparse.ArgumentParser(description="Decode saved S-Octree tokens to GLB meshes.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--tokens", type=Path, help="A UTF-8 file containing a <mesh_bos>...<mesh_eos> sequence.")
    inputs.add_argument("--input-dir", type=Path, help="Directory containing one <asset_id>.txt response per asset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("auto", "image", "text"), default="auto")
    parser.add_argument("--prompt", default="", help="Original text condition for a single mesh.")
    parser.add_argument("--image", help="Original condition image for a single mesh.")
    parser.add_argument(
        "--metadata-json", type=Path, help="Batch rows with name/asset_id and images/description fields."
    )
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    args = parser.parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        parser.error("Expected 0 <= rank < world-size and world-size >= 1.")
    if args.input_dir and args.metadata_json is None:
        parser.error("Batch decoding requires --metadata-json with the original conditions.")
    settings = _load_settings(args.config)
    generation = settings.get("mllm", {}).get("generation", {})
    mesh_settings = settings.get("mesh", {})
    metadata = {}
    if args.metadata_json:
        rows = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        metadata = {str(row.get("name", row.get("asset_id", ""))): row for row in rows}
    files = [args.tokens] if args.tokens else sorted(args.input_dir.glob("*.txt"))
    if not files:
        parser.error("No token files found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in files[args.rank :: args.world_size]:
        asset_id = path.stem.removesuffix(".tokens")
        row = metadata.get(asset_id, {})
        images = row.get("images", []) or []
        if isinstance(images, str):
            images = [images]
        if row.get("render_image_path"):
            images = [row["render_image_path"]]
        if args.image:
            images = [args.image]
        prompt = row.get("description") or row.get("text_description") or args.prompt
        sequence = extract_mesh_sequence(path.read_text(encoding="utf-8"))
        if sequence is None:
            raise ValueError(f"No complete S-Octree in {path}")
        voxel = _octree_tokens_to_voxel(
            sequence,
            depth=int(generation.get("max_layer", 6)),
            full_depth=int(generation.get("full_depth", 3)),
            device=str(mesh_settings.get("octree_device", "cpu")),
            threshold=float(mesh_settings.get("octree_threshold", 0.0)),
        )
        voxel = _complete_voxel(voxel, mesh_settings)
        glb = _generate_glb(
            voxel,
            prompt,
            images,
            mesh_settings,
            args.output_dir.resolve(),
            asset_id,
            args.condition,
            None,
            None,
        )
        print(f"GLB: {glb}")


if __name__ == "__main__":
    main()
