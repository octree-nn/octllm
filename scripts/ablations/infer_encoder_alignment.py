import argparse
import os
import sys
from typing import Optional, Sequence, Tuple

# Support direct execution from any working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

from scripts.model_sources import resolve_trellis_vae_directory

from trellis.data.voxel_dataset import _load_tensor, _postprocess_tensor
from trellis.models.sparse_structure_vae import SparseStructureDecoder, SparseStructureEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for encoder-align model.")
    parser.add_argument("--input-voxel", type=str, required=True, help="Path to input voxel file (.npy/.pt/.binvox).")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save reconstruction (.npy/.pt).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to encoder-align checkpoint (best.pt or latest.pt).")
    parser.add_argument("--pretrained-trellis-dir", type=str, default="microsoft/TRELLIS-image-large", help="Hugging Face TRELLIS repository ID or local pretrained weights directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference.")

    parser.add_argument("--encoder-channels", type=int, nargs="+", default=[32, 128, 512], help="Channels for encoder blocks.")
    parser.add_argument("--decoder-channels", type=int, nargs="+", default=None, help="Channels for decoder blocks. Defaults to reversed encoder channels.")
    parser.add_argument("--in-channels", type=int, default=1, help="Input voxel channels.")
    parser.add_argument("--out-channels", type=int, default=1, help="Output voxel channels.")
    parser.add_argument("--latent-channels", type=int, default=8, help="Latent channels.")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="Residual blocks per resolution.")
    parser.add_argument("--num-res-blocks-middle", type=int, default=2, help="Residual blocks in the bottleneck.")
    parser.add_argument("--norm-type", type=str, choices=["layer", "group"], default="layer", help="Normalization type.")
    parser.add_argument("--weight-fp16", action="store_true", help="Convert encoder/decoder weights to fp16.")
    parser.add_argument("--amp", action="store_true", help="Enable autocast for inference.")

    parser.add_argument("--binary-threshold", type=float, default=None, help="Optional binarization threshold for input voxels.")
    parser.add_argument("--normalize", type=str, choices=["0_1", "neg1_1"], default=None, help="Normalization mode for input voxels.")
    parser.add_argument("--no-clamp", action="store_true", help="Disable value clamping to [0, 1] after preprocessing.")
    parser.add_argument("--enforce-shape", type=int, nargs=3, default=[64, 64, 64], help="Expected voxel shape (D H W).")

    parser.add_argument("--output-mode", type=str, choices=["logits", "prob", "binary"], default="prob", help="Output voxel format.")
    parser.add_argument("--output-threshold", type=float, default=0.5, help="Threshold for binary output mode.")
    return parser.parse_args()


def build_encoder(args: argparse.Namespace) -> SparseStructureEncoder:
    return SparseStructureEncoder(
        in_channels=args.in_channels,
        latent_channels=args.latent_channels,
        num_res_blocks=args.num_res_blocks,
        channels=args.encoder_channels,
        num_res_blocks_middle=args.num_res_blocks_middle,
        norm_type=args.norm_type,
        use_fp16=args.weight_fp16,
    )


def build_decoder(args: argparse.Namespace) -> SparseStructureDecoder:
    decoder_channels = args.decoder_channels or list(reversed(args.encoder_channels))
    return SparseStructureDecoder(
        out_channels=args.out_channels,
        latent_channels=args.latent_channels,
        num_res_blocks=args.num_res_blocks,
        channels=decoder_channels,
        num_res_blocks_middle=args.num_res_blocks_middle,
        norm_type=args.norm_type,
        use_fp16=args.weight_fp16,
    )


def load_trellis_decoder_weights(decoder: SparseStructureDecoder, trellis_dir: str, strict: bool = False) -> None:
    import glob

    trellis_dir = str(resolve_trellis_vae_directory(trellis_dir))
    dec_files = glob.glob(os.path.join(trellis_dir, "ss_dec_conv3d_*.safetensors"))
    if not dec_files:
        raise FileNotFoundError(f"No decoder safetensors file found in {trellis_dir} matching pattern 'ss_dec_conv3d_*.safetensors'")
    dec_path = dec_files[0]
    print(f"Loading TRELLIS decoder weights from: {dec_path}")
    dec_weights = load_safetensors(dec_path)

    decoder_state = decoder.state_dict()
    dec_skipped = []
    for key, value in dec_weights.items():
        if key in decoder_state:
            if decoder_state[key].shape == value.shape:
                decoder_state[key] = value
            else:
                dec_skipped.append(f"{key} (shape mismatch: model {decoder_state[key].shape} vs pretrained {value.shape})")
        else:
            dec_skipped.append(f"{key} (not in model)")
    decoder.load_state_dict(decoder_state)

    if dec_skipped:
        print(f"  Skipped decoder keys: {dec_skipped[:5]}{'...' if len(dec_skipped) > 5 else ''}")
    if strict and dec_skipped:
        raise ValueError(f"Strict loading failed. Decoder skipped: {len(dec_skipped)}")


def load_encoder_checkpoint(encoder: SparseStructureEncoder, checkpoint: str) -> None:
    data = torch.load(checkpoint, map_location="cpu")
    if "encoder" not in data:
        raise KeyError(f"Checkpoint {checkpoint} does not contain 'encoder' weights.")
    encoder.load_state_dict(data["encoder"])


def preprocess_voxel(
    path: str,
    enforce_shape: Optional[Tuple[int, int, int]],
    binary_threshold: Optional[float],
    normalize: Optional[str],
    clamp: bool,
) -> torch.Tensor:
    tensor = _load_tensor(path)
    tensor = _postprocess_tensor(
        tensor,
        enforce_shape,
        torch.float32,
        binary_threshold,
        normalize,
        clamp,
    )
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    return tensor


def save_output(tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = path.split(".")[-1].lower()
    if ext == "pt":
        torch.save(tensor.cpu(), path)
    elif ext == "npy":
        np.save(path, tensor.cpu().numpy())
    else:
        raise ValueError(f"Unsupported output extension: {ext}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but device is set to cuda.")

    encoder = build_encoder(args)
    decoder = build_decoder(args)
    load_encoder_checkpoint(encoder, args.checkpoint)
    load_trellis_decoder_weights(decoder, args.pretrained_trellis_dir, strict=True)

    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    voxel = preprocess_voxel(
        args.input_voxel,
        enforce_shape=tuple(args.enforce_shape) if args.enforce_shape else None,
        binary_threshold=args.binary_threshold,
        normalize=args.normalize,
        clamp=not args.no_clamp,
    ).to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=args.amp, dtype=torch.float16):
            _, mean, _ = encoder(voxel, sample_posterior=False, return_raw=True)
            recon = decoder(mean)

    if args.output_mode == "logits":
        output = recon
    else:
        prob = torch.sigmoid(recon)
        if args.output_mode == "binary":
            output = (prob >= args.output_threshold).to(prob.dtype)
        else:
            output = prob

    save_output(output, args.output_path)
    print(f"Saved output to {args.output_path}")


if __name__ == "__main__":
    main()
