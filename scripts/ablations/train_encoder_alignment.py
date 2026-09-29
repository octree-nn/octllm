import argparse
import os
import sys
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Support direct execution from any working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.tensorboard import SummaryWriter

from scripts.model_sources import resolve_trellis_vae_directory

from trellis.data import (
    PairedVoxelDataset,
    PairedVoxelDatasetConfig,
    VoxelDatasetConfig,
    paired_voxel_collate_fn,
)
from trellis.models.sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_dice: float = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align Sparse Structure encoder latents on voxel data.")
    parser.add_argument("--train-input-dir", type=str, required=True, help="Directory containing incomplete voxels for training.")
    parser.add_argument("--train-target-dir", type=str, required=True, help="Directory containing completed voxels for training.")
    parser.add_argument("--val-input-dir", type=str, default=None, help="Optional directory containing incomplete voxels for validation.")
    parser.add_argument("--val-target-dir", type=str, default=None, help="Optional directory containing completed voxels for validation.")
    parser.add_argument("--output-dir", type=str, default="runs/encoder_align", help="Directory to save checkpoints and logs.")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory. Defaults to <output-dir>/logs.")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Frequency (in steps) to write checkpoints.")
    parser.add_argument("--save-best", action="store_true", help="Save best validation checkpoint when enabled.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume.")
    parser.add_argument("--resume-weights-only", action="store_true", help="Only load model weights from checkpoint.")
    parser.add_argument("--pretrained-trellis-dir", type=str, default="microsoft/TRELLIS-image-large",
                        help="Hugging Face TRELLIS repository ID or local pretrained weights directory.")
    parser.add_argument("--max-epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Validation batch size per GPU.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers per GPU.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--encoder-channels", type=int, nargs="+", default=[32, 128, 512], help="Channels for encoder blocks.")
    parser.add_argument("--decoder-channels", type=int, nargs="+", default=None, help="Channels for decoder blocks. Defaults to reversed encoder channels.")
    parser.add_argument("--in-channels", type=int, default=1, help="Input voxel channels.")
    parser.add_argument("--out-channels", type=int, default=1, help="Output voxel channels.")
    parser.add_argument("--latent-channels", type=int, default=8, help="Latent channels.")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="Residual blocks per resolution.")
    parser.add_argument("--num-res-blocks-middle", type=int, default=2, help="Residual blocks in the bottleneck.")
    parser.add_argument("--norm-type", type=str, choices=["layer", "group"], default="layer", help="Normalization type.")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training via autocast and GradScaler.")
    parser.add_argument("--weight-fp16", action="store_true", help="Convert encoder/decoder weights to fp16.")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping max norm.")
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="none", help="LR scheduler type.")
    parser.add_argument("--cosine-min-lr", type=float, default=1e-6, help="Minimum LR for cosine scheduler.")

    parser.add_argument("--binary-threshold", type=float, default=None, help="Optional binarization threshold for voxels.")
    parser.add_argument("--normalize", type=str, choices=["0_1", "neg1_1"], default=None, help="Normalization mode for voxels.")
    parser.add_argument("--no-clamp", action="store_true", help="Disable value clamping to [0, 1] after preprocessing.")
    parser.add_argument("--val-split", type=float, default=0.01, help="Fraction of training pairs reserved for validation when validation directories are not provided.")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="Distributed backend.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Steps to accumulate gradients before optimizer step.")
    parser.add_argument("--log-step", type=int, default=10, help="Log step.")
    return parser.parse_args()


def setup_distributed(backend: str) -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(args: argparse.Namespace, world_size: int):
    if args.train_input_dir is None or args.train_target_dir is None:
        raise ValueError("Both --train-input-dir and --train-target-dir must be provided.")
    if args.val_split is not None and args.val_split < 0:
        raise ValueError("val_split must be non-negative.")

    train_input_cfg = VoxelDatasetConfig(
        root=args.train_input_dir,
        binary_threshold=args.binary_threshold,
        normalize=args.normalize,
        clamp=not args.no_clamp,
    )
    train_target_cfg = VoxelDatasetConfig(
        root=args.train_target_dir,
        binary_threshold=args.binary_threshold,
        normalize=args.normalize,
        clamp=not args.no_clamp,
    )
    base_config = PairedVoxelDatasetConfig(input=train_input_cfg, target=train_target_cfg)
    full_dataset = PairedVoxelDataset(base_config)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_dataset = full_dataset
    val_dataset = None

    if args.val_input_dir or args.val_target_dir:
        if not (args.val_input_dir and args.val_target_dir):
            raise ValueError("Both --val-input-dir and --val-target-dir must be provided together.")
        val_input_cfg = VoxelDatasetConfig(
            root=args.val_input_dir,
            binary_threshold=args.binary_threshold,
            normalize=args.normalize,
            clamp=not args.no_clamp,
        )
        val_target_cfg = VoxelDatasetConfig(
            root=args.val_target_dir,
            binary_threshold=args.binary_threshold,
            normalize=args.normalize,
            clamp=not args.no_clamp,
        )
        val_dataset = PairedVoxelDataset(
            PairedVoxelDatasetConfig(input=val_input_cfg, target=val_target_cfg)
        )
    elif args.val_split and args.val_split > 0.0:
        total_samples = len(full_dataset)
        num_val = max(int(total_samples * args.val_split), 1)
        if num_val >= total_samples:
            raise ValueError("val_split is too large; resulting validation set would be empty or exceed dataset size.")
        permutation = torch.randperm(total_samples, generator=generator)
        val_indices = permutation[:num_val].tolist()
        train_indices = permutation[num_val:].tolist()
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

    rank = dist.get_rank() if world_size > 1 else 0
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=paired_voxel_collate_fn,
    )

    val_loader = None
    if val_dataset is not None:
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            if world_size > 1
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size or args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=paired_voxel_collate_fn,
        )
    return train_loader, val_loader


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


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.cosine_min_lr)
    else:
        scheduler = None
    return optimizer, scheduler


def load_trellis_weights(
    encoder: SparseStructureEncoder,
    decoder: SparseStructureDecoder,
    trellis_dir: str,
    strict: bool = False,
) -> Tuple[int, int]:
    import glob

    trellis_dir = str(resolve_trellis_vae_directory(trellis_dir))
    enc_files = glob.glob(os.path.join(trellis_dir, "ss_enc_conv3d_*.safetensors"))
    dec_files = glob.glob(os.path.join(trellis_dir, "ss_dec_conv3d_*.safetensors"))
    if not enc_files:
        raise FileNotFoundError(f"No encoder safetensors file found in {trellis_dir} matching pattern 'ss_enc_conv3d_*.safetensors'")
    if not dec_files:
        raise FileNotFoundError(f"No decoder safetensors file found in {trellis_dir} matching pattern 'ss_dec_conv3d_*.safetensors'")

    enc_path = enc_files[0]
    dec_path = dec_files[0]

    print(f"Loading TRELLIS encoder weights from: {enc_path}")
    print(f"Loading TRELLIS decoder weights from: {dec_path}")

    enc_weights = load_safetensors(enc_path)
    dec_weights = load_safetensors(dec_path)

    encoder_state = encoder.state_dict()
    decoder_state = decoder.state_dict()

    enc_loaded = 0
    enc_skipped = []
    for key, value in enc_weights.items():
        if key in encoder_state:
            if encoder_state[key].shape == value.shape:
                encoder_state[key] = value
                enc_loaded += 1
            else:
                enc_skipped.append(f"{key} (shape mismatch: model {encoder_state[key].shape} vs pretrained {value.shape})")
        else:
            enc_skipped.append(f"{key} (not in model)")

    dec_loaded = 0
    dec_skipped = []
    for key, value in dec_weights.items():
        if key in decoder_state:
            if decoder_state[key].shape == value.shape:
                decoder_state[key] = value
                dec_loaded += 1
            else:
                dec_skipped.append(f"{key} (shape mismatch: model {decoder_state[key].shape} vs pretrained {value.shape})")
        else:
            dec_skipped.append(f"{key} (not in model)")

    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)

    print(f"Encoder: loaded {enc_loaded}/{len(enc_weights)} weights")
    if enc_skipped:
        print(f"  Skipped encoder keys: {enc_skipped[:5]}{'...' if len(enc_skipped) > 5 else ''}")
    print(f"Decoder: loaded {dec_loaded}/{len(dec_weights)} weights")
    if dec_skipped:
        print(f"  Skipped decoder keys: {dec_skipped[:5]}{'...' if len(dec_skipped) > 5 else ''}")

    if strict and (enc_skipped or dec_skipped):
        raise ValueError(f"Strict loading failed. Encoder skipped: {len(enc_skipped)}, Decoder skipped: {len(dec_skipped)}")
    return enc_loaded, dec_loaded


def restore_checkpoint(
    path: str,
    encoder: torch.nn.Module,
    optimizer,
    scaler: GradScaler,
    state: TrainState,
    weights_only: bool = False,
) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder"])
    if weights_only:
        print("Only load encoder weights, reset optimizer and training state")
    else:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        state.epoch = checkpoint.get("epoch", 0)
        state.global_step = checkpoint.get("global_step", 0)
        state.best_dice = checkpoint.get("best_dice", -1.0)


def save_checkpoint(
    path: str,
    encoder: torch.nn.Module,
    optimizer,
    scaler: Optional[GradScaler],
    state: TrainState,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {
        "encoder": encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_dice": state.best_dice,
    }
    if scaler is not None:
        to_save["scaler"] = scaler.state_dict()
    torch.save(to_save, path)


def reduce_tensor(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


def dice_coefficient(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    smooth = 1e-5
    prob = torch.sigmoid(logits)
    prob_flat = prob.view(prob.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    intersection = (prob_flat * target_flat).sum(dim=1)
    union = prob_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def evaluate(
    encoder: SparseStructureEncoder,
    decoder: SparseStructureDecoder,
    dataloader: DataLoader,
) -> Dict[str, float]:
    if dataloader is None:
        return {}
    encoder.eval()
    decoder.eval()
    total_dice = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Validation batch must contain (inputs, targets).")
            inputs, targets = batch[0], batch[1]
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            _, mean_train, _ = encoder(inputs, sample_posterior=False, return_raw=True)
            recon = decoder(mean_train)
            dice = dice_coefficient(recon, targets)
            total_dice += reduce_tensor(dice.detach()).item()
            total_batches += 1
    if total_batches > 0:
        total_dice /= total_batches
    return {"dice": total_dice}


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed(args.dist_backend)
    set_seed(args.seed + rank)

    is_main_process = rank == 0
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir) if is_main_process else None

    train_loader, val_loader = create_dataloaders(args, world_size)
    torch.cuda.set_device(local_rank)

    encoder_ref = build_encoder(args)
    encoder_train = build_encoder(args)
    decoder_ref = build_decoder(args)

    if is_main_process:
        print(f"Initializing encoder/decoder with TRELLIS pretrained weights from: {args.pretrained_trellis_dir}")
    load_trellis_weights(encoder_ref, decoder_ref, args.pretrained_trellis_dir, strict=True)
    load_trellis_weights(encoder_train, decoder_ref, args.pretrained_trellis_dir, strict=True)
    if is_main_process:
        print("TRELLIS pretrained weights loaded successfully.")

    encoder_ref.cuda()
    encoder_train.cuda()
    decoder_ref.cuda()

    encoder_ref.eval()
    decoder_ref.eval()
    for param in encoder_ref.parameters():
        param.requires_grad = False
    for param in decoder_ref.parameters():
        param.requires_grad = False

    ddp_encoder = DDP(encoder_train, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if world_size > 1 else encoder_train
    module = ddp_encoder.module if isinstance(ddp_encoder, DDP) else ddp_encoder

    optimizer, scheduler = build_optimizer(module, args)
    scaler = GradScaler(enabled=args.amp)
    state = TrainState()

    if args.resume is not None and os.path.isfile(args.resume):
        if is_main_process:
            print(f"Resuming from checkpoint: {args.resume}")
        restore_checkpoint(
            args.resume,
            module,
            optimizer,
            scaler if args.amp else None,
            state,
            weights_only=args.resume_weights_only,
        )

    global_step = state.global_step
    for epoch in range(state.epoch, args.max_epochs):
        state.epoch = epoch
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        ddp_encoder.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Training batch must contain (inputs, targets).")
            inputs, targets = batch[0], batch[1]
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=args.amp, dtype=torch.float16):
                with torch.no_grad():
                    _, mean_ref, _ = encoder_ref(targets, sample_posterior=False, return_raw=True)
                _, mean_train, _ = module(inputs, sample_posterior=False, return_raw=True)
                align_loss = F.mse_loss(mean_train, mean_ref, reduction="mean")

            loss = align_loss / args.gradient_accumulation_steps
            if args.amp:
                scaler.scale(loss).backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(module.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(module.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                state.global_step = global_step
                reduced_loss = reduce_tensor(align_loss.detach()).item()
                if writer is not None and global_step % args.log_step == 0:
                    writer.add_scalar("train/align_loss", reduced_loss, global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    if is_main_process:
                        print(f"Step {global_step} - Align Loss: {reduced_loss:.4f} - Epoch {state.epoch} / {args.max_epochs}")
                if is_main_process and global_step % args.checkpoint_every == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"step_{global_step:07d}.pt")
                    save_checkpoint(checkpoint_path, module, optimizer, scaler if args.amp else None, state)
                    print(f"Saved checkpoint to {checkpoint_path}, epoch {state.epoch}, global_step {global_step}")

        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            if is_main_process:
                print(f"Evaluating on validation set, epoch {state.epoch}, global_step {global_step}")
            val_metrics = evaluate(module, decoder_ref, val_loader)
            if writer is not None:
                for key, value in val_metrics.items():
                    writer.add_scalar(f"val/{key}", value, global_step)
            if is_main_process:
                print(f"Validation Dice: {val_metrics['dice']:.4f}")
            if is_main_process and args.save_best and val_metrics:
                if val_metrics["dice"] > state.best_dice:
                    state.best_dice = val_metrics["dice"]
                    save_checkpoint(os.path.join(args.output_dir, "best.pt"), module, optimizer, scaler if args.amp else None, state)
                    print(f"Saved best checkpoint to {os.path.join(args.output_dir, 'best.pt')}, epoch {state.epoch}, global_step {global_step}")

        if is_main_process:
            latest_path = os.path.join(args.output_dir, "latest.pt")
            save_checkpoint(latest_path, module, optimizer, scaler if args.amp else None, state)
            print(f"Saved latest checkpoint to {latest_path}, epoch {state.epoch}, global_step {global_step}")

    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
