import argparse
import os
import sys
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Add parent directory to path for trellis module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.tensorboard import SummaryWriter

from trellis.data import (
    PairedVoxelDataset,
    PairedVoxelDatasetConfig,
    VoxelDatasetConfig,
    paired_voxel_collate_fn,
)
from trellis.models.sparse_structure_vae import SparseStructureVAE
from trellis.models.sparse_structure_unet import SparseStructureVAE as SparseStructureVAE_UNET


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val: float = float("inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sparse Structure VAE on voxel data.")
    parser.add_argument("--train-input-dir", type=str, required=True, help="Directory containing incomplete voxels for training.")
    parser.add_argument("--train-target-dir", type=str, required=True, help="Directory containing completed voxels for training.")
    parser.add_argument("--val-input-dir", type=str, default=None, help="Optional directory containing incomplete voxels for validation (overrides automatic split).")
    parser.add_argument("--val-target-dir", type=str, default=None, help="Optional directory containing completed voxels for validation (overrides automatic split).")
    parser.add_argument("--output-dir", type=str, default="runs/vae", help="Directory to save checkpoints and logs.")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory. Defaults to <output-dir>/logs.")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Frequency (in steps) to write checkpoints.")
    parser.add_argument("--save-best", action="store_true", help="Save best validation checkpoint when enabled.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume.")
    parser.add_argument("--resume-weights-only", action="store_true", help="Only load model weights from checkpoint, reset optimizer and training state (for fine-tuning on new dataset).")
    parser.add_argument("--max-epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Validation batch size per GPU.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers per GPU.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--model-type", type=str, choices=["vae", "unet"], default="unet", help="Model type.")
    parser.add_argument("--encoder-channels", type=int, nargs="+", default=[64, 128, 256, 256], help="Channels for encoder blocks.")
    parser.add_argument("--decoder-channels", type=int, nargs="+", default=None, help="Channels for decoder blocks. Defaults to reversed encoder channels.")
    parser.add_argument("--in-channels", type=int, default=1, help="Input voxel channels.")
    parser.add_argument("--out-channels", type=int, default=1, help="Output voxel channels.")
    parser.add_argument("--latent-channels", type=int, default=256, help="Latent channels.")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="Residual blocks per resolution.")
    parser.add_argument("--num-res-blocks-middle", type=int, default=2, help="Residual blocks in the bottleneck.")
    parser.add_argument("--norm-type", type=str, choices=["layer", "group"], default="layer", help="Normalization type.")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--recon-loss", type=str, choices=["bce", "dice"], default="dice", help="Reconstruction loss type.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training via autocast and GradScaler.")
    parser.add_argument("--weight-fp16", action="store_true", help="Convert encoder/decoder weights to fp16.")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping max norm.")
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="none", help="LR scheduler type.")
    parser.add_argument("--cosine-min-lr", type=float, default=1e-6, help="Minimum LR for cosine scheduler.")

    parser.add_argument("--binary-threshold", type=float, default=None, help="Optional binarization threshold for voxels.")
    parser.add_argument("--normalize", type=str, choices=["0_1", "neg1_1"], default=None, help="Normalization mode for voxels.")
    parser.add_argument("--no-clamp", action="store_true", help="Disable value clamping to [0, 1] after preprocessing.")
    parser.add_argument("--val-split", type=float, default=0.01, help="Fraction of training pairs reserved for validation when validation directories are not provided.")
    parser.add_argument("--reconstruction-samples", type=int, default=4, help="Number of samples to log for reconstruction preview.")

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


def build_model(args: argparse.Namespace) -> SparseStructureVAE:
    model = SparseStructureVAE(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        latent_channels=args.latent_channels,
        encoder_channels=args.encoder_channels,
        decoder_channels=args.decoder_channels,
        num_res_blocks=args.num_res_blocks,
        num_res_blocks_middle=args.num_res_blocks_middle,
        norm_type=args.norm_type,
        recon_loss=args.recon_loss,
        use_fp16=args.weight_fp16,
    )
    return model


def build_model_unet(args: argparse.Namespace) -> SparseStructureVAE_UNET:
    model = SparseStructureVAE_UNET(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        latent_channels=args.latent_channels,
        encoder_channels=args.encoder_channels,
        decoder_channels=args.decoder_channels,
        num_res_blocks=args.num_res_blocks,
        num_res_blocks_middle=args.num_res_blocks_middle,
        norm_type=args.norm_type,
        recon_loss=args.recon_loss,
        kl_weight=0.0,
        use_fp16=args.weight_fp16,
    )
    return model

def build_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.cosine_min_lr)
    else:
        scheduler = None
    return optimizer, scheduler


def restore_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer,
    scaler: GradScaler,
    state: TrainState,
    weights_only: bool = False,
) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    
    if weights_only:
        # only load model weights, reset optimizer and training state
        # for fine-tuning on new dataset
        print("only load model weights, reset optimizer and training state")
    else:
        # full restore, including optimizer、scaler and training state
        # for continuing training on the same dataset
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        state.epoch = checkpoint.get("epoch", 0)
        state.global_step = checkpoint.get("global_step", 0)
        state.best_val = checkpoint.get("best_val", float("inf"))


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer,
    scaler: Optional[GradScaler],
    state: TrainState,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_val": state.best_val,
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


def evaluate(model: SparseStructureVAE, dataloader: DataLoader, args: argparse.Namespace) -> Dict[str, float]:
    if dataloader is None:
        return {}
    model.eval()
    metrics = {"loss": 0.0, "reconstruction_loss": 0.0}
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Validation batch must contain (inputs, targets).")
            inputs, targets = batch[0], batch[1]
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            _, loss_dict = model.loss(inputs, target=targets, sample_posterior=False, reduction="mean")
            total_batches += 1
            for key in metrics:
                if key in loss_dict:
                    metrics[key] += reduce_tensor(loss_dict[key].detach()).item()
    if total_batches > 0:
        for key in metrics:
            metrics[key] /= total_batches
    return metrics


def log_reconstructions(
    writer: SummaryWriter,
    model: SparseStructureVAE,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    global_step: int,
    num_samples: int,
) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        recon = model.forward(inputs, sample_posterior=False)
        inputs = inputs[:num_samples]
        targets = targets[:num_samples]
        recon = recon[:num_samples]
        recon_prob = torch.sigmoid(recon)
        writer.add_histogram("voxels/input", inputs, global_step=global_step)
        writer.add_histogram("voxels/target", targets, global_step=global_step)
        writer.add_histogram("voxels/reconstruction", recon_prob, global_step=global_step)
        slice_idx = inputs.shape[-1] // 2
        writer.add_images("voxels/slices/input", inputs[:, :, :, :, slice_idx], global_step=global_step)
        writer.add_images("voxels/slices/target", targets[:, :, :, :, slice_idx], global_step=global_step)
        writer.add_images("voxels/slices/recon", recon_prob[:, :, :, :, slice_idx], global_step=global_step)
    if was_training:
        model.train()


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

    if args.model_type == "vae":
        model = build_model(args)
    elif args.model_type == "unet":
        model = build_model_unet(args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    model.cuda()
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if world_size > 1 else model
    module = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model

    optimizer, scheduler = build_optimizer(module, args)
    scaler = GradScaler(enabled=args.amp)
    state = TrainState()

    if args.resume is not None and os.path.isfile(args.resume):
        restore_checkpoint(
            args.resume, 
            module, 
            optimizer, 
            scaler if args.amp else None, 
            state,
            weights_only=args.resume_weights_only
        )

    global_step = state.global_step
    for epoch in range(state.epoch, args.max_epochs):
        state.epoch = epoch
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Training batch must contain (inputs, targets).")
            inputs, targets = batch[0], batch[1]
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            with module.autocast_context(enabled=args.amp):
                total_loss, loss_dict = module.loss(
                    inputs,
                    target=targets,
                    sample_posterior=True,
                    reduction="mean",
                )
            loss = total_loss / args.gradient_accumulation_steps

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
                reduced_metrics: Dict[str, float] = {}
                for name in ("loss", "reconstruction_loss"):
                    metric_tensor = loss_dict[name]
                    if metric_tensor.dim() > 0:
                        metric_tensor = metric_tensor.mean()
                    reduced = reduce_tensor(metric_tensor.detach())
                    reduced_metrics[name] = reduced.item()
                if writer is not None and global_step % args.log_step == 0:
                    writer.add_scalar("train/loss", reduced_metrics["loss"], global_step)
                    writer.add_scalar("train/reconstruction_loss", reduced_metrics["reconstruction_loss"], global_step)
                    # writer.add_scalar("train/kl_loss", reduced_metrics["kl_loss"], global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    if is_main_process:
                        print(f"Step {global_step} - Loss: {reduced_metrics['loss']:.4f}, Reconstruction Loss: {reduced_metrics['reconstruction_loss']:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                if is_main_process and global_step % args.checkpoint_every == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"step_{global_step:07d}.pt")
                    save_checkpoint(checkpoint_path, module, optimizer, scaler if args.amp else None, state)
                    print(f"Saved checkpoint to {checkpoint_path}, epoch {state.epoch}, global_step {global_step}")

        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            if is_main_process:
                print(f"Evaluating on validation set, epoch {state.epoch}, global_step {global_step}")
            val_metrics = evaluate(module, val_loader, args)
            if writer is not None:
                for key, value in val_metrics.items():
                    writer.add_scalar(f"val/{key}", value, global_step)
            if is_main_process and args.save_best and val_metrics:
                if val_metrics["loss"] < state.best_val:
                    state.best_val = val_metrics["loss"]
                    save_checkpoint(os.path.join(args.output_dir, "best.pt"), module, optimizer, scaler if args.amp else None, state)
                    print(f"Saved best checkpoint to {os.path.join(args.output_dir, 'best.pt')}, epoch {state.epoch}, global_step {global_step}")
        if writer is not None:
            sample_iter = iter(train_loader)
            sample_batch = next(sample_iter)
            if not isinstance(sample_batch, (list, tuple)) or len(sample_batch) < 2:
                raise ValueError("Sample batch must contain (inputs, targets).")
            sample_inputs = sample_batch[0][: args.reconstruction_samples].cuda(non_blocking=True)
            sample_targets = sample_batch[1][: args.reconstruction_samples].cuda(non_blocking=True)
            log_reconstructions(writer, module, sample_inputs, sample_targets, global_step, args.reconstruction_samples)
            if is_main_process:
                print(f"Logged {args.reconstruction_samples} reconstruction samples at step {global_step}")
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

