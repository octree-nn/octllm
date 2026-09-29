from contextlib import contextmanager
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.norm import GroupNorm32, ChannelLayerNorm32
from ..modules.spatial import pixel_shuffle_3d
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32


def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    Return a normalization layer.
    """
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
    ):
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif mode == "avgpool":
            assert in_channels == out_channels, "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels*8, 3, padding=1)
        elif mode == "nearest":
            assert in_channels == out_channels, "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        

class SparseStructureEncoder(nn.Module):
    r"""
    Encoder for Sparse Structure (\mathcal{E}_S in the paper Sec. 3.3).
    
    Args:
        in_channels (int): Channels of the input.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the encoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    DownsampleBlock3d(ch, channels[i+1])
                )
        
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1])
            for _ in range(num_res_blocks_middle)
        ])

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels*2, 3, padding=1)
        )

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False) -> torch.Tensor:
        h = self.input_layer(x)
        h = h.type(self.dtype)

        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)

        h = h.type(x.dtype)
        h = self.out_layer(h)

        mean, logvar = h.chunk(2, dim=1)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
            
        if return_raw:
            return z, mean, logvar
        return z
        

class SparseStructureDecoder(nn.Module):
    r"""
    Decoder for Sparse Structure (\mathcal{D}_S in the paper Sec. 3.3).
    
    Args:
        out_channels (int): Channels of the output.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """ 
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0])
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock3d(ch, channels[i+1])
                )

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1)
        )

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        
        h = h.type(self.dtype)
                
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)

        h = h.type(x.dtype)
        h = self.out_layer(h)
        return h


class SparseStructureVAE(nn.Module):
    """
    Sparse Structure VAE wrapper that couples encoder and decoder,
    computes reconstruction/KL losses, and provides AMP helpers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        encoder_channels: Sequence[int],
        decoder_channels: Optional[Sequence[int]] = None,
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        recon_loss: Literal["bce", "dice"] = "dice",
        use_fp16: bool = False,
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = list(reversed(encoder_channels))
        if len(encoder_channels) == 0:
            raise ValueError("encoder_channels must be a non-empty sequence.")
        if len(decoder_channels) == 0:
            raise ValueError("decoder_channels must be a non-empty sequence.")

        self.recon_loss = recon_loss
        self.use_amp = use_fp16
        self._latent_shape: Optional[Tuple[int, int, int]] = None

        self.encoder = SparseStructureEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            channels=list(encoder_channels),
            num_res_blocks_middle=num_res_blocks_middle,
            norm_type=norm_type,
            use_fp16=use_fp16,
        )
        self.decoder = SparseStructureDecoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            channels=list(decoder_channels),
            num_res_blocks_middle=num_res_blocks_middle,
            norm_type=norm_type,
            use_fp16=use_fp16,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(
        self, x: torch.Tensor, sample_posterior: bool = True, return_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        z, mean, logvar = self.encoder(x, sample_posterior=sample_posterior, return_raw=True)
        self._latent_shape = tuple(mean.shape[2:])
        if return_stats:
            return z, mean, logvar
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        return_latent: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        z, mean, logvar = self.encoder(x, sample_posterior=sample_posterior, return_raw=True)
        self._latent_shape = tuple(mean.shape[2:])
        recon = self.decoder(z)
        if return_latent:
            return recon, z, mean, logvar
        return recon

    def loss(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        sample_posterior: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        recon, z, mean, logvar = self.forward(x, sample_posterior=sample_posterior, return_latent=True)
        target_tensor = x if target is None else target
        target_tensor = target_tensor.to(recon.dtype)
        recon_loss_map = self._reconstruction_loss(recon, target_tensor)
        recon_loss_per_example = recon_loss_map.view(recon_loss_map.shape[0], -1).mean(dim=1)
        
        # Occupancy completion uses reconstruction loss without KL regularization.

        recon_loss_value = self._apply_reduction(recon_loss_per_example, reduction)
        total_loss = recon_loss_value 

        loss_dict: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "reconstruction_loss": recon_loss_value,
            "reconstruction_loss_per_example": recon_loss_per_example.detach(),
        }
        return total_loss, loss_dict

    def _reconstruction_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.recon_loss == "bce":
            loss = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
        elif self.recon_loss == "dice":
            prediction = F.sigmoid(prediction)
            pred_flat = prediction.view(prediction.shape[0], -1)
            target_flat = target.view(target.shape[0], -1)
            intersection = (pred_flat * target_flat).sum(dim=1)
            loss = 1 - (2 * intersection + 1) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + 1)
        else:
            raise ValueError(f"Unsupported reconstruction loss type: {self.recon_loss}")
        return loss

    @staticmethod
    def _kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        return kl.view(kl.shape[0], -1).sum(dim=1)

    @staticmethod
    def _apply_reduction(values: torch.Tensor, reduction: Literal["mean", "sum", "none"]) -> torch.Tensor:
        if reduction == "mean":
            return values.mean()
        if reduction == "sum":
            return values.sum()
        if reduction == "none":
            return values
        raise ValueError(f"Unsupported reduction: {reduction}")

    @contextmanager
    def autocast_context(
        self,
        enabled: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        enable_amp = self.use_amp if enabled is None else enabled
        if dtype is None:
            if device_type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.bfloat16
        with torch.autocast(device_type=device_type, enabled=enable_amp, dtype=dtype):
            yield

    def convert_to_fp16(self) -> None:
        self.use_amp = True
        self.encoder.convert_to_fp16()
        self.decoder.convert_to_fp16()

    def convert_to_fp32(self) -> None:
        self.use_amp = False
        self.encoder.convert_to_fp32()
        self.decoder.convert_to_fp32()

    @torch.inference_mode()
    def sample(
        self,
        num_samples: int,
        temperature: float = 1.0,
        latent_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        latent_shape = latent_shape or self._latent_shape
        if latent_shape is None:
            raise ValueError("latent_shape must be provided before any encoding has been performed.")
        z = torch.randn(
            num_samples,
            self.encoder.latent_channels,
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            device=self.device,
        )
        z = z * temperature
        return self.decode(z)
