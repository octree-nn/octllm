from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, List, Sequence, Union, Dict
from transformers import AutoTokenizer
from transformers.integrations import is_deepspeed_zero3_enabled
import torch
import torch.nn.functional as F
import transformers
# Import original model class
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel, Qwen2_5_VLModelOutputWithPast

from .bytes_to_split import bytes_to_binary_sequence, binary_to_split_tensor
from .incremental_octree_state import OctreeDecodeMetadata, OctreeTokenPosition
from .utils.utils import seq2octree
from .models.octformer import OctreeT
from .split_to_position_embedding import split_to_next_position_embedding
from .lora_token_mask_patch import enable_lora_token_mask_patch, set_lora_token_mask
from .qwen25_vl_3d_router import set_3d_route_mask
import ocnn
import os

from .models.positional_embedding import DepthPosEmb
from .model_utils.checkpoint import load_safetensors_weight_map

# Keep a reference to the original method so we can fall back when needed
_ORIGINAL_GET_ROPE_INDEX = Qwen2_5_VLModel.get_rope_index

# global position embedding model cache, avoid duplicate creation
_pos_emb_model_cache = {}
_pos_emb_loaded_state_dicts: Dict[int, dict] = {}

MeshTokenMetadata = list[list[dict[str, torch.Tensor]]]


@dataclass
class Qwen2_5_VL3DCausalLMOutputWithPast(
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLCausalLMOutputWithPast
):
    mesh_loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    mesh_loss_token_count: Optional[torch.LongTensor] = None
    text_loss_token_count: Optional[torch.LongTensor] = None


enable_lora_token_mask_patch()


def load_pos_emb_weights(checkpoint_dir: str, strict: bool = False) -> bool:
    """Load depth position embeddings from the complete model checkpoint."""
    models = _load_depth_pos_emb_state_from_safetensors(checkpoint_dir)
    if not models:
        return False
    _cache_loaded_depth_pos_emb_models(models, strict=strict)
    return True


def _cache_loaded_depth_pos_emb_models(models: dict, strict: bool = False) -> None:
    _pos_emb_loaded_state_dicts.clear()
    for num_embed, state_dict in models.items():
        # ensure tensors are on CPU, will be moved to device later
        _pos_emb_loaded_state_dicts[int(num_embed)] = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in state_dict.items()
        }

    # apply loaded weights to existing cached models (if created)
    for (num_embed, device), model in list(_pos_emb_model_cache.items()):
        to_load = _pos_emb_loaded_state_dicts.get(num_embed)
        if to_load:
            try:
                model.load_state_dict(to_load, strict=strict)
                print(f"loaded DepthPosEmb weights applied to cache model: num_embed={num_embed}, device={device}")
            except Exception as e:
                print(f"loaded weights applied to DepthPosEmb(num_embed={num_embed}) failed: {e}")


def _load_depth_pos_emb_state_from_safetensors(checkpoint_dir: str) -> dict[int, dict[str, torch.Tensor]]:
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return {}

    try:
        from safetensors import safe_open
    except ImportError:
        print("safetensors is not available; cannot recover DepthPosEmb weights from model shards.")
        return {}

    weight_map = load_safetensors_weight_map(checkpoint_dir)
    if not weight_map:
        return {}

    models: dict[int, dict[str, torch.Tensor]] = {}
    for source_key, shard_name in weight_map.items():
        target_key = _depth_pos_emb_key_from_safetensors_key(source_key)
        if target_key is None:
            continue

        shard_path = os.path.join(checkpoint_dir, shard_name)
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                tensor = shard.get_tensor(source_key)
        except Exception as exc:
            print(f"failed to read DepthPosEmb tensor `{source_key}` from `{shard_path}`: {exc}")
            continue

        if tensor.ndim < 2:
            print(f"skip invalid DepthPosEmb tensor `{source_key}` with shape {tuple(tensor.shape)}")
            continue

        num_embed = int(tensor.shape[-1])
        models.setdefault(num_embed, {})[target_key] = tensor

    return models


def _depth_pos_emb_key_from_safetensors_key(source_key: str) -> str | None:
    marker = "mesh_abs_pos_emb."
    idx = source_key.rfind(marker)
    if idx < 0:
        return None

    target_key = source_key[idx + len(marker):]
    return target_key or None


def _is_zero3_partitioned_param(param: torch.nn.Parameter | torch.Tensor) -> bool:
    return hasattr(param, "ds_id") or hasattr(param, "ds_status")


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


@contextmanager
def _maybe_gather_zero3_params(
    params: Sequence[torch.nn.Parameter | torch.Tensor],
    *,
    modifier_rank: int | None = None,
):
    zero3_params = [param for param in params if _is_zero3_partitioned_param(param)]
    if not zero3_params or not is_deepspeed_zero3_enabled():
        yield
        return

    try:
        import deepspeed
    except ImportError:
        yield
        return

    with deepspeed.zero.GatheredParameters(zero3_params, modifier_rank=modifier_rank):
        yield

def get_or_create_pos_emb_model(num_embed: int, device: str, full_depth: int = 3, max_depth: int = 6) -> Optional[torch.nn.Module]:
    """
    get or create position embedding model, use cache to avoid duplicate creation
    """

    cache_key = (num_embed, device)

    if cache_key not in _pos_emb_model_cache:
        try:
            pos_emb_model = DepthPosEmb(num_embed=num_embed, full_depth=full_depth, max_depth=max_depth)
            pos_emb_model.to(device)
            # if there is a loaded state_dict, apply it
            state_dict = _pos_emb_loaded_state_dicts.get(num_embed)
            if state_dict:
                try:
                    pos_emb_model.load_state_dict(state_dict, strict=False)
                    print(f"loaded DepthPosEmb weights applied to DepthPosEmb(num_embed={num_embed})")
                except Exception as e:
                    print(f"loaded DepthPosEmb weights applied to DepthPosEmb(num_embed={num_embed}) failed: {e}")
            _pos_emb_model_cache[cache_key] = pos_emb_model
        except Exception as e:
            print(f"created position embedding model failed: {e}")
            return None

    return _pos_emb_model_cache[cache_key]


def _average_incremental_octree_position(
    position: Optional[OctreeTokenPosition],
    pos_emb_model: torch.nn.Module,
    device: Union[str, torch.device],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Materialize the old eight-node mean from compact incremental metadata."""

    if position is None:
        return None, None

    # Do not replace this with a direct embedding-row lookup. The legacy path
    # averages eight identical float32 rows, and that reduction is not always
    # bitwise equal to the original row before the result is cast to bf16/fp16.
    depth_indices = torch.full(
        (8,),
        int(position.depth_index),
        dtype=torch.long,
        device=device,
    )
    averaged_embedding = pos_emb_model.depth_emb(depth_indices).mean(dim=0)
    averaged_xyz = torch.tensor(position.xyz, dtype=torch.long, device=device)
    return averaged_embedding, averaged_xyz


def _resolve_incremental_octree_positions(
    metadata: OctreeDecodeMetadata,
    pos_emb_model: torch.nn.Module,
    device: Union[str, torch.device],
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Return values in ``split_to_next_position_embedding`` order."""

    averaged_position_embedding, averaged_xyz = _average_incremental_octree_position(
        metadata.next_position,
        pos_emb_model,
        device,
    )
    averaged_pre_position_embedding, averaged_pre_xyz = _average_incremental_octree_position(
        metadata.previous_position,
        pos_emb_model,
        device,
    )
    return (
        averaged_position_embedding,
        averaged_xyz,
        averaged_pre_position_embedding,
        averaged_pre_xyz,
    )

def clear_pos_emb_model_cache():
    """
    clear the position embedding model cache
    """
    global _pos_emb_model_cache
    _pos_emb_model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def attach_pos_emb_to_model(model: torch.nn.Module, hidden_size: int, device: Union[str, torch.device], full_depth: int = 3, max_depth: int = 6) -> bool:
    """
    attach DepthPosEmb as a submodule to the given model, so that its parameters are automatically discovered and trained by the optimizer.
    the module name is fixed as `mesh_abs_pos_emb`.
    return whether the attachment is successful.
    """
    try:
        cache_key = (hidden_size, str(device))
        existing = getattr(model, "mesh_abs_pos_emb", None)
        if isinstance(existing, torch.nn.Module):
            _pos_emb_model_cache[cache_key] = existing
            return True

        pos = get_or_create_pos_emb_model(hidden_size, str(device), full_depth, max_depth)
        if pos is None:
            return False
        setattr(model, "mesh_abs_pos_emb", pos)
        return True
    except Exception as e:
        print(f"attach DepthPosEmb to model failed: {e}")
        return False


def _find_mesh_segments(
    input_tokens: List[int],
    bos_id: int,
    eos_id: int,
    start_index: int = 0,
) -> List[Tuple[int, int]]:
    """
    Find non-overlapping [bos, eos] segments in a token list.

    Returns a list of (bos_idx, eos_idx) pairs. Indices refer to the provided input_tokens list.
    """
    segments: List[Tuple[int, int]] = []
    idx = start_index
    n = len(input_tokens)
    while idx < n:
        try:
            bos_idx = input_tokens.index(bos_id, idx)
        except ValueError:
            break
        try:
            eos_idx = input_tokens.index(eos_id, bos_idx + 1)
        except ValueError:
            # No closing eos found after bos; stop searching further
            break
        segments.append((bos_idx, eos_idx))
        idx = eos_idx + 1
    return segments


def _as_long_on_device(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(t, dtype=like.dtype, device=like.device)


def _build_mesh_suppressed_4d_attention_mask(
    attention_mask: Optional[torch.Tensor],
    inserted_mask_token_positions: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build an inverted additive 4D causal mask that hides previous inserted <MASK> keys.

    The returned mask uses 0.0 for allowed attention and dtype min for blocked attention,
    matching Qwen2.5-VL's expected 4D attention mask format.
    """
    if inserted_mask_token_positions.dim() != 2:
        raise ValueError(
            "inserted_mask_token_positions must have shape (batch_size, seq_len), "
            f"got {tuple(inserted_mask_token_positions.shape)}."
        )

    batch_size, seq_len = inserted_mask_token_positions.shape
    device = inserted_mask_token_positions.device

    if attention_mask is None:
        attention_mask_2d = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    else:
        if attention_mask.dim() == 4:
            raise ValueError(
                "Mesh <MASK> suppression expects a 2D attention_mask before expansion. "
                "Disable block_diag_attn/neat_packing for this path or build a dedicated 4D merge."
            )
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be 2D or 4D, got {attention_mask.dim()}D.")
        if attention_mask.shape != inserted_mask_token_positions.shape:
            raise ValueError(
                "attention_mask and inserted_mask_token_positions must have the same shape, "
                f"got {tuple(attention_mask.shape)} and {tuple(inserted_mask_token_positions.shape)}."
            )
        attention_mask_2d = attention_mask.to(device)

    min_dtype = torch.finfo(dtype).min
    query_positions = torch.arange(seq_len, device=device).view(seq_len, 1)
    key_positions = torch.arange(seq_len, device=device).view(1, seq_len)

    causal_allowed = key_positions <= query_positions
    key_is_valid = attention_mask_2d.ne(0).view(batch_size, 1, seq_len)
    historical_inserted_mask_key = (
        inserted_mask_token_positions.bool().view(batch_size, 1, seq_len)
        & (key_positions < query_positions).view(1, seq_len, seq_len)
    )

    allowed = causal_allowed.view(1, seq_len, seq_len) & key_is_valid & ~historical_inserted_mask_key
    zero = torch.zeros((), dtype=dtype, device=device)
    min_value = torch.full((), min_dtype, dtype=dtype, device=device)
    return torch.where(allowed.unsqueeze(1), zero, min_value)


def _create_template_octree(depth: int, full_depth: int, device: Union[str, torch.device] = "cpu"):
    """
    Lightweight wrapper to create an empty Octree template used for reconstruction.
    """
    return ocnn.octree.init_octree(depth=depth, full_depth=full_depth, batch_size=1, device=str(device))


def _tokens_to_mesh_bytes(tokens: Sequence[str]) -> List[int]:
    """
    Parse tokens like "<mesh0>" ... "<mesh255>" into a list of integers in [0, 255].
    Ignores tokens that do not match the mesh pattern.
    """
    import re

    byte_vals: List[int] = []
    pattern = re.compile(r"<mesh(\d+)>")
    for tk in tokens:
        if not isinstance(tk, str):
            continue
        m = pattern.search(tk)
        if m is None:
            # tolerate leading spaces/subwords like '▁<mesh12>'
            tk_stripped = tk.replace("▁", "")
            m = pattern.search(tk_stripped)
            if m is None:
                continue
        val = int(m.group(1))
        if 0 <= val <= 255:
            byte_vals.append(val)
    return byte_vals


def _get_mesh_token_id_to_byte_map(tokenizer: AutoTokenizer) -> dict[int, int]:
    cache_attr = "_llamafactory_mesh_token_id_to_byte_map"
    cached = getattr(tokenizer, cache_attr, None)
    if isinstance(cached, dict):
        return cached

    id_to_byte: dict[int, int] = {}
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for byte_value in range(256):
        token_id = tokenizer.convert_tokens_to_ids(f"<mesh{byte_value}>")
        if token_id is None:
            continue

        token_id = int(token_id)
        if token_id < 0 or (unk_token_id is not None and token_id == int(unk_token_id)):
            continue

        id_to_byte[token_id] = byte_value

    try:
        setattr(tokenizer, cache_attr, id_to_byte)
    except Exception:
        pass

    return id_to_byte


def _mesh_inner_ids_to_byte_values(inner_ids: Sequence[int], tokenizer: AutoTokenizer) -> List[int]:
    id_to_byte = _get_mesh_token_id_to_byte_map(tokenizer)
    byte_vals: List[int] = []
    for token_id in inner_ids:
        byte_value = id_to_byte.get(int(token_id))
        if byte_value is None:
            inner_tokens = tokenizer.convert_ids_to_tokens(inner_ids)
            return _tokens_to_mesh_bytes(inner_tokens)

        byte_vals.append(byte_value)

    return byte_vals


def _build_mesh_segment_metadata_from_byte_values(
    byte_vals: Sequence[int],
    *,
    full_depth: int,
    max_depth: int,
    threshold: float,
    device: Union[str, torch.device],
) -> dict[str, torch.Tensor]:
    if len(byte_vals) == 0:
        return {
            "xyz": torch.zeros(0, 3, dtype=torch.long, device=device),
            "depth_idx": torch.zeros(0, dtype=torch.long, device=device),
        }

    binary_seq = bytes_to_binary_sequence(byte_vals)
    split_tensor = binary_to_split_tensor(binary_seq, dtype=torch.float32).to(device)
    octree_template = _create_template_octree(depth=max_depth, full_depth=full_depth, device=device)
    octree = seq2octree(octree_template, split_tensor, full_depth, max_depth, threshold)
    if octree is None:
        raise RuntimeError("seq2octree failed to reconstruct octree from split tensor")

    octree_t = OctreeT(octree=octree)
    xyz = octree_t.xyz.view(-1, 8, 3).mean(dim=1).div(2)
    xyz_long = torch.round(xyz).to(dtype=torch.long, device=device)
    depth_idx = octree_t.depth_idx.to(dtype=torch.long, device=device)
    return {"xyz": xyz_long, "depth_idx": depth_idx}


def build_mesh_token_metadata_from_input_ids(
    input_ids: torch.LongTensor,
    tokenizer: AutoTokenizer,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    depth: int = 6,
    full_depth: int = 3,
    threshold: float = 0.0,
    device: Optional[Union[str, torch.device]] = None,
) -> MeshTokenMetadata:
    if device is None:
        device = input_ids.device

    mesh_bos_id = tokenizer.convert_tokens_to_ids("<mesh_bos>")
    mesh_eos_id = tokenizer.convert_tokens_to_ids("<mesh_eos>")
    if mesh_bos_id is None or mesh_eos_id is None:
        return [[] for _ in range(input_ids.shape[0] if input_ids.dim() > 1 else 1)]

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    batch_metadata: MeshTokenMetadata = []
    for i, sample_input_ids in enumerate(input_ids):
        if attention_mask is not None:
            valid_mask = attention_mask[i].to(sample_input_ids.device) == 1
            filtered_ids = sample_input_ids[valid_mask]
        else:
            filtered_ids = sample_input_ids

        ids_list = filtered_ids.tolist()
        mesh_segments = _find_mesh_segments(ids_list, mesh_bos_id, mesh_eos_id, start_index=0)
        sample_metadata: list[dict[str, torch.Tensor]] = []
        for bos_idx, eos_idx in mesh_segments:
            inner_ids = ids_list[bos_idx + 1 : eos_idx]
            byte_vals = _mesh_inner_ids_to_byte_values(inner_ids, tokenizer)
            sample_metadata.append(
                _build_mesh_segment_metadata_from_byte_values(
                    byte_vals,
                    full_depth=full_depth,
                    max_depth=depth,
                    threshold=threshold,
                    device=device,
                )
            )

        batch_metadata.append(sample_metadata)

    return batch_metadata


def _get_mesh_token_metadata_segment(
    mesh_token_metadata: Optional[MeshTokenMetadata],
    batch_idx: int,
    segment_idx: int,
    device: Union[str, torch.device],
) -> Optional[dict[str, torch.Tensor]]:
    if mesh_token_metadata is None:
        return None
    if batch_idx >= len(mesh_token_metadata):
        raise ValueError("mesh_token_metadata batch size does not match input_ids batch size")

    sample_metadata = mesh_token_metadata[batch_idx]
    if segment_idx >= len(sample_metadata):
        raise ValueError("mesh_token_metadata segment count does not match detected mesh segments")

    metadata = sample_metadata[segment_idx]
    xyz = metadata.get("xyz")
    depth_idx = metadata.get("depth_idx")
    if not isinstance(xyz, torch.Tensor) or not isinstance(depth_idx, torch.Tensor):
        raise ValueError("mesh_token_metadata entries must contain tensor fields `xyz` and `depth_idx`")

    return {
        "xyz": xyz.to(device=device, dtype=torch.long),
        "depth_idx": depth_idx.to(device=device, dtype=torch.long),
    }


def _build_mesh_embedding_block(
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    old_pos: int,
    per_token_emb: torch.Tensor,
    mask_token_embed: torch.Tensor,
    sample_add_mask_token: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inner_len = per_token_emb.shape[0]
    mask_token_embed = mask_token_embed.reshape(-1, inputs_embeds.shape[-1])[0]
    mesh_token_embeds = inputs_embeds[old_pos : old_pos + inner_len] + per_token_emb
    if attention_mask is None:
        source_attention_mask = torch.ones(inner_len, dtype=torch.long, device=inputs_embeds.device)
    else:
        source_attention_mask = attention_mask[old_pos : old_pos + inner_len]

    if sample_add_mask_token:
        block = torch.empty(
            inner_len * 2,
            inputs_embeds.shape[-1],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        block[0::2] = mask_token_embed.unsqueeze(0) + per_token_emb
        block[1::2] = mesh_token_embeds
        block_attention_mask = source_attention_mask.repeat_interleave(2)
        inserted_mask_positions = torch.zeros(inner_len * 2, dtype=torch.bool, device=inputs_embeds.device)
        inserted_mask_positions[0::2] = True
    else:
        block = mesh_token_embeds + (mask_token_embed.unsqueeze(0) * 0.0)
        block_attention_mask = source_attention_mask
        inserted_mask_positions = torch.zeros(inner_len, dtype=torch.bool, device=inputs_embeds.device)

    block_route_mask = torch.ones(block.shape[0], dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    return block, block_attention_mask, block_route_mask, inserted_mask_positions


def _expand_mesh_xyz_for_rope(
    xyz: torch.Tensor,
    like: torch.Tensor,
    *,
    sample_add_mask_token: bool,
) -> torch.Tensor:
    xyz_long = _as_long_on_device(xyz, like).t().contiguous()
    if sample_add_mask_token:
        return xyz_long.repeat_interleave(2, dim=1)

    return xyz_long


def build_mesh_xyz_list_from_input_ids(
    input_ids: torch.LongTensor,
    tokenizer: AutoTokenizer,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    depth: int = 6,
    full_depth: int = 3,
    threshold: float = 0.0,
    device: Optional[Union[str, torch.device]] = None,
) -> List[List[torch.Tensor]]:
    """
    Infer mesh_xyz_list for each batch sample by scanning input_ids between
    mesh BOS/EOS tokens and reconstructing octree xyz using the split sequence
    derived from mesh byte tokens.

    Returns: list (per-batch) of lists (per-mesh-segment), where each inner
    tensor has shape (num_mesh_tokens, 3) and dtype = long.
    """
    if device is None:
        device = input_ids.device

    mesh_bos_id = tokenizer.convert_tokens_to_ids("<mesh_bos>")
    mesh_eos_id = tokenizer.convert_tokens_to_ids("<mesh_eos>")

    # Normalize to batch-first
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    batch_mesh_xyz: List[List[torch.Tensor]] = []

    for i, sample_input_ids in enumerate(input_ids):
        if attention_mask is not None:
            valid_mask = attention_mask[i] == 1
            valid_mask = valid_mask.to(sample_input_ids.device)
            filtered_ids = sample_input_ids[valid_mask]
        else:
            filtered_ids = sample_input_ids

        # locate mesh segments by id for robustness
        ids_list = filtered_ids.tolist()
        mesh_segments = _find_mesh_segments(ids_list, mesh_bos_id, mesh_eos_id, start_index=0)

        # Collect xyz per segment
        per_sample_xyz_list: List[torch.Tensor] = []
        for (bos_idx, eos_idx) in mesh_segments:
            inner_ids = ids_list[bos_idx + 1 : eos_idx]
            byte_vals = _mesh_inner_ids_to_byte_values(inner_ids, tokenizer)
            if len(byte_vals) == 0:
                # empty segment -> empty xyz
                per_sample_xyz_list.append(torch.zeros(0, 3, dtype=torch.long, device=device))
                continue

            metadata = _build_mesh_segment_metadata_from_byte_values(
                byte_vals,
                full_depth=full_depth,
                max_depth=depth,
                threshold=threshold,
                device=device,
            )
            per_sample_xyz_list.append(metadata["xyz"])

        batch_mesh_xyz.append(per_sample_xyz_list)

    return batch_mesh_xyz


def get_rope_index_with_mesh(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    # New optional parameter for mesh support
    tokenizer: Optional[AutoTokenizer] = None,
    mesh_xyz_list: Optional[Union[List[torch.Tensor], List[List[torch.Tensor]]]] = None,
    add_mask_token: bool = True,
    add_mask_token_flags: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extended get_rope_index with 3D octree (mesh) token support.

    Mesh handling rules:
    - Detect segments delimited by `mesh_bos_token_id` and `mesh_eos_token_id` (on self.config).
    - For tokens strictly between BOS and EOS, use provided xyz as 3D position IDs.
    - BOS/EOS themselves are treated as text positions (1D replicated to 3D).
    - The global position index space remains strictly increasing, similar to images/videos.

    If no mesh inputs exist, behavior falls back to the original implementation.
    """
    # If no mesh positions passed explicitly, try instance attribute (optional convenience)
    if mesh_xyz_list is None and hasattr(self, "mesh_xyz_list"):
        mesh_xyz_list = getattr(self, "mesh_xyz_list")

    # Will we attempt mesh handling?
    if tokenizer is None:
        raise ValueError("tokenizer is required to parse <mesh*> tokens from input_ids")

    mesh_bos_id = tokenizer.convert_tokens_to_ids("<mesh_bos>")
    mesh_eos_id = tokenizer.convert_tokens_to_ids("<mesh_eos>")
    enable_mesh = mesh_bos_id is not None and mesh_eos_id is not None and (
        mesh_xyz_list is not None
    )

    # Fast path: if mesh is not enabled, defer fully to the original implementation
    if not enable_mesh:
        return _ORIGINAL_GET_ROPE_INDEX(
            self,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )

    # Below replicates original logic, extended with mesh segments
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id


    if input_ids is None:
        # Original function requires input_ids to derive indices when multimodal; stay consistent
        return _ORIGINAL_GET_ROPE_INDEX(
            self,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )

    add_mask_token_flags_tensor: Optional[torch.Tensor] = None
    if add_mask_token_flags is not None:
        add_mask_token_flags_tensor = add_mask_token_flags.to(input_ids.device).bool()

    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)

    # Calculate expanded sequence length (including MASK tokens for mesh)
    batch_size = input_ids.shape[0]
    original_seq_len = input_ids.shape[1]

    # Count mesh tokens per sample to determine max expanded length
    max_expanded_len = original_seq_len
    if enable_mesh:
        for i in range(batch_size):
            filtered_ids = input_ids[i]
            ids_list = filtered_ids.tolist()
            mesh_segments = _find_mesh_segments(ids_list, mesh_bos_id, mesh_eos_id, start_index=0)
            sample_mesh_tokens = sum(eos_idx - bos_idx - 1 for (bos_idx, eos_idx) in mesh_segments)
            sample_add_mask_token = add_mask_token and (add_mask_token_flags_tensor is None or bool(add_mask_token_flags_tensor[i]))
            sample_expanded_len = len(ids_list) + (sample_mesh_tokens if sample_add_mask_token else 0)
            max_expanded_len = max(max_expanded_len, sample_expanded_len)

    expanded_seq_len = max_expanded_len

    # Allocate output tensor with expanded size
    position_ids = torch.ones(
        3,
        batch_size,
        expanded_seq_len,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    mrope_position_deltas: List[torch.Tensor] = []
    image_index, video_index = 0, 0
    attention_mask = attention_mask.to(total_input_ids.device)

    for i, sample_input_ids in enumerate(total_input_ids):
        # Filter out padding tokens like original
        valid_mask = attention_mask[i] == 1
        filtered_ids = sample_input_ids[valid_mask]

        # Count vision placeholders using vision_start markers (keep parity with upstream)
        vision_start_indices = torch.argwhere(filtered_ids == vision_start_token_id).squeeze(1)
        vision_tokens = filtered_ids[vision_start_indices + 1] if vision_start_indices.numel() > 0 else torch.empty(0, dtype=filtered_ids.dtype, device=filtered_ids.device)
        image_nums = (vision_tokens == image_token_id).sum().item()
        video_nums = (vision_tokens == video_token_id).sum().item()

        # Prepare Python list for index operations
        input_tokens: List[int] = filtered_ids.tolist()
        sample_add_mask_token = add_mask_token and (add_mask_token_flags_tensor is None or bool(add_mask_token_flags_tensor[i]))

        # Discover mesh segments in the filtered token list
        mesh_segments = _find_mesh_segments(input_tokens, mesh_bos_id, mesh_eos_id, start_index=0)

        # Select per-sample xyz list if batch-style provided
        sample_mesh_xyz: Optional[List[torch.Tensor]] = None
        if enable_mesh:
            if isinstance(mesh_xyz_list, (list, tuple)) and len(mesh_xyz_list) > 0 and isinstance(mesh_xyz_list[0], (list, tuple)):
                sample_mesh_xyz = mesh_xyz_list[i]
            else:
                sample_mesh_xyz = mesh_xyz_list  # treat as shared for all samples

        # Quick validations
        if len(mesh_segments) > 0 and (sample_mesh_xyz is None or len(sample_mesh_xyz) != len(mesh_segments)):
            raise ValueError(
                f"mesh segments detected ({len(mesh_segments)}), but provided xyz segments mismatch ({0 if sample_mesh_xyz is None else len(sample_mesh_xyz)})."
            )

        remain_images, remain_videos = image_nums, video_nums
        mesh_seg_idx = 0

        llm_pos_ids_list: List[torch.Tensor] = []
        st = 0

        # Walk through the sequence, interleaving image/video placeholders and mesh segments
        while st < len(input_tokens):
            # Next image or video placeholder positions after current st
            if remain_images > 0 and image_token_id in input_tokens:
                try:
                    ed_image = input_tokens.index(image_token_id, st)
                except ValueError:
                    ed_image = len(input_tokens) + 1
            else:
                ed_image = len(input_tokens) + 1

            if remain_videos > 0 and video_token_id in input_tokens:
                try:
                    ed_video = input_tokens.index(video_token_id, st)
                except ValueError:
                    ed_video = len(input_tokens) + 1
            else:
                ed_video = len(input_tokens) + 1

            # Next mesh BOS after current st
            if mesh_seg_idx < len(mesh_segments):
                next_bos, next_eos = mesh_segments[mesh_seg_idx]
                ed_mesh = next_bos if next_bos >= st else len(input_tokens) + 1
            else:
                ed_mesh = len(input_tokens) + 1

            # Determine next event among image, video, mesh-bos
            ed = min(ed_image, ed_video, ed_mesh)
            if ed >= len(input_tokens):
                # No more events, all remaining text
                text_len = len(input_tokens) - st
                if text_len > 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1).to(input_ids.device) + st_idx)
                break

            # Emit text block before the event
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            if text_len > 0:
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1).to(input_ids.device) + st_idx)

            # Handle the event type
            if ed == ed_image:
                # Image block: like upstream, t=1 for image (temporal = 1), use grid thw
                t = image_grid_thw[image_index][0]
                h = image_grid_thw[image_index][1]
                w = image_grid_thw[image_index][2]
                image_index += 1
                remain_images -= 1

                llm_grid_t = t.item()
                llm_grid_h = (h.item() // spatial_merge_size)
                llm_grid_w = (w.item() // spatial_merge_size)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                t_index = expanded_range.flatten().long()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]).to(input_ids.device) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            elif ed == ed_video:
                # Video block: follow upstream implementation (temporal step scaled by tokens_per_second)
                t = video_grid_thw[video_index][0]
                h = video_grid_thw[video_index][1]
                w = video_grid_thw[video_index][2]
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0
                video_index += 1
                remain_videos -= 1

                llm_grid_t = t.item()
                llm_grid_h = (h.item() // spatial_merge_size)
                llm_grid_w = (w.item() // spatial_merge_size)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                second_per_grid_t = torch.as_tensor(second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device)
                time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                t_index = time_tensor.long().flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]).to(input_ids.device) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            else:
                # Mesh block delimited by BOS/EOS; assign xyz to interior tokens only
                # Note: each mesh token now has a MASK token before it in inputs_embeds
                bos_idx, eos_idx = mesh_segments[mesh_seg_idx]
                if bos_idx != ed:
                    # Sanity check; should not happen
                    raise RuntimeError("Internal mesh scan mismatch: BOS index does not match next event.")
                interior_len = max(eos_idx - bos_idx - 1, 0)
                xyz = sample_mesh_xyz[mesh_seg_idx]
                if xyz is None:
                    raise ValueError(f"mesh_xyz_list[{mesh_seg_idx}] is None.")
                if xyz.dim() != 2 or xyz.size(-1) != 3:
                    raise ValueError(
                        f"mesh_xyz_list[{mesh_seg_idx}] must have shape (num_tokens, 3), got {tuple(xyz.shape)}."
                    )
                if interior_len != xyz.size(0):
                    raise ValueError(
                        f"Mesh segment length ({interior_len}) does not match xyz rows ({xyz.size(0)})."
                    )

                # Assign BOS as a text-style single token right before interior
                bos_block = torch.arange(1, device=filtered_ids.device, dtype=filtered_ids.dtype).view(1, -1).expand(3, -1).to(input_ids.device)
                llm_pos_ids_list.append(bos_block + text_len + st_idx)

                # xyz -> (3, L), then shift by current global st_idx + text_len + 1 (after BOS)
                if interior_len > 0:
                    mesh_positions = _expand_mesh_xyz_for_rope(
                        xyz,
                        filtered_ids,
                        sample_add_mask_token=sample_add_mask_token,
                    )
                    llm_pos_ids_list.append(mesh_positions + text_len + st_idx + 1)

                # Assign EOS as a text-style single token after interior
                st_idx_after = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else (text_len + st_idx + 1 + interior_len)
                eos_block = torch.arange(1, device=filtered_ids.device, dtype=filtered_ids.dtype).view(1, -1).expand(3, -1).to(input_ids.device)
                llm_pos_ids_list.append(eos_block + st_idx_after)

                # Advance beyond EOS in the token index space
                st = eos_idx + 1
                mesh_seg_idx += 1

        # Stitch positions and assign back like upstream
        if len(llm_pos_ids_list) == 0:
            # No content; create zeros of appropriate shape
            llm_positions = torch.zeros(3, 0, dtype=position_ids.dtype, device=position_ids.device)
        else:
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

        # Assign to position_ids (llm_positions already has the expanded length)
        actual_len = llm_positions.shape[1]
        position_ids[..., i, :actual_len] = llm_positions.to(position_ids.device)

        # For padding positions (if any), keep them as 1 (default value)
        # This handles cases where different samples have different expansion amounts

        # The mesh-aware path stores the absolute next RoPE position rather than
        # upstream Qwen's `max_position + 1 - sequence_length` delta. Inserted
        # MASK tokens and 3D coordinates make cache length an unsuitable source
        # for this offset during mesh generation.
        mrope_position_deltas.append(llm_positions.max() + 1)

    mrope_position_deltas_tensor = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas_tensor

def forward_qwen25_vl_model_with_mesh_extraction(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    ### Mesh related parameters
    tokenizer: Optional[AutoTokenizer] = None,
    is_train: bool = False,
    full_depth: int = 3,
    max_depth: int = 6,
    current_binary_sequence: Optional[torch.Tensor] = None,
    octree_decode_metadata: Optional[OctreeDecodeMetadata] = None,
    add_mask_token: bool = True,
    add_mask_token_flags: Optional[torch.Tensor] = None,
    mesh_token_metadata: Optional[MeshTokenMetadata] = None,
) -> Union[Tuple, Qwen2_5_VLModelOutputWithPast]:
    r"""
    pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
        The tensors corresponding to the input videos. Pixel values can be obtained using
        [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
        [`Qwen2_5_VLImageProcessor`] for processing videos.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    add_mask_token_flags_tensor: Optional[torch.Tensor] = None
    if input_ids is not None:
        batch_for_flags = input_ids.shape[0]
        if add_mask_token_flags is not None:
            add_mask_token_flags_tensor = add_mask_token_flags.to(input_ids.device).bool()
            if add_mask_token_flags_tensor.shape[0] != batch_for_flags:
                raise ValueError("add_mask_token_flags must match batch size")
        else:
            add_mask_token_flags_tensor = torch.full(
                (batch_for_flags,),
                bool(add_mask_token),
                device=input_ids.device,
                dtype=torch.bool,
            )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)
    elif input_ids is not None:
        attention_mask = torch.ones(
            input_ids.shape,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
    if attention_mask is not None and attention_mask.dim() != 2:
        raise ValueError(
            "qwen25_vl_3d_replace expects a 2D attention_mask so it can expand mesh <MASK> tokens "
            "and compute mesh RoPE positions. Disable block_diag_attn/neat_packing for this path."
        )

    # Insert MASK tokens before each mesh token and inject position embeddings
    # Save original attention_mask before expansion (needed for get_rope_index_with_mesh)
    original_attention_mask = attention_mask.clone() if attention_mask is not None else None
    lora_token_mask_tensor: Optional[torch.Tensor] = None
    inserted_mask_token_positions: Optional[torch.Tensor] = None
    has_inserted_mask_token = False
    mesh_xyz_list_for_rope: Optional[List[List[torch.Tensor]]] = None

    if (tokenizer is not None and input_ids is not None and (past_key_values is None or past_key_values.get_seq_length() == 0)) or is_train:
        mesh_bos_id = tokenizer.convert_tokens_to_ids("<mesh_bos>")
        mesh_eos_id = tokenizer.convert_tokens_to_ids("<mesh_eos>")
        mask_token_id = tokenizer.convert_tokens_to_ids("<MASK>")

        if (mesh_bos_id is not None) and (mesh_eos_id is not None) and (mask_token_id is not None):
            hidden_size = inputs_embeds.shape[-1]
            device_str = str(inputs_embeds.device)
            pos_model = get_or_create_pos_emb_model(hidden_size, device_str, full_depth=full_depth, max_depth=max_depth)
            if pos_model is not None:
                batch_size = input_ids.shape[0]
                add_mask_token_flags_tensor = add_mask_token_flags_tensor.to(inputs_embeds.device) if add_mask_token_flags_tensor is not None else add_mask_token_flags_tensor
                mask_counts_all = []
                # First pass: count total MASK tokens to insert per batch
                mask_counts = []
                for b in range(batch_size):
                    ids_b = input_ids[b].tolist()
                    segments = _find_mesh_segments(ids_b, mesh_bos_id, mesh_eos_id, start_index=0)
                    total_mask = sum(eos_idx - bos_idx - 1 for bos_idx, eos_idx in segments)
                    mask_counts_all.append(total_mask)
                    sample_add_mask_token = add_mask_token and (add_mask_token_flags_tensor is None or bool(add_mask_token_flags_tensor[b]))
                    mask_counts.append(total_mask if sample_add_mask_token else 0)


                max_mask_count_all = max(mask_counts_all) if mask_counts_all else 0
                max_mask_count = max(mask_counts) if mask_counts else 0
                if max_mask_count_all > 0: # NOTE: if depth embedding is used, we still need to go through this process
                    has_inserted_mask_token = max_mask_count > 0
                    # Expand inputs_embeds and attention_mask to accommodate MASK tokens
                    batch_size, seq_len, hidden_size = inputs_embeds.shape
                    new_seq_len = seq_len + max_mask_count
                    mesh_xyz_list_for_rope = [[] for _ in range(batch_size)]
                    new_inputs_embeds = torch.zeros(batch_size, new_seq_len, hidden_size,
                                                    dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                    if attention_mask is not None:
                        new_attention_mask = torch.zeros(batch_size, new_seq_len,
                                                         dtype=attention_mask.dtype, device=attention_mask.device)
                    else:
                        new_attention_mask = torch.ones(batch_size, new_seq_len,
                                                       dtype=torch.long, device=inputs_embeds.device)
                    new_lora_token_mask = torch.zeros(
                        batch_size,
                        new_seq_len,
                        dtype=inputs_embeds.dtype,
                        device=inputs_embeds.device,
                    )
                    new_inserted_mask_token_positions = torch.zeros(
                        batch_size,
                        new_seq_len,
                        dtype=torch.bool,
                        device=inputs_embeds.device,
                    )

                    # Get MASK token embedding
                    mask_token_embed = self.get_input_embeddings()(torch.tensor([mask_token_id], device=inputs_embeds.device))

                    # Second pass: insert MASK tokens and add depth embeddings
                    for b in range(batch_size):
                        ids_b = input_ids[b].tolist()
                        segments = _find_mesh_segments(ids_b, mesh_bos_id, mesh_eos_id, start_index=0)
                        sample_add_mask_token = add_mask_token and (add_mask_token_flags_tensor is None or bool(add_mask_token_flags_tensor[b]))

                        # Build mapping from old positions to new positions
                        new_pos = 0
                        old_pos = 0

                        if len(segments) == 0:
                            # No mesh segments, keep all tokens as is
                            new_inputs_embeds[b, :seq_len] = inputs_embeds[b]
                            if attention_mask is not None:
                                new_attention_mask[b, :seq_len] = attention_mask[b]
                            new_lora_token_mask[b, :seq_len] = 0.0
                            continue

                        for seg_idx, (bos_idx, eos_idx) in enumerate(segments):
                            # Process octree for depth embedding
                            inner_ids = ids_b[bos_idx + 1:eos_idx]
                            if len(inner_ids) == 0:
                                raise ValueError("Inner ids are empty")

                            segment_metadata = _get_mesh_token_metadata_segment(
                                mesh_token_metadata,
                                b,
                                seg_idx,
                                inputs_embeds.device,
                            )
                            if segment_metadata is None:
                                byte_vals = _mesh_inner_ids_to_byte_values(inner_ids, tokenizer)
                                if len(byte_vals) == 0:
                                    raise ValueError("Byte values are empty")
                                segment_metadata = _build_mesh_segment_metadata_from_byte_values(
                                    byte_vals,
                                    full_depth=full_depth,
                                    max_depth=max_depth,
                                    threshold=0.0,
                                    device=inputs_embeds.device,
                                )

                            xyz_long = segment_metadata["xyz"]
                            mesh_xyz_list_for_rope[b].append(xyz_long)
                            depth_idx = segment_metadata["depth_idx"]
                            depth_emb = pos_model.depth_emb(depth_idx)  # (N_pos, hidden)
                            if depth_emb.dim() != 2 or depth_emb.size(-1) != hidden_size:
                                raise ValueError(f"Depth embedding has invalid shape: {depth_emb.shape}")
                            Npos = depth_emb.size(0)
                            inner_len = eos_idx - bos_idx - 1
                            # Map N_pos to inner_len tokens
                            if Npos % 8 == 0 and (Npos // 8) == inner_len:
                                per_token_emb = depth_emb.view(-1, 8, hidden_size)[:, 0, :] # pick the first token of each 8 tokens
                            else:
                                raise ValueError(f"Depth embedding has invalid shape: {depth_emb.shape} for length {inner_len}")
                            per_token_emb = per_token_emb.to(inputs_embeds.dtype)

                            # Copy tokens before mesh segment
                            copy_len = bos_idx - old_pos
                            new_inputs_embeds[b, new_pos:new_pos + copy_len] = inputs_embeds[b, old_pos:bos_idx]
                            if attention_mask is not None:
                                new_attention_mask[b, new_pos:new_pos + copy_len] = attention_mask[b, old_pos:bos_idx]
                            new_pos += copy_len
                            old_pos = bos_idx

                            # Copy BOS token
                            new_inputs_embeds[b, new_pos] = inputs_embeds[b, old_pos]
                            if attention_mask is not None:
                                new_attention_mask[b, new_pos] = attention_mask[b, old_pos]
                            new_lora_token_mask[b, new_pos] = 0.0
                            new_pos += 1
                            old_pos += 1

                            mesh_block, mesh_block_attention, mesh_block_route, mesh_block_inserted = (
                                _build_mesh_embedding_block(
                                    inputs_embeds[b],
                                    attention_mask[b] if attention_mask is not None else None,
                                    old_pos=old_pos,
                                    per_token_emb=per_token_emb,
                                    mask_token_embed=mask_token_embed[0],
                                    sample_add_mask_token=sample_add_mask_token,
                                )
                            )
                            block_len = mesh_block.shape[0]
                            new_inputs_embeds[b, new_pos:new_pos + block_len] = mesh_block
                            new_attention_mask[b, new_pos:new_pos + block_len] = mesh_block_attention
                            new_lora_token_mask[b, new_pos:new_pos + block_len] = mesh_block_route
                            new_inserted_mask_token_positions[b, new_pos:new_pos + block_len] = mesh_block_inserted
                            new_pos += block_len
                            old_pos += inner_len

                            # Copy EOS token
                            new_inputs_embeds[b, new_pos] = inputs_embeds[b, old_pos]
                            if attention_mask is not None:
                                new_attention_mask[b, new_pos] = attention_mask[b, old_pos]
                            new_lora_token_mask[b, new_pos] = 0.0
                            new_pos += 1
                            old_pos += 1

                        # Copy remaining tokens
                        if old_pos < seq_len:
                            copy_len = seq_len - old_pos
                            new_inputs_embeds[b, new_pos:new_pos + copy_len] = inputs_embeds[b, old_pos:]
                            if attention_mask is not None:
                                new_attention_mask[b, new_pos:new_pos + copy_len] = attention_mask[b, old_pos:]
                            new_lora_token_mask[b, new_pos:new_pos + copy_len] = 0.0

                    # Update inputs_embeds and attention_mask
                    inputs_embeds = new_inputs_embeds
                    attention_mask = new_attention_mask
                    lora_token_mask_tensor = new_lora_token_mask
                    inserted_mask_token_positions = new_inserted_mask_token_positions
                else:
                    lora_token_mask_tensor = torch.zeros(
                        batch_size,
                        inputs_embeds.shape[1],
                        dtype=inputs_embeds.dtype,
                        device=inputs_embeds.device,
                    )
            else:
                raise ValueError("pos_model is None")

    # calculate RoPE index once per generation in the pre-fill stage only
    if (
        (cache_position is not None and cache_position[0] == 0)
        or self.rope_deltas is None
        or (past_key_values is None or past_key_values.get_seq_length() == 0)
        or is_train
    ):
        # Use original_attention_mask (before MASK token expansion) for mesh xyz extraction and RoPE calculation
        mask_for_rope = original_attention_mask if original_attention_mask is not None else attention_mask
        if mesh_xyz_list_for_rope is None:
            mesh_xyz_list = build_mesh_xyz_list_from_input_ids(
                input_ids,
                tokenizer,
                attention_mask=mask_for_rope,
                depth=max_depth,
                full_depth=full_depth,
                threshold=0.0,
                device=inputs_embeds.device,
            )
        else:
            mesh_xyz_list = mesh_xyz_list_for_rope
        position_ids, rope_deltas = get_rope_index_with_mesh(
            self,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            mask_for_rope,
            tokenizer,
            mesh_xyz_list,
            add_mask_token=add_mask_token,
            add_mask_token_flags=add_mask_token_flags_tensor,
        )
        self.rope_deltas = rope_deltas
    # then use the prev pre-calculated rope-deltas to get the correct position ids
    else:
        batch_size, seq_length, _ = inputs_embeds.shape
        # `self.rope_deltas` is the absolute next RoPE position on this custom
        # path (see get_rope_index_with_mesh), not the relative delta used by
        # upstream Qwen2.5-VL. Do not add cache_position again here.
        delta = (
            (self.rope_deltas).to(inputs_embeds.device)
            if cache_position is not None
            else 0
        )
        if inputs_embeds.shape[1] == 2 and (
            octree_decode_metadata is not None or current_binary_sequence is not None
        ):
            pos_emb_model = get_or_create_pos_emb_model(
                inputs_embeds.shape[2],
                str(inputs_embeds.device),
                full_depth=full_depth,
                max_depth=max_depth,
            )
            if octree_decode_metadata is not None:
                res = (
                    _resolve_incremental_octree_positions(
                        octree_decode_metadata,
                        pos_emb_model,
                        inputs_embeds.device,
                    )
                    if pos_emb_model is not None
                    else None
                )
            else:
                res = split_to_next_position_embedding(
                    current_binary_sequence,
                    depth=max_depth,
                    full_depth=full_depth,
                    threshold=0.0,
                    device=str(inputs_embeds.device),
                    pos_emb_model=pos_emb_model,
                )
            averaged_position_embedding = None
            averaged_xyz = None
            averaged_pre_position_embedding = None
            averaged_pre_xyz = None
            if res is not None:
                try:
                    averaged_position_embedding, averaged_xyz, averaged_pre_position_embedding, averaged_pre_xyz = res
                except Exception:
                    # Defensive: unexpected return structure
                    averaged_position_embedding, averaged_xyz, averaged_pre_position_embedding, averaged_pre_xyz = None, None, None, None

            # Cast embeddings to match inputs_embeds dtype/device when available
            if averaged_position_embedding is not None:
                averaged_position_embedding = averaged_position_embedding.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
            if averaged_pre_position_embedding is not None:
                averaged_pre_position_embedding = averaged_pre_position_embedding.to(inputs_embeds.device, dtype=inputs_embeds.dtype)

            # Case A: we have both previous and next token information
            if (averaged_pre_position_embedding is not None) and (averaged_pre_xyz is not None) and (averaged_xyz is not None):
                # Build position_ids of shape (3, B, 2)
                pos_pair = torch.stack([
                    averaged_pre_xyz.to(inputs_embeds.device),
                    averaged_xyz.to(inputs_embeds.device)
                ], dim=1)  # [3, 2]
                position_ids = pos_pair.unsqueeze(1).expand(-1, batch_size, -1)  # [3, B, 2]
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)  # broadcast delta: [B,1] -> [3,B,2]

                # Add depth embeddings token-wise: build [B,2,H]
                add_pair = torch.stack([
                    averaged_pre_position_embedding,
                    averaged_position_embedding
                ], dim=0).unsqueeze(0).expand(batch_size, -1, -1)
                inputs_embeds = inputs_embeds + add_pair

                lora_token_mask_tensor = torch.ones(batch_size, 2, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

            # Case B: only have next token info -> use 1D pos for first token, 3D for second
            elif (averaged_xyz is not None) and (averaged_position_embedding is not None):
                base = torch.arange(seq_length - 1, device=inputs_embeds.device)
                base = base.view(1, -1).expand(batch_size, -1)
                pos_1d = base.unsqueeze(0).expand(3, -1, -1)  # [3, B, 1]

                pos_last = averaged_xyz.to(inputs_embeds.device).view(3, 1, 1).expand(-1, batch_size, -1)  # [3,B,1]
                position_ids = torch.cat([pos_1d, pos_last], dim=2)  # [3, B, 2]
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)

                # Add depth embedding to the last token only
                inputs_embeds[:, -1] = inputs_embeds[:, -1] + averaged_position_embedding

                lora_token_mask_tensor = torch.ones(batch_size, 2, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                lora_token_mask_tensor[:, 0] = 0.0

            # Case C: no 3D info available -> fallback to pure sequential 1D replicated to 3 channels
            else:
                position_ids = torch.arange(seq_length, device=inputs_embeds.device, dtype=delta.dtype)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

                lora_token_mask_tensor = torch.zeros(batch_size, 2, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        else:
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            # Advance the absolute next position for ordinary one-token text
            # decoding. Mesh decoding computes its coordinates in the branches
            # above instead of deriving them from the KV-cache length.
            self.rope_deltas = self.rope_deltas + 1

            lora_token_mask_tensor = torch.zeros(batch_size, 1, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

    lora_token_mask_tensor = lora_token_mask_tensor.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
    route_3d_token_mask_tensor = lora_token_mask_tensor.detach().to(dtype=torch.bool)
    language_attention_mask = attention_mask
    if inserted_mask_token_positions is not None and has_inserted_mask_token:
        attn_implementation = getattr(
            getattr(self.language_model, "config", self.config),
            "_attn_implementation",
            None,
        )
        if attn_implementation in {"flash_attention_2", "flex_attention"}:
            raise ValueError(
                "Mesh <MASK> suppression builds a 4D additive attention mask, which is only supported "
                "on the eager/sdpa Qwen2.5-VL attention paths. Disable flash/flex attention for 3D SFT."
            )
        language_attention_mask = _build_mesh_suppressed_4d_attention_mask(
            attention_mask,
            inserted_mask_token_positions,
            inputs_embeds.dtype,
        )


    # Set token mask directly on LoRA layers (more reliable than ContextVar in distributed training)
    # NOTE: Do NOT clear mask after forward! Gradient checkpointing will re-run forward during
    # backward pass, and the mask must remain available. The mask will be overwritten on the
    # next forward call anyway.
    set_lora_token_mask(self.language_model, lora_token_mask_tensor)
    set_3d_route_mask(self.language_model, route_3d_token_mask_tensor)
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=language_attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


def replace_qwen25_vl_model_forward_with_mesh_extraction(
    tokenizer: Optional[AutoTokenizer] = None,
    is_train: bool = False,
    full_depth: int = 3,
    max_depth: int = 6,
    add_mask_token: bool = True,
):
    def wrapped_forward(self, *args, **kwargs):
        kwargs['tokenizer'] = tokenizer
        kwargs['is_train'] = is_train
        kwargs['full_depth'] = full_depth
        kwargs['max_depth'] = max_depth
        # current_binary_sequence is passed directly via kwargs from caller
        # If not provided, default to None
        kwargs.setdefault('current_binary_sequence', None)
        kwargs.setdefault('octree_decode_metadata', None)
        kwargs.setdefault('add_mask_token', add_mask_token)
        kwargs.setdefault('add_mask_token_flags', None)
        return forward_qwen25_vl_model_with_mesh_extraction(self, *args, **kwargs)

    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = wrapped_forward

    # Also update _old_forward if it exists (used by accelerate hooks)
    if hasattr(transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel, '_old_forward'):
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._old_forward = wrapped_forward


def _expand_labels_with_mesh_masks(
    labels: Optional[torch.LongTensor],
    input_ids: Optional[torch.LongTensor],
    tokenizer: Optional[AutoTokenizer],
    add_mask_token: bool = True,
    add_mask_token_flags: Optional[torch.Tensor] = None,
    loss_mode: str = "text_only", # text_only or text_and_mesh
) -> Optional[torch.LongTensor]:
    """
    Expand labels by inserting -100 before every mesh token strictly between <mesh_bos> and <mesh_eos>.

    The expansion mirrors the sequence length increase in Qwen2_5_VLModel.forward where a <MASK> token is
    inserted before each mesh token. We do not train the model to generate <MASK>, thus we insert -100 as
    a placeholder label, which is ignored by the loss function.

    Args:
        labels: (batch, seq_len) long tensor or None.
        input_ids: (batch, seq_len) long tensor used to locate mesh segments.
        tokenizer: tokenizer to resolve <mesh_bos> and <mesh_eos> ids.

    Returns:
        expanded_labels: (batch, new_seq_len) long tensor, padded with -100 so that
        new_seq_len == seq_len + max_mesh_tokens_in_batch.
    """
    if labels is None or input_ids is None or tokenizer is None:
        return labels

    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    batch_size, seq_len = labels.shape[0], labels.shape[1]
    add_mask_token_flags_list: Optional[List[bool]] = None
    if add_mask_token_flags is not None:
        if isinstance(add_mask_token_flags, torch.Tensor):
            add_mask_token_flags_list = add_mask_token_flags.detach().cpu().view(-1).tolist()
        else:
            add_mask_token_flags_list = list(add_mask_token_flags)  # pyright: ignore[reportUnreachable]
        if len(add_mask_token_flags_list) != batch_size:
            raise ValueError("add_mask_token_flags must match batch size")

    mesh_bos_id = tokenizer.convert_tokens_to_ids("<mesh_bos>")
    mesh_eos_id = tokenizer.convert_tokens_to_ids("<mesh_eos>")
    if mesh_bos_id is None or mesh_eos_id is None:
        return labels

    # Count mesh tokens inside BOS/EOS per sample to compute max expansion
    per_sample_extra_counts: list[int] = []
    has_mesh_segment = False
    for b in range(batch_size):
        ids_b = input_ids[b].tolist()
        segments = _find_mesh_segments(ids_b, mesh_bos_id, mesh_eos_id, start_index=0)
        interior_count = sum(max(eos_idx - bos_idx - 1, 0) for bos_idx, eos_idx in segments)
        sample_add_mask_token = add_mask_token and (add_mask_token_flags_list is None or bool(add_mask_token_flags_list[b]))
        per_sample_extra_counts.append(interior_count if sample_add_mask_token else 0)
        if interior_count > 0:
            has_mesh_segment = True

    max_extra = max(per_sample_extra_counts) if per_sample_extra_counts else 0
    if max_extra == 0 and not has_mesh_segment:
        return labels

    new_seq_len = seq_len + max_extra
    expanded = labels.new_full((batch_size, new_seq_len), fill_value=-100)

    for b in range(batch_size):
        ids_b = input_ids[b].tolist()
        segments = _find_mesh_segments(ids_b, mesh_bos_id, mesh_eos_id, start_index=0)
        sample_add_mask_token = add_mask_token and (add_mask_token_flags_list is None or bool(add_mask_token_flags_list[b]))

        # Build a boolean map for interior positions to know where to insert -100
        interior_pos = [False] * seq_len
        for bos_idx, eos_idx in segments:
            for pos in range(bos_idx + 1, max(min(eos_idx, seq_len), bos_idx + 1)):
                interior_pos[pos] = True

        write_pos = 0
        bos_indices = [bos for bos, _ in segments]
        for pos in range(seq_len):
            label_val = labels[b, pos]
            if ids_b[pos] == mesh_eos_id:
                label_val = -100

            # label conversion based on loss_mode requirements
            if loss_mode == "text_only":
                # After each <mesh_bos> (excluding itself), set to -100
                if any(pos > bos for bos in bos_indices):
                    label_val = -100
            elif loss_mode == "text_and_mesh":
                # Before each <mesh_bos> (excluding itself), set to -100
                if any(pos < bos for bos in bos_indices):
                    label_val = -100

            if interior_pos[pos] and sample_add_mask_token:
                # Insert one -100 before the current mesh token's label
                expanded[b, write_pos] = -100
                write_pos += 1
            expanded[b, write_pos] = label_val
            write_pos += 1
        # Remaining tail positions (if any) stay as -100 by initialization

    return expanded


def _get_mesh_byte_token_ids(tokenizer: AutoTokenizer) -> set[int]:
    mesh_token_ids = set()
    for idx in range(256):
        token_id = tokenizer.convert_tokens_to_ids(f"<mesh{idx}>")
        if token_id is not None:
            mesh_token_ids.add(int(token_id))

    return mesh_token_ids


def _normalize_add_mask_token_flags(
    add_mask_token_flags: Optional[torch.Tensor],
    batch_size: int,
) -> Optional[list[bool]]:
    if add_mask_token_flags is None:
        return None

    if isinstance(add_mask_token_flags, torch.Tensor):
        flags = add_mask_token_flags.detach().cpu().view(-1).tolist()
    else:
        flags = list(add_mask_token_flags)  # pyright: ignore[reportUnreachable]

    if len(flags) != batch_size:
        raise ValueError("add_mask_token_flags must match batch size")

    return [bool(flag) for flag in flags]


def _expand_labels_for_mesh_eval_scopes(
    labels: Optional[torch.LongTensor],
    input_ids: Optional[torch.LongTensor],
    tokenizer: Optional[AutoTokenizer],
    add_mask_token: bool = True,
    add_mask_token_flags: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.LongTensor], Optional[torch.LongTensor]]:
    """
    Build two expanded label tensors for 3D evaluation.

    mesh labels keep only mesh byte tokens inside <mesh_bos>...</mesh_eos>. text labels keep only non-mesh
    supervised tokens. Inserted <MASK> positions and mesh boundary tokens are always ignored.
    """
    if labels is None or input_ids is None or tokenizer is None:
        return None, None

    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    batch_size, seq_len = labels.shape[0], labels.shape[1]
    add_mask_token_flags_list = _normalize_add_mask_token_flags(add_mask_token_flags, batch_size)

    mesh_bos_id = tokenizer.convert_tokens_to_ids("<mesh_bos>")
    mesh_eos_id = tokenizer.convert_tokens_to_ids("<mesh_eos>")
    if mesh_bos_id is None or mesh_eos_id is None:
        return labels.new_full(labels.shape, fill_value=-100), labels.clone()

    mesh_token_ids = _get_mesh_byte_token_ids(tokenizer)

    per_sample_extra_counts: list[int] = []
    has_mesh_segment = False
    for b in range(batch_size):
        ids_b = input_ids[b].tolist()
        segments = _find_mesh_segments(ids_b, mesh_bos_id, mesh_eos_id, start_index=0)
        interior_count = sum(max(eos_idx - bos_idx - 1, 0) for bos_idx, eos_idx in segments)
        sample_add_mask_token = add_mask_token and (
            add_mask_token_flags_list is None or bool(add_mask_token_flags_list[b])
        )
        per_sample_extra_counts.append(interior_count if sample_add_mask_token else 0)
        has_mesh_segment = has_mesh_segment or interior_count > 0

    max_extra = max(per_sample_extra_counts) if per_sample_extra_counts else 0
    new_seq_len = seq_len + max_extra
    mesh_labels = labels.new_full((batch_size, new_seq_len), fill_value=-100)
    text_labels = labels.new_full((batch_size, new_seq_len), fill_value=-100)

    for b in range(batch_size):
        ids_b = input_ids[b].tolist()
        segments = _find_mesh_segments(ids_b, mesh_bos_id, mesh_eos_id, start_index=0)
        sample_add_mask_token = add_mask_token and (
            add_mask_token_flags_list is None or bool(add_mask_token_flags_list[b])
        )
        last_mesh_eos_pos = max((eos_idx for _bos_idx, eos_idx in segments), default=-1)

        interior_pos = [False] * seq_len
        for bos_idx, eos_idx in segments:
            upper = min(eos_idx, seq_len)
            for pos in range(bos_idx + 1, max(upper, bos_idx + 1)):
                interior_pos[pos] = True

        write_pos = 0
        for pos in range(seq_len):
            token_id = int(ids_b[pos])
            label_val = labels[b, pos]

            if interior_pos[pos] and sample_add_mask_token:
                write_pos += 1

            if label_val != -100:
                if interior_pos[pos] and token_id in mesh_token_ids:
                    mesh_labels[b, write_pos] = label_val
                elif (
                    (not interior_pos[pos])
                    and token_id not in {mesh_bos_id, mesh_eos_id}
                    and (last_mesh_eos_pos < 0 or pos > last_mesh_eos_pos)
                ):
                    text_labels[b, write_pos] = label_val

            write_pos += 1

    if not has_mesh_segment and max_extra == 0:
        mesh_labels = labels.new_full(labels.shape, fill_value=-100)
        text_labels = labels.clone()

    return mesh_labels, text_labels


def _count_causal_loss_tokens(labels: Optional[torch.LongTensor]) -> torch.Tensor:
    if labels is None:
        return torch.tensor(0, dtype=torch.long)

    if labels.dim() == 1:
        labels = labels.unsqueeze(0)

    if labels.shape[-1] <= 1:
        return torch.zeros((), dtype=torch.long, device=labels.device)

    return labels[..., 1:].ne(-100).sum()


def _compute_causal_lm_loss(
    model,
    logits: torch.Tensor,
    labels: torch.LongTensor,
    tokenizer: Optional[AutoTokenizer],
    *,
    use_separate_new_token_loss: bool = False,
) -> torch.Tensor:
    use_mesh_byte_split_loss = use_separate_new_token_loss or bool(
        getattr(model, "_lf_use_separate_new_token_embeddings", False)
    )
    if not use_mesh_byte_split_loss:
        return model.loss_function(logits=logits, labels=labels, vocab_size=logits.shape[-1])

    if tokenizer is None:
        return model.loss_function(logits=logits, labels=labels, vocab_size=logits.shape[-1])

    mesh_byte_token_ids = sorted(token_id for token_id in _get_mesh_byte_token_ids(tokenizer) if token_id >= 0)
    if not mesh_byte_token_ids:
        return model.loss_function(logits=logits, labels=labels, vocab_size=logits.shape[-1])

    max_mesh_byte_token_id = mesh_byte_token_ids[-1]
    if logits.shape[-1] <= max_mesh_byte_token_id:
        return model.loss_function(logits=logits, labels=labels, vocab_size=logits.shape[-1])

    return _compute_separate_new_token_causal_lm_loss(
        logits,
        labels,
        mesh_byte_token_ids=mesh_byte_token_ids,
    )


def _compute_separate_new_token_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    *,
    mesh_byte_token_ids: Sequence[int],
) -> torch.Tensor:
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)

    labels = labels.to(logits.device)
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    valid_mask = shift_labels.ne(-100)
    if not bool(valid_mask.any()):
        return shift_logits.sum() * 0.0

    first_mesh_byte_id = int(mesh_byte_token_ids[0])
    last_mesh_byte_id = int(mesh_byte_token_ids[-1])
    mesh_ids_are_contiguous = len(mesh_byte_token_ids) == last_mesh_byte_id - first_mesh_byte_id + 1
    if mesh_ids_are_contiguous:
        mesh_byte_mask = valid_mask & shift_labels.ge(first_mesh_byte_id) & shift_labels.le(last_mesh_byte_id)
    else:
        mesh_byte_ids = torch.tensor(mesh_byte_token_ids, device=shift_labels.device, dtype=shift_labels.dtype)
        mesh_byte_mask = valid_mask & torch.isin(shift_labels, mesh_byte_ids)

    full_vocab_mask = valid_mask & ~mesh_byte_mask

    loss_sum = shift_logits.new_zeros((), dtype=torch.float32)
    token_count = valid_mask.sum().to(dtype=torch.float32)
    if bool(full_vocab_mask.any()):
        full_vocab_logits = shift_logits[full_vocab_mask]
        full_vocab_labels = shift_labels[full_vocab_mask]
        loss_sum = loss_sum + F.cross_entropy(full_vocab_logits.float(), full_vocab_labels, reduction="sum")

    if bool(mesh_byte_mask.any()):
        # Mesh byte targets use the independent new-token head instead of competing with frozen base-vocab logits.
        if mesh_ids_are_contiguous:
            mesh_logits = shift_logits[mesh_byte_mask][:, first_mesh_byte_id : last_mesh_byte_id + 1]
            mesh_labels = shift_labels[mesh_byte_mask] - first_mesh_byte_id
        else:
            mesh_byte_ids = torch.tensor(mesh_byte_token_ids, device=shift_labels.device, dtype=shift_labels.dtype)
            mesh_logits = shift_logits[mesh_byte_mask].index_select(dim=-1, index=mesh_byte_ids)
            remap = shift_labels.new_full((last_mesh_byte_id + 1,), fill_value=-100)
            remap[mesh_byte_ids] = torch.arange(len(mesh_byte_token_ids), device=shift_labels.device)
            mesh_labels = remap[shift_labels[mesh_byte_mask]]

        loss_sum = loss_sum + F.cross_entropy(mesh_logits.float(), mesh_labels, reduction="sum")

    return loss_sum / token_count.clamp_min(1.0)


def forward_qwen25_vl_for_causal_lm_with_mesh_mask_loss(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    is_train: bool = False,
    full_depth: int = 3,
    max_depth: int = 6,
    current_binary_sequence: Optional[torch.Tensor] = None,
    octree_decode_metadata: Optional[OctreeDecodeMetadata] = None,
    add_mask_token: bool = True,
    loss_mode: str = "text_only", # text_only, mesh_only or text_and_mesh
    use_separate_new_token_loss: bool = False,
    mesh_token_metadata: Optional[MeshTokenMetadata] = None,
) -> Union[Tuple, transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLCausalLMOutputWithPast]:
    """
    Wrapped forward for Qwen2_5_VLForConditionalGeneration that expands labels by inserting -100 before
    each mesh token inside <mesh_bos> ... <mesh_eos>, to match the logits length after the model inserts
    <MASK> tokens in its forward pass. Loss on inserted <MASK> positions is ignored.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    add_mask_token_flags = None
    if add_mask_token and labels is not None and input_ids is not None and tokenizer is not None:
        mesh_bos_id = tokenizer.convert_tokens_to_ids("<mesh_bos>")
        mesh_eos_id = tokenizer.convert_tokens_to_ids("<mesh_eos>")
        if mesh_bos_id is not None and mesh_eos_id is not None:
            ids_for_flag = input_ids if input_ids.dim() > 1 else input_ids.unsqueeze(0)
            labels_for_flag = labels if labels.dim() > 1 else labels.unsqueeze(0)
            if labels_for_flag.shape[0] != ids_for_flag.shape[0]:
                raise ValueError("labels batch size must match input_ids batch size")
            flags: List[bool] = []
            for b in range(ids_for_flag.shape[0]):
                ids_b = ids_for_flag[b].tolist()
                labels_b = labels_for_flag[b]
                segments = _find_mesh_segments(ids_b, mesh_bos_id, mesh_eos_id, start_index=0)
                need_mask = False
                for bos_idx, eos_idx in segments:
                    upper = min(eos_idx, labels_b.shape[0])
                    if upper <= bos_idx + 1:
                        continue
                    if torch.any(labels_b[bos_idx + 1:upper] != -100):
                        need_mask = True
                        break
                flags.append(need_mask)
            add_mask_token_flags = torch.tensor(flags, device=input_ids.device, dtype=torch.bool)

    # Execute model forward (the model's forward is expected to handle MASK insertion)
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        rope_deltas=rope_deltas,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        tokenizer=tokenizer,
        is_train=is_train,
        full_depth=full_depth,
        max_depth=max_depth,
        current_binary_sequence=current_binary_sequence,
        octree_decode_metadata=octree_decode_metadata,
        add_mask_token=add_mask_token,
        add_mask_token_flags=add_mask_token_flags,
        mesh_token_metadata=mesh_token_metadata,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    mesh_loss = None
    text_loss = None
    mesh_loss_token_count = None
    text_loss_token_count = None
    if labels is not None:
        expanded_labels = _expand_labels_with_mesh_masks(
            labels,
            input_ids,
            tokenizer,
            add_mask_token=add_mask_token,
            add_mask_token_flags=add_mask_token_flags,
            loss_mode=loss_mode,
        )
        loss = _compute_causal_lm_loss(
            self,
            logits,
            expanded_labels,
            tokenizer,
            use_separate_new_token_loss=use_separate_new_token_loss,
        )
        if not self.training:
            mesh_labels, text_labels = _expand_labels_for_mesh_eval_scopes(
                labels,
                input_ids,
                tokenizer,
                add_mask_token=add_mask_token,
                add_mask_token_flags=add_mask_token_flags,
            )
            mesh_loss_token_count = _count_causal_loss_tokens(mesh_labels).to(logits.device)
            text_loss_token_count = _count_causal_loss_tokens(text_labels).to(logits.device)
            if int(mesh_loss_token_count.item()) > 0:
                mesh_loss = _compute_causal_lm_loss(
                    self,
                    logits,
                    mesh_labels,
                    tokenizer,
                    use_separate_new_token_loss=use_separate_new_token_loss,
                )
            if int(text_loss_token_count.item()) > 0:
                text_loss = _compute_causal_lm_loss(
                    self,
                    logits,
                    text_labels,
                    tokenizer,
                    use_separate_new_token_loss=use_separate_new_token_loss,
                )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VL3DCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        mesh_loss=mesh_loss,
        text_loss=text_loss,
        mesh_loss_token_count=mesh_loss_token_count,
        text_loss_token_count=text_loss_token_count,
    )


def replace_qwen25_vl_for_conditional_generation_forward_with_mesh_mask_loss(
    tokenizer: Optional[AutoTokenizer] = None,
    *,
    is_train: bool = False,
    full_depth: int = 3,
    max_depth: int = 6,
    add_mask_token: bool = True,
    loss_mode: str = "text_only", # text_only or text_and_mesh
    use_separate_new_token_loss: bool = False,
):
    """
    Install monkey patches:
      1) Replace Qwen2_5_VLModel.forward to insert <MASK> before each mesh token and compute 3D RoPE.
      2) Replace Qwen2_5_VLForConditionalGeneration.forward to expand labels with -100 at inserted MASK positions
         so that loss ignores MASK generation.
    """
    # Ensure model forward replacement is applied (idempotent if called multiple times)
    replace_qwen25_vl_model_forward_with_mesh_extraction(
        tokenizer=tokenizer,
        is_train=is_train,
        full_depth=full_depth,
        max_depth=max_depth,
        add_mask_token=add_mask_token,
    )

    def wrapped_forward(self, *args, **kwargs):
        kwargs['tokenizer'] = tokenizer
        kwargs['is_train'] = is_train
        kwargs['full_depth'] = full_depth
        kwargs['max_depth'] = max_depth
        # current_binary_sequence is now passed directly via kwargs from caller
        # If not provided, default to None
        kwargs.setdefault('current_binary_sequence', None)
        kwargs.setdefault('octree_decode_metadata', None)
        kwargs['add_mask_token'] = add_mask_token
        kwargs['loss_mode'] = loss_mode
        kwargs['use_separate_new_token_loss'] = use_separate_new_token_loss
        return forward_qwen25_vl_for_causal_lm_with_mesh_mask_loss(self, *args, **kwargs)

    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = (
        wrapped_forward
    )

    # Also update _old_forward if it exists (used by accelerate hooks)
    # This is critical because accelerate saves the original forward as _old_forward before wrapping
    if hasattr(transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration, '_old_forward'):
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration._old_forward = (
            wrapped_forward
        )
