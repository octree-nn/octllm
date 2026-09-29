import glob
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ResumeMode(str, Enum):
    FULL = "full"
    MODEL_ONLY = "model_only"


@dataclass(frozen=True)
class ResumeDecision:
    mode: ResumeMode
    reason: str = ""
    saved_world_size: Optional[int] = None


_RNG_STATE_PATTERN = re.compile(r"^rng_state_(\d+)\.pth$")


def _get_rng_world_size(checkpoint: str) -> Optional[int]:
    if not os.path.isdir(checkpoint):
        return None

    ranks = []
    for filename in os.listdir(checkpoint):
        match = _RNG_STATE_PATTERN.match(filename)
        if match is not None:
            ranks.append(int(match.group(1)))

    if not ranks:
        return None

    return max(ranks) + 1


def _get_deepspeed_optimizer_world_size(checkpoint: str) -> Optional[int]:
    shard_counts = []
    for step_dir in glob.glob(os.path.join(checkpoint, "global_step*")):
        if os.path.isdir(step_dir):
            shards = glob.glob(os.path.join(step_dir, "*optim_states.pt"))
            if shards:
                shard_counts.append(len(shards))

    if not shard_counts:
        return None

    return max(shard_counts)


def get_resume_mode(
    checkpoint: str,
    current_world_size: int,
    requested_mode: str = "auto",
    is_deepspeed_enabled: bool = False,
) -> ResumeDecision:
    requested_mode = (requested_mode or "auto").lower()
    if requested_mode == ResumeMode.MODEL_ONLY.value:
        return ResumeDecision(
            mode=ResumeMode.MODEL_ONLY,
            reason="resume_from_checkpoint_mode=model_only was requested",
        )

    if requested_mode == ResumeMode.FULL.value:
        return ResumeDecision(mode=ResumeMode.FULL)

    if requested_mode != "auto":
        raise ValueError("`resume_from_checkpoint_mode` must be one of: auto, full, model_only.")

    rng_world_size = _get_rng_world_size(checkpoint)
    if rng_world_size is not None and rng_world_size != current_world_size:
        return ResumeDecision(
            mode=ResumeMode.MODEL_ONLY,
            reason=(
                f"rng_state files were saved for world size {rng_world_size}, "
                f"but current world size is {current_world_size}"
            ),
            saved_world_size=rng_world_size,
        )

    deepspeed_world_size = _get_deepspeed_optimizer_world_size(checkpoint) if is_deepspeed_enabled else None
    if deepspeed_world_size is not None and deepspeed_world_size != current_world_size:
        return ResumeDecision(
            mode=ResumeMode.MODEL_ONLY,
            reason=(
                f"DeepSpeed optimizer shards were saved for world size {deepspeed_world_size}, "
                f"but current world size is {current_world_size}"
            ),
            saved_world_size=deepspeed_world_size,
        )

    return ResumeDecision(mode=ResumeMode.FULL, saved_world_size=rng_world_size or deepspeed_world_size)
