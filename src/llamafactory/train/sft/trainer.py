# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ...model.model_utils.new_tokens import set_no_weight_decay_for_marked_params
from ...model.qwen25_vl_3d_replace import (
    attach_pos_emb_to_model as attach_pos_emb_to_model_qwen25_vl,
)
from ..callbacks import SaveProcessorCallback
from ..resume_utils import ResumeMode, get_resume_mode
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self._ensure_label_names_include_labels()
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def _resolve_resume_checkpoint(self, resume_from_checkpoint: Optional[Union[str, bool]]) -> Optional[str]:
        if resume_from_checkpoint is False or resume_from_checkpoint is None:
            return None

        if resume_from_checkpoint is True:
            return get_last_checkpoint(self.args.output_dir)

        return resume_from_checkpoint

    @override
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Any = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs,
    ):
        checkpoint = self._resolve_resume_checkpoint(resume_from_checkpoint)
        if checkpoint is not None:
            resume_decision = get_resume_mode(
                checkpoint,
                current_world_size=self.args.world_size,
                requested_mode=getattr(self.args, "resume_from_checkpoint_mode", "auto"),
                is_deepspeed_enabled=self.is_deepspeed_enabled,
            )
            if resume_decision.mode == ResumeMode.MODEL_ONLY:
                logger.warning_rank0(
                    "Falling back to model-only resume from checkpoint. Model weights will be loaded from "
                    f"`{checkpoint}`, while optimizer, scheduler, Trainer state, and RNG state will be initialized "
                    f"for the current run. Reason: {resume_decision.reason}."
                )
                self._load_from_checkpoint(checkpoint)
                return super().train(
                    resume_from_checkpoint=None,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                    **kwargs,
                )

        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )

    def _ensure_label_names_include_labels(self) -> None:
        label_names = list(getattr(self, "label_names", []))
        if "labels" not in label_names:
            label_names.append("labels")
            self.label_names = label_names

    def _ensure_3d_position_embedding_attached(self) -> None:
        if not self.finetuning_args.use_3d_position_embedding:
            return

        hidden_size = getattr(self.model.config, "hidden_size", None)
        device = getattr(getattr(self, "accelerator", None), "device", None)
        if hidden_size is None or device is None:
            return

        attach_pos_emb_to_model_qwen25_vl(
            self.model,
            hidden_size,
            device,
            full_depth=self.finetuning_args.full_depth,
            max_depth=self.finetuning_args.max_depth,
        )

        if hasattr(self.model, "mesh_abs_pos_emb"):
            for param in self.model.mesh_abs_pos_emb.parameters():
                if self.finetuning_args.freeze_3d_position_embedding:
                    logger.info_rank0("Freezing 3D position embedding parameters.")
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        # Register depth position embeddings before constructing the optimizer.
        self._ensure_3d_position_embedding_attached()

        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)

        optimizer = super().create_optimizer()

        # Prevent AdamW-style weight decay from modifying frozen original token rows.
        set_no_weight_decay_for_marked_params(optimizer, self.model)

        return optimizer

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        """When saving the model, make sure 3D modules are attached so they enter the main model shards."""
        self._ensure_3d_position_embedding_attached()
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def _reset_3d_eval_loss_meters(self) -> None:
        self._collect_3d_eval_losses = self.finetuning_args.use_3d_position_embedding
        self._3d_eval_loss_meters = {
            "mesh": {"loss_sum": 0.0, "tokens": 0.0},
            "text": {"loss_sum": 0.0, "tokens": 0.0},
        }

    def _get_output_value(self, outputs: Any, key: str) -> Any:
        value = getattr(outputs, key, None)
        if value is None and isinstance(outputs, dict):
            value = outputs.get(key)

        return value

    def _accumulate_3d_eval_losses(self, outputs: Any) -> None:
        if not getattr(self, "_collect_3d_eval_losses", False):
            return

        for scope in ("mesh", "text"):
            loss = self._get_output_value(outputs, f"{scope}_loss")
            token_count = self._get_output_value(outputs, f"{scope}_loss_token_count")
            if loss is None or token_count is None:
                continue

            count_value = float(token_count.detach().sum().item())
            if count_value <= 0:
                continue

            loss_value = float(loss.detach().mean().item())
            self._3d_eval_loss_meters[scope]["loss_sum"] += loss_value * count_value
            self._3d_eval_loss_meters[scope]["tokens"] += count_value

    def _gather_3d_eval_loss_pair(self, loss_sum: float, token_count: float) -> tuple[float, float]:
        pair = torch.tensor([loss_sum, token_count], dtype=torch.float64, device=self.args.device)
        if hasattr(self, "_nested_gather"):
            gathered = self._nested_gather(pair)
        else:
            gathered = self.accelerator.gather(pair)

        gathered = gathered.reshape(-1, 2).sum(dim=0).detach().cpu()
        return float(gathered[0].item()), float(gathered[1].item())

    def _add_3d_eval_loss_metrics(self, metrics: dict[str, float], metric_key_prefix: str) -> None:
        if not getattr(self, "_collect_3d_eval_losses", False):
            return

        for scope, meter in self._3d_eval_loss_meters.items():
            loss_sum, token_count = self._gather_3d_eval_loss_pair(meter["loss_sum"], meter["tokens"])
            metrics[f"{metric_key_prefix}_{scope}_tokens"] = token_count
            if token_count > 0:
                metrics[f"{metric_key_prefix}_{scope}_loss"] = loss_sum / token_count

    def _get_eval_metric_key_prefix(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        if "metric_key_prefix" in kwargs:
            return kwargs["metric_key_prefix"]
        if len(args) >= 5:
            return args[4]

        return "eval"

    @override
    def evaluation_loop(self, *args, **kwargs):
        metric_key_prefix = self._get_eval_metric_key_prefix(args, kwargs)
        self._reset_3d_eval_loss_meters()
        try:
            output = super().evaluation_loop(*args, **kwargs)
            self._add_3d_eval_loss_metrics(output.metrics, metric_key_prefix)
            return output
        finally:
            self._collect_3d_eval_losses = False

    @override
    def prediction_loop(self, *args, **kwargs):
        metric_key_prefix = self._get_eval_metric_key_prefix(args, kwargs)
        self._reset_3d_eval_loss_meters()
        try:
            output = super().prediction_loop(*args, **kwargs)
            self._add_3d_eval_loss_metrics(output.metrics, metric_key_prefix)
            return output
        finally:
            self._collect_3d_eval_losses = False

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        outputs = super().compute_loss(model, inputs, *args, **kwargs)
        if kwargs.get("return_outputs", False):
            self._accumulate_3d_eval_losses(outputs[1])

        return outputs

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
