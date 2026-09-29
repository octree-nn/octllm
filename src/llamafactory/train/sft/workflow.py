# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ...model.model_utils.separate_new_tokens import load_separate_new_token_weights
from ...model.qwen25_vl_3d_replace import (
    attach_pos_emb_to_model as attach_pos_emb_to_model_qwen25_vl,
)
from ...model.qwen25_vl_3d_replace import (
    load_pos_emb_weights as load_pos_emb_weights_qwen25_vl,
)
from ...model.qwen25_vl_3d_replace import (
    replace_qwen25_vl_for_conditional_generation_forward_with_mesh_mask_loss,
)
from ...model.qwen25_vl_3d_router import install_qwen25_vl_3d_router, load_3d_router_weights
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def _get_model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device

    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _attach_3d_position_embedding_for_checkpoint(model, finetuning_args: "FinetuningArguments") -> None:
    if not finetuning_args.use_3d_position_embedding:
        return

    hidden_size = getattr(model.config, "hidden_size", None)
    device = _get_model_device(model)
    if hidden_size is None or device is None:
        return

    attach_pos_emb_to_model_qwen25_vl(
        model,
        hidden_size,
        device,
        full_depth=finetuning_args.full_depth,
        max_depth=finetuning_args.max_depth,
    )


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if model_args.use_separate_new_token_embeddings and training_args.resume_from_checkpoint:
        loaded = load_separate_new_token_weights(
            model,
            training_args.resume_from_checkpoint,
            strict=False,
        )
        if loaded:
            logger.info_rank0(
                f"Loaded separate new-token weights from `{training_args.resume_from_checkpoint}` for resume."
            )

    if finetuning_args.use_3d_position_embedding:
        logger.info_rank0(
            "Replace Qwen2.5-VLModel with 3D position embedding, "
            f"full_depth={finetuning_args.full_depth}, max_depth={finetuning_args.max_depth}"
        )
        if model_args.use_separate_new_token_embeddings:
            logger.info_rank0("Using separate new-token mesh-byte loss for Qwen2.5-VL 3D training.")
        replace_qwen25_vl_for_conditional_generation_forward_with_mesh_mask_loss(
            tokenizer=tokenizer,
            is_train=training_args.do_train,
            full_depth=finetuning_args.full_depth,
            max_depth=finetuning_args.max_depth,
            add_mask_token=finetuning_args.add_mask_token,
            loss_mode=finetuning_args.loss_mode,
            use_separate_new_token_loss=model_args.use_separate_new_token_embeddings,
        )
        logger.info_rank0("3D position embedding has been replaced with Qwen2.5-VLModel")

        router_load_path = training_args.resume_from_checkpoint or model_args.model_name_or_path
        if model_args.adapter_name_or_path and not training_args.resume_from_checkpoint:
            loaded = load_pos_emb_weights_qwen25_vl(
                model_args.adapter_name_or_path[-1],
                strict=False,
            )
            if loaded:
                logger.info_rank0(
                    f"Loaded AbsPosEmb weights from `{model_args.adapter_name_or_path[-1]} for Qwen2.5-VLModel`"
                )
        elif router_load_path:
            loaded = load_pos_emb_weights_qwen25_vl(router_load_path, strict=False)
            if loaded:
                logger.info_rank0(f"Loaded AbsPosEmb weights from `{router_load_path} for Qwen2.5-VLModel`")

        _attach_3d_position_embedding_for_checkpoint(model, finetuning_args)

    if finetuning_args.use_3d_token_router:
        logger.info_rank0(
            f"Installing Qwen2.5-VL 3D token router: layer_scope={finetuning_args.route_3d_layer_scope}, last_n={finetuning_args.route_3d_last_n_layers}, layer_ids={finetuning_args.route_3d_layer_ids}, "
            f"replace_ffn={finetuning_args.route_3d_replace_ffn}, replace_attn_proj={finetuning_args.route_3d_replace_attn_proj}, attn_proj_mode={finetuning_args.route_3d_attn_proj_mode}"
        )
        selected_layer_ids = install_qwen25_vl_3d_router(
            model,
            replace_ffn=finetuning_args.route_3d_replace_ffn,
            replace_attn_proj=finetuning_args.route_3d_replace_attn_proj,
            attn_proj_mode=finetuning_args.route_3d_attn_proj_mode,
            layer_scope=finetuning_args.route_3d_layer_scope,
            last_n_layers=finetuning_args.route_3d_last_n_layers,
            layer_ids=finetuning_args.route_3d_layer_ids,
            mlp_ratio=finetuning_args.route_3d_mlp_ratio,
            init_from_base=finetuning_args.route_3d_init_from_base,
            freeze_base=finetuning_args.route_3d_freeze_base,
        )
        logger.info_rank0(f"Installed 3D token router on decoder layers: {selected_layer_ids}")

        router_load_path = training_args.resume_from_checkpoint or model_args.model_name_or_path
        if router_load_path:
            loaded = load_3d_router_weights(
                model,
                router_load_path,
                strict=False,
            )
            if loaded:
                logger.info_rank0(f"Loaded 3D token router weights from `{router_load_path}`")

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        precompute_mesh_token_metadata=(finetuning_args.use_3d_position_embedding),
        mesh_full_depth=finetuning_args.full_depth,
        mesh_max_depth=finetuning_args.max_depth,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [
                        [
                            f"eval_{key}_loss",
                            f"eval_{key}_mesh_loss",
                            f"eval_{key}_text_loss",
                            f"eval_{key}_accuracy",
                        ]
                        for key in dataset_module["eval_dataset"].keys()
                    ],
                    [],
                )
            else:
                keys += ["eval_loss", "eval_mesh_loss", "eval_text_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
