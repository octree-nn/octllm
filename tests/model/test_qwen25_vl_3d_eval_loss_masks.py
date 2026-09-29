from types import SimpleNamespace

import pytest
import torch


qwen25_vl_3d_replace = pytest.importorskip("llamafactory.model.qwen25_vl_3d_replace")


class DummyTokenizer:
    def __init__(self):
        self.token_to_id = {
            "<mesh_bos>": 1000,
            "<mesh_eos>": 1001,
        }
        self.token_to_id.update({f"<mesh{i}>": 2000 + i for i in range(256)})

    def convert_tokens_to_ids(self, token: str):
        return self.token_to_id.get(token)


def test_expand_labels_for_mesh_eval_scopes_splits_mesh_and_text_targets():
    tokenizer = DummyTokenizer()
    mesh0 = tokenizer.convert_tokens_to_ids("<mesh0>")
    mesh1 = tokenizer.convert_tokens_to_ids("<mesh1>")
    bos = tokenizer.convert_tokens_to_ids("<mesh_bos>")
    eos = tokenizer.convert_tokens_to_ids("<mesh_eos>")

    input_ids = torch.tensor([[10, bos, mesh0, mesh1, eos, 11, 12]])
    labels = torch.tensor([[10, -100, mesh0, mesh1, eos, 11, 12]])

    mesh_labels, text_labels = qwen25_vl_3d_replace._expand_labels_for_mesh_eval_scopes(
        labels,
        input_ids,
        tokenizer,
        add_mask_token=True,
    )

    assert mesh_labels.tolist() == [[-100, -100, -100, mesh0, -100, mesh1, -100, -100, -100]]
    assert text_labels.tolist() == [[-100, -100, -100, -100, -100, -100, -100, 11, 12]]
    assert int(qwen25_vl_3d_replace._count_causal_loss_tokens(mesh_labels).item()) == 2
    assert int(qwen25_vl_3d_replace._count_causal_loss_tokens(text_labels).item()) == 2


def test_expand_labels_for_mesh_eval_scopes_keeps_plain_text_when_no_mesh_segment():
    tokenizer = DummyTokenizer()
    input_ids = torch.tensor([[10, 11, 12]])
    labels = torch.tensor([[-100, 11, 12]])

    mesh_labels, text_labels = qwen25_vl_3d_replace._expand_labels_for_mesh_eval_scopes(
        labels,
        input_ids,
        tokenizer,
        add_mask_token=True,
    )

    assert mesh_labels.tolist() == [[-100, -100, -100]]
    assert text_labels.tolist() == labels.tolist()


def test_expand_labels_for_mesh_eval_scopes_respects_disabled_mask_insertion():
    tokenizer = DummyTokenizer()
    mesh0 = tokenizer.convert_tokens_to_ids("<mesh0>")
    bos = tokenizer.convert_tokens_to_ids("<mesh_bos>")
    eos = tokenizer.convert_tokens_to_ids("<mesh_eos>")

    input_ids = torch.tensor([[bos, mesh0, eos, 11]])
    labels = torch.tensor([[-100, -100, -100, 11]])

    mesh_labels, text_labels = qwen25_vl_3d_replace._expand_labels_for_mesh_eval_scopes(
        labels,
        input_ids,
        tokenizer,
        add_mask_token=True,
        add_mask_token_flags=torch.tensor([False]),
    )

    assert mesh_labels.shape == labels.shape
    assert mesh_labels.tolist() == [[-100, -100, -100, -100]]
    assert text_labels.tolist() == [[-100, -100, -100, 11]]


def test_3d_eval_loss_output_fields_survive_accelerate_fp32_conversion():
    operations = pytest.importorskip("accelerate.utils.operations")

    output = qwen25_vl_3d_replace.Qwen2_5_VL3DCausalLMOutputWithPast(
        loss=torch.tensor(1.0, dtype=torch.bfloat16),
        logits=torch.zeros(1, 2, 3, dtype=torch.bfloat16),
        mesh_loss=torch.tensor(2.0, dtype=torch.bfloat16),
        text_loss=torch.tensor(4.0, dtype=torch.bfloat16),
        mesh_loss_token_count=torch.tensor(3),
        text_loss_token_count=torch.tensor(5),
    )

    converted = operations.convert_to_fp32(output)

    assert converted.mesh_loss.dtype == torch.float32
    assert converted.text_loss.dtype == torch.float32
    assert int(converted.mesh_loss_token_count.item()) == 3
    assert int(converted.text_loss_token_count.item()) == 5
    assert converted["mesh_loss"] is converted.mesh_loss
    assert converted["text_loss_token_count"] is converted.text_loss_token_count


def test_sft_trainer_aggregates_3d_eval_losses_by_token_count():
    trainer_module = pytest.importorskip("llamafactory.train.sft.trainer")
    trainer = trainer_module.CustomSeq2SeqTrainer.__new__(trainer_module.CustomSeq2SeqTrainer)
    trainer.args = SimpleNamespace(device=torch.device("cpu"))
    trainer.finetuning_args = SimpleNamespace(
        use_3d_position_embedding=True,
    )
    trainer._nested_gather = lambda tensor: tensor

    trainer._reset_3d_eval_loss_meters()
    trainer._accumulate_3d_eval_losses(
        SimpleNamespace(
            mesh_loss=torch.tensor(2.0),
            mesh_loss_token_count=torch.tensor(3),
            text_loss=torch.tensor(4.0),
            text_loss_token_count=torch.tensor(2),
        )
    )
    trainer._accumulate_3d_eval_losses(
        SimpleNamespace(
            mesh_loss=torch.tensor(5.0),
            mesh_loss_token_count=torch.tensor(1),
            text_loss=None,
            text_loss_token_count=torch.tensor(0),
        )
    )

    metrics = {}
    trainer._add_3d_eval_loss_metrics(metrics, "eval")

    assert metrics["eval_mesh_tokens"] == 4.0
    assert metrics["eval_text_tokens"] == 2.0
    assert metrics["eval_mesh_loss"] == pytest.approx((2.0 * 3 + 5.0) / 4)
    assert metrics["eval_text_loss"] == pytest.approx(4.0)
