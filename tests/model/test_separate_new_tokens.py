import torch

from llamafactory.model.model_utils.separate_new_tokens import (
    install_separate_new_token_embeddings,
    load_separate_new_token_weights,
)


class DummyTokenizer:
    def __len__(self):
        return 7


class PaddedTokenizer:
    _lf_separate_new_token_ids = [5, 6, 7]

    def __len__(self):
        return 8


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=5):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, 4)
        self.lm_head = torch.nn.Linear(4, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_separate_new_token_modules_keep_base_weights_frozen():
    model = DummyModel()

    installed = install_separate_new_token_embeddings(model, DummyTokenizer(), is_trainable=True)

    assert installed is True
    assert model.embed_tokens.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is False
    assert model.separate_new_token_embeddings.weight.requires_grad is True
    assert model.separate_new_token_lm_head.weight.requires_grad is True

    input_ids = torch.tensor([[0, 5, 6]])
    hidden_states = model.embed_tokens(input_ids)
    logits = model.lm_head(hidden_states)

    assert hidden_states.shape == (1, 3, 4)
    assert logits.shape == (1, 3, 7)

    logits[..., -2:].sum().backward()

    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
    assert model.separate_new_token_lm_head.weight.grad is not None


def test_separate_new_token_modules_shadow_padded_vocab_rows():
    model = DummyModel(vocab_size=10)
    base_new_row = model.embed_tokens.weight[5].detach().clone()

    installed = install_separate_new_token_embeddings(model, PaddedTokenizer(), is_trainable=True)

    assert installed is True
    assert model._lf_separate_new_token_ids == (5, 6, 7)

    hidden_states = model.embed_tokens(torch.tensor([[4, 5, 7, 8]]))
    logits = model.lm_head(hidden_states)

    assert logits.shape == (1, 4, 10)
    assert not torch.allclose(hidden_states[0, 1], base_new_row)
    assert torch.allclose(hidden_states[0, 1], model.separate_new_token_embeddings.weight[0].to(hidden_states.dtype))
    assert torch.allclose(
        logits[..., 5:8],
        model.separate_new_token_lm_head(hidden_states).to(logits.dtype),
    )


def test_separate_new_token_modules_are_in_model_state_dict():
    model = DummyModel()

    assert install_separate_new_token_embeddings(model, DummyTokenizer(), is_trainable=True)

    state_dict = model.state_dict()
    assert "separate_new_token_embeddings.weight" in state_dict
    assert "separate_new_token_lm_head.weight" in state_dict


def test_load_separate_new_token_weights_from_model_safetensors(model_checkpoint):
    model = DummyModel()
    assert install_separate_new_token_embeddings(model, DummyTokenizer(), is_trainable=True)

    expected_embedding = torch.full_like(model.separate_new_token_embeddings.weight, 0.25)
    expected_lm_head = torch.full_like(model.separate_new_token_lm_head.weight, 0.75)
    checkpoint = model_checkpoint(
        {
            "separate_new_token_embeddings.weight": expected_embedding,
            "separate_new_token_lm_head.weight": expected_lm_head,
        },
    )

    assert load_separate_new_token_weights(model, checkpoint, strict=False)
    assert torch.allclose(model.separate_new_token_embeddings.weight, expected_embedding)
    assert torch.allclose(model.separate_new_token_lm_head.weight, expected_lm_head)
