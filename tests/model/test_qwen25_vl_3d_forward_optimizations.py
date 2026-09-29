import pytest
import torch


qwen25_vl_3d_replace = pytest.importorskip("llamafactory.model.qwen25_vl_3d_replace")


class DummyMeshTokenizer:
    def __init__(self):
        self.token_to_id = {f"<mesh{i}>": 1000 + i for i in range(256)}

    def convert_tokens_to_ids(self, token: str):
        return self.token_to_id.get(token)

    def convert_ids_to_tokens(self, ids):
        raise AssertionError("mesh byte ids should be parsed without tokenizer string conversion")


def test_mesh_inner_ids_to_byte_values_uses_direct_token_id_lookup():
    tokenizer = DummyMeshTokenizer()

    byte_values = qwen25_vl_3d_replace._mesh_inner_ids_to_byte_values(
        [tokenizer.convert_tokens_to_ids("<mesh0>"), tokenizer.convert_tokens_to_ids("<mesh17>")],
        tokenizer,
    )

    assert byte_values == [0, 17]


def test_build_mesh_embedding_block_interleaves_mask_and_mesh_embeddings():
    inputs_embeds = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
            [4.0, 5.0, 6.0],
        ]
    )
    attention_mask = torch.tensor([1, 1, 0, 1])
    per_token_emb = torch.tensor([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
    mask_token_embed = torch.tensor([100.0, 200.0, 300.0])

    block, block_attention_mask, block_route_mask, inserted_mask_positions = (
        qwen25_vl_3d_replace._build_mesh_embedding_block(
            inputs_embeds,
            attention_mask,
            old_pos=1,
            per_token_emb=per_token_emb,
            mask_token_embed=mask_token_embed,
            sample_add_mask_token=True,
        )
    )

    expected = torch.stack(
        [
            mask_token_embed + per_token_emb[0],
            inputs_embeds[1] + per_token_emb[0],
            mask_token_embed + per_token_emb[1],
            inputs_embeds[2] + per_token_emb[1],
        ]
    )
    assert torch.allclose(block, expected)
    assert block_attention_mask.tolist() == [1, 1, 0, 0]
    assert block_route_mask.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert inserted_mask_positions.tolist() == [True, False, True, False]


def test_build_mesh_embedding_block_keeps_mesh_tokens_when_mask_disabled():
    inputs_embeds = torch.tensor(
        [
            [1.0, 2.0],
            [10.0, 20.0],
            [30.0, 40.0],
        ]
    )
    attention_mask = torch.tensor([1, 0, 1])
    per_token_emb = torch.tensor([[0.5, 0.25], [1.5, 1.25]])
    mask_token_embed = torch.tensor([100.0, 200.0])

    block, block_attention_mask, block_route_mask, inserted_mask_positions = (
        qwen25_vl_3d_replace._build_mesh_embedding_block(
            inputs_embeds,
            attention_mask,
            old_pos=1,
            per_token_emb=per_token_emb,
            mask_token_embed=mask_token_embed,
            sample_add_mask_token=False,
        )
    )

    assert torch.allclose(block, inputs_embeds[1:3] + per_token_emb)
    assert block_attention_mask.tolist() == [0, 1]
    assert block_route_mask.tolist() == [1.0, 1.0]
    assert inserted_mask_positions.tolist() == [False, False]


def test_expand_mesh_xyz_for_rope_duplicates_positions_for_mask_tokens():
    xyz = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    like = torch.tensor([0], dtype=torch.long)

    expanded = qwen25_vl_3d_replace._expand_mesh_xyz_for_rope(
        xyz,
        like,
        sample_add_mask_token=True,
    )

    assert expanded.tolist() == [
        [1, 1, 4, 4],
        [2, 2, 5, 5],
        [3, 3, 6, 6],
    ]

    not_expanded = qwen25_vl_3d_replace._expand_mesh_xyz_for_rope(
        xyz,
        like,
        sample_add_mask_token=False,
    )
    assert not_expanded.tolist() == [[1, 4], [2, 5], [3, 6]]
