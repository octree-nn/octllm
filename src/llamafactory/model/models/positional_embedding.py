"""
This code was originally obtained from:
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

import torch
from ocnn.octree import Octree

FULL_DEPTH = 3
MAX_DEPTH = 6


class DepthPosEmb(torch.nn.Module):
    def __init__(self, num_embed: int, full_depth: int = FULL_DEPTH, max_depth: int = MAX_DEPTH):
        super().__init__()
        self.num_embed = num_embed
        self.full_depth = full_depth
        self.max_depth = max_depth
        self.depth_emb = torch.nn.Embedding(self.max_depth - self.full_depth + 1, num_embed)

    def forward(self, octree: Octree):
        depth_embedding = self.depth_emb(octree.depth_idx)
        return depth_embedding
