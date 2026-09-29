# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

# autopep8: off
import torch
import torch.autograd
import copy

from ocnn.octree import Octree
from ocnn.nn import octree_pad
# autopep8: on


def _sync_octree_lookup_tables(octree: Octree) -> None:
    device = torch.device(octree.device)
    octree.lut_parent = octree.lut_parent.to(device)
    octree.lut_child = octree.lut_child.to(device)
    octree.lut_kernel = {key: value.to(device) for key, value in octree.lut_kernel.items()}


def octree2split(octree, depth_low, depth_high, shift=False):
    child = octree.children[depth_high - 1]
    split = (child >= 0).unsqueeze(-1)

    for d in range(depth_low, depth_high - 1)[::-1]:
        split_dim = 2 ** (3 * (depth_high - d - 1))
        split = split.reshape(-1, split_dim)
        split = octree_pad(data=split, octree=octree, depth=d)

    split = split.float()
    if shift:
        split = 2 * split - 1  # scale to [-1, 1]

    return split


def octree2seq(octree: Octree, depth_low: int, depth_high: int, shift: bool = False):
    seq = torch.cat(octree.children[depth_low:depth_high])
    seq = (seq >= 0).long()

    if shift:  # scale to [-1, 1]
        seq = 2 * seq - 1
    return seq


def seq2octree(octree, seq, depth_low, depth_high, threshold=0.0):
    discrete_seq = (seq > threshold).long()

    octree_out = copy.deepcopy(octree)
    _sync_octree_lookup_tables(octree_out)
    cur_nnum = 0
    for d in range(depth_low, depth_high):
        nnum_d = octree_out.nnum[d]
        label = copy.deepcopy(discrete_seq[cur_nnum : cur_nnum + nnum_d])
        cur_nnum += nnum_d
        if torch.numel(label) == 0:
            label = torch.zeros((8,), dtype=torch.long, device=octree.device)
        octree_out.octree_split(label, depth=d)
        octree_out.octree_grow(d + 1)
    return octree_out
