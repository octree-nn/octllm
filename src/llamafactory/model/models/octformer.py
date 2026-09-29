# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from ocnn.octree import Octree
from .positional_embedding import MAX_DEPTH


class OctreeT(Octree):
    def __init__(
        self,
        octree: Octree,
        depth_list: list = None,
        data_mask: torch.Tensor = None,
        buffer_size: int = 0,
    ):
        super().__init__(octree.depth, octree.full_depth)
        self.__dict__.update(octree.__dict__)

        self.depth_list = depth_list if depth_list is not None else list(range(self.full_depth, self.depth + 1))
        self.data_mask = data_mask
        self.buffer_size = buffer_size
        self.build_t()

    def build_t(self):
        self.depth_idx = self.build_depth_idx()
        self.xyz = self.build_xyz(self.depth)

    def build_depth_idx(self):
        depth_idx = torch.cat(
            [
                torch.ones(self.nnum[self.depth_list[i]], device=self.device).long() * i
                for i in range(len(self.depth_list))
            ]
        )
        depth_idx = torch.cat([torch.zeros(self.buffer_size * self.batch_size, device=self.device).long(), depth_idx])
        if self.data_mask is not None:
            depth_idx = depth_idx[~self.data_mask]
        return depth_idx

    def build_xyz(self, max_depth=MAX_DEPTH):
        max_scale = 2 ** (max_depth + 1)

        def rescale_pos(x, scale):
            x = x * max_scale // scale
            x += max_scale // scale // 2
            return x

        xyz = []
        for d in self.depth_list:
            scale = 2**d
            x, y, z, b = self.xyzb(d)
            x = rescale_pos(x, scale)
            y = rescale_pos(y, scale)
            z = rescale_pos(z, scale)
            xyz.append(torch.stack([x, y, z], dim=1))
        xyz = torch.cat(xyz, dim=0).float()
        xyz = torch.cat([torch.zeros(self.buffer_size * self.batch_size, 3, device=self.device), xyz])
        if self.data_mask is not None:
            xyz = xyz[~self.data_mask]
        return xyz
