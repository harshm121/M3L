#Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.
#
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch.nn as nn
import torch

num_parallel = 2


class LinearFuse(nn.Module):
    def __init__(self, ratio):
        super(LinearFuse, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        # x: [B, N, C], mask: [B, N]
        # mask_threshold = torch.rand(1)[0] * (high - low) + low
        # mask = [torch.rand(x[0].shape[0], x[0].shape[1]), torch.rand(x[0].shape[0], x[0].shape[1])]
        x0, x1 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x0 = self.ratio*x[0] + (1-self.ratio)*x[1]
        x1 = (1-self.ratio)*x[0] + self.ratio*x[1]
        return [x0, x1]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'ln_' + str(i), nn.LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]