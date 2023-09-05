"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseNet
from models.modules import ResBlock
from models import cbam

class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x

class Net2(BaseNet):

    def __init__(self, cfg, backbone):
        super(Net2, self).__init__()

        self.cfg = cfg
        self.backbone = backbone
        self.fuser = FeatureFusionBlock(512+self.backbone.fdim, 512)

    def lr_mult(self):
        """Learning rate multiplier for weights.
        Returns: [old, new]"""
        return 1., 1.

    def lr_mult_bias(self):
        """Learning rate multiplier for bias.
        Returns: [old, new]"""
        return 2., 2.

    def forward(self, frames, mask, f4, norm=True):
        """Forward pass to extract projection and task features"""

        # extracting the time dimension
        res3 = self.backbone(frames, mask)
        x = self.fuser(res3, f4)

        return x
