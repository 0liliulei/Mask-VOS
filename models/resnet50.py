"""
Based on Jabri et al., (2020)
Credit: https://github.com/ajabri/videowalk.git
License: MIT
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import BasicBlock, Bottleneck

class Block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class ResNet(torch_resnet.ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self.down_dim4 = Block(1024, 512)
#         self.pos_embed1 = nn.Parameter(torch.zeros(1, 64, 64, 64))
    

    def filter_layers(self, x):
        return [l for l in x if getattr(self, l) is not None]

    def remove_layers(self, remove_layers=[]):
        # Remove extraneous layers
        remove_layers += ['fc', 'avgpool']
        for layer in self.filter_layers(remove_layers):
            setattr(self, layer, None)
        setattr(self, 'layer4', None)

    def modify(self):

        # Set stride of layer3 and layer 4 to 1 (from 2)
        for layer in self.filter_layers(['layer3']):
            for m in getattr(self, layer).modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.stride = (1, 1)

        for layer in self.filter_layers(['layer4']):
            for m in getattr(self, layer).modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.stride = (1, 1)


    def forward(self, x, ape=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x if self.maxpool is None else self.maxpool(x)
        
#         pos_embed1 = F.interpolate(self.pos_embed1, size=(x.size(2), x.size(3)), mode='bicubic', align_corners=True)
#         if ape:
#             x = x + pos_embed1

        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2) 
#         x4 = self.layer4(x3)
        x3 = self.down_dim4(x3)
        
        return x3, x2, x

def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
#     state_dict = torch.load("moco_v2_800ep_pretrain.pth", map_location="cuda:0")
#     for ii in list(state_dict['state_dict'].keys()):
#         state_dict['state_dict'][ii[17:]]=state_dict['state_dict'][ii]
#         del state_dict['state_dict'][ii]
#     print(state_dict['state_dict'].keys())
#     print(model.state_dict().keys())
#     model.load_state_dict(state_dict['state_dict'], strict=False)
#     state_dict = torch.load("dino_resnet50_pretrain.pth", map_location="cuda:0")
#     print(state_dict.keys())
#     print(model.state_dict().keys())
#     model.load_state_dict(state_dict, strict=False)
#     state_dict = torch.load("pixpro_base_r50_400ep_md5_919c6612.pth", map_location="cuda:0")['model']
#     for ii in list(state_dict.keys()):
#         if ii.startswith('module.encoder_k'):
#             del state_dict[ii]
#         else:    
#             state_dict[ii[15:]]=state_dict[ii]
#             del state_dict[ii]
#     print(state_dict.keys())
#     print(model.state_dict().keys())
#     model.load_state_dict(state_dict, strict=False)
#     state_dict = torch.load("dino_resnet50_pretrain.pth", map_location="cuda:0")
#     print(state_dict.keys())
#     print(model.state_dict().keys())
#     model.load_state_dict(state_dict, strict=False)
    return model

def resnet50(pretrained='', remove_layers=[], train=True, **kwargs):
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)
    model.modify()

    model.remove_layers(remove_layers)
    setattr(model, "fdim", 512)
    return model
