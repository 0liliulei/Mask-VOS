import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import Bottleneck
import math

class ResNet(torch_resnet.ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3+1, 64, kernel_size=7, stride=2, padding=3)
        self.down_dim = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)

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


    def forward(self, image, mask):
        x = torch.cat([image, mask], 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x if self.maxpool is None else self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x) 
        x3 = self.down_dim(x3)
#         x4 = self.layer4(x3)

        return x3
        
def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
#     state_dict = torch.load("epoch120_score0.675_key.pth", map_location="cuda:0")['model']
#     for item in list(state_dict.keys()):
#         state_dict[item[18:]]=state_dict[item]
#         del state_dict[item]
# #     print(model.state_dict().keys())
# #     print(state_dict.keys())
# #     state_dict = torch.load("dino_resnet50_pretrain.pth", map_location="cuda:0")
# #     print(model.state_dict().keys())
# #     print(state_dict.keys())
#     del state_dict['conv1.weight']
#     model.load_state_dict(state_dict, strict=False)
    return model

def resnet50_mod(pretrained='', remove_layers=[], train=True, **kwargs):
   
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)
    model.modify()

    model.remove_layers(remove_layers)
    setattr(model, "fdim", 512)
    return model