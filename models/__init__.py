"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

from .resnet18 import resnet18
from .resnet50 import resnet50
from .resnet18_mod import resnet18_mod
from .resnet50_mod import resnet50_mod
from .net import Net
from .net2 import Net2
from .framework import Framework
from .framework_quan import FrameworkQuan
from .framework_mask import FrameworkMask

def get_model(cfg, *args, **kwargs):

    if cfg.TRAIN.TASK == 'YTVOS_Quan':
        backbones = {
            'resnet18': resnet18,
            'resnet50': resnet50
        }
        encoders = {
            'resnet18': resnet18_mod,
            'resnet50': resnet50_mod
        }
        def create_net():
            backbone = backbones['resnet18'](*args, **kwargs)
            return Net(cfg, backbone)
        net = create_net()
        def create_net2():
            backbone = encoders[cfg.MODEL.ARCH.lower()](*args, **kwargs)
            return Net2(cfg, backbone)
        mask_encoder = create_net2()
        return FrameworkQuan(cfg, net, mask_encoder)
    elif cfg.TRAIN.TASK == 'YTVOS_Mask':
        backbones = {
            'resnet18': resnet18,
            'resnet50': resnet50
        }
        encoders = {
            'resnet18': resnet18_mod,
            'resnet50': resnet50_mod
        }
        def create_net():
            backbone = backbones['resnet50'](*args, **kwargs)
            return Net(cfg, backbone)
        net = create_net()
        def create_net2():
            backbone = encoders['resnet50'](*args, **kwargs)
            return Net2(cfg, backbone)
        mask_encoder = create_net2()
        return FrameworkMask(cfg, net, mask_encoder)
    else:
        backbones = {
            'resnet18': resnet18,
            'resnet50': resnet50
        }
        def create_net():
            backbone = backbones['resnet50'](*args, **kwargs)
            return Net(cfg, backbone)
        net = create_net()
        return Framework(cfg, net)
