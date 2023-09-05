"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseNet

from sklearn.cluster import KMeans
import numpy as np
import cv2

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
    
class MLP(nn.Sequential):

    def __init__(self, n_in, n_out):
        super().__init__()

        self.add_module("conv1", nn.Conv2d(n_in, n_in, 1, 1))
        self.add_module("bn1", nn.BatchNorm2d(n_in))
        self.add_module("relu", nn.ReLU(True))
        self.add_module("conv2", nn.Conv2d(n_in, n_out, 1, 1))

class Net(BaseNet):

    def __init__(self, cfg, backbone):
        super(Net, self).__init__()

        self.cfg = cfg
        self.backbone = backbone
        self.emb_q = MLP(backbone.fdim, cfg.MODEL.FEATURE_DIM)
        self.emb_v = MLP(backbone.fdim, 512)
        self.down_dim3 = Block(512, 512)
        self.down_dim2 = Block(256, 256)

    def lr_mult(self):
        """Learning rate multiplier for weights.
        Returns: [old, new]"""
        return 1., 1.

    def lr_mult_bias(self):
        """Learning rate multiplier for bias.
        Returns: [old, new]"""
        return 2., 2.

    def forward(self, frames, norm=True):
        """Forward pass to extract projection and task features"""

        # extracting the time dimension
        if self.cfg.TRAIN.TASK == 'YTVOS_Mask':
            with torch.no_grad():
                res4, res3, res2 = self.backbone(frames)  
        else:
            res4, res3, res2 = self.backbone(frames)

        # B,K,H,W
        query = self.emb_q(res4)
        value = self.emb_v(res4)
        res3 = self.down_dim3(res3)
        res2 = self.down_dim2(res2)
        if norm:
            query_n = F.normalize(query, p=2, dim=1)
            value_n = F.normalize(value, p=2, dim=1)
            res4_n = F.normalize(res4, p=2, dim=1)

        return query_n, res4_n, query_n, value_n, res4, res3, res2
    
    
#     def forward(self, frames, norm=True):
#         """Forward pass to extract projection and task features"""

#         # extracting the time dimension
#         res4, res3, res2 = self.backbone(frames)

#         # B,K,H,W
#         query = self.emb_q(res4)

#         if norm:
#             query = F.normalize(query, p=2, dim=1)
#             res3 = F.normalize(res3, p=2, dim=1)
#             res4 = F.normalize(res4, p=2, dim=1)
        
        
# #         n_cluster = 5
# #         B,C,H,W = query.size()
# #         X = F.interpolate(query, scale_factor=4, mode='bilinear', align_corners=False)
# #         X = X[0:1].flatten(2).squeeze().transpose(0,1).cpu().detach().numpy()
# #         km3 = KMeans(n_clusters=n_cluster)
# #         km3.fit(X)
# #         km3_labels = km3.labels_
# #         print(km3_labels.shape)
        
# #         for y in range(n_cluster):
# #             a1 = []
# #             for i,x in enumerate(km3_labels):
# #                 if x==y:
# #                     a1.append([1])
# #                 else:
# #                     a1.append([0])
# #             a2=np.array(a1)
# #             a3=a2.reshape([H*4,W*4])*255
# # #             a3 = cv2.resize(a3.astype('uint8'), (W*8, H*8), interpolation=cv2.INTER_LINEAR)
# #             cv2.imwrite(str(y)+'.png', a3)
# #         print(a4)


#         return query, res3, res4
