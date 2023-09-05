import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import BasicBlock
from models.base import BaseNet
import math
from labelprop.crw import MaskedAttention
    
def compute_tensor_iu(seg, gt):
    intersection = (seg & gt).float().sum()
    union = (seg | gt).float().sum()

    return intersection, union

def compute_tensor_iou(seg, gt):
    intersection, union = compute_tensor_iu(seg, gt)
    iou = (intersection + 1e-6) / (union + 1e-6)
     
    return iou 


def pad_divide_by(in_img, d, in_size=None):
    if in_size is None:
        h, w = in_img.shape[-2:]
    else:
        h, w = in_size

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array

def unpad(img, pad):
    if pad[2]+pad[3] > 0:
        img = img[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        img = img[:,:,:,pad[0]:-pad[1]]
    return img

def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    # The types should be the same already
    # some people report an error here so an additional guard is added
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

    return x

def softmax_w_top_ensem(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    # The types should be the same already
    # some people report an error here so an additional guard is added
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

    return x, x_exp, indices

class MemoryBank_dotproduct:
    def __init__(self, k, radius, top_k=20):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None
        self.long_mem_k = None
        self.long_mem_v = None
        self.short_mem_k = None
        self.short_mem_v = None

        self.num_objects = k
        self.mask=None
        self.mask_hw=None
        self.radius = radius

    def _global_matching(self, mk, qk, mask, TEM=0.05):
        #=================================================================================
        # memory efficient
        bsize, pbsize = 2, 100
        for b in range(0, mk.shape[2], bsize):
            # B, C, 1, N, HW
            # B, C, 1, HW
            _k, _q = mk[:, :, b:b+bsize].cuda(), qk[:, :, b:b+bsize].cuda()
            a_s, w_s, i_s = [], [], []
            for pb in range(0, _k.shape[-1], pbsize):
                A = torch.einsum('ijklm,ijkn->iklmn', _k, _q[..., pb:pb+pbsize]) 
                A[0, :, 1:] += mask[..., pb:pb+pbsize]
                # 1, 1, N, HW, bsize
                _, N, T, h1w1, hw = A.shape
                A = A.view(N, T*h1w1, hw)
                A /= TEM
                affinity, values, indices = softmax_w_top_ensem(A, top=self.top_k)  # B, NE, HW
                # 1, topk, bsize
                a_s.append(affinity)
                w_s.append(values)
                i_s.append(indices)
            # B, THW, HW 
            affinity = torch.cat(a_s, dim=-1)
            values = torch.cat(w_s, dim=-1)
            indices = torch.cat(i_s, dim=-1)
            
#         #=================================================================================
#         #normal
#         A = torch.einsum('ijklm,ijkn->iklmn', mk, qk)
#         _, N, T, h1w1, hw = A.shape
#         A[0, :, 1:] += mask.cuda()
#         A = A.view(N, T*h1w1, hw)
#         A /= TEM
#         affinity, values, indices = softmax_w_top_ensem(A, top=self.top_k)  # B, NE, HW

        return affinity, values, indices

    def _readout(self, affinity, mv):
        B, CV, T, HW = mv.shape
        mo = mv.view(B, CV, T*HW) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        return mem

    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape

        if self.mask is None or self.mask_hw != (h, w):
            restrict = MaskedAttention(self.radius, flat=False)
            D = restrict.mask(h, w)[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D==0] = -1e10; D[D==1] = 0
            self.mask = D.cuda()
            self.mask_hw = (h, w)
        
        qk = qk.flatten(start_dim=2)
        
        mk = self.mem_k
        mv = self.mem_v

        affinity, values, indices = self._global_matching(mk.unsqueeze(2), qk.unsqueeze(2), self.mask)

        # One affinity for all
        readout_mem = self._readout(affinity.expand(k,-1,-1), mv)

        return readout_mem.view(k, self.CV, h, w), values, indices

    def add_memory(self, key, value, is_long=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        key = key.unsqueeze(2).flatten(start_dim=3)
        value = value.unsqueeze(2).flatten(start_dim=3)

        if is_long:
            if self.long_mem_k is None:
                # First frame, just shove it in
                self.long_mem_k = key
                self.long_mem_v = value
                self.CK = key.shape[1]
                self.CV = value.shape[1]
            else:
                self.long_mem_k = torch.cat([self.long_mem_k, key], 2)
                self.long_mem_v = torch.cat([self.long_mem_v, value], 2)
        else:
            if self.short_mem_k is None:
                self.short_mem_k = key
                self.short_mem_v = value
            else:
                if self.mem_k.size(2)>=6:
                    self.short_mem_k = torch.cat([self.short_mem_k[:,:,1:], key], 2)
                    self.short_mem_v = torch.cat([self.short_mem_v[:,:,1:], value], 2)
                else:
                    self.short_mem_k = torch.cat([self.short_mem_k, key], 2)
                    self.short_mem_v = torch.cat([self.short_mem_v, value], 2)
            
        if self.mem_k is None:
            self.mem_k = self.long_mem_k
            self.mem_v = self.long_mem_v
        else:
            self.mem_k = torch.cat([self.long_mem_k, self.short_mem_k], 2)
            self.mem_v = torch.cat([self.long_mem_v, self.short_mem_v], 2)

# class MemoryBank_dotproduct:
#     def __init__(self, k, radius, top_k=20):
#         self.top_k = top_k

#         self.CK = None
#         self.CV = None

#         self.mem_k = None
#         self.mem_v = None

#         self.num_objects = k
#         self.mask=None
#         self.mask_hw=None
#         self.radius = radius

#     def _global_matching(self, mk, qk, mask, TEM=0.05):
#         #=================================================================================
#         # memory efficient
#         bsize, pbsize = 2, 100
#         for b in range(0, mk.shape[2], bsize):
#             # B, C, 1, N, HW
#             # B, C, 1, HW
#             _k, _q = mk[:, :, b:b+bsize].cuda(), qk[:, :, b:b+bsize].cuda()
#             a_s, w_s, i_s = [], [], []
#             for pb in range(0, _k.shape[-1], pbsize):
#                 A = torch.einsum('ijklm,ijkn->iklmn', _k, _q[..., pb:pb+pbsize]) 
#                 A[0, :, 1:] += mask[..., pb:pb+pbsize]
#                 # 1, 1, N, HW, bsize
#                 _, N, T, h1w1, hw = A.shape
#                 A = A.view(N, T*h1w1, hw)
#                 A /= TEM
#                 affinity, values, indices = softmax_w_top_ensem(A, top=self.top_k)  # B, NE, HW
#                 # 1, topk, bsize
#                 a_s.append(affinity)
#                 w_s.append(values)
#                 i_s.append(indices)
#             # B, THW, HW 
#             affinity = torch.cat(a_s, dim=-1)
#             values = torch.cat(w_s, dim=-1)
#             indices = torch.cat(i_s, dim=-1)
            
# #         #=================================================================================
# #         #normal
# #         A = torch.einsum('ijklm,ijkn->iklmn', mk, qk)
# #         _, N, T, h1w1, hw = A.shape
# #         A[0, :, 1:] += mask.cuda()
# #         A = A.view(N, T*h1w1, hw)
# #         A /= TEM
# #         affinity, values, indices = softmax_w_top_ensem(A, top=self.top_k)  # B, NE, HW

#         return affinity, values, indices

#     def _readout(self, affinity, mv):
#         B, CV, T, HW = mv.shape
#         mo = mv.view(B, CV, T*HW) 
#         mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
#         return mem

#     def match_memory(self, qk):
#         k = self.num_objects
#         _, _, h, w = qk.shape

#         if self.mask is None or self.mask_hw != (h, w):
#             restrict = MaskedAttention(self.radius, flat=False)
#             D = restrict.mask(h, w)[None]
#             D = D.flatten(-4, -3).flatten(-2)
#             D[D==0] = -1e10; D[D==1] = 0
#             self.mask = D.cuda()
#             self.mask_hw = (h, w)
        
#         qk = qk.flatten(start_dim=2)
        
#         if self.temp_k is not None:
#             mk = torch.cat([self.mem_k, self.temp_k], 2)
#             mv = torch.cat([self.mem_v, self.temp_v], 2)
#         else:
#             mk = self.mem_k
#             mv = self.mem_v

#         affinity, values, indices = self._global_matching(mk.unsqueeze(2), qk.unsqueeze(2), self.mask)

#         # One affinity for all
#         readout_mem = self._readout(affinity.expand(k,-1,-1), mv)

#         return readout_mem.view(k, self.CV, h, w), values, indices

#     def add_memory(self, key, value, is_temp=False):
#         # Temp is for "last frame"
#         # Not always used
#         # But can always be flushed
#         self.temp_k = None
#         self.temp_v = None
#         key = key.unsqueeze(2).flatten(start_dim=3)
#         value = value.unsqueeze(2).flatten(start_dim=3)

#         if self.mem_k is None:
#             # First frame, just shove it in
#             self.mem_k = key
#             self.mem_v = value
#             self.CK = key.shape[1]
#             self.CV = value.shape[1]
#         else:
#             if is_temp:
#                 self.temp_k = key
#                 self.temp_v = value
#             else:
#                 if self.mem_k.size(2)>=21:
#                     self.mem_k = torch.cat([self.mem_k[:,:,0:1], self.mem_k[:,:,2:], key], 2)
#                     self.mem_v = torch.cat([self.mem_v[:,:,0:1], self.mem_v[:,:,2:], value], 2)
#                 else:
#                     self.mem_k = torch.cat([self.mem_k, key], 2)
#                     self.mem_v = torch.cat([self.mem_v, value], 2)
                

class MemoryBank:
    def __init__(self, k, top_k=20):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None

        self.num_objects = k

    def _global_matching(self, mk, qk):
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, NE, HW
        affinity = softmax_w_top(affinity, top=self.top_k)  # B, NE, HW

        return affinity

    def _readout(self, affinity, mv):
        return torch.bmm(mv, affinity)

    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)
        
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v

        affinity = self._global_matching(mk, qk)

        # One affinity for all
        readout_mem = self._readout(affinity.expand(k,-1,-1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def add_memory(self, key, value, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r
    

class UpsampleBlock2(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x
    
class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + up_f
        x = self.out_conv(x)
        return x
    
class Decoder(BaseNet):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/8 -> 1/8
        self.up_8_4 = UpsampleBlock2(256, 256, 128) # 1/8 -> 1/4
        
        self.pred = nn.Conv2d(128, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x
    
# class Decoder(BaseNet):
#     def __init__(self):
#         super().__init__()
#         self.compress = ResBlock(1024, 256)
#         self.up_16_8 = UpsampleBlock(256, 256, 128) # 1/8 -> 1/4
#         self.pred = nn.Conv2d(128, 1, kernel_size=(3,3), padding=(1,1), stride=1)

#     def forward(self, f16, f8):
#         x = self.compress(f16)
#         x = self.up_16_8(f8, x)

#         x = self.pred(F.relu(x))
        
#         x2 = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
#         return x2, x
    
class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
 
    def get_affinity(self, mk, qk):
        mk = mk.permute(0,2,1,3,4).contiguous()
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        return affinity
    
    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out
    