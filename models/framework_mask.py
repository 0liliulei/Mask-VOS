# """
# Copyright (c) 2021 TU Darmstadt
# Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
# License: Apache License 2.0
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.base import BaseNet
# from models.modules import MemoryReader, Decoder, compute_tensor_iu

# import cv2

# class BootstrappedCE(nn.Module):
#     def __init__(self, start_warm=2000, end_warm=12000, top_p=0.15):
#         super().__init__()

#         self.start_warm = start_warm
#         self.end_warm = end_warm
#         self.top_p = top_p

#     def forward(self, input, target, it):
#         if it < self.start_warm:
#             return F.cross_entropy(input, target), 1.0

#         raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
#         num_pixels = raw_loss.numel()

#         if it > self.end_warm:
#             this_p = self.top_p
#         else:
#             this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
#         loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
#         return loss.mean(), this_p
    
# def batched_affinity(keys, query, TEM=0.05):
#     '''
#     Mini-batched computation of affinity, for memory efficiency
#     (less aggressively mini-batched)
#     '''
#     mk = keys.transpose(1,2).unsqueeze(2).flatten(4)
#     qk = query.unsqueeze(2).flatten(3)
    
#     bsize, pbsize = 2, 100
#     for b in range(0, mk.shape[2], bsize):
#         # B, C, 1, N, HW
#         # B, C, 1, HW
#         _k, _q = mk[:, :, b:b+bsize].cuda(), qk[:, :, b:b+bsize].cuda()
#         a_s, w_s, i_s = [], [], []
#         for pb in range(0, _k.shape[-1], pbsize):
#             A = torch.einsum('ijklm,ijkn->iklmn', _k, _q[..., pb:pb+pbsize]) 
#             B, N, T, h1w1, hw = A.shape
#             A = A.view(B, T*h1w1, hw)
#             A /= TEM
#             affinity = F.softmax(A, dim=-2)
#             a_s.append(affinity)
#         affinity = torch.cat(a_s, dim=-1)

#     return affinity

# class FrameworkMask(BaseNet):

#     def __init__(self, cfg, net, mask_encoder):
#         super(FrameworkMask, self).__init__()

#         self.cfg = cfg
#         self.fast_net = net
#         self.mask_encoder = mask_encoder
#         self.memory = MemoryReader()
#         self.decoder = Decoder()
# #         self.bce = BootstrappedCE()
# #         self.register_buffer("count", torch.zeros(1))

#     def parameter_groups(self, base_lr, wd):
#         return self.fast_net.parameter_groups(base_lr*0.1, wd)+ self.mask_encoder.parameter_groups(base_lr*0.1, wd) + self.decoder.parameter_groups(base_lr*0.1, wd)

#     def aggregate(self, prob, dim=1):
#         new_prob = torch.cat([
#             torch.prod(1-prob, dim=dim, keepdim=True),
#             prob
#         ], dim).clamp(1e-7, 1-1e-7)
#         logits = torch.log((new_prob /(1-new_prob)))
#         return logits
    
#     def segment_with_memory(self, mem_bank, qk, qv, f3, f2, memory_masks=None):
#         readout_mem, values, indices = mem_bank.match_memory(qk)
            
# #         h, w = memory_masks[0].shape[-2:]
# #         ctx_lbls = torch.cat(memory_masks, 0).permute([0,2,3,1])
# #         # C, H*W*N
# #         ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)
# #         pred = (ctx_lbls[:, indices[0]] * values[0][None]).sum(1)
# #         pred = pred.view(-1, h, w)
# #         pred = pred[None, ...]

#         qv = qv.expand(readout_mem.size(0), -1, -1, -1)
#         qv16 = torch.cat([readout_mem, qv], 1)
#         prob1 = self.decoder(qv16, f3, f2)
#         prob1 = torch.sigmoid(prob1)
#         logits1 = self.aggregate(prob1, 0)
#         return logits1, None
    
#     def encoder_value(self, frames, mask, f4):
#         value4 = self.mask_encoder(frames.repeat(mask.size(0),1,1,1), mask, f4.repeat(mask.size(0),1,1,1))
#         return value4
        
#     def forward(self, frames, mask1=None, frames2=None, mask=None, T=None, affine=None, affine2=None, embd_only=False, norm=True, dbg=False):
#         """Extract temporal correspondences
#         Args:
#             frames: [B,T,C,H,W]

#         Returns:
#             losses: a dictionary with the embedding loss
#             net_outs: feature embeddings
        
#         """     
#         _, C, H, W = frames.shape
#         # embedding for self-supervised learning
#         key1, res4, k, v, f4, f3, f2 = self.fast_net(frames, norm)
        
#         outs, losses = {}, {}
#         if embd_only: # only embedding
#             return key1, res4, k, v, f4, f3, f2
#         else:
#             #==========================================================================================
#             # mask [B, C, H, W]
#             _, C1 ,H1, W1 = f4.shape
#             _, C2 ,H2, W2 = k.shape
#             _, C3 ,H3, W3 = v.shape
#             _, C4 ,H4, W4 = f3.shape
#             _, C5 ,H5, W5 = f2.shape
#             value41 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,0], mask1[:,0,0:1], f4.reshape(-1,T,C1,H1,W1)[:,0])
#             value41_ = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,0], mask1[:,0,1:2],f4.reshape(-1,T,C1,H1,W1)[:,0])
#             pre_values = torch.stack([value41, value41_], 2).unsqueeze(3) # B C O T H W
#             # Segment frame 1 with frame 0 
#             # mem_key, current_key
#             affinity = batched_affinity(k.reshape(-1,T,C2,H2,W2)[:,0:1], k.reshape(-1,T,C2,H2,W2)[:,1])
#             # affinity = self.memory.get_affinity(k.reshape(-1,T+1,C2,H2,W2)[:,0:1], k.reshape(-1,T+1,C2,H2,W2)[:,1])
#             # aff, previous_value, current_features
#             logits = torch.cat([self.decoder(self.memory.readout(affinity, pre_values[:,:, 0], v.reshape(-1,T,C3,H3,W3)[:,1]), f3.reshape(-1,T,C4,H4,W4)[:,1], f2.reshape(-1,T,C5,H5,W5)[:,1]),
#                                 self.decoder(self.memory.readout(affinity, pre_values[:,:, 1], v.reshape(-1,T,C3,H3,W3)[:,1]), f3.reshape(-1,T,C4,H4,W4)[:,1], f2.reshape(-1,T,C5,H5,W5)[:,1])], 1)
#             prob1 = torch.sigmoid(logits)
#             logits1 = self.aggregate(prob1)
#             prob1 = F.softmax(logits1, dim=1)[:, 1:]

#             value42 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,1], mask1[:,1,0:1], f4.reshape(-1,T,C1,H1,W1)[:,1])
#             value42_ = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,1], mask1[:,1,1:2],f4.reshape(-1,T,C1,H1,W1)[:,1])
#             cur_values = torch.stack([value42, value42_], 2).unsqueeze(3) # B C O T H W
#             values = torch.cat([pre_values, cur_values], 3) # B C O T H W
#             # Segment frame 2 with frame 0 and 1
#             # mem_key, current_key
#             affinity = batched_affinity(k.reshape(-1,T,C2,H2,W2)[:,0:2], k.reshape(-1,T,C2,H2,W2)[:,2])
#             # affinity = self.memory.get_affinity(k.reshape(-1,T+1,C2,H2,W2)[:,0:2], k.reshape(-1,T+1,C2,H2,W2)[:,2])
#             # aff, previous_value, current_features
#             logits = torch.cat([self.decoder(self.memory.readout(affinity, values[:,:,0], v.reshape(-1,T,C3,H3,W3)[:,2]), f3.reshape(-1,T,C4,H4,W4)[:,2], f2.reshape(-1,T,C5,H5,W5)[:,2]),
#                                 self.decoder(self.memory.readout(affinity, values[:,:,1], v.reshape(-1,T,C3,H3,W3)[:,2]), f3.reshape(-1,T,C4,H4,W4)[:,2], f2.reshape(-1,T,C5,H5,W5)[:,2])], 1)
#             prob2 = torch.sigmoid(logits)
#             logits2 = self.aggregate(prob2)
#             prob2 = F.softmax(logits2, dim=1)[:, 1:]
#             losses['mask1'] = F.cross_entropy(logits1, mask1[:,1,0].long()+mask1[:,1,1].long()*2)
#             losses['mask2'] = F.cross_entropy(logits2, mask1[:,2,0].long()+mask1[:,2,1].long()*2)
# #             losses['mask1'], P = self.bce(logits1, mask1[:,1,0].long()+mask1[:,1,1].long()*2, self.count[0])
# #             losses['mask2'], P = self.bce(logits2, mask1[:,2,0].long()+mask1[:,2,1].long()*2, self.count[0])
# #             self.count[0]+=1
            
#             #==========================================================================================

#             # computing the main loss
#             losses["main"] = 0.5*losses['mask1'] + 0.5*losses['mask2']

#         return losses, outs


"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseNet
from models.modules import MemoryReader, Decoder, compute_tensor_iu

import cv2

class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=2000, end_warm=12000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p
    
def batched_affinity(keys, query, TEM=0.05):
    '''
    Mini-batched computation of affinity, for memory efficiency
    (less aggressively mini-batched)
    '''
    mk = keys.transpose(1,2).unsqueeze(2).flatten(4)
    qk = query.unsqueeze(2).flatten(3)
    
    bsize, pbsize = 2, 100
    for b in range(0, mk.shape[2], bsize):
        # B, C, 1, N, HW
        # B, C, 1, HW
        _k, _q = mk[:, :, b:b+bsize].cuda(), qk[:, :, b:b+bsize].cuda()
        a_s, w_s, i_s = [], [], []
        for pb in range(0, _k.shape[-1], pbsize):
            A = torch.einsum('ijklm,ijkn->iklmn', _k, _q[..., pb:pb+pbsize]) 
            B, N, T, h1w1, hw = A.shape
            A = A.view(B, T*h1w1, hw)
            A /= TEM
            affinity = F.softmax(A, dim=-2)
            a_s.append(affinity)
        affinity = torch.cat(a_s, dim=-1)

    return affinity

class FrameworkMask(BaseNet):

    def __init__(self, cfg, net, mask_encoder):
        super(FrameworkMask, self).__init__()

        self.cfg = cfg
        self.fast_net = net
        self.mask_encoder = mask_encoder
        self.memory = MemoryReader()
        self.decoder = Decoder()
#         self.bce = BootstrappedCE()
#         self.register_buffer("count", torch.zeros(1))

    def parameter_groups(self, base_lr, wd):
        return self.fast_net.parameter_groups(base_lr*0.1, wd)+ self.mask_encoder.parameter_groups(base_lr*0.1, wd) + self.decoder.parameter_groups(base_lr*0.1, wd)

    def aggregate(self, prob, dim=1):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=dim, keepdim=True),
            prob
        ], dim).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits
    
    def segment_with_memory(self, mem_bank, qk, qv, f3, f2, memory_masks=None, frames=None, f4=None):
        readout_mem, values, indices = mem_bank.match_memory(qk)
        
        h, w = memory_masks[0].shape[-2:]
        ctx_lbls = torch.cat(memory_masks, 0).permute([0,2,3,1])
        # C, H*W*N
        ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)
        pred = (ctx_lbls[:, indices[0]] * values[0][None]).sum(1)
        pred = pred.view(-1, h, w)
        pred = pred[None, ...]
        
        qv = self.mask_encoder(frames.repeat(pred.size(1)-1,1,1,1), F.interpolate(pred[:, 1:].transpose(0,1), scale_factor=8, mode="bilinear", align_corners=True), f4.repeat(pred.size(1)-1,1,1,1))
  
        qv16 = torch.cat([readout_mem, qv], 1)
        prob1 = self.decoder(qv16, f3, f2)
        prob1 = torch.sigmoid(prob1)
        logits1 = self.aggregate(prob1, 0)
        return logits1, pred
    
    def encoder_value(self, frames, mask, f4):
        value4 = self.mask_encoder(frames.repeat(mask.size(0),1,1,1), mask, f4.repeat(mask.size(0),1,1,1))
        return value4
        
    def forward(self, frames, mask1=None, frames2=None, mask=None, T=None, affine=None, affine2=None, embd_only=False, norm=True, dbg=False):
        """Extract temporal correspondences
        Args:
            frames: [B,T,C,H,W]

        Returns:
            losses: a dictionary with the embedding loss
            net_outs: feature embeddings
        
        """     
        _, C, H, W = frames.shape
        # embedding for self-supervised learning
        key1, res4, k, v, f4, f3, f2 = self.fast_net(frames, norm)
        
        outs, losses = {}, {}
        if embd_only: # only embedding
            return key1, res4, k, v, f4, f3, f2
        else:
            #==========================================================================================
            # mask [B, C, H, W]
            B, N, _, h, w = mask1.shape
            # B, N, HW, C 
            mask = F.one_hot((mask1[:,:,0].long()+mask1[:,:,1].long()*2).flatten(2), 3)
            mask = mask.permute([0,1,3,2]).reshape(B*N, 3, h, w)
            mask = F.interpolate(mask.float(), (h//8, w//8), mode="bilinear", align_corners=True).reshape(B,N, 3, h//8, w//8).permute([0,2,1,3,4])
            _, C1 ,H1, W1 = f4.shape
            _, C2 ,H2, W2 = k.shape
            _, C3 ,H3, W3 = v.shape
            _, C4 ,H4, W4 = f3.shape
            _, C5 ,H5, W5 = f2.shape
            value41 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,0], mask1[:,0,0:1], f4.reshape(-1,T,C1,H1,W1)[:,0])
            value41_ = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,0], mask1[:,0,1:2],f4.reshape(-1,T,C1,H1,W1)[:,0])
            pre_values = torch.stack([value41, value41_], 2).unsqueeze(3) # B C O T H W
            # Segment frame 1 with frame 0 
            # mem_key, current_key
            affinity = batched_affinity(k.reshape(-1,T,C2,H2,W2)[:,0:1], k.reshape(-1,T,C2,H2,W2)[:,1])
            # affinity = self.memory.get_affinity(k.reshape(-1,T+1,C2,H2,W2)[:,0:1], k.reshape(-1,T+1,C2,H2,W2)[:,1])
            # aff, previous_value, current_features
            
            # B, C, H*W*N affinity B, NHW, HW
            ctx_lbls = mask[:,:,0:1].flatten(2)
            pred = torch.bmm(ctx_lbls,affinity).detach()
            pred = F.one_hot(pred.view(-1, 3, h//8, w//8).argmax(1).flatten(1), 3).permute([0,2,1]).reshape(B, 3, h//8, w//8)
#             cv2.imwrite('ex_.png', (mask[:,:,1]).permute(0,2,3,1).detach().cpu().numpy()[0]*255)
#             cv2.imwrite('ex.png', pred.permute(0,2,3,1).detach().cpu().numpy()[0]*255)
#             print(ex)
            v1 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,1], F.interpolate(pred[:, 1:2].float(), scale_factor=8, mode="bilinear", align_corners=True), f4.reshape(-1,T,C1,H1,W1)[:,1])
            v2 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,1], F.interpolate(pred[:, 2:3].float(), scale_factor=8, mode="bilinear", align_corners=True), f4.reshape(-1,T,C1,H1,W1)[:,1])
#             v1 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,1], mask1[:,1,0:1], f4.reshape(-1,T,C1,H1,W1)[:,1])
#             v2 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,1], mask1[:,1,1:2], f4.reshape(-1,T,C1,H1,W1)[:,1])
            
            logits = torch.cat([self.decoder(self.memory.readout(affinity, pre_values[:,:, 0], v1.reshape(-1,C3,H3,W3)), f3.reshape(-1,T,C4,H4,W4)[:,1], f2.reshape(-1,T,C5,H5,W5)[:,1]),
                                self.decoder(self.memory.readout(affinity, pre_values[:,:, 1], v2.reshape(-1,C3,H3,W3)), f3.reshape(-1,T,C4,H4,W4)[:,1], f2.reshape(-1,T,C5,H5,W5)[:,1])], 1)
            prob1 = torch.sigmoid(logits)
            logits1 = self.aggregate(prob1)
            prob1 = F.softmax(logits1, dim=1)[:, 1:]

            
            value42 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,1], mask1[:,1,0:1], f4.reshape(-1,T,C1,H1,W1)[:,1])
            value42_ = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,1], mask1[:,1,1:2],f4.reshape(-1,T,C1,H1,W1)[:,1])
            cur_values = torch.stack([value42, value42_], 2).unsqueeze(3) # B C O T H W
            values = torch.cat([pre_values, cur_values], 3) # B C O T H W
            # Segment frame 2 with frame 0 and 1
            # mem_key, current_key
            affinity = batched_affinity(k.reshape(-1,T,C2,H2,W2)[:,0:2], k.reshape(-1,T,C2,H2,W2)[:,2])
            # affinity = self.memory.get_affinity(k.reshape(-1,T+1,C2,H2,W2)[:,0:2], k.reshape(-1,T+1,C2,H2,W2)[:,2])
            # aff, previous_value, current_features
            
            # B, C, H*W*N affinity B, NHW, HW
            ctx_lbls = mask[:,:,0:2].flatten(2)
            pred = torch.bmm(ctx_lbls,affinity).detach()
            pred = F.one_hot(pred.view(-1, 3, h//8, w//8).argmax(1).flatten(1), 3).permute([0,2,1]).reshape(B, 3, h//8, w//8)
            v1 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,2], F.interpolate(pred[:, 1:2].float(), scale_factor=8, mode="bilinear", align_corners=True), f4.reshape(-1,T,C1,H1,W1)[:,2])
            v2 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,2], F.interpolate(pred[:, 2:3].float(), scale_factor=8, mode="bilinear", align_corners=True), f4.reshape(-1,T,C1,H1,W1)[:,2])
#             v1 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,2], mask1[:,2,0:1], f4.reshape(-1,T,C1,H1,W1)[:,2])
#             v2 = self.mask_encoder(frames.reshape(-1,T,C,H,W)[:,2], mask1[:,2,1:2], f4.reshape(-1,T,C1,H1,W1)[:,2])
            
            logits = torch.cat([self.decoder(self.memory.readout(affinity, values[:,:,0], v1.reshape(-1,C3,H3,W3)), f3.reshape(-1,T,C4,H4,W4)[:,2], f2.reshape(-1,T,C5,H5,W5)[:,2]),
                                self.decoder(self.memory.readout(affinity, values[:,:,1], v2.reshape(-1,C3,H3,W3)), f3.reshape(-1,T,C4,H4,W4)[:,2], f2.reshape(-1,T,C5,H5,W5)[:,2])], 1)
            prob2 = torch.sigmoid(logits)
            logits2 = self.aggregate(prob2)
            prob2 = F.softmax(logits2, dim=1)[:, 1:]
            losses['mask1'] = F.cross_entropy(logits1, mask1[:,1,0].long()+mask1[:,1,1].long()*2)
            losses['mask2'] = F.cross_entropy(logits2, mask1[:,2,0].long()+mask1[:,2,1].long()*2)
#             losses['mask1'], P = self.bce(logits1, mask1[:,1,0].long()+mask1[:,1,1].long()*2, self.count[0])
#             losses['mask2'], P = self.bce(logits2, mask1[:,2,0].long()+mask1[:,2,1].long()*2, self.count[0])
#             self.count[0]+=1
            
            #==========================================================================================

            # computing the main loss
            losses["main"] = 0.5*losses['mask1'] + 0.5*losses['mask2']

        return losses, outs
