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

def batched_affinity(keys, query, TEM=0.05):
    '''
    Mini-batched computation of affinity, for memory efficiency
    (less aggressively mini-batched)
    '''
    keys = keys.transpose(1,2).unsqueeze(2).flatten(4)
    query = query.unsqueeze(2).flatten(3)
    
    A = torch.einsum('ijklm,ijkn->iklmn', keys, query)

    B, N, T, h1w1, hw = A.shape
    A = A.view(B, T*h1w1, hw)
    A /= TEM
    
    affinity = F.softmax(A, dim=-2)

    return affinity

class FrameworkQuan(BaseNet):

    def __init__(self, cfg, net, mask_encoder):
        super(FrameworkQuan, self).__init__()

        self.cfg = cfg
        self.fast_net = net
        self.mask_encoder = mask_encoder
        self.memory = MemoryReader()
        self.decoder = Decoder()
        self.eye = None

    def parameter_groups(self, base_lr, wd):
        return self.fast_net.parameter_groups(base_lr*0.1, wd) + self.mask_encoder.parameter_groups(base_lr*0.1, wd) + self.decoder.parameter_groups(base_lr*0.1, wd)

    def _align(self, x, t):
        tf = F.affine_grid(t, size=x.size(), align_corners=False)
        return F.grid_sample(x, tf, align_corners=False, mode="nearest")

    def _key_val(self, ctr, q):
        """
        Args:
            ctr: [N,K]
            q: [BHW,K]
        Returns:
            val: [BHW,N]
        """

        # [BHW,K] x [N,K].t -> [BHWxN]
        vals = torch.mm(q, ctr.t()) # [BHW,N]

        # normalising attention
        return vals / self.cfg.TEST.TEMP

    def _sample_index(self, x, T, N):
        """Sample indices of the anchors

        Args:
            x: [BT,K,H,W]
        Returns:
            index: [B,N*N,K]
        """

        BT,K,H,W = x.shape
        B = x.view(-1,T,K,H*W).shape[0]

        # sample indices from a uniform grid
        xs, ys = W // N, H // N
        x_sample = torch.arange(0, W, xs).view(1, 1, N)
        y_sample = torch.arange(0, H, ys).view(1, N, 1)

        # Random offsets
        # [B x 1 x N]
        x_sample = x_sample + torch.randint(0, xs, (B, 1, 1))
        # [B x N x 1]
        y_sample = y_sample + torch.randint(0, ys, (B, 1, 1))

        # batch index
        # [B x N x N]
        hw_index = torch.LongTensor(x_sample + y_sample * W)

        return hw_index

    def _sample_from(self, x, index, T, N):
        """Gather the features based on the index

        Args:
            x: [BT,K,H,W]
            index: [B,N,N] defines the indices of NxN grid for a single
                           frame in each of B video clips
        Returns:
            anchors: [BNN,K] sampled features given by index from x
        """

        BT,K,H,W = x.shape
        x = x.view(-1,T,K,H*W)
        B = x.shape[0]

        # > [B,T,K,HW] > [B,T,HW,K] > [B,THW,K]
        x = x.permute([0,1,3,2]).reshape(B,-1,K)

        # every video clip will have the same samples
        # on the grid
        # [B x N x N] -> [B x N*N x 1] -> [B x N*N x K]
        index = index.view(B,-1,1).expand(-1,-1,K)

        # selecting from the uniform grid
        y = x.gather(1, index.to(x.device))

        # [BNN,K]
        return y.flatten(0,1)

    def _mark_from(self, x, index, T, N, fill_value=0):
        """This is analogous to _sample_from except that
        here we simply "mark" the sampled positions in the tensor
        Used for visualisation only.
        Since it is a binary mask, K == 1

        Args:
            x: [BT,1,H,W] binary mask
            index: [B,N,N] defines the indices of NxN grid for a single
                           frame in each of B video clips
        Returns:
            y: [BT,1,H,W] marked sample positions
        """

        BT,K,H,W = x.shape
        assert K == 1, "Expected binary mask"
        x = x.view(-1,T,K,H*W)
        B = x.shape[0]

        # > [B,T,K,HW] > [B,T,HW,K] > [B,THW,K]
        x = x.permute([0,1,3,2]).reshape(B,-1,K)

        # every video clip will have the same samples
        # on the grid
        # [B x N x N] -> [B x N*N x 1] -> [B x N*N x K]
        index = index.view(B,-1,1).expand(-1,-1,K)

        # selecting from the uniform grid
        # [B x T*H*W x K]
        y = x.scatter(1, index.to(x.device), fill_value)

        # [B x T*H*W x K] -> [BT x K x H x W]
        return y.view(-1,H*W,K).permute([0,2,1]).view(-1,K,H,W)

    def _cluster_grid(self, k1, k2, aff1, aff2, T, index=None):
        """ Selecting clusters within a sequence
        Args:
            k1: [BT,K,H,W]
            k2: [BT,K,H,W]
        """

        BT,K,H,W = k1.shape
        assert BT % T == 0, "Batch not divisible by sequence length"
        B = BT // T

        # N = [G x G]
        N = self.cfg.MODEL.GRID_SIZE ** 2

        # [BT,K,H,W] -> [BTHW,K]
        flatten = lambda x: x.flatten(2,3).permute([0,2,1]).flatten(0,1)

        # [BTHW,BN] -> [BT,BN,H,W]
        def unflatten(x, aff=None):
            x = x.view(BT,H*W,-1).permute([0,2,1]).view(BT,-1,H,W)
            if aff is None:
                return x
            return self._align(x, aff)

        index = self._sample_index(k1, T, N = self.cfg.MODEL.GRID_SIZE)
        # [BNN,K]
        query1 = self._sample_from(k1, index, T, N = self.cfg.MODEL.GRID_SIZE)

        """Computing the distances and pseudo labels"""

        # [BTHW,K]
        k1_ = flatten(k1)
        k2_ = flatten(k2)

        # [BTHW,BN] -> [BTHW,BN] -> [BT,BN,H,W]
        vals_soft = unflatten(self._key_val(query1, k1_), aff1)
        vals_pseudo = unflatten(self._key_val(query1, k2_), aff2)

        # [BT,BN,H,W]
        probs_pseudo = self._pseudo_mask(vals_pseudo, T)
        probs_pseudo2 = self._pseudo_mask(vals_soft, T)

        pseudo = probs_pseudo.argmax(1)
        pseudo2 = probs_pseudo2.argmax(1)

        # mask
        def grid_mask():
            grid_mask = torch.ones(BT,1,H,W).to(pseudo.device)
            return self._mark_from(grid_mask, index, T, N = self.cfg.MODEL.GRID_SIZE)

        return vals_soft, pseudo, index, [vals_pseudo, pseudo2, grid_mask]

    # sampling affinity
    def _aff_sample(self, k1, k2, T):
        BT,K,h,w = k1.size()
        B = BT // T
        hw = h*w

        def gen(query):
            grid_mask = torch.ones(B,1,hw).to(k1.device)
            # generating random indices
            indices = torch.randint(0, hw, (B,1,1)).to(k1.device)
            grid_mask.scatter_(2, indices, 0)

            # [B,K,H,W] -> [B,K,1]
            query_ = query[::T].view(B,K,-1).gather(2, indices.expand(-1,K,-1))

            def aff(keys):
                k = keys.view(B,T,K,-1)
                # [B,T,K,HW] x [B,1,K,HW] -> [B,T,HW]
                aff = (k * query_[:,None,:,:]).sum(2)
                return (aff + 1) / 2


            aff1 = aff(k1)
            aff2 = aff(k2)

            return grid_mask.view(B,h,w), aff1.view(BT,h,w), aff2.view(BT,h,w)

        grid_mask1, aff1_1, aff1_2 = gen(k1)
        grid_mask2, aff2_1, aff2_2 = gen(k2)

        return grid_mask1, aff1_1, aff1_2, \
                grid_mask2, aff2_1, aff2_2

    def _pseudo_mask(self, logits, T):
        BT,K,h,w = logits.shape
        assert BT % T == 0, "Batch not divisible by sequence length"
        B = BT // T

        # N = [G x G]
        N = self.cfg.MODEL.GRID_SIZE ** 2

        # generating a pseudo label
        # first we need to mask out the affinities across the batch
        if self.eye is None or self.eye.shape[0] != B*T \
                            or self.eye.shape[1] != B*N:
            eye = torch.eye(B)[:,:,None].expand(-1,-1,N).reshape(B,-1)
            eye = eye.unsqueeze(1).expand(-1,T,-1).reshape(B*T, B*N, 1, 1)
            self.eye = eye.to(logits.device)

        probs = F.softmax(logits, 1)
        return probs * self.eye

    def _ref_loss(self, x, y, N = 4):
        B,_,h,w = x.shape

        index = self._sample_index(x, T=1, N=N)
        x1 = self._sample_from(x, index, T=1, N=N)
        y1 = self._sample_from(y, index, T=1, N=N)
        logits = torch.mm(x1, y1.t()) / self.cfg.TEST.TEMP

        labels = torch.arange(logits.size(1)).to(logits.device)
        return F.cross_entropy(logits, labels)

    def _ce_loss(self, x, pseudo_map, T, eps=1e-5):
        error_map = F.cross_entropy(x, pseudo_map, reduction="none", ignore_index=-1)

        BT,h,w = error_map.shape
        errors = error_map.view(-1,T,h,w)
        error_ref, error_t = errors[:,0], errors[:,1:]

        return error_ref.mean(), error_t.mean(), error_map

    def _forward_reg(self, frames2, norm):
        losses = {}

        if not self.cfg.TRAIN.STOP_GRAD:
            key2, res4, qk, qv, f4, f3 = self.fast_net(frames2, norm)
            return key2, res4, losses

        training = self.fast_net.training
        if self.cfg.TRAIN.BLOCK_BN:
            self.fast_net.eval()

        with torch.no_grad():
            key2, res4, qk, qv, f4, f3 = self.fast_net(frames2, norm)

        if self.cfg.TRAIN.BLOCK_BN:
            self.fast_net.train(training)

        return key2, res4, losses

    def fetch_first(self, x1, x2, T):
        assert x1.shape[1:] == x2.shape[1:]
        c,h,w = x1.shape[1:]

        x1 = x1.view(-1,T+1,c,h,w)
        x2 = x2.view(-1,T-1,c,h,w)

        x2 = torch.cat((x1[:,-1:], x2), 1)
        x1 = x1[:,:-1]

        return x1.flatten(0,1), x2.flatten(0,1)

    def aggregate(self, prob, dim=1):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=dim, keepdim=True),
            prob
        ], dim).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits
    
    def segment_with_memory(self, mem_bank, qk, qv, f3):
        readout_mem = mem_bank.match_memory(qk)
        qv = qv.expand(readout_mem.size(0), -1, -1, -1)
        qv16 = torch.cat([readout_mem, qv], 1)
        prob1 = torch.sigmoid(self.decoder(qv16, f3))
        logits1 = self.aggregate(prob1, 0)
        return logits1
    
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
        key1, res4, k, v, f4, f3 = self.fast_net(frames, norm)
        
        outs, losses = {}, {}
        if embd_only: # only embedding
            return key1, res4, k, v, f4, f3
        else:
            key2, res4_2, losses = self._forward_reg(frames2, norm)
            
            #==========================================================================================
            # mask [B, C, H, W]
            _, C1 ,H1, W1 = f4.shape
            _, C2 ,H2, W2 = k.shape
            _, C3 ,H3, W3 = v.shape
            _, C4 ,H4, W4 = f3.shape
            value41 = self.mask_encoder(frames.reshape(-1,T+1,C,H,W)[:,0], mask1[:,0,0:1], f4.reshape(-1,T+1,C1,H1,W1)[:,0])
            value41_ = self.mask_encoder(frames.reshape(-1,T+1,C,H,W)[:,0], mask1[:,0,1:2],f4.reshape(-1,T+1,C1,H1,W1)[:,0])
            pre_values = torch.stack([value41, value41_], 2).unsqueeze(3) # B C O T H W
            # Segment frame 1 with frame 0 
            # mem_key, current_key
            affinity = batched_affinity(k.reshape(-1,T+1,C2,H2,W2)[:,0:1], k.reshape(-1,T+1,C2,H2,W2)[:,1])
            # affinity = self.memory.get_affinity(k.reshape(-1,T+1,C2,H2,W2)[:,0:1], k.reshape(-1,T+1,C2,H2,W2)[:,1])
            # aff, previous_value, current_features
            logits = torch.cat([self.decoder(self.memory.readout(affinity, pre_values[:,:, 0], v.reshape(-1,T+1,C3,H3,W3)[:,1]), f3.reshape(-1,T+1,C4,H4,W4)[:,1]),
                                self.decoder(self.memory.readout(affinity, pre_values[:,:, 1], v.reshape(-1,T+1,C3,H3,W3)[:,1]), f3.reshape(-1,T+1,C4,H4,W4)[:,1])], 1)
            prob1 = torch.sigmoid(logits)
            logits1 = self.aggregate(prob1)
            prob1 = F.softmax(logits1, dim=1)[:, 1:]

            value42 = self.mask_encoder(frames.reshape(-1,T+1,C,H,W)[:,1], mask1[:,1,0:1], f4.reshape(-1,T+1,C1,H1,W1)[:,1])
            value42_ = self.mask_encoder(frames.reshape(-1,T+1,C,H,W)[:,1], mask1[:,1,1:2],f4.reshape(-1,T+1,C1,H1,W1)[:,1])
            cur_values = torch.stack([value42, value42_], 2).unsqueeze(3) # B C O T H W
            values = torch.cat([pre_values, cur_values], 3) # B C O T H W
            # Segment frame 2 with frame 0 and 1
            # mem_key, current_key
            affinity = batched_affinity(k.reshape(-1,T+1,C2,H2,W2)[:,0:2], k.reshape(-1,T+1,C2,H2,W2)[:,2])
            # affinity = self.memory.get_affinity(k.reshape(-1,T+1,C2,H2,W2)[:,0:2], k.reshape(-1,T+1,C2,H2,W2)[:,2])
            # aff, previous_value, current_features
            logits = torch.cat([self.decoder(self.memory.readout(affinity, values[:,:,0], v.reshape(-1,T+1,C3,H3,W3)[:,2]), f3.reshape(-1,T+1,C4,H4,W4)[:,2]),
                                self.decoder(self.memory.readout(affinity, values[:,:,1], v.reshape(-1,T+1,C3,H3,W3)[:,2]), f3.reshape(-1,T+1,C4,H4,W4)[:,2])], 1)
            prob2 = torch.sigmoid(logits)
            logits2 = self.aggregate(prob2)
            prob2 = F.softmax(logits2, dim=1)[:, 1:]
#             cv2.imwrite( 'a.jpg', mask1[0,1,0].long().cpu().numpy()*255)
#             cv2.imwrite( 'b.jpg', mask1[0,2,0].long().cpu().numpy()*255)
#             cv2.imwrite( 'a1.jpg', frames.reshape(-1,T+1,C,H,W)[0,1].permute(1,2,0).cpu().numpy()*255)
#             cv2.imwrite( 'b1.jpg', frames.reshape(-1,T+1,C,H,W)[0,2].permute(1,2,0).cpu().numpy()*255)
#             print(asss)
            losses['mask1'] = F.cross_entropy(logits1, mask1[:,1,0].long()+mask1[:,1,1].long()*2)
            losses['mask2'] = F.cross_entropy(logits2, mask1[:,2,0].long()+mask1[:,2,1].long()*2)
            #==========================================================================================

            # fetching the first frame from the second view
            key1, key2 = self.fetch_first(key1, key2, T)

            vals, pseudo, index, dbg_info = self._cluster_grid(key1, key2, affine, affine2, T)

            vals_pseudo, pseudo2, grid_mask = dbg_info

            key1_aligned = self._align(key1, affine)
            key2_aligned = self._align(key2, affine2)

            n_ref = self.cfg.MODEL.GRID_SIZE_REF
            losses["cross_key"] = self._ref_loss(key1_aligned[::T], key2_aligned[::T], N = n_ref)

            # losses
            _, losses["temp"], outs["error_map"] = self._ce_loss(vals, pseudo, T)

            # computing the main loss
            losses["main"] = self.cfg.MODEL.CE_REF * losses["cross_key"] + losses["temp"] + 0.5*losses['mask1'] + 0.5*losses['mask2']

            if dbg:
                vals = F.softmax(vals, 1)
                vals_pseudo = F.softmax(vals_pseudo, 1)

                frames, frames2 = self.fetch_first(frames, frames2, T)
                outs["frames_orig"] = frames
                outs["frames"] = self._align(frames, affine)
                outs["frames2"] = self._align(frames2, affine2)

                outs["map_soft"] = vals
                outs["map"] = pseudo
                outs["map_target_soft"] = vals_pseudo
                outs["map_target"] = pseudo2
                outs["grid_mask"] = grid_mask()

                outs["aff_mask1"], outs["aff11"], outs["aff12"], \
                        outs["aff_mask2"], outs["aff21"], outs["aff22"] = self._aff_sample(key1, key2, T)

        return losses, outs
