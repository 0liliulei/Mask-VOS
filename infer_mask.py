"""
Single-scale inference
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import sys
import numpy as np
import imageio
import time

import torch.multiprocessing as mp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from opts import get_arguments
from core.config import cfg, cfg_from_file, cfg_from_list
from models import get_model
from utils.timer import Timer
from utils.sys_tools import check_dir
from utils.palette_davis import palette as davis_palette

from torch.utils.data import DataLoader
from datasets.dataloader_infer import DataSeg

# deterministic inference
from torch.backends import cudnn
import cv2
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

VERBOSE = False
#==========================================================================================
from models.modules import MemoryBank_dotproduct, pad_divide_by, unpad

def mask2rgb(mask, palette):
    mask_rgb = palette(mask)
    mask_rgb = mask_rgb[:,:,:3]
    return mask_rgb

def mask_overlay(mask, image, palette):
    """Creates an overlayed mask visualisation"""
    mask_rgb = mask2rgb(mask, palette)
    return 0.3 * image + 0.7 * mask_rgb

class ResultWriter:
    
    def __init__(self, key, palette, out_path):
        self.key = key
        self.palette = palette
        self.out_path = out_path
        self.verbose = VERBOSE

    def save(self, frames, masks_pred, masks_gt, flags, fn, seq_name, start=0):

        subdir_vos = os.path.join(self.out_path, "{}_vos".format(self.key))
        check_dir(subdir_vos, seq_name)

        subdir_vis = os.path.join(self.out_path, "{}_vis".format(self.key))
        check_dir(subdir_vis, seq_name)

        for frame_id, mask in enumerate(masks_pred.split(1, 0)):

            mask = mask[0].numpy().astype(np.uint8)
            filepath = os.path.join(subdir_vos, seq_name, "{}.png".format(fn[frame_id+start][0]))

            # saving only every 5th frame
            if flags[frame_id] != 0:
                imageio.imwrite(filepath, mask)

            if self.verbose:
                frame = frames[frame_id].numpy()
                #mask_gt = masks_gt[frame_id].numpy().astype(np.uint8)
                #masks = np.concatenate([mask, mask_gt], 1)
                #frame = np.concatenate([frame, frame], 2)
                frame = np.transpose(frame, [1,2,0])

                overlay = mask_overlay(mask, frame, self.palette)
                filepath = os.path.join(subdir_vis, seq_name, "{}.png".format(fn[frame_id+start][0]))
                imageio.imwrite(filepath, (overlay * 255.).astype(np.uint8))


def convert_dict(state_dict):
    new_dict = {}
    for k,v in state_dict.items():
        new_key = k.replace("module.", "")
        new_dict[new_key] = v
    return new_dict

def mask2tensor(mask, idx, num_classes=cfg.DATASET.NUM_CLASSES):
    h,w = mask.shape
    mask_t = torch.zeros(1,num_classes,h,w)
    mask_t[0, idx] = mask
    return mask_t

def configure_tracks(masks_gt, tracks, num_objects):
    """Selecting masks for initialisation

    Args:
        masks_gt: [T,H,W]
        tracks: [T,2]

    """
    init_masks = {}

    # we always have first mask
    # if there are no instances, it will be simply zero
    H,W = masks_gt[0].shape[-2:]
    init_masks[0] = torch.zeros(1, cfg.DATASET.NUM_CLASSES, H, W)

    for oid in range(cfg.DATASET.NUM_CLASSES):

        t = tracks[oid].item()
        if not t in init_masks:
            init_masks[t] = mask2tensor(masks_gt[oid], oid)
        else:
            init_masks[t] += mask2tensor(masks_gt[oid], oid)

    return init_masks

def scale_smallest(frame, a):
    H,W = frame.shape[-2:]
    s = a / min(H, W)
    h, w = int(s * H), int(s * W)
    return F.interpolate(frame, (h, w), mode="bilinear", align_corners=True)

def step_seg(cfg, net, frames, mask_init, n_obj, start, tracks):

    # dense tracking: start from the 1st frame
    # keep track of new objects

    T = frames.shape[0]
    frames = frames.cuda()

    # scale smallest
    if cfg.TEST.INPUT_SIZE > 0:
        frames = scale_smallest(frames, cfg.TEST.INPUT_SIZE)

    for t in mask_init.keys():
        mask_init[t] = mask_init[t].cuda()
        
        
        
    ori_H, ori_W = mask_init[start].shape[-2:]
    if not ori_W<960:
        frames = F.interpolate(frames, (frames.size(2)//2, frames.size(3)//2), mode='bilinear')
        for t in mask_init.keys():
            mask_init[t] = mask_init[t][:,:,::2,::2]
    mem_bank = MemoryBank_dotproduct(n_obj.item()-1, cfg.TEST.RADIUS, cfg.TEST.KNN)
    #==========================================================================================
    frames, pad = pad_divide_by(frames, 8)
    H,W = frames.shape[-2:]
    for t in mask_init.keys():
        mask_init[t], _ = pad_divide_by(mask_init[t], 8)

    scale = lambda x, hw: F.interpolate(x, hw, mode="bilinear", align_corners=True)
    
    # initialising
    all_masks=[mask_init[start][:,:n_obj]]
    long_memory_masks = [scale(mask_init[start][:,:n_obj], (H//8, W//8))]
    short_memory_masks = []
    memory_masks = long_memory_masks+short_memory_masks

    print(">", end='')
    for t in range(0, T):
        print(".", end='')
        sys.stdout.flush()
        # next frame
        frames_batch = frames[t:t+1]
        key1, res4, qk, qv, f4, f3, f2 = net(frames_batch, embd_only=True)
        # results
        if t != 0:
            out_mask, pre = net.segment_with_memory(mem_bank, qk, qv, f3, f2, memory_masks, frames_batch, f4)
            if (t+start) in tracks.numpy():
                middle = (out_mask.transpose(0,1)+scale(pre, (H, W)))/2
                label_id = tracks.numpy().tolist().index(t+start)
                middle[:,label_id] = net.aggregate(mask_init[t+start][:, label_id], 0)[1:,]
                all_masks.append(middle)  
            else:
                all_masks.append((out_mask.transpose(0,1)+scale(pre, (H, W)))/2)   
        value = net.encoder_value(frames_batch, all_masks[t][:, 1:].transpose(0,1), f4)
        if (t+start) in tracks.numpy():
            mem_bank.add_memory(qk, value, is_long=True)
        else:
            mem_bank.add_memory(qk, value, is_long=False)
        if t!=0:
            if (t+start) in tracks.numpy():
                pre[:,label_id:label_id+1] = scale(mask_init[t+start][:, label_id:label_id+1], (H//8, W//8))
                long_memory_masks.append(pre)
            if len(memory_masks)>=6:
                short_memory_masks = short_memory_masks[1:]+[pre]
                memory_masks = long_memory_masks+short_memory_masks
            else:
                short_memory_masks.append(pre)
                memory_masks = long_memory_masks+short_memory_masks
        if t==0:
            for ss in range(5):
                short_memory_masks.append(scale(mask_init[start][:,:n_obj], (H//8, W//8)))
                memory_masks = long_memory_masks+short_memory_masks
                mem_bank.add_memory(qk, value, is_long=False)
            
    print('<')
    masks_pred = torch.cat(all_masks, 0)                                                                            
    masks_pred = unpad(masks_pred, pad)  
    #==========================================================================================

    return masks_pred


if __name__ == '__main__':

    # loading the model
    args = get_arguments(sys.argv[1:])

    # reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # initialising the dirs
    check_dir(args.mask_output_dir, "{}_vis".format(cfg.TEST.KEY))
    check_dir(args.mask_output_dir, "{}_vos".format(cfg.TEST.KEY))

    # Loading the model
    model = get_model(cfg, remove_layers=cfg.MODEL.REMOVE_LAYERS)

    if not os.path.isfile(args.resume):
        print("[W]: ", "Snapshot not found: {}".format(args.resume))
        print("[W]: Using a random model")
    else:
        state_dict = convert_dict(torch.load(args.resume)["model"])
#         state_dict3 = convert_dict(torch.load('epoch120_score0.675_key.pth')["model"])
#         for item in list(state_dict.keys()):
#             if item.startswith('fast_net.emb_q'):
#                 print(item)
#                 state_dict[item] = state_dict3[item]
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print("Error while loading the snapshot:\n", str(e))
            print("Resuming...")
            model.load_state_dict(state_dict, strict=False)

    for p in model.parameters():
        p.requires_grad = False

    # setting the evaluation mode
    model.eval()
    model = model.cuda()
    dataset = DataSeg(cfg, args.infer_list)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, \
                                    drop_last=False) #, num_workers=args.workers)
    palette = dataloader.dataset.get_palette()

    timer = Timer()
    N = len(dataloader)

    pool = mp.Pool(processes=args.workers)
    writer = ResultWriter(cfg.TEST.KEY, davis_palette, args.mask_output_dir)

    scale = lambda x, hw: F.interpolate(x, hw, mode="bilinear", align_corners=True)
    for iter, batch in enumerate(dataloader):
        frames_orig, frames, masks_gt, tracks, num_ids, fns, flags, seq_name = batch
        
        print("Sequence {:02d} | {}".format(iter, seq_name[0]))
        start = tracks[0][0].item()
        print(tracks[0])
        masks_gt = masks_gt.flatten(0,1)
        frames_orig = frames_orig.flatten(0,1)[start:]
        frames = frames.flatten(0,1)[start:]
        tracks = tracks.flatten(0,1)
        flags = flags.flatten(0,1)[start:]
        if seq_name[0]=='48b5ddd1f9' or seq_name[0]=='6031809500':
            num_ids[0] = num_ids[0]+1

        init_masks = configure_tracks(masks_gt, tracks, num_ids[0])
#         if start!=0:
#             init_masks.pop(0)
        assert 0 in init_masks, "initial frame has no instances"

        with torch.no_grad():
            init_masks_ = {}
            for t in init_masks.keys():
                init_masks_[t] = init_masks[t].flip([3])
            ori_H, ori_W = init_masks[start].shape[-2:]
            masks_pred = step_seg(cfg, model, frames, init_masks, num_ids[0], start, tracks)
#             print(masks_pred.shape)
#             np.save('pro.npy',masks_pred.cpu().numpy())
#             masks_pred2 = step_seg(cfg, model, frames.flip([3]), init_masks_, num_ids[0], start, tracks)
#             masks_pred = masks_pred1 + masks_pred2.flip([3])
            
            if masks_pred.size(0)>160:
#                 new_masks_pred = []
#                 masks_pred = masks_pred.argmax(1).cpu().numpy().astype("uint8")
#                 for ii in range(masks_pred.shape[0]):
#                     new_masks_pred.append(torch.tensor(cv2.resize(masks_pred[ii], (ori_W, ori_H))))
#                 masks_pred = torch.stack(new_masks_pred, dim=0)
                masks_pred1 = scale(masks_pred[:80], (ori_H, ori_W))
                masks_pred1 = masks_pred1.argmax(1) 
                masks_pred2 = scale(masks_pred[80:], (ori_H, ori_W))
                masks_pred2 = masks_pred2.argmax(1)
                masks_pred = torch.cat([masks_pred1, masks_pred2], dim=0)
            else:
                masks_pred = scale(masks_pred, (ori_H, ori_W))
                masks_pred = masks_pred.argmax(1) 
        
        frames_orig = dataset.denorm(frames_orig)
        pool.apply_async(writer.save, args=(frames_orig, masks_pred.cpu(), masks_gt.cpu(), flags, fns, seq_name[0], start))

    timer.stage("Inference completed")
    pool.close()
    pool.join()