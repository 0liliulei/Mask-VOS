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

from torch.utils.data import DataLoader
from datasets.dataloader_infer import DataSeg

# deterministic inference
from torch.backends import cudnn

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

VERBOSE = True
#==========================================================================================
from models.modules import pad_divide_by, unpad
from sklearn.cluster import KMeans
# from kmeans_pytorch import kmeans
import cv2

def convert_dict(state_dict):
    new_dict = {}
    for k,v in state_dict.items():
        new_key = k.replace("module.", "")
        new_dict[new_key] = v
    return new_dict

def write_mask(n_cluster, km3_labels, H, W, T):
    km3_labels = km3_labels.reshape([H,W])
    for y in range(n_cluster):
        a2 = km3_labels==y
        a3=a2*255
        cv2.imwrite(str(T)+'_'+str(y)+'.png', a3)

# def step_seg(cfg, net, frames, fns, seq_name, pre_fix):
    
#     T = frames.shape[0]
#     frames = frames.cuda()
    
#     fetch = {"res3": lambda x: x[0], \
#              "res4": lambda x: x[1], \
#              "key": lambda x: x[2]}
    
#     H,W = frames.shape[-2:]
    
#     frames_batch = frames[0:1]
#     nxt_embds = net(frames_batch, embd_only=True, norm=True)
#     query = fetch[cfg.TEST.KEY](nxt_embds)
    
#     n_cluster = 5
#     B,C,H,W = query.size()
#     X = F.interpolate(query, scale_factor=4, mode='bilinear', align_corners=True)
#     X = X[0:1].flatten(2).squeeze().transpose(0,1).cpu().detach().numpy()
#     km3 = KMeans(n_clusters=n_cluster)
#     km3.fit(X)
#     km3_labels = km3.labels_
#     center_coord = torch.tensor(km3.cluster_centers_).cuda()
    
#     os.makedirs(pre_fix+'/'+seq_name[0], exist_ok=True)
#     write_mask(n_cluster, km3_labels, H, W, pre_fix+'/'+seq_name[0]+'/'+fns[0][0])

    
#     print(">", end='')
#     for t in range(1, T, 4):
#         print(".", end='')
#         sys.stdout.flush()

#         frames_batch = frames[t:t+4]
#         nxt_embds = net(frames_batch, embd_only=True, norm=True)
#         query = fetch[cfg.TEST.KEY](nxt_embds)
#         query = F.interpolate(query, scale_factor=4, mode='bilinear', align_corners=True)
#         query = query.flatten(2).transpose(1,2)
#         distance = torch.zeros((query.shape[0], km3_labels.shape[0], n_cluster))
#         # B, W*H, C, n_cluster.    1, 1, C, n_cluster
#         ab = (query.unsqueeze(-1)-center_coord.transpose(0,1).unsqueeze(0).unsqueeze(0))**2
#         # B, W*H, n_cluster
#         distance  = torch.mean(ab, dim=2).squeeze(2)
#         index = np.argmin(distance.cpu().numpy(), axis=2)
#         for ii in range(index.shape[0]):
#             write_mask(n_cluster, index[ii], H, W, pre_fix+'/'+seq_name[0]+'/'+fns[t+ii][0])

#     print('<')


#     return None,None

def step_seg(cfg, net, frames, fns, seq_name, pre_fix):
    
    T = frames.shape[0]
    frames = frames.cuda()
    
    fetch = {"res3": lambda x: x[0], \
             "res4": lambda x: x[1], \
             "key": lambda x: x[2]}
    
    H,W = frames.shape[-2:]
    

    querys = []
    print(">", end='')
    for t in range(0, T, 4):
        print(".", end='')
        sys.stdout.flush()

        frames_batch = frames[t:t+4]
        nxt_embds = net(frames_batch, embd_only=True, norm=True)
        query = fetch[cfg.TEST.KEY](nxt_embds)
#         query = F.interpolate(query, scale_factor=2, mode='bilinear', align_corners=True)
        querys.append(query)
    querys = torch.cat(querys, dim=0)
    query = querys[::8]
    n_cluster = 7
    B,C,H,W = query.size()
    X = query.transpose(0,1).flatten(1).transpose(0,1)
    
#     km3_labels, center_coord = kmeans(
#         X=X, num_clusters=n_cluster, distance='euclidean', device = torch.device('cuda:0')
#     )
#     km3_labels = km3_labels.reshape(B,H,W)
#     print(km3_labels.shape)
#     print(center_coord.shape)
#     cluster_ids_y = kmeans_predict(
#         y, cluster_centers, 'euclidean', device=torch.device('cuda:0')
#     )

    km3 = KMeans(n_clusters=n_cluster)
    # https://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance
    km3.fit(X.cpu().detach().numpy())
    km3_labels = km3.labels_.reshape(B,H,W)
#     print(km3.n_iter_)
#     print(km3_labels.shape)
    center_coord = torch.tensor(km3.cluster_centers_).cuda()
#     print(center_coord.shape)
    
    os.makedirs(pre_fix+'/'+seq_name[0], exist_ok=True)
    print(">", end='')
    for t in range(0, T):
        print(".", end='')
        sys.stdout.flush()
        query = querys[t:t+1]
        query = query.flatten(2).transpose(1,2)
        distance = torch.zeros((query.shape[0], km3_labels.shape[0], n_cluster))
        # B, W*H, C, n_cluster.    1, 1, C, n_cluster
        ab = (query.unsqueeze(-1)-center_coord.transpose(0,1).unsqueeze(0).unsqueeze(0))**2
        # B, W*H, n_cluster
        distance  = torch.mean(ab, dim=2).squeeze(2)
        index = np.argmin(distance.cpu().numpy(), axis=2)
        for ii in range(index.shape[0]):
            write_mask(n_cluster, index[ii], H, W, pre_fix+'/'+seq_name[0]+'/'+fns[t+ii][0])

    print('<')


    return None,None

if __name__ == '__main__':

    # loading the model
    args = get_arguments(sys.argv[1:])

    # reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # initialising the dirs
    check_dir(args.mask_output_dir, "{}_mask".format(cfg.TEST.KEY))

    # Loading the model
    model = get_model(cfg, remove_layers=cfg.MODEL.REMOVE_LAYERS)

    if not os.path.isfile(args.resume):
        print("[W]: ", "Snapshot not found: {}".format(args.resume))
        print("[W]: Using a random model")
    else:
        state_dict = convert_dict(torch.load(args.resume)["model"])
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

    for iter, batch in enumerate(dataloader):
        frames_orig, frames, masks_gt, tracks, num_ids, fns, flags, seq_name = batch

        print("Sequence {:02d} | {}".format(iter, seq_name[0]))

        frames_orig = frames_orig.flatten(0,1)
        frames = frames.flatten(0,1)
        
        prefix = os.path.join(args.mask_output_dir, "{}_mask".format(cfg.TEST.KEY))
        with torch.no_grad():
            masks_pred, masks_conf = step_seg(cfg, model, frames, fns, seq_name, prefix)
            
        timer.stage("Inference completed")

    timer.stage("Inference completed")

