# """
# Copyright (c) 2021 TU Darmstadt
# Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
# License: Apache License 2.0
# """

# from __future__ import print_function

# import os
# import sys
# import numpy as np
# import time
# import random
# import setproctitle

# from functools import partial

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from datasets import *
# from models import get_model

# from base_trainer import BaseTrainer

# from opts import get_arguments
# from core.config import cfg, cfg_from_file, cfg_from_list
# from utils.timer import Timer
# from utils.stat_manager import StatManager
# from utils.davis2017 import evaluate_semi
# from labelprop.crw import CRW

# from models.modules import MemoryBank_dotproduct, pad_divide_by, unpad

# from torch.utils.tensorboard import SummaryWriter

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False


# import cv2
# from PIL import Image as PILImage
# def  get_pseudo_color_map(pred):
#     pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
#     color_map = get_color_map_list(256)
# #     color_map = get_cityscapes_colors()
#     pred_mask.putpalette(color_map)
#     return pred_mask
# def get_color_map_list(num_classes):
#     """
#     Returns the color map for visualizing the segmentation mask,
#     which can support arbitrary number of classes.

#     Args:
#         num_classes (int): Number of classes.

#     Returns:
#         (list). The color map.
#     """

#     num_classes += 1
#     color_map = num_classes * [0, 0, 0]
#     for i in range(0, num_classes):
#         j = 0
#         lab = i
#         while lab:
#             color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
#             color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
#             color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
#             j += 1
#             lab >>= 3
#     # color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
#     color_map = color_map[3:]
#     return color_map



# class Trainer(BaseTrainer):

#     def __init__(self, args, cfg):
        
#         #---------------------------------------------------------------------------
#         self.increase_epoch = [50,100,150,200,420,500]
#         self.skip_value = [3,4,5,6,3,3]
#         self.args = args
#         self.cfg = cfg

#         super(Trainer, self).__init__(args, cfg)
        
#         #---------------------------------------------------------------------------
#         self.loader = get_dataloader(args, cfg, 'train',2)

#         # alias
#         self.denorm = self.loader.dataset.denorm 

#         # val loaders for source and target domains
#         self.valloaders = get_dataloader(args, cfg, 'val')

#         # writers (only main)
#         self.writer_val = {}
#         for val_set in self.valloaders.keys():
#             logdir_val = os.path.join(args.logdir, val_set)
#             self.writer_val[val_set] = SummaryWriter(logdir_val)

#         # model
#         self.net = get_model(cfg, remove_layers=cfg.MODEL.REMOVE_LAYERS)

#         print("Train Net: ")
#         print(self.net)

#         # optimizer using different LR
#         net_params = self.net.parameter_groups(cfg.MODEL.LR, cfg.MODEL.WEIGHT_DECAY)

#         print("Optimising parameter groups: ")
#         for i, g in enumerate(net_params):
#             print("[{}]: # parameters: {}, lr = {:4.3e}".format(i, len(g["params"]), g["lr"]))

#         self.optim = self.get_optim(net_params, cfg.MODEL)

#         print("# of params: ", len(list(self.net.parameters())))

#         # LR scheduler
#         if cfg.MODEL.LR_SCHEDULER == "step":
#             self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, \
#                                                              step_size=cfg.MODEL.LR_STEP, \
#                                                              gamma=cfg.MODEL.LR_GAMMA)
#         elif cfg.MODEL.LR_SCHEDULER == "linear": # linear decay

#             def lr_lambda(epoch):
#                 mult = 1 - epoch / (float(self.cfg.TRAIN.NUM_EPOCHS) - self.start_epoch)
#                 mult = mult ** self.cfg.MODEL.LR_POWER
#                 #print("Linear Scheduler: mult = {:4.3f}".format(mult))
#                 return mult

#             self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
#         else:
#             self.scheduler = None

#         self.vis_batch = None

#         # using cuda
#         self.net.cuda()
#         self.crw = CRW(cfg.TEST)

#         # checkpoint management
#         self.checkpoint.create_model(self.net, self.optim)
#         if not args.resume is None:
#             self.start_epoch, self.best_score = self.checkpoint.load(args.resume, "cuda:0")
#         if not args.pretrain is None:
#             print("loading pretrain weights from {}".format(args.pretrain))
#             self.checkpoint.load(args.pretrain, "cuda:0")

#     def step_seg(self, epoch, batch_src, key, temp=None, train=False, visualise=False, \
#                  save_batch=False, writer=None, tag="train_src"):

#         frames, masks_gt, n_obj, seq_name = batch_src

#         # semi-supervised: select only the first
#         frames = frames.flatten(0,1)
#         masks_gt = masks_gt.flatten(0,1)
#         masks_gt = masks_gt[:, :n_obj.item()]

#         masks_ref = masks_gt.clone()
#         masks_ref[1:] *= 0
#         masks_ref_origin = masks_ref.cuda()
#         self.mem_bank = MemoryBank_dotproduct(n_obj.item()-1, 20)
        
#         #==========================================================================================
#         frames, self.pad = pad_divide_by(frames, 8)
#         masks_ref, self.pad = pad_divide_by(masks_ref_origin, 8)
#         #==========================================================================================
        
#         T = frames.shape[0]

#         fetch = {"res3": lambda x: x[5], \
#                  "res4": lambda x: x[1], \
#                  "key": lambda x: x[0]}

#         # number of iterations
#         bs = self.cfg.TRAIN.BATCH_SIZE
#         feats = []
#         t0 = time.time()
        
#         torch.cuda.empty_cache()

#         for t in range(0, T):
#             # next frame
#             frames_batch = frames[t:t+1].cuda()
#             feats_ = self.net(frames_batch, embd_only=True)
#             feats.append(fetch[key](feats_).cpu())
            
#             #==========================================================================================
#             key1, res4, qk, qv, f4, f3, f2 = feats_
#             if t != 0:
#                 out_mask, pre = self.net.segment_with_memory(self.mem_bank, qk, qv, f3, f2)
#                 masks_ref[t] = out_mask.squeeze()
#                 masks_ref_origin[t] = unpad(out_mask, self.pad).squeeze()
# #                 wr = masks_ref_origin[t].argmax(0)
# #                 pred_mask = get_pseudo_color_map(wr.cpu().numpy())
# #                 pred_mask.save('tmp/b'+str(t)+'.png')
#             if t%5==0:
#                 value = self.net.encoder_value(frames_batch, masks_ref[t:t+1, 1:].transpose(0,1), f4)
#                 self.mem_bank.add_memory(qk, value, is_temp=False)
                
# #             value = self.net.encoder_value(frames_batch, masks_ref[t:t+1, 1:].transpose(0,1), f4)
# #             self.mem_bank.add_memory(qk, value, is_temp=False)
# #             if t==0:
# #                 for ss in range(20):
# #                     self.mem_bank.add_memory(qk, value, is_temp=False)
#             #==========================================================================================        

#         feats = torch.cat(feats, 0)
#         print("Inference: {:4.3f}s".format(time.time() - t0))
#         sys.stdout.flush()
#         t0 = time.time()
# #         outs = self.crw.forward(feats, masks_ref)
#         outs={}
#         print("CRW propagation: {:4.3f}s".format(time.time() - t0))
#         sys.stdout.flush()
#         outs["masks_gt"] = masks_gt.argmax(1)
#         outs["masks_pred_idx"] = masks_ref_origin.argmax(1)

#         if visualise:
#             outs["frames"] = unpad(frames, self.pad)
#             self._visualise_seg(epoch, outs, writer, tag)

#         if save_batch:
#             self.save_vis_batch(tag, batch_src)

#         return outs

#     def step(self, epoch, batch_in, train=False, visualise=False, save_batch=False, writer=None, tag="train"):

#         frames1, mask1, frames2, affine1, affine2 = batch_in
  
#         assert frames1.size() == frames2.size(), "Frames shape mismatch"

#         B,T,C,H,W = frames1.shape
#         images1 = frames1.flatten(0,1).cuda()
#         images2 = frames2[:, 1:].flatten(0,1).cuda()

#         affine1 = affine1.flatten(0,1).cuda()
#         affine2 = affine2.flatten(0,1).cuda()

#         # source forward pass
#         losses, outs = self.net(images1, mask1.cuda(), frames2=images2, T=T, \
#                                 affine=affine1, affine2=affine2, \
#                                 dbg=visualise)

#         if train:
#             self.optim.zero_grad()
#             losses["main"].backward()
#             self.optim.step()

#         if visualise:
#             self._visualise(epoch, outs, T, writer, tag)

#         if save_batch:
#             # Saving batch for visualisation
#             self.save_vis_batch(tag, batch_in)

#         # summarising the losses into python scalars
#         losses_ret = {}
#         for key, val in losses.items():
#             losses_ret[key] = val.mean().item()

#         return losses_ret, outs

#     def train_epoch(self, epoch):
# #         #---------------------------------------------------------------------------
# #         if epoch>self.increase_epoch[0]:
# #             self.loader = get_dataloader(self.args, self.cfg, 'train',self.skip_value[0])
# #             self.skip_value = self.skip_value[1:]
# #             self.increase_epoch =self.increase_epoch[1:]

#         stat = StatManager()

#         # adding stats for classes
#         timer = Timer("Epoch {}".format(epoch))
#         step = partial(self.step, train=True, visualise=False)

#         # training mode
#         self.net.train()
#         self.net.fast_net.backbone.eval()
# #         self.net.fast_net.emb_q.eval()
# #         for m in self.net.mask_encoder.modules():
# #             if isinstance(m, nn.BatchNorm2d):
# #                 m.eval()

#         for i, batch in enumerate(self.loader):

#             save_batch = i == 0

#             losses, _ = step(epoch, batch, save_batch=save_batch, tag="train")

#             for loss_key, loss_val in losses.items():
#                 stat.update_stats(loss_key, loss_val)

#             # intermediate logging
#             if i % 10 == 0:
#                 msg =  "Loss [{:04d}]: ".format(i)
#                 for loss_key, loss_val in losses.items():
#                     msg += " {} {:.4f} | ".format(loss_key, loss_val)
#                 msg += " | Im/Sec: {:.1f}".format(i * self.cfg.TRAIN.BATCH_SIZE / timer.get_stage_elapsed())
#                 print(msg)
#                 sys.stdout.flush()
        
#         for name, val in stat.items():
#             print("{}: {:4.3f}".format(name, val))
#             self.writer.add_scalar('all/{}'.format(name), val, epoch)

#         # plotting learning rate
#         for ii, l in enumerate(self.optim.param_groups):
#             print("Learning rate [{}]: {:4.3e}".format(ii, l['lr']))
#             self.writer.add_scalar('lr/enc_group_%02d' % ii, l['lr'], epoch)

#         # plotting moment distance
#         if stat.has_vals("lr_gamma"):
#             self.writer.add_scalar('hyper/gamma', stat.summarize_key("lr_gamma"), epoch)


#     def validation_seg(self, epoch, writer, loader, key="all", temp=None, tag=None, max_iter=None):

#         vis = key == "res4"
#         stat = StatManager()

#         if max_iter is None:
#             max_iter = len(loader)

#         if temp is None:
#             temp = self.cfg.TEST.TEMP

#         step_fn = partial(self.step_seg, key=key, temp=temp, train=False, visualise=vis, writer=writer)

#         # Fast test during the training
#         def eval_batch(n, batch):
#             tag_n = tag + "_{:02d}".format(n)
#             masks = step_fn(epoch, batch, tag=tag_n)
#             return masks

#         self.net.eval()

#         def davis_mask(masks):
#             masks = masks.cpu() 
#             num_objects = int(masks.max())
#             tmp = torch.ones(num_objects, *masks.shape)
#             tmp = tmp * torch.arange(1, num_objects + 1)[:, None, None, None]
#             return (tmp == masks[None, ...]).long().numpy()

#         Js = {"M": [], "R": [], "D": []}
#         Fs = {"M": [], "R": [], "D": []}

#         timer = Timer("[Epoch {}] Validation-Seg".format(epoch))
#         tag_key = "{}_{}_{:3.2f}".format(tag, key, temp)
#         for n, batch in enumerate(loader):
#             seq_name = batch[-1][0]
#             print("Sequence: ", seq_name)
#             sys.stdout.flush()

#             with torch.no_grad():
#                 masks_out = eval_batch(n, batch)

#             # second element is assumed to be always GT masks
#             masks_gt = davis_mask(masks_out["masks_gt"])
#             masks_pred = davis_mask(masks_out["masks_pred_idx"])
#             assert masks_gt.shape == masks_pred.shape

#             # converting to a digestible format
#             # [num_objects, seq_length, height, width]

#             if not tag_key is None and not self.has_vis_batch(tag_key):
#                 self.save_vis_batch(tag_key, batch)

#             start_t = time.time()
#             metrics_res = evaluate_semi((masks_gt, ), (masks_pred, ))
#             J, F = metrics_res['J'], metrics_res['F']

#             print("Evaluation: {:4.3f}s".format(time.time() - start_t))
#             print("Jaccard: ", J["M"])
#             print("F-Score: ", F["M"])

#             for l in ("M", "R", "D"):
#                 Js[l] += J[l]
#                 Fs[l] += F[l]

#             msg = "{} | Im/Sec: {:.1f}".format(n, n * batch[0].shape[1] / timer.get_stage_elapsed())
#             print(msg)
#             sys.stdout.flush()

#         g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']

#         # Generate dataframe for the general results
#         final_mean = (np.mean(Js["M"]) + np.mean(Fs["M"])) / 2.
#         g_res = [final_mean, \
#                  np.mean(Js["M"]), np.mean(Js["R"]), np.mean(Js["D"]), \
#                  np.mean(Fs["M"]), np.mean(Fs["R"]), np.mean(Fs["D"])]

#         for (name, val) in zip(g_measures, g_res):
#             writer.add_scalar('{}_{:3.2f}/{}'.format(key, temp, name), val, epoch)
#             print('{}: {:4.3f}'.format(name, val))

#         return final_mean


# def train(args, cfg):

#     setproctitle.setproctitle("dense-ulearn | {}".format(args.run))

#     if args.seed is not None:
#         print("Setting the seed: {}".format(args.seed))
#         random.seed(args.seed)
#         np.random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         torch.cuda.manual_seed(args.seed)
#         torch.cuda.manual_seed_all(args.seed)

#     trainer = Trainer(args, cfg)

#     timer = Timer()
#     def time_call(func, msg, *args, **kwargs):
#         timer.reset_stage()
#         val = func(*args, **kwargs)
#         print(msg + (" {:3.2}m".format(timer.get_stage_elapsed() / 60.)))
#         return val

#     for epoch in range(trainer.start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):
        
#         # training 1 epoch
#         time_call(trainer.train_epoch, "Train epoch: ", epoch)

#         print("Epoch >>> {:02d} <<< ".format(epoch))
#         if epoch <300:
#             if epoch % cfg.LOG.ITER_VAL == 0:
#                 best_layer = None
#                 best_score = -1e10
#                 for val_set in ("val_video_seg", ):
#                     writer = trainer.writer_val[val_set]
#                     loader = trainer.valloaders[val_set]
#                     #==========================================================================================   
#     #                 for layer in ("key", "res4"):
#                     layer = 'key'
#                     #==========================================================================================   
#                     msg = ">>> Validation {} / {} <<<".format(layer, val_set)
#                     score = time_call(trainer.validation_seg, msg, epoch, writer, loader, key=layer, tag=val_set)
#                     if score > best_score:
#                         best_score = score
#                         best_layer = layer

#                     print("Best score / layer: {:4.2f} / {}".format(best_score, best_layer))
#                     if val_set =="val_video_seg":
#                         trainer.checkpoint_best(best_score, epoch, best_layer)
                   
#         else:
#             if epoch % 4 == 0:
#                 best_layer = None
#                 best_score = -1e10
#                 for val_set in ("val_video_seg", ):
#                     writer = trainer.writer_val[val_set]
#                     loader = trainer.valloaders[val_set]
#                     #==========================================================================================   
#     #                 for layer in ("key", "res4"):
#                     layer = 'key'
#                     #==========================================================================================   
#                     msg = ">>> Validation {} / {} <<<".format(layer, val_set)
#                     score = time_call(trainer.validation_seg, msg, epoch, writer, loader, key=layer, tag=val_set)
#                     if score > best_score:
#                         best_score = score
#                         best_layer = layer

#                     print("Best score / layer: {:4.2f} / {}".format(best_score, best_layer))
#                     if val_set =="val_video_seg":
#                         trainer.checkpoint_best(best_score, epoch, best_layer)

#         if not trainer.scheduler is None and cfg.MODEL.LR_SCHED_USE_EPOCH:
#             trainer.scheduler.step()

# def main():
#     args = get_arguments(sys.argv[1:])

#     # Reading the config
#     cfg_from_file(args.cfg_file)
#     if args.set_cfgs is not None:
#         cfg_from_list(args.set_cfgs)

#     train(args, cfg)

# if __name__ == "__main__":
#     main()


"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

from __future__ import print_function

import os
import sys
import numpy as np
import time
import random
import setproctitle

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import *
from models import get_model

from base_trainer import BaseTrainer

from opts import get_arguments
from core.config import cfg, cfg_from_file, cfg_from_list
from utils.timer import Timer
from utils.stat_manager import StatManager
from utils.davis2017 import evaluate_semi
from labelprop.crw import CRW

from models.modules import MemoryBank_dotproduct, pad_divide_by, unpad

from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


import cv2
from PIL import Image as PILImage
def  get_pseudo_color_map(pred):
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    color_map = get_color_map_list(256)
#     color_map = get_cityscapes_colors()
    pred_mask.putpalette(color_map)
    return pred_mask
def get_color_map_list(num_classes):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    # color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = color_map[3:]
    return color_map



class Trainer(BaseTrainer):

    def __init__(self, args, cfg):
        
        #---------------------------------------------------------------------------
        self.increase_epoch = [50,100,150,200,420,500]
        self.skip_value = [3,4,5,6,3,3]
        self.args = args
        self.cfg = cfg

        super(Trainer, self).__init__(args, cfg)
        
        #---------------------------------------------------------------------------
        self.loader = get_dataloader(args, cfg, 'train',2)

        # alias
        self.denorm = self.loader.dataset.denorm 

        # val loaders for source and target domains
        self.valloaders = get_dataloader(args, cfg, 'val')

        # writers (only main)
        self.writer_val = {}
        for val_set in self.valloaders.keys():
            logdir_val = os.path.join(args.logdir, val_set)
            self.writer_val[val_set] = SummaryWriter(logdir_val)

        # model
        self.net = get_model(cfg, remove_layers=cfg.MODEL.REMOVE_LAYERS)

        print("Train Net: ")
        print(self.net)

        # optimizer using different LR
        net_params = self.net.parameter_groups(cfg.MODEL.LR, cfg.MODEL.WEIGHT_DECAY)

        print("Optimising parameter groups: ")
        for i, g in enumerate(net_params):
            print("[{}]: # parameters: {}, lr = {:4.3e}".format(i, len(g["params"]), g["lr"]))

        self.optim = self.get_optim(net_params, cfg.MODEL)

        print("# of params: ", len(list(self.net.parameters())))

        # LR scheduler
        if cfg.MODEL.LR_SCHEDULER == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, \
                                                             step_size=cfg.MODEL.LR_STEP, \
                                                             gamma=cfg.MODEL.LR_GAMMA)
        elif cfg.MODEL.LR_SCHEDULER == "linear": # linear decay

            def lr_lambda(epoch):
                mult = 1 - epoch / (float(self.cfg.TRAIN.NUM_EPOCHS) - self.start_epoch)
                mult = mult ** self.cfg.MODEL.LR_POWER
                #print("Linear Scheduler: mult = {:4.3f}".format(mult))
                return mult

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        else:
            self.scheduler = None

        self.vis_batch = None

        # using cuda
        self.net.cuda()
        self.crw = CRW(cfg.TEST)

        # checkpoint management
        self.checkpoint.create_model(self.net, self.optim)
        if not args.resume is None:
            self.start_epoch, self.best_score = self.checkpoint.load(args.resume, "cuda:0")
        if not args.pretrain is None:
            print("loading pretrain weights from {}".format(args.pretrain))
            self.checkpoint.load(args.pretrain, "cuda:0")

    def step_seg(self, epoch, batch_src, key, temp=None, train=False, visualise=False, \
                 save_batch=False, writer=None, tag="train_src"):

        frames, masks_gt, n_obj, seq_name = batch_src

        # semi-supervised: select only the first
        frames = frames.flatten(0,1)
        masks_gt = masks_gt.flatten(0,1)
        masks_gt = masks_gt[:, :n_obj.item()]

        masks_ref = masks_gt.clone()
        masks_ref[1:] *= 0
        masks_ref_origin = masks_ref.cuda()
        self.mem_bank = MemoryBank_dotproduct(n_obj.item()-1, 20)
        
        #==========================================================================================
        scale = lambda x, hw: F.interpolate(x, hw, mode="bilinear", align_corners=True)
        frames, self.pad = pad_divide_by(frames, 8)
        masks_ref, self.pad = pad_divide_by(masks_ref_origin, 8)
        H,W = frames.shape[-2:]
        memory_masks = [scale(masks_ref[0:1,:n_obj], (H//8, W//8))]
        #==========================================================================================
        
        T = frames.shape[0]

        fetch = {"res3": lambda x: x[5], \
                 "res4": lambda x: x[1], \
                 "key": lambda x: x[0]}

        # number of iterations
        bs = self.cfg.TRAIN.BATCH_SIZE
        feats = []
        t0 = time.time()
        
        torch.cuda.empty_cache()

        for t in range(0, T):
            # next frame
            frames_batch = frames[t:t+1].cuda()
            feats_ = self.net(frames_batch, embd_only=True)
            feats.append(fetch[key](feats_).cpu())
            
            #==========================================================================================
            key1, res4, qk, qv, f4, f3, f2 = feats_
            if t != 0:
                out_mask, pre = self.net.segment_with_memory(self.mem_bank, qk, qv, f3, f2, memory_masks, frames_batch, f4)
                masks_ref[t] = out_mask.squeeze()
                masks_ref_origin[t] = unpad(out_mask, self.pad).squeeze()
#                 wr = masks_ref_origin[t].argmax(0)
#                 pred_mask = get_pseudo_color_map(wr.cpu().numpy())
#                 pred_mask.save('tmp/b'+str(t)+'.png')
#             if t%5==0:
#                 value = self.net.encoder_value(frames_batch, masks_ref[t:t+1, 1:].transpose(0,1), f4)
#                 self.mem_bank.add_memory(qk, value, is_temp=False)
                
            value = self.net.encoder_value(frames_batch, masks_ref[t:t+1, 1:].transpose(0,1), f4)
            self.mem_bank.add_memory(qk, value, is_temp=False)
            if t!=0:
                if len(memory_masks)>=21:
                    memory_masks = memory_masks[0:1]+ memory_masks[2:]+[pre]
                else:
                    memory_masks.append(pre)
            if t==0:
                for ss in range(20):
                    memory_masks.append(scale(masks_ref[0:1,:n_obj], (H//8, W//8)))
                    self.mem_bank.add_memory(qk, value, is_temp=False)
            #==========================================================================================        

        feats = torch.cat(feats, 0)
        print("Inference: {:4.3f}s".format(time.time() - t0))
        sys.stdout.flush()
        t0 = time.time()
#         outs = self.crw.forward(feats, masks_ref)
        outs={}
        print("CRW propagation: {:4.3f}s".format(time.time() - t0))
        sys.stdout.flush()
        outs["masks_gt"] = masks_gt.argmax(1)
        outs["masks_pred_idx"] = masks_ref_origin.argmax(1)

        if visualise:
            outs["frames"] = unpad(frames, self.pad)
            self._visualise_seg(epoch, outs, writer, tag)

        if save_batch:
            self.save_vis_batch(tag, batch_src)

        return outs

    def step(self, epoch, batch_in, train=False, visualise=False, save_batch=False, writer=None, tag="train"):

        frames1, mask1, frames2, affine1, affine2 = batch_in
  
        assert frames1.size() == frames2.size(), "Frames shape mismatch"

        B,T,C,H,W = frames1.shape
        images1 = frames1.flatten(0,1).cuda()
        images2 = frames2[:, 1:].flatten(0,1).cuda()

        affine1 = affine1.flatten(0,1).cuda()
        affine2 = affine2.flatten(0,1).cuda()

        # source forward pass
        losses, outs = self.net(images1, mask1.cuda(), frames2=images2, T=T, \
                                affine=affine1, affine2=affine2, \
                                dbg=visualise)

        if train:
            self.optim.zero_grad()
            losses["main"].backward()
            self.optim.step()

        if visualise:
            self._visualise(epoch, outs, T, writer, tag)

        if save_batch:
            # Saving batch for visualisation
            self.save_vis_batch(tag, batch_in)

        # summarising the losses into python scalars
        losses_ret = {}
        for key, val in losses.items():
            losses_ret[key] = val.mean().item()

        return losses_ret, outs

    def train_epoch(self, epoch):
#         #---------------------------------------------------------------------------
#         if epoch>self.increase_epoch[0]:
#             self.loader = get_dataloader(self.args, self.cfg, 'train',self.skip_value[0])
#             self.skip_value = self.skip_value[1:]
#             self.increase_epoch =self.increase_epoch[1:]

        stat = StatManager()

        # adding stats for classes
        timer = Timer("Epoch {}".format(epoch))
        step = partial(self.step, train=True, visualise=False)

        # training mode
        self.net.train()
        self.net.fast_net.backbone.eval()
#         self.net.fast_net.emb_q.eval()
#         for m in self.net.mask_encoder.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()

        for i, batch in enumerate(self.loader):

            save_batch = i == 0

            losses, _ = step(epoch, batch, save_batch=save_batch, tag="train")

            for loss_key, loss_val in losses.items():
                stat.update_stats(loss_key, loss_val)

            # intermediate logging
            if i % 10 == 0:
                msg =  "Loss [{:04d}]: ".format(i)
                for loss_key, loss_val in losses.items():
                    msg += " {} {:.4f} | ".format(loss_key, loss_val)
                msg += " | Im/Sec: {:.1f}".format(i * self.cfg.TRAIN.BATCH_SIZE / timer.get_stage_elapsed())
                print(msg)
                sys.stdout.flush()
        
        for name, val in stat.items():
            print("{}: {:4.3f}".format(name, val))
            self.writer.add_scalar('all/{}'.format(name), val, epoch)

        # plotting learning rate
        for ii, l in enumerate(self.optim.param_groups):
            print("Learning rate [{}]: {:4.3e}".format(ii, l['lr']))
            self.writer.add_scalar('lr/enc_group_%02d' % ii, l['lr'], epoch)

        # plotting moment distance
        if stat.has_vals("lr_gamma"):
            self.writer.add_scalar('hyper/gamma', stat.summarize_key("lr_gamma"), epoch)


    def validation_seg(self, epoch, writer, loader, key="all", temp=None, tag=None, max_iter=None):

        vis = key == "res4"
        stat = StatManager()

        if max_iter is None:
            max_iter = len(loader)

        if temp is None:
            temp = self.cfg.TEST.TEMP

        step_fn = partial(self.step_seg, key=key, temp=temp, train=False, visualise=vis, writer=writer)

        # Fast test during the training
        def eval_batch(n, batch):
            tag_n = tag + "_{:02d}".format(n)
            masks = step_fn(epoch, batch, tag=tag_n)
            return masks

        self.net.eval()

        def davis_mask(masks):
            masks = masks.cpu() 
            num_objects = int(masks.max())
            tmp = torch.ones(num_objects, *masks.shape)
            tmp = tmp * torch.arange(1, num_objects + 1)[:, None, None, None]
            return (tmp == masks[None, ...]).long().numpy()

        Js = {"M": [], "R": [], "D": []}
        Fs = {"M": [], "R": [], "D": []}

        timer = Timer("[Epoch {}] Validation-Seg".format(epoch))
        tag_key = "{}_{}_{:3.2f}".format(tag, key, temp)
        for n, batch in enumerate(loader):
            seq_name = batch[-1][0]
            print("Sequence: ", seq_name)
            sys.stdout.flush()

            with torch.no_grad():
                masks_out = eval_batch(n, batch)

            # second element is assumed to be always GT masks
            masks_gt = davis_mask(masks_out["masks_gt"])
            masks_pred = davis_mask(masks_out["masks_pred_idx"])
            assert masks_gt.shape == masks_pred.shape

            # converting to a digestible format
            # [num_objects, seq_length, height, width]

            if not tag_key is None and not self.has_vis_batch(tag_key):
                self.save_vis_batch(tag_key, batch)

            start_t = time.time()
            metrics_res = evaluate_semi((masks_gt, ), (masks_pred, ))
            J, F = metrics_res['J'], metrics_res['F']

            print("Evaluation: {:4.3f}s".format(time.time() - start_t))
            print("Jaccard: ", J["M"])
            print("F-Score: ", F["M"])

            for l in ("M", "R", "D"):
                Js[l] += J[l]
                Fs[l] += F[l]

            msg = "{} | Im/Sec: {:.1f}".format(n, n * batch[0].shape[1] / timer.get_stage_elapsed())
            print(msg)
            sys.stdout.flush()

        g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']

        # Generate dataframe for the general results
        final_mean = (np.mean(Js["M"]) + np.mean(Fs["M"])) / 2.
        g_res = [final_mean, \
                 np.mean(Js["M"]), np.mean(Js["R"]), np.mean(Js["D"]), \
                 np.mean(Fs["M"]), np.mean(Fs["R"]), np.mean(Fs["D"])]

        for (name, val) in zip(g_measures, g_res):
            writer.add_scalar('{}_{:3.2f}/{}'.format(key, temp, name), val, epoch)
            print('{}: {:4.3f}'.format(name, val))

        return final_mean


def train(args, cfg):

    setproctitle.setproctitle("dense-ulearn | {}".format(args.run))

    if args.seed is not None:
        print("Setting the seed: {}".format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args, cfg)

    timer = Timer()
    def time_call(func, msg, *args, **kwargs):
        timer.reset_stage()
        val = func(*args, **kwargs)
        print(msg + (" {:3.2}m".format(timer.get_stage_elapsed() / 60.)))
        return val

    for epoch in range(trainer.start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):
        
        # training 1 epoch
        time_call(trainer.train_epoch, "Train epoch: ", epoch)

        print("Epoch >>> {:02d} <<< ".format(epoch))
        if epoch <300:
            if epoch % cfg.LOG.ITER_VAL == 0:
                best_layer = None
                best_score = -1e10
                for val_set in ("val_video_seg", ):
                    writer = trainer.writer_val[val_set]
                    loader = trainer.valloaders[val_set]
                    #==========================================================================================   
    #                 for layer in ("key", "res4"):
                    layer = 'key'
                    #==========================================================================================   
                    msg = ">>> Validation {} / {} <<<".format(layer, val_set)
                    score = time_call(trainer.validation_seg, msg, epoch, writer, loader, key=layer, tag=val_set)
                    if score > best_score:
                        best_score = score
                        best_layer = layer

                    print("Best score / layer: {:4.2f} / {}".format(best_score, best_layer))
                    if val_set =="val_video_seg":
                        trainer.checkpoint_best(best_score, epoch, best_layer)
                   
        else:
            if epoch % 4 == 0:
                best_layer = None
                best_score = -1e10
                for val_set in ("val_video_seg", ):
                    writer = trainer.writer_val[val_set]
                    loader = trainer.valloaders[val_set]
                    #==========================================================================================   
    #                 for layer in ("key", "res4"):
                    layer = 'key'
                    #==========================================================================================   
                    msg = ">>> Validation {} / {} <<<".format(layer, val_set)
                    score = time_call(trainer.validation_seg, msg, epoch, writer, loader, key=layer, tag=val_set)
                    if score > best_score:
                        best_score = score
                        best_layer = layer

                    print("Best score / layer: {:4.2f} / {}".format(best_score, best_layer))
                    if val_set =="val_video_seg":
                        trainer.checkpoint_best(best_score, epoch, best_layer)

        if not trainer.scheduler is None and cfg.MODEL.LR_SCHED_USE_EPOCH:
            trainer.scheduler.step()

def main():
    args = get_arguments(sys.argv[1:])

    # Reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    train(args, cfg)

if __name__ == "__main__":
    main()