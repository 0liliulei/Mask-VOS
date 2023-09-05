"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import random
import torch
import math
import glob

from PIL import Image
import numpy as np

from .dataloader_base import DLBase
import datasets.daugm_video as tf

class DataVideoMask(DLBase):

    def __init__(self, cfg, split, max_skip=2, val=False):
        super(DataVideoMask, self).__init__()

        self.cfg = cfg
        self.split = split
        self.max_skip = max_skip
        self.val = val

        self.cfg_frame_gap = cfg.DATASET.VAL_FRAME_GAP if val else cfg.DATASET.FRAME_GAP

        self._init_palette(cfg.TRAIN.BATCH_SIZE * cfg.MODEL.GRID_SIZE**2)

        # train/val/test splits are pre-cut
        split_fn = os.path.join(cfg.DATASET.ROOT, "filelists", self.split + ".txt")
        assert os.path.isfile(split_fn)

        self.images = []

        token = None # sequence token (new video when it changes)

        subsequence = []
        ignored = [0]
        num_frames = [0]

        def add_sequence():

            if cfg.DATASET.VIDEO_LEN > len(subsequence):
                # found a very short sequence
                ignored[0] += 1
            else:
                # adding the subsequence
                self.images.append(tuple(subsequence))
                num_frames[0] += len(subsequence)

            del subsequence[:]

        with open(split_fn, "r") as lines:
            for line in lines:
                _line = line.strip("\n").split(' ')

                assert len(_line) > 0, "Expected at least one path"
                _image = _line[0]

                # each sequence may have a different length
                # do some book-keeping e.g. to ensure we have
                # sequences long enough for subsequent sampling
                _token = _image.split("/")[-2] # parent directory

                # sequence ID is in the filename
                #_token = os.path.basename(_image).split("_")[0] 
                if token != _token:
                    if not token is None:
                        add_sequence()
                    token = _token
    
                # image 1
                _image = os.path.join(cfg.DATASET.ROOT, _image.lstrip('/'))
                #assert os.path.isfile(_image), '%s not found' % _image
                subsequence.append(_image)

        # update the last sequence
        # returns the total amount of frames
        add_sequence()
        print("Dataloader: {}".format(split), " / Frame Gap: ", self.cfg_frame_gap)
        print("Loaded {} sequences / {} ignored / {} frames".format(len(self.images), \
                                                                    ignored[0], \
                                                                    num_frames[0]))

        self._num_samples = num_frames[0]
        self._init_augm(cfg)

    def _init_augm(self, cfg):

        # general (unguided) affine transformations
        tfs_pre = []
        self.tf_pre = tf.Compose(tfs_pre)

        # photometric noise
        tfs_affine = []

        # guided affine transformations
        tfs_augm = []

        # 1.
        # general affine transformations
        #
        tfs_pre.append(tf.MaskScaleSmallest(cfg.DATASET.SMALLEST_RANGE))
        
        if cfg.DATASET.RND_CROP:
            tfs_pre.append(tf.MaskRandCrop(cfg.DATASET.CROP_SIZE, pad_if_needed=True))
        else:
            tfs_pre.append(tf.MaskCenterCrop(cfg.DATASET.CROP_SIZE))

        if cfg.DATASET.RND_HFLIP:
            tfs_pre.append(tf.MaskRandHFlip())

        # 2.
        # Guided affine transformation
        #
        if cfg.DATASET.GUIDED_HFLIP:
            tfs_affine.append(tf.GuidedRandHFlip())

        # this will add affine transformation
        if cfg.DATASET.RND_ZOOM:
            tfs_affine.append(tf.MaskRandScaleCrop2(*cfg.DATASET.RND_ZOOM_RANGE))

        self.tf_affine = tf.Compose(tfs_affine)
        self.tf_affine2 = tf.Compose([tf.AffineIdentity()])

        tfs_post = [tf.ToTensorMask(),
                    tf.Normalize(mean=self.MEAN, std=self.STD)]

        # image to the teacher will have no noise
        self.tf_post = tf.Compose(tfs_post)

    def set_num_samples(self, n):
        print("Re-setting # of samples: {:d} -> {:d}".format(self._num_samples, n))
        self._num_samples = n

    def __len__(self):
        return len(self.images) #self._num_samples

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image

    def _get_affine(self, params):

        N = len(params)

        # construct affine operator
        affine = torch.zeros(N, 2, 3)

        aspect_ratio = float(self.cfg.DATASET.CROP_SIZE[0]) / \
                            float(self.cfg.DATASET.CROP_SIZE[1])

        for i, (dy,dx,alpha,scale,flip) in enumerate(params):

            # R inverse
            sin = math.sin(alpha * math.pi / 180.)
            cos = math.cos(alpha * math.pi / 180.)

            # inverse, note how flipping is incorporated
            affine[i,0,0], affine[i,0,1] = flip * cos, sin * aspect_ratio
            affine[i,1,0], affine[i,1,1] = -sin / aspect_ratio, cos

            # T inverse Rinv * t == R^T * t
            affine[i,0,2] = -1. * (cos * dx + sin * dy)
            affine[i,1,2] = -1. * (-sin * dx + cos * dy)

            # T
            affine[i,0,2] /= float(self.cfg.DATASET.CROP_SIZE[1] // 2)
            affine[i,1,2] /= float(self.cfg.DATASET.CROP_SIZE[0] // 2)

            # scaling
            affine[i] *= scale

        return affine

    def _get_affine_inv(self, affine, params):

        aspect_ratio = float(self.cfg.DATASET.CROP_SIZE[0]) / \
                            float(self.cfg.DATASET.CROP_SIZE[1])

        affine_inv = affine.clone()
        affine_inv[:,0,1] = affine[:,1,0] * aspect_ratio**2
        affine_inv[:,1,0] = affine[:,0,1] / aspect_ratio**2
        affine_inv[:,0,2] = -1 * (affine_inv[:,0,0] * affine[:,0,2] + affine_inv[:,0,1] * affine[:,1,2])
        affine_inv[:,1,2] = -1 * (affine_inv[:,1,0] * affine[:,0,2] + affine_inv[:,1,1] * affine[:,1,2])

        # scaling
        affine_inv /= torch.Tensor(params)[:,3].view(-1,1,1)**2

        return affine_inv

    def __getitem__(self, index):

        # searching for the video clip ID
        sequence = self.images[index] # % len(self.images)]
        seqlen = len(sequence)

#         assert self.cfg_frame_gap > 0, "Frame gap should be positive"
#         t_window = self.cfg_frame_gap * self.cfg.DATASET.VIDEO_LEN

#         # reduce sampling gap for short clips
#         t_window = min(seqlen, t_window)
#         frame_gap = t_window // self.cfg.DATASET.VIDEO_LEN

#         # strided slice
#         frame_ids = torch.arange(t_window)[::frame_gap]
#         frame_ids = frame_ids[:self.cfg.DATASET.VIDEO_LEN]
#         assert len(frame_ids) == self.cfg.DATASET.VIDEO_LEN

#         # selecting a random start
#         index_start = random.randint(0, seqlen - frame_ids[-1] - 1)
#         # permuting the frames in the batch
#         random_ids = torch.randperm(self.cfg.DATASET.VIDEO_LEN)
#         # adding the offset
#         frame_ids = frame_ids[random_ids] + index_start

        this_max_jump = min(seqlen, self.max_skip)
        start_idx = np.random.randint(seqlen-this_max_jump+1)
        f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1
        f1_idx = min(f1_idx, seqlen-this_max_jump, seqlen-1)

        f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
        f2_idx = min(f2_idx, seqlen-this_max_jump//2, seqlen-1)
        
        if np.random.random()>0.5:
            frame_ids = [start_idx, f1_idx, f2_idx]
        else:
            frame_ids = [f2_idx, f1_idx, start_idx]

        
        # forward sequence
        images = []
        masks = []
        for frame_id in frame_ids:
            fn = sequence[frame_id]
            images.append(Image.open(fn).convert('RGB'))
            fn_quan = fn.replace('JPEGImages', 'Annotations_pseudo')
            masks.append(Image.open(fn_quan[:-3]+'png'))

        # 1. general transforms
        frames, valid = self.tf_pre(images, masks) 

        # 1.1 creating two sequences in forward/backward order
        frames1, valid1 = frames[:], valid[:]

        # second copy
        frames2 = [f.copy() for f in frames]
        valid2 = [v.copy() for v in valid]

        # 2. guided affine transforms
        frames1, valid1, affine_params1 = self.tf_affine(frames1, valid1)
        frames2, valid2, affine_params2 = self.tf_affine2(frames2, valid2)

        # convert to tensor, zero out the values
        frames1, valid1 = self.tf_post(frames1, valid1)
        frames2, valid2 = self.tf_post(frames2, valid2)

        # converting the affine transforms
        aff_reg = self._get_affine(affine_params1)
        aff_main = self._get_affine(affine_params2)

        aff_reg_inv = self._get_affine_inv(aff_reg, affine_params1)

        aff_reg = aff_main # identity affine2_inv
        aff_main = aff_reg_inv

        frames1 = torch.stack(frames1, 0)
        frames2 = torch.stack(frames2, 0)
        valid2_0 = (torch.stack(valid2, 0)==1).float()
        valid2_1 = (torch.stack(valid2, 0)==2).float()
        valid2 = torch.stack([valid2_0,valid2_1], dim=1)

        assert frames1.shape == frames2.shape

        return frames2, valid2, frames1, aff_main, aff_reg

