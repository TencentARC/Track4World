import os
import torch
import cv2
import imageio
import numpy as np
from torchvision.transforms import ColorJitter, GaussianBlur
from PIL import Image
import holi4d.utils.data

class PointDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        use_augs=False,
    ):
        super(PointDataset, self).__init__()
        self.data_root = data_root
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.use_augs = use_augs
        # photometric augmentation
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))
        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.5
        self.replace_bounds = [2, 100]
        self.replace_max = 10

        # spatial augmentations
        self.pad_bounds = [10, 100]
        self.crop_size = crop_size
        self.resize_lim = [0.25, 2.0]  # sample resizes from here
        self.resize_delta = 0.2
        self.max_crop_offset = 50

        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5
        self.rot_prob = 0.5

    def getitem_helper(self, index):
        return NotImplementedError

    def __getitem__(self, index):
        gotit = False
        fails = 0
        while not gotit and fails < 4:
            sample, gotit = self.getitem_helper(index)
            if gotit:
                return sample, gotit
            else:
                fails += 1
                index = np.random.randint(len(self))
                del sample
        if fails > 1:
            print('note: sampling failed %d times' % fails)

        if self.seq_len is not None:
            S = self.seq_len
        else:
            S = 11
        # fake sample, so we can still collate
        sample = holi4d.utils.data.VideoData(
            video=torch.zeros((S, 3, self.crop_size[0], self.crop_size[1])),
            trajs=torch.zeros((S, self.traj_per_sample, 2)),
            visibs=torch.zeros((S, self.traj_per_sample)),
            valids=torch.zeros((S, self.traj_per_sample)),
        )
        return sample, gotit

    def add_photometric_augs(self, rgbs, trajs, visibles, eraser=True, replace=True, augscale=1.0):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        if eraser:
            ############ eraser transform (per image after the first) ############
            eraser_bounds = [eb*augscale for eb in self.eraser_bounds]
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(np.random.randint(1, self.eraser_max + 1)):
                        # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(eraser_bounds[0], eraser_bounds[1])
                        dy = np.random.randint(eraser_bounds[0], eraser_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        mean_color = np.mean(
                            rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0
                        )
                        rgbs[i][y0:y1, x0:x1, :] = mean_color

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        if replace:
            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs_alt]

            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            replace_bounds = [rb*augscale for rb in self.replace_bounds]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.replace_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(replace_bounds[0], replace_bounds[1])
                        dy = np.random.randint(replace_bounds[0], replace_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = np.random.randint(0, H - hei)
                        x00 = np.random.randint(0, W - wid)
                        fr = np.random.randint(0, S)
                        rep = rgbs_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        rgbs[i][y0:y1, x0:x1, :] = rep

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        return rgbs, trajs, visibles


    def add_spatial_augs(self, rgbs, trajs, visibles, crop_size, augscale=1.0):
        T, N, __ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T
        
        rgbs = [rgb.astype(np.float32) for rgb in rgbs]

        trajs = trajs.astype(np.float64)
        
        target_H, target_W = crop_size
        if target_H > H or target_W > W:
            scale = max(target_H / H, target_W / W)
            new_H, new_W = int(np.ceil(H * scale)), int(np.ceil(W * scale))
            rgbs = [cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
            trajs = trajs * scale

        ############ spatial transform ############

        # padding
        pad_bounds = [int(pb*augscale) for pb in self.pad_bounds]
        pad_x0 = np.random.randint(pad_bounds[0], pad_bounds[1])
        pad_x1 = np.random.randint(pad_bounds[0], pad_bounds[1])
        pad_y0 = np.random.randint(pad_bounds[0], pad_bounds[1])
        pad_y1 = np.random.randint(pad_bounds[0], pad_bounds[1])

        rgbs = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs]
        trajs[:, :, 0] += pad_x0
        trajs[:, :, 1] += pad_y0
        H, W = rgbs[0].shape[:2]

        # scaling + stretching
        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        scale_x = scale
        scale_y = scale
        H_new = H
        W_new = W

        scale_delta_x = 0.0
        scale_delta_y = 0.0

        rgbs_scaled = []
        resize_delta = self.resize_delta * augscale
        for s in range(S):
            if s == 1:
                scale_delta_x = np.random.uniform(-resize_delta, resize_delta)
                scale_delta_y = np.random.uniform(-resize_delta, resize_delta)
            elif s > 1:
                scale_delta_x = (
                    scale_delta_x * 0.8
                    + np.random.uniform(-resize_delta, resize_delta) * 0.2
                )
                scale_delta_y = (
                    scale_delta_y * 0.8
                    + np.random.uniform(-resize_delta, resize_delta) * 0.2
                )
            scale_x = scale_x + scale_delta_x
            scale_y = scale_y + scale_delta_y

            # bring h/w closer
            scale_xy = (scale_x + scale_y) * 0.5
            scale_x = scale_x * 0.5 + scale_xy * 0.5
            scale_y = scale_y * 0.5 + scale_xy * 0.5

            # don't get too crazy
            scale_x = np.clip(scale_x, 0.2, 2.0)
            scale_y = np.clip(scale_y, 0.2, 2.0)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # make it at least slightly bigger than the crop area,
            # so that the random cropping can add diversity
            H_new = np.clip(H_new, crop_size[0] + 10, None)
            W_new = np.clip(W_new, crop_size[1] + 10, None)
            # recompute scale in case we clipped
            scale_x = (W_new - 1) / float(W - 1)
            scale_y = (H_new - 1) / float(H - 1)
            rgbs_scaled.append(
                cv2.resize(rgbs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
            )
            trajs[s, :, 0] *= scale_x
            trajs[s, :, 1] *= scale_y
        rgbs = rgbs_scaled
        ok_inds = visibles[0, :] > 0
        vis_trajs = trajs[:, ok_inds]  # S,?,2

        if vis_trajs.shape[1] > 0:
            mid_x = np.mean(vis_trajs[0, :, 0])
            mid_y = np.mean(vis_trajs[0, :, 1])
        else:
            mid_y = crop_size[0]
            mid_x = crop_size[1]

        x0 = int(mid_x - crop_size[1] // 2)
        y0 = int(mid_y - crop_size[0] // 2)

        offset_x = 0
        offset_y = 0
        max_crop_offset = int(self.max_crop_offset*augscale)

        for s in range(S):
            # on each frame, shift a bit more
            if s == 1:
                offset_x = np.random.randint(-max_crop_offset, max_crop_offset)
                offset_y = np.random.randint(-max_crop_offset, max_crop_offset)
            elif s > 1:
                offset_x = int(
                    offset_x * 0.8
                    + np.random.randint(-max_crop_offset, max_crop_offset + 1)
                    * 0.2
                )
                offset_y = int(
                    offset_y * 0.8
                    + np.random.randint(-max_crop_offset, max_crop_offset + 1)
                    * 0.2
                )
            x0 = x0 + offset_x
            y0 = y0 + offset_y

            H_new, W_new = rgbs[s].shape[:2]
            if H_new == crop_size[0]:
                y0 = 0
            else:
                y0 = min(max(0, y0), H_new - crop_size[0] - 1)

            if W_new == crop_size[1]:
                x0 = 0
            else:
                x0 = min(max(0, x0), W_new - crop_size[1] - 1)

            rgbs[s] = rgbs[s][y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
            trajs[s, :, 0] -= x0
            trajs[s, :, 1] -= y0

        H = crop_size[0]
        W = crop_size[1]

        if np.random.rand() < self.h_flip_prob:
            rgbs = [rgb[:, ::-1].copy() for rgb in rgbs]
            trajs[:, :, 0] = W-1 - trajs[:, :, 0]
        if np.random.rand() < self.v_flip_prob:
            rgbs = [rgb[::-1].copy() for rgb in rgbs]
            trajs[:, :, 1] = H-1 - trajs[:, :, 1]
        return np.stack(rgbs), trajs.astype(np.float32)

    def crop(self, rgbs, trajs, crop_size):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        target_H, target_W = crop_size
        if target_H > H or target_W > W:
            scale = max(target_H / H, target_W / W)
            new_H, new_W = int(np.ceil(H * scale)), int(np.ceil(W * scale))
            rgbs = [cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
            trajs = trajs * scale
            H, W = rgbs[0].shape[:2]

        # simple random crop
        y0 = 0 if crop_size[0] >= H else (H - crop_size[0]) // 2
        x0 = 0 if crop_size[1] >= W else np.random.randint(0, W - crop_size[1])
        rgbs = [rgb[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]] for rgb in rgbs]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        return np.stack(rgbs), trajs

    def follow_crop(self, rgbs, trajs, visibs, crop_size):
        T, N, _ = trajs.shape

        rgbs = [rgb for rgb in rgbs] # unstack so we can change them one by one

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        vels = trajs[1:]-trajs[:-1] 
        accels = vels[1:]-vels[:-1]
        vis_ = visibs[1:]*visibs[:-1]
        vis__ = vis_[1:]*vis_[:-1]
        travel = np.sum(np.sum(np.abs(accels)*vis__[:,:,None], axis=2), axis=0)
        num_interesting = np.sum(travel > 0).round()
        inds = np.argsort(-travel)[:max(num_interesting//32,32)]

        trajs_interesting = trajs[:,inds] # S,?,2

        # pick a random one to focus on, for variety
        smooth_xys = trajs_interesting[:,np.random.randint(len(inds))]

        crop_H, crop_W = crop_size

        smooth_xys = np.clip(smooth_xys, [crop_W // 2, crop_H // 2], [W - crop_W // 2, H - crop_H // 2])

        def smooth_path(xys, num_passes):
            kernel = np.array([0.25, 0.5, 0.25])
            for _ in range(num_passes):
                padded = np.pad(xys, ((1, 1), (0, 0)), mode='edge')
                xys = (
                    kernel[0] * padded[:-2] +
                    kernel[1] * padded[1:-1] +
                    kernel[2] * padded[2:]
                )
                return xys
        num_passes = np.random.randint(4, S) # 1 is perfect follow; S is near linear
        smooth_xys = smooth_path(smooth_xys, num_passes)
        
        for si in range(S):
            x0, y0 = smooth_xys[si].round().astype(np.int32)
            x0 -= crop_W//2
            y0 -= crop_H//2
            rgbs[si] = rgbs[si][y0:y0+crop_H, x0:x0+crop_W]
            trajs[si,:,0] -= x0
            trajs[si,:,1] -= y0

        return np.stack(rgbs), trajs

