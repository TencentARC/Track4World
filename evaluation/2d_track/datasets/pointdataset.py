import os
import torch
import cv2
import imageio
import numpy as np
from torchvision.transforms import ColorJitter, GaussianBlur
from PIL import Image
import track4world.utils.data

class PointDataset(torch.utils.data.Dataset):
    """
    A base dataset class for 4D point tracking. 
    Handles complex photometric and spatial augmentations for video sequences and trajectories.
    """
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
        
        # --- Photometric Augmentation Config ---
        # Randomly jitter brightness, contrast, saturation, and hue
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))
        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25

        # --- Eraser Augmentation (Occlusion) ---
        # Simulates occlusion by placing random colored rectangles over the image
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [2, 100]  # Min/Max size of the eraser block
        self.eraser_max = 10           # Max number of blocks to place

        # --- Replace Augmentation (Patch Swap) ---
        # Simulates occlusion by replacing patches with content from other frames
        self.replace_aug_prob = 0.5
        self.replace_bounds = [2, 100]
        self.replace_max = 10

        # --- Spatial Augmentation Config ---
        self.pad_bounds = [10, 100]
        self.crop_size = crop_size
        self.resize_lim = [0.25, 2.0]  # Global resize scale
        self.resize_delta = 0.2        # Per-frame scale drift
        self.max_crop_offset = 50      # Per-frame crop drift (jitter)

        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5
        self.rot_prob = 0.5

    def getitem_helper(self, index):
        """Method to be implemented by child classes to load specific data."""
        return NotImplementedError

    def __getitem__(self, index):
        """
        Loads a sample. Includes a retry mechanism: if loading fails, 
        it tries a different random index.
        """
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

        # Fallback: return a zero-filled sample if all retries fail
        S = self.seq_len if self.seq_len is not None else 11
        sample = track4world.utils.data.VideoData(
            video=torch.zeros((S, 3, self.crop_size[0], self.crop_size[1])),
            trajs=torch.zeros((S, self.traj_per_sample, 2)),
            visibs=torch.zeros((S, self.traj_per_sample)),
            valids=torch.zeros((S, self.traj_per_sample)),
        )
        return sample, gotit

    def add_photometric_augs(self, rgbs, trajs, visibles, eraser=True, replace=True, augscale=1.0):
        """
        Applies pixel-level noise: Eraser, Patch Replacement, Color Jitter, and Blur.
        Crucially, it updates the 'visibles' mask if a point is covered by an occlusion.
        """
        T, N, _ = trajs.shape
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        if eraser:
            # Randomly black out (or mean-color out) areas to simulate occlusions
            eraser_bounds = [eb*augscale for eb in self.eraser_bounds]
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(np.random.randint(1, self.eraser_max + 1)):
                        xc, yc = np.random.randint(0, W), np.random.randint(0, H)
                        dx, dy = np.random.randint(eraser_bounds[0], eraser_bounds[1], size=2)
                        x0, x1 = np.clip([xc-dx//2, xc+dx//2], 0, W-1).astype(np.int32)
                        y0, y1 = np.clip([yc-dy//2, yc+dy//2], 0, H-1).astype(np.int32)

                        # Fill area with its mean color
                        mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0)
                        rgbs[i][y0:y1, x0:x1, :] = mean_color

                        # Update visibility: points inside the box are now invisible
                        occ_inds = (trajs[i, :, 0] >= x0) & (trajs[i, :, 0] < x1) & \
                                   (trajs[i, :, 1] >= y0) & (trajs[i, :, 1] < y1)
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        if replace:
            # Similar to eraser, but copies a patch from a random frame to the current frame
            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            replace_bounds = [rb*augscale for rb in self.replace_bounds]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(np.random.randint(1, self.replace_max + 1)):
                        xc, yc = np.random.randint(0, W), np.random.randint(0, H)
                        dx, dy = np.random.randint(replace_bounds[0], replace_bounds[1], size=2)
                        x0, x1 = np.clip([xc-dx//2, xc+dx//2], 0, W-1).astype(np.int32)
                        y0, y1 = np.clip([yc-dy//2, yc+dy//2], 0, H-1).astype(np.int32)

                        wid, hei = x1 - x0, y1 - y0
                        if wid > 0 and hei > 0:
                            y00, x00 = np.random.randint(0, H - hei), np.random.randint(0, W - wid)
                            fr = np.random.randint(0, S)
                            rgbs[i][y0:y1, x0:x1, :] = rgbs_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        
                        occ_inds = (trajs[i, :, 0] >= x0) & (trajs[i, :, 0] < x1) & \
                                   (trajs[i, :, 1] >= y0) & (trajs[i, :, 1] < y1)
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        # Apply standard photometric jitter and blur
        if np.random.rand() < self.color_aug_prob:
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
        if np.random.rand() < self.blur_aug_prob:
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        return rgbs, trajs, visibles

    def add_spatial_augs(self, rgbs, trajs, visibles, crop_size, augscale=1.0):
        """
        Applies geometric transforms: Padding, Dynamic Scaling (zoom drift), 
        Dynamic Cropping (pan drift), and Flipping.
        """
        T, N, __ = trajs.shape
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        
        rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        trajs = trajs.astype(np.float64)
        
        # Initial resize to ensure image is at least as large as the crop_size
        target_H, target_W = crop_size
        if target_H > H or target_W > W:
            scale = max(target_H / H, target_W / W)
            rgbs = [cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
            trajs = trajs * scale

        # --- Padding ---
        pad_bounds = [int(pb*augscale) for pb in self.pad_bounds]
        px0, px1, py0, py1 = [np.random.randint(pad_bounds[0], pad_bounds[1]) for _ in range(4)]
        rgbs = [np.pad(rgb, ((py0, py1), (px0, px1), (0, 0))) for rgb in rgbs]
        trajs[:, :, 0] += px0
        trajs[:, :, 1] += py0
        H, W = rgbs[0].shape[:2]

        # --- Scaling + Stretching (Zoom Drift) ---
        # The scale factors change slightly over time to simulate a zooming camera
        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        sx, sy = scale, scale
        sdx, sdy = 0.0, 0.0
        rgbs_scaled = []
        res_delta = self.resize_delta * augscale

        for s in range(S):
            # Apply temporal jitter/shift starting from the second frame (s > 0)
            if s > 0:
                # Generate new random displacement within range [-res_delta, res_delta]
                noise_x = np.random.uniform(-res_delta, res_delta)
                noise_y = np.random.uniform(-res_delta, res_delta)

                if s > 1:
                    # Smooth the transition (80% previous direction, 20% new random noise)
                    sdx = sdx * 0.8 + noise_x * 0.2
                    sdy = sdy * 0.8 + noise_y * 0.2
                else:
                    # For the very first transition, initialize with pure random noise
                    sdx = noise_x
                    sdy = noise_y
                        
            sx, sy = np.clip(sx + sdx, 0.2, 2.0), np.clip(sy + sdy, 0.2, 2.0)
            
            # Keep aspect ratio somewhat sane
            avg_s = (sx + sy) * 0.5
            sx, sy = sx * 0.5 + avg_s * 0.5, sy * 0.5 + avg_s * 0.5

            h_new, w_new = max(int(H * sy), crop_size[0] + 10), max(int(W * sx), crop_size[1] + 10)
            rgbs_scaled.append(cv2.resize(rgbs[s], (w_new, h_new), interpolation=cv2.INTER_LINEAR))
            trajs[s, :, 0] *= (w_new - 1) / (W - 1)
            trajs[s, :, 1] *= (h_new - 1) / (H - 1)
        
        rgbs = rgbs_scaled

        # --- Dynamic Cropping (Pan/Jitter Drift) ---
        # Selects a window that moves slightly per frame
        visible_mask = visibles[0] > 0

        if np.any(visible_mask):
            # Calculate the centroid (mean X, mean Y) of visible points
            mid_x = np.mean(trajs[0, visible_mask, 0])
            mid_y = np.mean(trajs[0, visible_mask, 1])
        else:
            # Fallback: Default to image center if no visible points exist
            mid_y, mid_x = crop_size[0], crop_size[1]
        x0, y0 = int(mid_x - crop_size[1] // 2), int(mid_y - crop_size[0] // 2)
        off_x, off_y = 0, 0
        max_off = int(self.max_crop_offset * augscale)

        for s in range(S):
            if s > 0:
                off_x = int(off_x * 0.8 + np.random.randint(-max_off, max_off + 1) * 0.2)
                off_y = int(off_y * 0.8 + np.random.randint(-max_off, max_off + 1) * 0.2)
            
            x0, y0 = x0 + off_x, y0 + off_y
            h_s, w_s = rgbs[s].shape[:2]
            y0 = 0 if h_s == crop_size[0] else min(max(0, y0), h_s - crop_size[0] - 1)
            x0 = 0 if w_s == crop_size[1] else min(max(0, x0), w_s - crop_size[1] - 1)

            rgbs[s] = rgbs[s][y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
            trajs[s, :, 0] -= x0
            trajs[s, :, 1] -= y0

        # --- Mirroring ---
        if np.random.rand() < self.h_flip_prob:
            rgbs = [rgb[:, ::-1].copy() for rgb in rgbs]
            trajs[:, :, 0] = (crop_size[1] - 1) - trajs[:, :, 0]
        if np.random.rand() < self.v_flip_prob:
            rgbs = [rgb[::-1].copy() for rgb in rgbs]
            trajs[:, :, 1] = (crop_size[0] - 1) - trajs[:, :, 1]

        return np.stack(rgbs), trajs.astype(np.float32)

    def crop(self, rgbs, trajs, crop_size):
        """Standard static random crop for all frames."""
        T, N, _ = trajs.shape
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        
        target_H, target_W = crop_size
        if target_H > H or target_W > W:
            scale = max(target_H / H, target_W / W)
            rgbs = [cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
            trajs = trajs * scale
            H, W = rgbs[0].shape[:2]

        y0 = 0 if crop_size[0] >= H else (H - crop_size[0]) // 2
        x0 = 0 if crop_size[1] >= W else np.random.randint(0, W - crop_size[1])
        
        rgbs = [rgb[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]] for rgb in rgbs]
        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        return np.stack(rgbs), trajs

    def follow_crop(self, rgbs, trajs, visibs, crop_size):
        """
        A smart crop that 'follows' the most interesting (high motion/acceleration) 
        points in the sequence, creating a stabilized 'action camera' effect.
        """
        T, N, _ = trajs.shape
        rgbs = [rgb for rgb in rgbs]
        S, H, W = len(rgbs), rgbs[0].shape[0], rgbs[0].shape[1]

        # Calculate acceleration to find 'interesting' moving points
        vels = trajs[1:] - trajs[:-1] 
        accels = vels[1:] - vels[:-1]
        vis__ = visibs[1:-1] * visibs[:-2] * visibs[2:] # Points visible across 3 frames
        travel = np.sum(np.sum(np.abs(accels) * vis__[:,:,None], axis=2), axis=0)
        
        # Pick a point with high acceleration to focus on
        inds = np.argsort(-travel)[:max(int(np.sum(travel > 0) // 32), 32)]
        smooth_xys = trajs[:, np.random.choice(inds)]

        # Smooth the path so the camera doesn't shake too violently
        def smooth_path(xys, num_passes):
            kernel = np.array([0.25, 0.5, 0.25])
            for _ in range(num_passes):
                padded = np.pad(xys, ((1, 1), (0, 0)), mode='edge')
                xys = kernel[0] * padded[:-2] + kernel[1] * padded[1:-1] + kernel[2] * padded[2:]
            return xys

        num_passes = np.random.randint(4, S)
        smooth_xys = smooth_path(smooth_xys, num_passes)
        smooth_xys = np.clip(smooth_xys, [crop_size[1]//2, crop_size[0]//2], [W - crop_size[1]//2, H - crop_size[0]//2])

        for si in range(S):
            x0, y0 = (smooth_xys[si] - [crop_size[1]//2, crop_size[0]//2]).round().astype(np.int32)
            rgbs[si] = rgbs[si][y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
            trajs[si, :, 0] -= x0
            trajs[si, :, 1] -= y0

        return np.stack(rgbs), trajs