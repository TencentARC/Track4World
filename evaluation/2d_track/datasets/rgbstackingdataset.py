import torch
import numpy as np
import pickle
from datasets.pointdataset import PointDataset
import holi4d.utils.data
import cv2

class RGBStackingDataset(PointDataset):
    def __init__(
            self,
            data_root='../datasets/tapvid_rgbstacking',
            crop_size=(384,512),
            seq_len=None,
            only_first=False,
    ):
        # Initialize the parent PointDataset with common parameters
        super(RGBStackingDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )

        # Logging dataset loading
        print('loading TAPVID-RGB-Stacking dataset...')

        # Dataset name identifier
        self.dname = 'rgbstacking'

        # Whether to only use the first frame (for evaluation protocols)
        self.only_first = only_first
        
        # Load the preprocessed TAPVID RGB stacking pickle file
        input_path = '%s/tapvid_rgb_stacking.pkl' % data_root
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            # If the pickle is a dictionary, convert to a list of video samples
            if isinstance(data, dict):
                data = list(data.values())

        # Store all video samples
        self.data = data
        print('found %d videos in %s' % (len(self.data), data_root))
        
    def __getitem__(self, index):
        # Retrieve one video sample
        dat_ = self.data[index]

        # List of RGB frames, each with shape (H, W, C) and dtype uint8
        rgbs = dat_['video']

        # Point trajectories with shape (N, S, 2)
        trajs = dat_['points']

        # Visibility mask: 1 means visible, 0 means occluded
        visibs_ = 1 - dat_['occluded']

        # Validity mask (annotations are only valid when visible)
        valids = visibs_.copy()
        
        # Transpose trajectories to shape (S, N, 2)
        trajs = trajs.transpose(1, 0, 2)

        # Transpose visibility and validity masks to shape (S, N)
        visibs_ = visibs_.transpose(1, 0)
        valids = valids.transpose(1, 0)

        # Standardize data format (temporal trimming, first-frame selection, etc.)
        rgbs, trajs, visibs_, valids = holi4d.utils.data.standardize_test_data(
            rgbs, trajs, visibs_, 
            valids, only_first=self.only_first,
            seq_len=self.seq_len
        )
        
        # Resize all RGB frames to the target crop size
        rgbs = [
            cv2.resize(
                rgb, (self.crop_size[1], self.crop_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            for rgb in rgbs
        ]

        # Scale normalized trajectory coordinates to pixel coordinates
        # (1.0, 1.0) corresponds to the bottom-right pixel
        H, W = rgbs[0].shape[:2]
        trajs[:, :, 0] *= W - 1
        trajs[:, :, 1] *= H - 1
        
        # Convert RGB frames to a PyTorch tensor with shape (S, C, H, W)
        rgbs = (
            torch.from_numpy(np.stack(rgbs, 0))
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
        )

        # Convert trajectories and masks to PyTorch tensors
        trajs = torch.from_numpy(trajs).float()    # (S, N, 2)
        visibs_ = torch.from_numpy(visibs_).float()  # (S, N)
        valids = torch.from_numpy(valids).float()  # (S, N)

        # Package everything into a VideoData object
        sample = holi4d.utils.data.VideoData(
            video=rgbs,
            trajs=trajs,
            visibs=visibs_,
            valids=valids, 
            dname=self.dname,
        )

        # Return sample and a dummy success flag
        return sample, True

    def __len__(self):
        # Number of video samples in the dataset
        return len(self.data)
