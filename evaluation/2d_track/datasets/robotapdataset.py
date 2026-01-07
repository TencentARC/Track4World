import torch
import numpy as np
import pickle
from datasets.pointdataset import PointDataset
import holi4d.utils.data
import cv2

class RobotapDataset(PointDataset):
    def __init__(
            self,
            data_root='../datasets/robotap',
            crop_size=(384,512),
            seq_len=None,
            only_first=False,
    ):
        """
        Dataset wrapper for the RoboTap benchmark.

        This dataset provides video sequences with sparse point trajectories
        and visibility annotations, typically used for evaluating 2D/3D
        tracking and motion estimation methods.

        Args:
            data_root: Root directory containing the Robotap dataset and split files.
            crop_size: Spatial resolution (H, W) to which all frames are resized.
            seq_len: Optional temporal length to truncate each video sequence.
            only_first: If True, only the first frame is kept (used for specific eval settings).
        """
        super(RobotapDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )

        # Dataset name identifier
        self.dname = 'robo'
        self.only_first = only_first
        
        # Validation splits (training splits are commented out)
        # self.train_pkls = ['robotap_split0.pkl', 'robotap_split1.pkl', 'robotap_split2.pkl']
        self.val_pkls = ['robotap_split3.pkl', 'robotap_split4.pkl']

        print("loading robotap dataset...")

        # Load all validation videos into memory
        self.data = []
        for vid_pkl in self.val_pkls:
            print(vid_pkl)
            input_path = "%s/%s" % (data_root, vid_pkl)
            with open(input_path, "rb") as f:
                data = pickle.load(f)

            # Each pickle contains a dict of videos; flatten into a list
            keys = list(data.keys())
            self.data += [data[key] for key in keys]

        print("found %d videos in %s" % (len(self.data), data_root))

    def __len__(self):
        """
        Returns the total number of video samples in the dataset.
        """
        return len(self.data)

    def getitem_helper(self, index):
        """
        Load and preprocess a single video sample.

        This function:
            - reads RGB frames, point trajectories, and occlusion flags
            - converts annotations to a unified temporal layout
            - rescales trajectories to match resized image resolution
            - converts everything into PyTorch tensors

        Args:
            index: Index of the video sample.

        Returns:
            sample: A VideoData object containing video frames, trajectories,
                    visibility flags, and validity masks.
            True: A flag indicating successful loading (used by the data pipeline).
        """
        dat = self.data[index]

        # Raw RGB frames (list of H x W x 3 uint8 images)
        rgbs = dat["video"]

        # Point trajectories: (N, S, 2) where N = number of points, S = sequence length
        trajs = dat["points"]

        # Visibility flags derived from occlusion annotations
        visibs = 1 - dat["occluded"]

        # Transpose to temporal-major format: (S, N, 2) and (S, N)
        trajs = trajs.transpose(1,0,2)
        visibs = visibs.transpose(1,0)

        # Validity mask initially matches visibility
        valids = visibs.copy()

        # Standardize data layout, optionally truncate sequence or keep only first frame
        rgbs, trajs, visibs, valids = holi4d.utils.data.standardize_test_data(
            rgbs, trajs, visibs, valids,
            only_first=self.only_first,
            seq_len=self.seq_len
        )
        
        # Resize all frames to the target crop size
        rgbs = [
            cv2.resize(
                rgb,
                (self.crop_size[1], self.crop_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            for rgb in rgbs
        ]

        # Update trajectory coordinates to match resized image resolution
        # Convention: (1.0, 1.0) corresponds to the bottom-right pixel
        H, W = rgbs[0].shape[:2]
        trajs[:,:,0] *= W - 1
        trajs[:,:,1] *= H - 1
        
        # Convert to PyTorch tensors
        rgbs = torch.from_numpy(np.stack(rgbs,0)) \
                    .permute(0,3,1,2) \
                    .contiguous() \
                    .float()          # (S, C, H, W)

        trajs = torch.from_numpy(trajs).float()   # (S, N, 2)
        visibs = torch.from_numpy(visibs).float() # (S, N)
        valids = torch.from_numpy(valids).float() # (S, N)

        # Explicit temporal truncation if seq_len is specified
        if self.seq_len is not None:
            rgbs = rgbs[:self.seq_len]
            trajs = trajs[:self.seq_len]
            valids = valids[:self.seq_len]
            visibs = visibs[:self.seq_len]

        # Pack everything into a unified VideoData container
        sample = holi4d.utils.data.VideoData(
            video=rgbs,
            trajs=trajs,
            visibs=visibs,
            valids=valids,
            dname=self.dname,
        )

        return sample, True
