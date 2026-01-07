import numpy as np
from PIL import Image
from os.path import *
import re
import cv2
from scipy.spatial.transform import Rotation
from typing import List, Dict, Union, Optional, Tuple
import torch

TAG_CHAR = np.array([202021.25], np.float32)

def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    try:
        dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    except:
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())

    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readDPT(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    TAG_FLOAT = 202021.25
    TAG_CHAR = 'PIEH'
    # Check if the flow file tag matches the expected float tag (sanity check)
    assert check == TAG_FLOAT, (
        f"depth_read:: Wrong tag in flow file "
        f"(should be: {TAG_FLOAT}, is: {check}). Big-endian machine?"
    )
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    # Verify that the input image dimensions and total file size are within valid ranges
    assert (width > 0 and height > 0 and 1 < size < 100_000_000), (
        f"depth_read:: Wrong input size (width = {width}, height = {height})."
    )
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    M is the intrinsic matrix, N is the extrinsic matrix, so that
    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates."""
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))

    E = np.eye(4)
    E[0:3,:] = N

    fx, fy, cx, cy = M[0,0], M[1,1], M[0,2], M[1,2]
    kvec = np.array([fx, fy, cx, cy])

    q = Rotation.from_matrix(E[:3,:3]).as_quat()
    pvec = np.concatenate([E[:3,3], q], 0)

    return pvec, kvec


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        return readPFM(file_name).astype(np.float32)
    elif ext == '.dpt':
        return readDPT(file_name).astype(np.float32)
    elif ext == '.cam':
        return cam_read(file_name)
    return []

# ==============================================================================
# Metric Computation Functions
# ==============================================================================

def compute_optical_flow_metrics(
    pred_flow: torch.Tensor, 
    gt_flow: torch.Tensor, 
    valid_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Computes 2D Optical Flow metrics (EPE, Accuracy, Outliers).
    """
    if pred_flow.dim() == 3:
        pred_flow = pred_flow.unsqueeze(0)
        gt_flow = gt_flow.unsqueeze(0)
        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(0)

    B, H, W, _ = pred_flow.shape

    if valid_mask is None:
        valid_mask = torch.ones((B, H, W), dtype=torch.bool, device=pred_flow.device)
    else:
        valid_mask = valid_mask.squeeze(1).bool()

    # End-Point Error
    epe = torch.norm(pred_flow - gt_flow, dim=-1)

    epe_valid = epe[valid_mask]
    gt_valid = gt_flow[valid_mask]
    gt_norm = torch.norm(gt_valid, dim=-1)

    relative_err = epe_valid / (gt_norm + 1e-8)

    return {
        "EPE2D": epe_valid.sum().item(),
        "ACC1_2D": (epe_valid < 1.0).float().sum().item(),
        "ACC3_2D": (epe_valid < 3.0).float().sum().item(),
        "Outlier_2D": ((epe_valid > 3.0) & (relative_err > 0.05)).float().sum().item(),
    }


def compute_pc_metrics(
    pred: torch.Tensor, 
    gt: torch.Tensor, 
    valid_mask: torch.Tensor
) -> Dict[str, float]:
    """
    Computes Point Cloud reconstruction metrics.
    """
    dist_gt = torch.norm(gt[valid_mask], dim=-1)
    dist_err = torch.norm(pred[valid_mask] - gt[valid_mask], dim=-1)
    
    # Absolute Relative Error
    abs_rel = (dist_err / (dist_gt + 1e-6)).sum().item()

    dist_pred = torch.norm(pred[valid_mask], dim=-1)
    
    # Threshold Accuracy
    threshold_1 = (dist_err < 0.25 * torch.minimum(dist_gt, dist_pred)).float().sum().item()
    
    return {
        'abs_rel': abs_rel,
        'threshold_1': threshold_1
    }


def compute_scene_flow_metrics(
    pred_flow3d: torch.Tensor, 
    gt_flow3d: torch.Tensor, 
    valid_mask: torch.Tensor
) -> Dict[str, float]:
    """
    Computes 3D Scene Flow metrics.
    """
    diff = (pred_flow3d - gt_flow3d)[valid_mask]
    epe3d = torch.norm(diff, dim=-1)
    gt_norm = torch.norm(gt_flow3d[valid_mask], dim=-1)
    relative_err = epe3d / (gt_norm + 1e-6)

    return {
        'EPE3D': epe3d.sum().item(),
        'Acc3D_strict': ((epe3d < 0.05) | (relative_err < 0.05)).float().sum().item(),
        'Acc3D_relax': ((epe3d < 0.1) | (relative_err < 0.1)).float().sum().item(),
        'Outlier': ((epe3d > 0.3) | (relative_err > 0.1)).float().sum().item(),
    }