import glob
import math
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Optional, List, Union

EPS = 1e-6

# Coordinate bounds
XMIN, XMAX = -64.0, 64.0  # Right (neg is left)
YMIN, YMAX = -64.0, 64.0  # Down (neg is up)
ZMIN, ZMAX = -64.0, 64.0  # Forward

# ==============================================================================
# Basic Statistics & Masked Operations
# ==============================================================================

def print_stats(name: str, tensor: np.ndarray):
    tensor = tensor.astype(np.float32)
    print(f'{name} min = {np.min(tensor):.2f}, mean = {np.mean(tensor):.2f}, '
          f'max = {np.max(tensor):.2f}, shape = {tensor.shape}')

def reduce_masked_mean(x: np.ndarray, mask: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    prod = x * mask
    numer = np.sum(prod, axis=axis, keepdims=keepdims)
    denom = np.sum(mask, axis=axis, keepdims=keepdims) + EPS
    return numer / denom

def reduce_masked_sum(x: np.ndarray, mask: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    prod = x * mask
    return np.sum(prod, axis=axis, keepdims=keepdims)

def reduce_masked_median(x: np.ndarray, mask: np.ndarray, keep_batch=False) -> np.ndarray:
    assert x.shape == mask.shape, f"Shapes mismatch: {x.shape} vs {mask.shape}"

    if keep_batch:
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        mask_flat = mask.reshape(B, -1)
        meds = np.zeros(B, dtype=np.float32)
        
        for b in range(B):
            xb = x_flat[b]
            mb = mask_flat[b]
            valid_data = xb[mb > 0]
            if valid_data.size > 0:
                meds[b] = np.median(valid_data)
            else:
                meds[b] = np.nan
        return meds
    else:
        valid_data = x[mask > 0]
        if valid_data.size > 0:
            med = np.median(valid_data)
        else:
            med = np.nan
        return np.array([med], dtype=np.float32)

def get_nFiles(path: str) -> int:
    return len(glob.glob(path))

def get_file_list(path: str) -> List[str]:
    return glob.glob(path)

# ==============================================================================
# Rotation & Geometry
# ==============================================================================

def rotm2eul(R: np.ndarray) -> Tuple[float, float, float]:
    """Converts a 3x3 rotation matrix to Euler angles (x, y, z)."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return x, y, z

def rad2deg(rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return rad * 180.0 / np.pi

def deg2rad(deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return deg / 180.0 * np.pi

def eul2rotm(rx: float, ry: float, rz: float) -> np.ndarray:
    """Converts Euler angles to a 3x3 rotation matrix."""
    sinz, cosz = np.sin(rz), np.cos(rz)
    siny, cosy = np.sin(ry), np.cos(ry)
    sinx, cosx = np.sin(rx), np.cos(rx)

    r11 = cosy * cosz
    r12 = sinx * siny * cosz - cosx * sinz
    r13 = cosx * siny * cosz + sinx * sinz
    r21 = cosy * sinz
    r22 = sinx * siny * sinz + cosx * cosz
    r23 = cosx * siny * sinz - sinx * cosz
    r31 = -siny
    r32 = sinx * cosy
    r33 = cosx * cosy

    R = np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ], dtype=np.float32)
    return R

def wrap2pi(rad_angle: np.ndarray) -> np.ndarray:
    """Wraps angle to [-pi, pi]."""
    return np.arctan2(np.sin(rad_angle), np.cos(rad_angle))

def rot2view(rx, ry, rz, x, y, z):
    """Calculates viewpoint angles from rotation and position."""
    az = wrap2pi(ry - (-np.arctan2(z, x) - 1.5 * np.pi))
    el = -wrap2pi(rx - (-np.arctan2(z, y) - 1.5 * np.pi))
    th = -rz
    return az, el, th

def invAxB(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relative 3d transformation from a to b (inv(a) @ b)."""
    return np.linalg.inv(a) @ b

def merge_rt(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Merges 3x3 R and 3x1 t into 4x4 RT."""
    t = np.reshape(t, [3, 1])
    rt = np.concatenate((r, t), axis=1)
    bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
    return np.concatenate((rt, bottom), axis=0)

def split_rt(rt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Splits 4x4 RT into 3x3 R and 3x1 t."""
    r = rt[:3, :3]
    t = rt[:3, 3].reshape(3, 1)
    return r, t

def safe_inverse(a: np.ndarray) -> np.ndarray:
    """Computes inverse of a rigid body transform 4x4 matrix efficiently."""
    r, t = split_rt(a)
    r_transpose = r.T
    inv = np.concatenate([r_transpose, -r_transpose @ t], axis=1)
    bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
    return np.concatenate([inv, bottom], axis=0)

# ==============================================================================
# Intrinsics & Projections
# ==============================================================================

def split_intrinsics(K: np.ndarray) -> Tuple[float, float, float, float]:
    return K[0, 0], K[1, 1], K[0, 2], K[1, 2]

def merge_intrinsics(fx, fy, x0, y0) -> np.ndarray:
    K = np.eye(4, dtype=np.float32)
    K[0, 0], K[1, 1] = fx, fy
    K[0, 2], K[1, 2] = x0, y0
    return K

def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    fx, fy, x0, y0 = split_intrinsics(K)
    return merge_intrinsics(fx * sx, fy * sy, x0 * sx, y0 * sy)

def apply_pix_T_cam(pix_T_cam: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Projects 3D points (N, 3) to pixels (N, 2)."""
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = np.split(xyz, 3, axis=-1)
    
    z = np.clip(z, 1e-4, None)
    x = (x * fx) / z + x0
    y = (y * fy) / z + y0
    return np.concatenate([x, y], axis=-1)

def apply_4x4(RT: np.ndarray, XYZ: np.ndarray) -> np.ndarray:
    """Applies 4x4 transform to N x 3 points."""
    # XYZ: N x 3
    # RT: 4 x 4
    N = XYZ.shape[0]
    ones = np.ones((N, 1), dtype=XYZ.dtype)
    XYZ1 = np.concatenate([XYZ, ones], axis=1) # N x 4
    
    # (4x4 @ 4xN).T -> N x 4
    XYZ2 = (RT @ XYZ1.T).T
    return XYZ2[:, :3]

def pixels2camera(x, y, z, fx, fy, x0, y0):
    """Unprojects pixels to 3D camera coordinates."""
    # x, y, z are flattened arrays
    x = ((z + EPS) / fx) * (x - x0)
    y = ((z + EPS) / fy) * (y - y0)
    return np.stack([x, y, z], axis=1)

def camera2pixels(xyz: np.ndarray, pix_T_cam: np.ndarray) -> np.ndarray:
    """Projects 3D points to 2D pixels."""
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    z = np.clip(z, 1e-4, None)
    x = (x * fx) / z + x0
    y = (y * fy) / z + y0
    return np.stack([x, y], axis=-1)

def depth2pointcloud(z: np.ndarray, pix_T_cam: np.ndarray) -> np.ndarray:
    H, W = z.shape
    grid_y, grid_x = meshgrid2d(H, W)
    
    z_flat = z.reshape(-1)
    x_flat = grid_x.reshape(-1)
    y_flat = grid_y.reshape(-1)
    
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    return pixels2camera(x_flat, y_flat, z_flat, fx, fy, x0, y0)

# ==============================================================================
# Voxel & Grid Operations
# ==============================================================================

def get_mem_T_ref(Z, Y, X):
    """Constructs transform from Reference to Memory (Voxel) coordinates."""
    center_T_ref = np.eye(4, dtype=np.float32)
    center_T_ref[:3, 3] = [-XMIN, -YMIN, -ZMIN]

    vox_size_x = (XMAX - XMIN) / float(X)
    vox_size_y = (YMAX - YMIN) / float(Y)
    vox_size_z = (ZMAX - ZMIN) / float(Z)
    
    mem_T_center = np.eye(4, dtype=np.float32)
    mem_T_center[0, 0] = 1. / vox_size_x
    mem_T_center[1, 1] = 1. / vox_size_y
    mem_T_center[2, 2] = 1. / vox_size_z
    
    return mem_T_center @ center_T_ref

def get_ref_T_mem(Z, Y, X):
    mem_T_ref = get_mem_T_ref(Z, Y, X) # Note: Z passed as first arg in original logic
    return np.linalg.inv(mem_T_ref)

def Ref2Mem(xyz, Z, Y, X):
    mem_T_ref = get_mem_T_ref(Z, Y, X)
    return apply_4x4(mem_T_ref, xyz)

def get_inbounds(xyz, Z, Y, X, already_mem=False):
    if not already_mem:
        xyz = Ref2Mem(xyz, Z, Y, X)
    
    x_valid = (xyz[:, 0] >= -0.5) & (xyz[:, 0] < float(X) - 0.5)
    y_valid = (xyz[:, 1] >= -0.5) & (xyz[:, 1] < float(Y) - 0.5)
    z_valid = (xyz[:, 2] >= -0.5) & (xyz[:, 2] < float(Z) - 0.5)
    
    return x_valid & y_valid & z_valid

def get_occupancy(xyz_mem, Z, Y, X):
    inbounds = get_inbounds(xyz_mem, Z, Y, X, already_mem=True)
    xyz_valid = xyz_mem[inbounds]
    
    # Round to nearest voxel index
    xyz_valid = np.round(xyz_valid).astype(np.int32)
    
    voxels = np.zeros((Z, Y, X), dtype=np.float32)
    # Advanced indexing
    voxels[xyz_valid[:, 2], xyz_valid[:, 1], xyz_valid[:, 0]] = 1.0
    return voxels

def voxelize_xyz(xyz_ref, Z, Y, X):
    xyz_mem = Ref2Mem(xyz_ref, Z, Y, X)
    voxels = get_occupancy(xyz_mem, Z, Y, X)
    return voxels.reshape(Z, Y, X, 1)

def convert_occ_to_height(occ):
    Z, Y, X, C = occ.shape
    assert C == 1
    
    height_vals = np.linspace(float(Y), 1.0, Y).reshape(1, Y, 1, 1)
    height_map = np.max(occ * height_vals, axis=1) / float(Y)
    return height_map.reshape(Z, X, C)

def meshgrid2d(Y, X):
    """Returns 2D meshgrid (Y, X)."""
    grid_y, grid_x = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')
    return grid_y.astype(np.float32), grid_x.astype(np.float32)

def gridcloud2d(Y, X):
    grid_y, grid_x = meshgrid2d(Y, X)
    x = grid_x.reshape(-1)
    y = grid_y.reshape(-1)
    return np.stack([x, y], axis=1)

def gridcloud3d(Y, X, Z):
    grid_y, grid_x, grid_z = np.meshgrid(np.arange(Y), np.arange(X), np.arange(Z), indexing='ij')
    x = grid_x.reshape(-1)
    y = grid_y.reshape(-1)
    z = grid_z.reshape(-1)
    return np.stack([x, y, z], axis=1).astype(np.float32)

def sub2ind(height, width, y, x):
    return (y * width + x).astype(np.int32)

# ==============================================================================
# Image Processing & Visualization
# ==============================================================================

def normalize_image(im):
    im_min, im_max = np.min(im), np.max(im)
    return (im - im_min) / (im_max - im_min + EPS)

def preprocess_color(x):
    return x.astype(np.float32) * (1. / 255) - 0.5

def create_depth_image(xy, Z, H, W):
    """Creates a depth image from sparse points (vectorized)."""
    xy = np.round(xy).astype(np.int32)
    
    # Filter valid points
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < W) & \
            (xy[:, 1] >= 0) & (xy[:, 1] < H) & \
            (Z > 0)
            
    xy = xy[valid]
    Z = Z[valid]
    
    depth = np.full((H, W), 70.0, dtype=np.float32)
    # Advanced indexing for speed (last write wins)
    depth[xy[:, 1], xy[:, 0]] = Z
    
    return depth

def vis_depth(depth, maxdepth=80.0, log_vis=True):
    depth = depth.copy()
    depth[depth <= 0.0] = maxdepth
    
    if log_vis:
        depth = np.log(depth)
        depth = np.clip(depth, 0, np.log(maxdepth))
        depth = depth / np.log(maxdepth) # Normalize to 0-1
    else:
        depth = np.clip(depth, 0, maxdepth) / maxdepth
        
    return (depth * 255.0).astype(np.uint8)

def im2col(im, psize):
    """
    Rearranges image blocks into columns.
    Note: For heavy usage, consider using skimage.util.view_as_windows.
    """
    if im.ndim == 2:
        im = im[np.newaxis, :, :]
    
    n_channels, rows, cols = im.shape
    
    # Pad image
    pad_h = int(math.ceil(rows / psize) * psize)
    pad_w = int(math.ceil(cols / psize) * psize)
    im_pad = np.zeros((n_channels, pad_h, pad_w), dtype=im.dtype)
    im_pad[:, :rows, :cols] = im

    # Extract blocks
    # This implementation is functionally correct but slow for large images
    # compared to strided tricks. Keeping logic as requested but cleaning up.
    n_h = pad_h // psize
    n_w = pad_w // psize
    
    final = np.zeros((n_h, n_w, n_channels, psize, psize), dtype=im.dtype)
    
    for c in range(n_channels):
        for x in range(psize):
            for y in range(psize):
                # Shift and sample
                im_shift = im_pad[c, x:, y:]
                # Subsample
                sampled = im_shift[::psize, ::psize]
                # Crop to fit if necessary (though padding handles most)
                h_s, w_s = sampled.shape
                final[:h_s, :w_s, c, x, y] = sampled

    return np.squeeze(final[:rows - psize + 1, :cols - psize + 1])

def filter_discontinuities(depth, filter_size=9, thresh=10):
    H, W = depth.shape
    assert filter_size % 2 == 1, "Can only use odd filter sizes."

    offset = (filter_size - 1) // 2
    patches = im2col(depth, filter_size) # (H, W, 1, K, K)
    
    # Center pixel
    mids = patches[:, :, 0, offset, offset]
    mins = np.min(patches, axis=(2, 3, 4))
    maxes = np.max(patches, axis=(2, 3, 4))

    discont = np.maximum(np.abs(mins - mids), np.abs(maxes - mids))
    mark = discont > thresh

    final_mark = np.zeros((H, W), dtype=bool)
    # Adjust for padding/cropping effects of im2col if needed, 
    # but im2col output usually matches input dims if stride=1.
    # Assuming im2col returns (H, W, ...)
    final_mark = mark

    return depth * (1 - final_mark)

def plot_traj_3d(traj):
    """Plots a 3D trajectory and returns it as an image array."""
    S, C = traj.shape
    assert C == 3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = [plt.cm.RdYlBu(i) for i in np.linspace(0, 1, S)]
    
    xs, ys, zs = traj[:, 0], -traj[:, 1], traj[:, 2]

    ax.scatter(xs, zs, ys, s=30, c=colors, marker='o', alpha=1.0, edgecolors=(0, 0, 0))
        
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-1, 0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    image = np.array(Image.open(buf))
    plt.close(fig) # Important to close memory
    
    return image[:, :, :3]

# ==============================================================================
# Optical Flow Visualization
# ==============================================================================

def make_colorwheel():
    """Generates a color wheel for optical flow visualization."""
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75
        
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
        
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    assert flow_uv.ndim == 3 and flow_uv.shape[2] == 2
    
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow) / clip_flow
        
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    
    u = u / (rad_max + 1e-5)
    v = v / (rad_max + 1e-5)
    return flow_uv_to_colors(u, v, convert_to_bgr)