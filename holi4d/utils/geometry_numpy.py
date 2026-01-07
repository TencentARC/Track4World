from typing import *
from functools import partial
import math

import cv2
import numpy as np
from scipy.signal import fftconvolve
import utils3d


def weighted_mean_numpy(
    x: np.ndarray,
    w: np.ndarray = None,
    axis: Union[int, Tuple[int, ...]] = None,
    keepdims: bool = False,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Compute weighted mean of a numpy array.

    Args:
        x: input array
        w: optional weights of the same shape as x
        axis: axis or axes along which to compute mean
        keepdims: whether to keep reduced dimensions
        eps: small value to prevent division by zero

    Returns:
        weighted mean along specified axis
    """
    if w is None:
        return np.mean(x, axis=axis)
    else:
        w = w.astype(x.dtype)
        # Calculate weighted sum and divide by sum of weights (clamped to eps)
        weighted_sum = (x * w).mean(axis=axis)
        weights_mean = np.clip(w.mean(axis=axis), eps, None)
        return weighted_sum / weights_mean


def harmonic_mean_numpy(
    x: np.ndarray,
    w: np.ndarray = None,
    axis: Union[int, Tuple[int, ...]] = None,
    keepdims: bool = False,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Compute (weighted) harmonic mean of a numpy array.

    The harmonic mean is defined as: H = 1 / mean(1 / x)

    Args:
        x: input array
        w: optional weights of the same shape as x
        axis: axis or axes along which to compute mean
        keepdims: whether to keep reduced dimensions
        eps: small value to prevent division by zero

    Returns:
        harmonic mean along specified axis
    """
    if w is None:
        return 1 / (1 / np.clip(x, eps, None)).mean(axis=axis)
    else:
        w = w.astype(x.dtype)
        # Calculate weighted mean of the inverse values
        inv_weighted_mean = weighted_mean_numpy(
            1 / (x + eps), w, axis=axis, keepdims=keepdims, eps=eps
        )
        return 1 / (inv_weighted_mean + eps)


def normalized_view_plane_uv_numpy(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Generate normalized UV coordinates on the camera view plane.

    The coordinates are in a normalized range based on the image diagonal:
    - Top-left corner maps to (-width/diagonal, -height/diagonal)
    - Bottom-right corner maps to (width/diagonal, height/diagonal)

    Args:
        width: image width
        height: image height
        aspect_ratio: optional aspect ratio (width / height). 
                      If None, computed from width and height
        dtype: data type of output array

    Returns:
        uv: (H, W, 2) array of normalized coordinates
    """
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    # Calculate span based on aspect ratio
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    # Generate linear spaces for U and V
    u = np.linspace(
        -span_x * (width - 1) / width, 
        span_x * (width - 1) / width, 
        width, 
        dtype=dtype
    )
    v = np.linspace(
        -span_y * (height - 1) / height, 
        span_y * (height - 1) / height, 
        height, 
        dtype=dtype
    )
    
    u, v = np.meshgrid(u, v, indexing='xy')
    uv = np.stack([u, v], axis=-1)
    return uv


def focal_to_fov_numpy(focal: np.ndarray):
    """
    Convert focal length (in pixels) to field-of-view (radians).

    Args:
        focal: focal length along x or y axis

    Returns:
        fov: corresponding field-of-view angle
    """
    return 2 * np.arctan(0.5 / focal)


def fov_to_focal_numpy(fov: np.ndarray):
    """
    Convert field-of-view (radians) to focal length in pixels.

    Args:
        fov: field-of-view along x or y axis

    Returns:
        focal length in pixels
    """
    return 0.5 / np.tan(fov / 2)


def intrinsics_to_fov_numpy(intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert camera intrinsic matrix to FOV angles along x and y axes.

    Args:
        intrinsics: [..., 3, 3] camera intrinsics matrix

    Returns:
        fov_x, fov_y: field-of-view angles in radians
    """
    fov_x = focal_to_fov_numpy(intrinsics[..., 0, 0])
    fov_y = focal_to_fov_numpy(intrinsics[..., 1, 1])
    return fov_x, fov_y


def point_map_to_depth_legacy_numpy(points: np.ndarray):
    """
    Estimate depth, FOV, and depth shift from point cloud map using 
    legacy least-squares.

    Args:
        points: [..., H, W, 3] 3D points

    Returns:
        depth: corrected depth map [..., H, W]
        fov_x: estimated horizontal FOV
        fov_y: estimated vertical FOV
        shift: global depth shift applied
    """
    height, width = points.shape[-3:-1]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    
    # Generate normalized UV coordinates
    uv = normalized_view_plane_uv_numpy(width, height, dtype=points.dtype)
    _, uv = np.broadcast_arrays(points[..., :2], uv)

    # Prepare linear system Ax = b
    # b: (..., H * W * 2)
    b = (uv * points[..., 2:]).reshape(*points.shape[:-3], -1)
    
    # A: (..., H * W * 2, 2)
    A = np.stack([points[..., :2], -uv], axis=-1)
    A = A.reshape(*points.shape[:-3], -1, 2)

    # Solve least squares: M = A^T * A
    M = A.swapaxes(-2, -1) @ A 
    
    # solution = inv(M) @ A^T @ b
    inv_M = np.linalg.inv(M + 1e-6 * np.eye(2))
    At_b = (A.swapaxes(-2, -1) @ b[..., None])
    solution = (inv_M @ At_b).squeeze(-1)
    
    focal, shift = solution

    depth = points[..., 2] + shift[..., None, None]
    fov_x = np.arctan(width / diagonal / focal) * 2
    fov_y = np.arctan(height / diagonal / focal) * 2
    return depth, fov_x, fov_y, shift


def solve_optimal_focal(uv: np.ndarray, xyz: np.ndarray):
    """
    Solve least-squares optimal focal length (without depth shift).

    Solve min ||focal * (xy / z) - uv||^2

    Args:
        uv: pixel coordinates (N, 2)
        xyz: corresponding 3D points (N, 3)

    Returns:
        optimal focal length (scalar)
    """
    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    # Projection (without shift)
    xy_proj = xy / z[:, None]

    # Optimal focal (Least Squares solution)
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_focal.astype(np.float32)


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    """
    Solve jointly for optimal focal length and global depth shift.

    Solve min ||focal * (xy / (z + shift)) - uv||^2 using Levenberg-Marquardt.

    Args:
        uv: pixel coordinates (N, 2)
        xyz: corresponding 3D points (N, 3)

    Returns:
        optim_shift: scalar depth shift
        optim_focal: scalar focal length
    """
    from scipy.optimize import least_squares
    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        # Project points with current shift
        xy_proj = xy / (z + shift)[: , None]
        # Solve for optimal focal given this shift
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        # Calculate residuals
        err = (f * xy_proj - uv).ravel()
        return err

    # Optimize shift
    solution = least_squares(
        partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm'
    )
    optim_shift = solution['x'].squeeze().astype(np.float32)

    # Recompute optimal focal with the optimized shift
    xy_proj = xy / (z + optim_shift)[: , None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    """
    Solve for optimal depth shift given a fixed focal length.

    Solve min ||focal * (xy / (z + shift)) - uv||^2

    Args:
        uv: pixel coordinates (N, 2)
        xyz: corresponding 3D points (N, 3)
        focal: fixed focal length

    Returns:
        optim_shift: scalar depth shift
    """
    from scipy.optimize import least_squares
    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[: , None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(
        partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm'
    )
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift


def recover_focal_shift_numpy(
    points: np.ndarray,
    mask: np.ndarray = None,
    focal: float = None,
    downsample_size: Tuple[int, int] = (64, 64)
):
    """
    Estimate optimal focal length and depth shift from 3D points.

    Args:
        points: (H, W, 3) 3D points
        mask: optional boolean mask for valid points
        focal: optional known focal length; if None, focal is estimated
        downsample_size: resolution for efficient estimation

    Returns:
        focal: estimated focal length
        shift: estimated depth shift
    """
    import cv2
    assert points.shape[-1] == 3, "Points should (H, W, 3)"

    height, width = points.shape[-3], points.shape[-2]
    
    uv = normalized_view_plane_uv_numpy(width=width, height=height)
    
    # Downsample points and UVs for faster optimization
    if mask is None:
        points_lr = cv2.resize(
            points, downsample_size, interpolation=cv2.INTER_LINEAR
        ).reshape(-1, 3)
        uv_lr = cv2.resize(
            uv, downsample_size, interpolation=cv2.INTER_LINEAR
        ).reshape(-1, 2)
    else:
        # Use mask-aware resizing to avoid interpolating invalid points
        (points_lr, uv_lr), mask_lr = mask_aware_nearest_resize_numpy(
            (points, uv), mask, downsample_size
        )
    
    if points_lr.size < 2:
        return 1., 0.
    
    if focal is None:
        focal, shift = solve_optimal_focal_shift(uv_lr, points_lr)
    else:
        shift = solve_optimal_shift(uv_lr, points_lr, focal)

    return focal, shift


def mask_aware_nearest_resize_numpy(
    inputs: Union[np.ndarray, Tuple[np.ndarray, ...], None],
    mask: np.ndarray, 
    size: Tuple[int, int], 
    return_index: bool = False
) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...], None], np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Resize 2D map by nearest interpolation. Return the nearest neighbor index 
    and mask of the resized map.

    ### Parameters
    - `inputs`: a single or a list of input 2D map(s) of shape (..., H, W, ...). 
    - `mask`: input 2D mask of shape (..., H, W)
    - `size`: target size (width, height)

    ### Returns
    - `*resized_maps`: resized map(s) of shape (..., target_height, target_width, ...). 
    - `resized_mask`: mask of the resized map of shape (..., target_height, target_width)
    - `nearest_idx`: if return_index is True, nearest neighbor index of the resized map 
       of shape (..., target_height, target_width) for each dimension.
    """
    height, width = mask.shape[-2:]
    target_width, target_height = size
    filter_h_f = max(1, height / target_height)
    filter_w_f = max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1
    
    # Window the original mask and uv
    uv = utils3d.numpy.image_pixel_center(
        width=width, height=height, dtype=np.float32
    )
    indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
    
    # Pad UV coordinates
    padded_uv = np.full(
        (height + 2 * padding_h, width + 2 * padding_w, 2), 0, dtype=np.float32
    )
    padded_uv[padding_h:padding_h + height, padding_w:padding_w + width] = uv
    
    # Pad Mask
    padded_mask = np.full(
        (*mask.shape[:-2], height + 2 * padding_h, width + 2 * padding_w), 
        False, dtype=bool
    )
    padded_mask[..., padding_h:padding_h + height, padding_w:padding_w + width] = mask
    
    # Pad Indices
    padded_indices = np.full(
        (height + 2 * padding_h, width + 2 * padding_w), 0, dtype=np.int32
    )
    padded_indices[padding_h:padding_h + height, padding_w:padding_w + width] = indices
    
    # Create sliding windows
    windowed_uv = utils3d.numpy.sliding_window_2d(
        padded_uv, (filter_h_i, filter_w_i), 1, axis=(0, 1)
    )
    windowed_mask = utils3d.numpy.sliding_window_2d(
        padded_mask, (filter_h_i, filter_w_i), 1, axis=(-2, -1)
    )
    windowed_indices = utils3d.numpy.sliding_window_2d(
        padded_indices, (filter_h_i, filter_w_i), 1, axis=(0, 1)
    )

    # Gather the target pixels's local window
    target_centers = utils3d.numpy.image_uv(
        width=target_width, height=target_height, dtype=np.float32
    ) * np.array([width, height], dtype=np.float32)
    
    target_lefttop = target_centers - np.array(
        (filter_w_f / 2, filter_h_f / 2), dtype=np.float32
    )
    target_window = np.round(target_lefttop).astype(np.int32) + \
                    np.array((padding_w, padding_h), dtype=np.int32)

    # Extract windows corresponding to target pixels
    target_window_centers = windowed_uv[
        target_window[..., 1], target_window[..., 0], :, :, :
    ].reshape(target_height, target_width, 2, filter_size)
    
    target_window_mask = windowed_mask[
        ..., target_window[..., 1], target_window[..., 0], :, :
    ].reshape(*mask.shape[:-2], target_height, target_width, filter_size)
    
    target_window_indices = windowed_indices[
        target_window[..., 1], target_window[..., 0], :, :
    ].reshape(
        *([-1] * (mask.ndim - 2)), target_height, target_width, filter_size
    )

    # Compute nearest neighbor in the local window for each pixel 
    dist = np.square(target_window_centers - target_centers[..., None])
    dist = dist[..., 0, :] + dist[..., 1, :]
    # Set distance to infinity for masked pixels
    dist = np.where(target_window_mask, dist, np.inf)
    
    nearest_in_window = np.argmin(dist, axis=-1, keepdims=True)
    nearest_idx = np.take_along_axis(
        target_window_indices, nearest_in_window, axis=-1
    ).squeeze(-1)
    
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    target_mask = np.any(target_window_mask, axis=-1)
    
    # Construct batch indices for advanced indexing
    batch_indices = [
        np.arange(n).reshape([1] * i + [n] + [1] * (mask.ndim - i - 1)) 
        for i, n in enumerate(mask.shape[:-2])
    ]

    index = (*batch_indices, nearest_i, nearest_j)
    
    if inputs is None:
        outputs = None
    elif isinstance(inputs, np.ndarray):
        outputs = inputs[index]
    elif isinstance(inputs, Sequence):
        outputs = tuple(x[index] for x in inputs)
    else:
        raise ValueError(f'Invalid input type: {type(inputs)}')
    
    if return_index:
        return outputs, target_mask, index
    else:
        return outputs, target_mask


def mask_aware_area_resize_numpy(
    image: np.ndarray, 
    mask: np.ndarray, 
    target_width: int, 
    target_height: int
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """
    Resize 2D map by nearest interpolation. Return the nearest neighbor index 
    and mask of the resized map.

    ### Parameters
    - `image`: Input 2D image of shape (..., H, W, C)
    - `mask`: Input 2D mask of shape (..., H, W)
    - `target_width`: target width of the resized map
    - `target_height`: target height of the resized map

    ### Returns
    - `nearest_idx`: Nearest neighbor index of the resized map of shape 
       (..., target_height, target_width). 
    - `target_mask`: Mask of the resized map of shape (..., target_height, target_width)
    """
    height, width = mask.shape[-2:]

    if image.shape[-2:] == (height, width):
        omit_channel_dim = True
    else:
        omit_channel_dim = False
    if omit_channel_dim:
        image = image[..., None]

    image = np.where(mask[..., None], image, 0)

    filter_h_f = max(1, height / target_height)
    filter_w_f = max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f) + 1, math.ceil(filter_w_f) + 1
    filter_size = filter_h_i * filter_w_i
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1
    
    # Window the original mask and uv (non-copy)
    uv = utils3d.numpy.image_pixel_center(
        width=width, height=height, dtype=np.float32
    )
    indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
    
    padded_uv = np.full(
        (height + 2 * padding_h, width + 2 * padding_w, 2), 0, dtype=np.float32
    )
    padded_uv[padding_h:padding_h + height, padding_w:padding_w + width] = uv
    
    padded_mask = np.full(
        (*mask.shape[:-2], height + 2 * padding_h, width + 2 * padding_w), 
        False, dtype=bool
    )
    padded_mask[..., padding_h:padding_h + height, padding_w:padding_w + width] = mask
    
    padded_indices = np.full(
        (height + 2 * padding_h, width + 2 * padding_w), 0, dtype=np.int32
    )
    padded_indices[padding_h:padding_h + height, padding_w:padding_w + width] = indices
    
    windowed_uv = utils3d.numpy.sliding_window_2d(
        padded_uv, (filter_h_i, filter_w_i), 1, axis=(0, 1)
    )
    windowed_mask = utils3d.numpy.sliding_window_2d(
        padded_mask, (filter_h_i, filter_w_i), 1, axis=(-2, -1)
    )
    windowed_indices = utils3d.numpy.sliding_window_2d(
        padded_indices, (filter_h_i, filter_w_i), 1, axis=(0, 1)
    )

    # Gather the target pixels's local window
    target_center = utils3d.numpy.image_uv(
        width=target_width, height=target_height, dtype=np.float32
    ) * np.array([width, height], dtype=np.float32)
    
    target_lefttop = target_center - np.array(
        (filter_w_f / 2, filter_h_f / 2), dtype=np.float32
    )
    target_bottomright = target_center + np.array(
        (filter_w_f / 2, filter_h_f / 2), dtype=np.float32
    )
    target_window = np.floor(target_lefttop).astype(np.int32) + \
                    np.array((padding_w, padding_h), dtype=np.int32)

    target_window_centers = windowed_uv[
        target_window[..., 1], target_window[..., 0], :, :, :
    ].reshape(target_height, target_width, 2, filter_size)
    
    target_window_mask = windowed_mask[
        ..., target_window[..., 1], target_window[..., 0], :, :
    ].reshape(*mask.shape[:-2], target_height, target_width, filter_size)
    
    target_window_indices = windowed_indices[
        target_window[..., 1], target_window[..., 0], :, :
    ].reshape(target_height, target_width, filter_size)

    # Compute pixel area in the local windows
    target_window_lefttop = np.maximum(
        target_window_centers - 0.5, target_lefttop[..., None]
    )
    target_window_bottomright = np.minimum(
        target_window_centers + 0.5, target_bottomright[..., None]
    )
    target_window_area = (
        target_window_bottomright - target_window_lefttop
    ).clip(0, None)
    
    target_window_area = np.where(
        target_window_mask, 
        target_window_area[..., 0, :] * target_window_area[..., 1, :], 
        0
    )
    
    # Weighted sum by area
    target_window_image = image.reshape(
        *image.shape[:-3], height * width, -1
    )[..., target_window_indices, :].swapaxes(-2, -1)
    
    target_mask = np.sum(target_window_area, axis=-1) >= 0.25
    target_image = weighted_mean_numpy(
        target_window_image, target_window_area[..., None, :], axis=-1
    )
    
    if omit_channel_dim:
        target_image = target_image[..., 0]

    return target_image, target_mask


def norm3d(x: np.ndarray) -> np.ndarray:
    "Faster `np.linalg.norm(x, axis=-1)` for 3D vectors"
    return np.sqrt(
        np.square(x[..., 0]) + np.square(x[..., 1]) + np.square(x[..., 2])
    )
    

def depth_occlusion_edge_numpy(
    depth: np.ndarray, 
    mask: np.ndarray, 
    thickness: int = 1, 
    tol: float = 0.1
):
    disp = np.where(mask, 1 / depth, 0)
    disp_pad = np.pad(disp, (thickness, thickness), constant_values=0)
    mask_pad = np.pad(mask, (thickness, thickness), constant_values=False)
    kernel_size = 2 * thickness + 1
    
    # Create sliding windows for disparity and mask
    disp_window = utils3d.numpy.sliding_window_2d(
        disp_pad, (kernel_size, kernel_size), 1, axis=(-2, -1)
    )
    mask_window = utils3d.numpy.sliding_window_2d(
        mask_pad, (kernel_size, kernel_size), 1, axis=(-2, -1)
    )

    disp_mean = weighted_mean_numpy(disp_window, mask_window, axis=(-2, -1))
    
    # Detect edges based on disparity difference
    fg_edge_mask = mask & (disp > (1 + tol) * disp_mean)
    bg_edge_mask = mask & (disp_mean > (1 + tol) * disp)

    # Dilate edges
    dilated_fg = cv2.dilate(
        fg_edge_mask.astype(np.uint8), 
        np.ones((3, 3), dtype=np.uint8), 
        iterations=thickness
    )
    dilated_bg = cv2.dilate(
        bg_edge_mask.astype(np.uint8), 
        np.ones((3, 3), dtype=np.uint8), 
        iterations=thickness
    )

    edge_mask = (dilated_fg > 0) & (dilated_bg > 0)

    return edge_mask


def disk_kernel(radius: int) -> np.ndarray:
    """
    Generate disk kernel with given radius.
    
    Args:
        radius (int): Radius of the disk (in pixels).
    
    Returns:
        np.ndarray: (2*radius+1, 2*radius+1) normalized convolution kernel.
    """
    # Create coordinate grid centered at (0,0)
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    # Generate disk: region inside circle with radius R is 1
    kernel = ((X**2 + Y**2) <= radius**2).astype(np.float32)
    # Normalize the kernel
    kernel /= np.sum(kernel)
    return kernel


def disk_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Apply disk blur to an image using FFT convolution.

    Args:
        image (np.ndarray): Input image, can be grayscale or color.
        radius (int): Blur radius (in pixels).

    Returns:
        np.ndarray: Blurred image.
    """
    if radius == 0:
        return image
    kernel = disk_kernel(radius)
    if image.ndim == 2:
        blurred = fftconvolve(image, kernel, mode='same')
    elif image.ndim == 3:
        channels = []
        for i in range(image.shape[2]):
            blurred_channel = fftconvolve(image[..., i], kernel, mode='same')
            channels.append(blurred_channel)
        blurred = np.stack(channels, axis=-1)
    else:
        raise ValueError("Image must be 2D or 3D.")
    return blurred


def depth_of_field(
    img: np.ndarray, 
    disp: np.ndarray, 
    focus_disp : float, 
    max_blur_radius : int = 10,
) -> np.ndarray:
    """
    Apply depth of field effect to an image.

    Args:
        img (numpy.ndarray): (H, W, 3) input image.
        disp (numpy.ndarray): (H, W) disparity map of the scene.
        focus_disp (float): Focus disparity of the lens.
        max_blur_radius (int): Maximum blur radius (in pixels).
        
    Returns:
        numpy.ndarray: (H, W, 3) output image with depth of field effect applied.
    """
    # Precalculate dilated depth map for each blur radius
    max_disp = np.max(disp)
    disp = disp / max_disp
    focus_disp = focus_disp / max_disp
    dilated_disp = []
    
    for radius in range(max_blur_radius + 1):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)
        )
        dilated_disp.append(cv2.dilate(disp, kernel, iterations=1))
        
    # Determine the blur radius for each pixel based on the depth map
    blur_radii = np.clip(
        abs(disp - focus_disp) * max_blur_radius, 0, max_blur_radius
    ).astype(np.int32)
    
    for radius in range(max_blur_radius + 1):
        dialted_blur_radii = np.clip(
            abs(dilated_disp[radius] - focus_disp) * max_blur_radius, 
            0, max_blur_radius
        ).astype(np.int32)
        
        mask = (dialted_blur_radii >= radius) & \
               (dialted_blur_radii >= blur_radii) & \
               (dilated_disp[radius] > disp)
        blur_radii[mask] = dialted_blur_radii[mask]
        
    blur_radii = np.clip(blur_radii, 0, max_blur_radius)
    blur_radii = cv2.blur(blur_radii, (5, 5))

    # Precalculate the blurred image for each blur radius
    unique_radii = np.unique(blur_radii)
    precomputed = {}
    for radius in range(max_blur_radius + 1):
        if radius not in unique_radii:
            continue
        precomputed[radius] = disk_blur(img, radius)
        
    # Composite the blurred image for each pixel
    output = np.zeros_like(img)
    for r in unique_radii:
        mask = blur_radii == r
        output[mask] = precomputed[r][mask]
        
    return output