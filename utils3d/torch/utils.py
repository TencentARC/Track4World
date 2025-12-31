from typing import *
from numbers import Number 
import torch
import torch.nn.functional as F
from torch import Tensor
from . import transforms
from . import mesh
from ._helpers import batched
from .._helpers import no_warnings


__all__ = [
    'sliding_window',
    'sliding_window_1d',
    'sliding_window_2d',
    'sliding_window_nd',
    'image_uv',
    'image_pixel_center',
    'image_mesh',
    'chessboard',
    'depth_edge',
    'depth_aliasing',
    'image_mesh_from_depth',
    'points_to_normals',
    'depth_to_points',
    'depth_to_normals',
    'masked_min',
    'masked_max',
    'bounding_rect',
    'lookup',
    'csr_eliminate_zeros',
    'csr_matrix_from_dense_indices',
    'segment_roll'

]

def lookup(key: Tensor, query: Tensor) -> torch.LongTensor:
    """Look up `query` in `key` like a dictionary. Useful for COO indexing.

    ## Parameters
    - `key` (Tensor): shape `(K, ...)`, the array to search in
    - `query` (Tensor): shape `(Q, ...)`, the array to search for

    ## Returns
    - `indices` (Tensor): shape `(Q,)` indices of `query` in `key`. If a query is not found in key, the corresponding index will be -1.

    ## NOTE
    `O((Q + K) * log(Q + K))` complexity.
    """
    unique, inverse = torch.unique(
        torch.cat([key, query], dim=0),
        dim=0,
        return_inverse=True
    )
    index = torch.full((unique.shape[0],), -1, dtype=torch.long, device=key.device)
    index.scatter_(0, inverse[:key.shape[0]], torch.arange(key.shape[0], device=key.device))
    result = index.index_select(0, inverse[key.shape[0]:])
    return torch.where(result < key.shape[0], result, -1)

def csr_matrix_from_dense_indices(indices: Tensor, n_cols: int) -> Tensor:
    """Convert a regular indices array to a sparse CSR adjacency matrix format

    ## Parameters
        - `indices` (Tensor): shape (N, M) dense tensor. Each one in `N` has `M` connections.
        - `values` (Tensor): shape (N, M) values of the connections
        - `n_cols` (int): total number of columns in the adjacency matrix

    ## Returns
        Tensor: shape `(N, n_cols)` sparse CSR adjacency matrix
    """
    return torch.sparse_csr_tensor(
        crow_indices=torch.arange(0, indices.numel() + 1, indices.shape[1], device=indices.device),
        col_indices=indices.view(-1),
        values=torch.ones_like(indices, dtype=torch.bool).view(-1),
        size=(indices.shape[0], n_cols)
    )


def csr_eliminate_zeros(input: Tensor):
    """Remove zero elements from a sparse CSR tensor.
    """
    nonzero = input.values() != 0
    nonzero_element_indices = nonzero.nonzero(as_tuple=False).flatten()
    row_nonzero_count = torch.sparse_csr_tensor(
        input.crow_indices(), 
        input.col_indices(), 
        nonzero, 
        input.size()
    ).long().sum(dim=-1, keepdim=True).to_dense().flatten()
    crow_indices = torch.cat([torch.tensor([0], device=input.device), torch.cumsum(row_nonzero_count, dim=0)])
    col_indices = input.col_indices()[nonzero_element_indices]
    values = input.values()[nonzero_element_indices]
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, input.size())

def segment_roll(data: torch.Tensor, offsets: torch.Tensor, shift: int) -> Tensor:
    """Roll the data within each segment.
    """
    lengths = torch.diff(offsets)
    start = offsets[:-1].repeat_interleave(lengths)
    elem_indices = start + (torch.arange(data.shape[0], dtype=offsets.dtype) - start - shift) % lengths.repeat_interleave(lengths)
    data = data.gather(0, elem_indices)
    return data

def sliding_window(
    x: Tensor, 
    window_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    dilation: Optional[Union[int, Tuple[int, ...]]] = None,
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    pad_mode: str = 'constant',
    pad_value: Number = 0,
    dim: Tuple[int, ...] = None
) -> Tensor:
    """
    Get a sliding window of the input array.
    This function is a wrapper of `torch.nn.functional.unfold` with additional support for padding and stride.

    ## Parameters
    - `x` (Tensor): Input tensor.
    - `window_size` (int or Tuple[int,...]): Size of the sliding window. If int
        is provided, the same size is used for all specified axes.
    - `stride` (Optional[Tuple[int,...]]): Stride between the sliding windows. If None,
        no stride is applied. If int is provided, the same stride is used for all specified axes.
    - `dilation` (Optional[Tuple[int,...]]): Dilation in each sliding window. If None,
        no dilation is applied. If int is provided, the same dilation is used for all specified axes.
    - `pad_size` (Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]]): Size of padding to apply before sliding window.
        Corresponding to `axis`.
        - General format is `((before_1, after_1), (before_2, after_2), ...)`.
        - Shortcut formats: 
            - `int` -> same padding before and after for all axes;
            - `(int, int)` -> same padding before and after for each axis;
            - `((int,), (int,) ...)` -> specify padding for each axis, same before and after.
    - `pad_mode` (str): Padding mode to use. Refer to `numpy.pad` for more details.
    - `pad_value` (Union[int, float]): Value to use for constant padding. Only used
        when `pad_mode` is 'constant'.
    - `axis` (Optional[Tuple[int,...]]): Axes to apply the sliding window. If None, all axes are used.

    ## Returns
    - (Tensor): Sliding window of the input array. 
        - If no padding, the output is a view of the input array with zero copy.
        - Otherwise, the output is no longer a view but a copy of the padded array.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if isinstance(dim, Integral):
        dim = (dim,)
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    if isinstance(window_size, Integral):
        window_size = (window_size,) * len(dim)
    if stride is None:
        stride = (1,) * len(dim)
    elif isinstance(stride, Integral):
        stride = (stride,) * len(dim)
    if dilation is None:
        dilation = (1,) * len(dim)
    elif isinstance(dilation, Integral):
        dilation = (dilation,) * len(dim)
    assert len(window_size) == len(stride) == len(dim)

    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, Integral):
            pad_size = ((pad_size, pad_size),) * len(dim)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, Integral) for p in pad_size):
            pad_size = (pad_size,) * len(dim)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(dim)
            else:
                assert len(pad_size) == len(dim), f"pad_size {pad_size} must match the number of axes {len(dim)}"
            pad_size = tuple(p * 2 if len(p) == 1 else p for p in pad_size)
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in dim else pad_size[dim.index(i)] for i in range(x.ndim)]
        full_pad = tuple(chain(*reversed(full_pad)))
        x = F.pad(x, full_pad, mode=pad_mode, value=pad_value)
    
    for i in range(len(window_size)):
        x = x.unfold(dim[i], (window_size[i] - 1) * dilation[i] + 1, stride[i])[..., ::dilation[i]]
    return x

def sliding_window_1d(x: torch.Tensor, window_size: int, stride: int = 1, dim: int = -1) -> torch.Tensor:
    """
    Sliding window view of the input tensor. The dimension of the sliding window is appended to the end of the input tensor's shape.
    NOTE: Since Pytorch has `unfold` function, 1D sliding window view is just a wrapper of it.
    """
    return x.unfold(dim, window_size, stride)


def sliding_window_nd(x: torch.Tensor, window_size: Tuple[int, ...], stride: Tuple[int, ...], dim: Tuple[int, ...]) -> torch.Tensor:
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    assert len(window_size) == len(stride) == len(dim)
    for i in range(len(window_size)):
        x = sliding_window_1d(x, window_size[i], stride[i], dim[i])
    return x


def sliding_window_2d(x: torch.Tensor, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], dim: Union[int, Tuple[int, int]] = (-2, -1)) -> torch.Tensor:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, dim)


def image_uv(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Get image space UV grid, ranging in [0, 1]. 

    >>> image_uv(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        torch.Tensor: shape (height, width, 2)
    """
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, device=device, dtype=dtype)
    v = torch.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def image_pixel_center(
    height: int,
    width: int,
    left: int = None,
    top: int = None,
    right: int = None,
    bottom: int = None,
    dtype: torch.dtype = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    Get image pixel center coordinates, ranging in [0, width] and [0, height].
    `image[i, j]` has pixel center coordinates `(j + 0.5, i + 0.5)`.

    >>> image_pixel_center(10, 10):
    [[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
     [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
      ...             ...                  ...
    [[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        torch.Tensor: shape (height, width, 2)
    """
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace(left + 0.5, right - 0.5, right - left, dtype=dtype, device=device)
    v = torch.linspace(top + 0.5, bottom - 0.5, bottom - top, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=2)


def image_mesh(height: int, width: int, mask: torch.Tensor = None, device: torch.device = None, dtype: torch.dtype = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a quad mesh regarding image pixel uv coordinates as vertices and image grid as faces.

    Args:
        width (int): image width
        height (int): image height
        mask (torch.Tensor, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

    Returns:
        uv (torch.Tensor): uv corresponding to pixels as described in image_uv()
        faces (torch.Tensor): quad faces connecting neighboring pixels
        indices (torch.Tensor, optional): indices of vertices in the original mesh
    """
    if device is None and mask is not None:
        device = mask.device
    if mask is not None:
        assert mask.shape[0] == height and mask.shape[1] == width
        assert mask.dtype == torch.bool
    uv = image_uv(height, width, device=device, dtype=dtype).reshape((-1, 2))
    row_faces = torch.stack([
        torch.arange(0, width - 1, dtype=torch.int32, device=device), 
        torch.arange(width, 2 * width - 1, dtype=torch.int32, device=device), 
        torch.arange(1 + width, 2 * width, dtype=torch.int32, device=device), 
        torch.arange(1, width, dtype=torch.int32, device=device)
    ], dim=1)
    faces = (torch.arange(0, (height - 1) * width, width, device=device, dtype=torch.int32)[:, None, None] + row_faces[None, :, :]).reshape((-1, 4))
    if mask is not None:
        quad_mask = (mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]).ravel()
        faces = faces[quad_mask]
        faces, uv, indices = mesh.remove_unreferenced_vertices(faces, uv, return_indices=True)
        return uv, faces, indices
    return uv, faces


def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
    
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge


def depth_aliasing(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
    """
    Compute the map that indicates the aliasing of a depth map. The aliasing is defined as the pixels which neither close to the maximum nor the minimum of its neighbors.
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff_max = F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) - depth
        diff_min = F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2) + depth
    else:
        diff_max = F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) - depth
        diff_min = F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + depth
    diff = torch.minimum(diff_max, diff_min)

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge


def image_mesh_from_depth(
    depth: torch.Tensor,
    extrinsics: torch.Tensor = None,
    intrinsics: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = depth.shape
    uv, faces = image_mesh(height, width)
    faces = faces.reshape(-1, 4)
    depth = depth.reshape(-1)
    pts = transforms.unproject_cv(image_uv, depth, extrinsics, intrinsics)
    faces = mesh.triangulate(faces, vertices=pts)
    return pts, faces


@batched(3, 2, 2)
def points_to_normals(point: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate normal map from point map. Value range is [-1, 1]. Normal direction in OpenGL identity camera's coordinate system.

    Args:
        point (torch.Tensor): shape (..., height, width, 3), point map
    Returns:
        normal (torch.Tensor): shape (..., height, width, 3), normal map. 
    """
    has_mask = mask is not None

    if mask is None:
        mask = torch.ones_like(point[..., 0], dtype=torch.bool)
    mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)

    pts = F.pad(point.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='constant', value=1).permute(0, 2, 3, 1)
    up = pts[:, :-2, 1:-1, :] - pts[:, 1:-1, 1:-1, :]
    left = pts[:, 1:-1, :-2, :] - pts[:, 1:-1, 1:-1, :]
    down = pts[:, 2:, 1:-1, :] - pts[:, 1:-1, 1:-1, :]
    right = pts[:, 1:-1, 2:, :] - pts[:, 1:-1, 1:-1, :]
    normal = torch.stack([
        torch.cross(up, left, dim=-1),
        torch.cross(left, down, dim=-1),
        torch.cross(down, right, dim=-1),
        torch.cross(right, up, dim=-1),
    ])
    normal = F.normalize(normal, dim=-1)
    valid = torch.stack([
        mask[:, :-2, 1:-1] & mask[:, 1:-1, :-2],
        mask[:, 1:-1, :-2] & mask[:, 2:, 1:-1],
        mask[:, 2:, 1:-1] & mask[:, 1:-1, 2:],
        mask[:, 1:-1, 2:] & mask[:, :-2, 1:-1],
    ]) & mask[None, :, 1:-1, 1:-1]
    normal = (normal * valid[..., None]).sum(dim=0)
    normal = F.normalize(normal, dim=-1)
    
    if has_mask:
        return normal, valid.any(dim=0)
    else:
        return normal


def depth_to_normals(depth: torch.Tensor, intrinsics: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenGL identity camera's coordinate system.

    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        intrinsics (torch.Tensor): shape (..., 3, 3), intrinsics matrix
    Returns:
        normal (torch.Tensor): shape (..., 3, height, width), normal map. 
    """
    pts = depth_to_points(depth, intrinsics)
    return points_to_normals(pts, mask)


def depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor = None, use_ray: bool = False):
    height, width = depth.shape[-2:]
    uv = image_uv(width=width, height=height, dtype=depth.dtype, device=depth.device)
    pts = transforms.unproject_cv(uv, depth, intrinsics=intrinsics[..., None, :, :], extrinsics=extrinsics[..., None, :, :] if extrinsics is not None else None, use_ray=use_ray)
    return pts


def masked_min(input: torch.Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Similar to torch.min, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min()
    else:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min(dim=dim, keepdim=keepdim)


def masked_max(input: torch.Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Similar to torch.max, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max()
    else:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max(dim=dim, keepdim=keepdim)
    

def bounding_rect(mask: torch.BoolTensor):
    """get bounding rectangle of a mask

    Args:
        mask (torch.Tensor): shape (..., height, width), mask

    Returns:
        rect (torch.Tensor): shape (..., 4), bounding rectangle (left, top, right, bottom)
    """
    height, width = mask.shape[-2:]
    mask = mask.flatten(-2).unsqueeze(-1)
    uv = image_uv(height, width).to(mask.device).reshape(-1, 2)
    left_top = masked_min(uv, mask, dim=-2)[0]
    right_bottom = masked_max(uv, mask, dim=-2)[0]
    return torch.cat([left_top, right_bottom], dim=-1)


def chessboard(width: int, height: int, grid_size: int, color_a: torch.Tensor, color_b: torch.Tensor) -> torch.Tensor:
    """get a chessboard image

    Args:
        width (int): image width
        height (int): image height
        grid_size (int): size of chessboard grid
        color_a (torch.Tensor): shape (chanenls,), color of the grid at the top-left corner
        color_b (torch.Tensor): shape (chanenls,), color in complementary grids

    Returns:
        image (torch.Tensor): shape (height, width, channels), chessboard image
    """
    x = torch.div(torch.arange(width), grid_size, rounding_mode='floor')
    y = torch.div(torch.arange(height), grid_size, rounding_mode='floor')
    mask = ((x[None, :] + y[:, None]) % 2).to(color_a)
    image = (1 - mask[..., None]) * color_a + mask[..., None] * color_b
    return image