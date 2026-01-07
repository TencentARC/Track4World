from typing import *
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types


def scatter_min(
    size: int, dim: int, index: torch.LongTensor, src: torch.Tensor
) -> torch.return_types.min:
    """
    Scatter the minimum value along the given dimension of `src` into a new tensor.
    Unlike standard scatter, this also recovers the original indices of the minima.
    """
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1:]
    
    # Initialize with infinity to ensure any value in src is smaller
    minimum = torch.full(shape, float('inf'), dtype=src.dtype, device=src.device)
    minimum = minimum.scatter_reduce(
        dim=dim, index=index, src=src, reduce='amin', include_self=False
    )
    
    # Find where the original src matches the computed minimum to find indices
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    
    # Map the found minima back to their source index
    idx_tuple = (
        *minimum_where[:dim], 
        index[minimum_where], 
        *minimum_where[dim + 1:]
    )
    indices[idx_tuple] = minimum_where[dim]
    
    return torch.return_types.min((minimum, indices))
    

def split_batch_fwd(fn: Callable, chunk_size: int, *args, **kwargs):
    """
    Splits a batch-processing function into smaller chunks to prevent OOM (Out of Memory).
    """
    # Infer batch size from the first tensor found in arguments
    batch_size = next(
        x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)
    ).shape[0]
    
    n_chunks = batch_size // chunk_size + (batch_size % chunk_size > 0)
    
    # Split tensors into chunks; keep non-tensors as-is (duplicated for each chunk)
    splited_args = tuple(
        arg.split(chunk_size, dim=0) if isinstance(arg, torch.Tensor) 
        else [arg] * n_chunks for arg in args
    )
    splited_kwargs = {
        k: [v.split(chunk_size, dim=0) if isinstance(v, torch.Tensor) 
        else [v] * n_chunks] for k, v in kwargs.items()
    }
    
    results = []
    for i in range(n_chunks):
        chunk_args = tuple(arg[i] for arg in splited_args)
        chunk_kwargs = {k: v[i] for k, v in splited_kwargs.items()}
        results.append(fn(*chunk_args, **chunk_kwargs))

    # Concatenate results back into a single tensor or tuple of tensors
    if isinstance(results[0], tuple):
        return tuple(torch.cat(r, dim=0) for r in zip(*results))
    else:
        return torch.cat(results, dim=0)

def _pad_inf(x_: torch.Tensor):
    """Pads a tensor with -inf at the start and +inf at the end."""
    return torch.cat([
        torch.full_like(x_[..., :1], -torch.inf), 
        x_, 
        torch.full_like(x_[..., :1], torch.inf)
    ], dim=-1)


def _pad_cumsum(cumsum: torch.Tensor):
    """Pads a cumsum tensor with a 0 at the start and repeats the last value."""
    return torch.cat([
        torch.zeros_like(cumsum[..., :1]), 
        cumsum, 
        cumsum[..., -1:]
    ], dim=-1)


def _compute_residual(a: torch.Tensor, xyw: torch.Tensor, trunc: float):
    """Computes the weighted L1 loss with truncation: sum(min(trunc, w * |a*x - y|))."""
    return a.mul(xyw[..., 0]).sub_(xyw[..., 1]).abs_().mul_(xyw[..., 2]) \
            .clamp_max_(trunc).sum(dim=-1)


def align(
    x: torch.Tensor, 
    y: torch.Tensor, 
    w: torch.Tensor, 
    trunc: Optional[Union[float, torch.Tensor]] = None, 
    eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """
    Solves the alignment problem: find 'a' that minimizes weighted L1 distance.
    If trunc is provided, it solves a Robust L1 problem using an exhaustive search over 
    critical points (extrema) in the derivative of the piecewise linear loss.
    """
    if trunc is None:
        # Standard Weighted Median approach for min sum(w * |a*x - y|)
        x, y, w = torch.broadcast_tensors(x, y, w)
        sign = torch.sign(x)
        x, y = x * sign, y * sign
        y_div_x = y / x.clamp_min(eps)
        y_div_x, argsort = y_div_x.sort(dim=-1)

        # The optimal 'a' is the weighted median of y/x
        wx = torch.gather(x * w, dim=-1, index=argsort)
        derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
        search = torch.searchsorted(
            derivatives, 
            torch.zeros_like(derivatives[..., :1]), 
            side='left'
        ).clamp_max(derivatives.shape[-1] - 1)

        a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
        index = argsort.gather(dim=-1, index=search).squeeze(-1)
        loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)
        
    else:
        # Robust alignment with Truncated L1 loss
        x, y, w = torch.broadcast_tensors(x, y, w)
        batch_shape = x.shape[:-1]
        batch_size = math.prod(batch_shape)
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        w = w.reshape(-1, w.shape[-1])

        sign = torch.sign(x)
        x, y = x * sign, y * sign
        wx, wy = w * x, w * y
        xyw = torch.stack([x, y, w], dim=-1)

        # Critical points are where residuals switch signs or hit the truncation threshold
        y_div_x = A = y / x.clamp_min(eps)
        B = (wy - trunc) / wx.clamp_min(eps)
        C = (wy + trunc) / wx.clamp_min(eps)
        
        with torch.no_grad():
            # Sort critical points to calculate cumulative sums for piecewise derivatives
            A, A_argsort = A.sort(dim=-1)
            Q_A = torch.cumsum(torch.gather(wx, dim=-1, index=A_argsort), dim=-1)
            A, Q_A = _pad_inf(A), _pad_cumsum(Q_A)

            B, B_argsort = B.sort(dim=-1)
            Q_B = torch.cumsum(torch.gather(wx, dim=-1, index=B_argsort), dim=-1)
            B, Q_B = _pad_inf(B), _pad_cumsum(Q_B)

            C, C_argsort = C.sort(dim=-1)
            Q_C = torch.cumsum(torch.gather(wx, dim=-1, index=C_argsort), dim=-1)
            C, Q_C = _pad_inf(C), _pad_cumsum(Q_C)
            
            # Use searchsorted to find positions of y/x candidates in sorted critical points
            j_A = torch.searchsorted(A, y_div_x, side='left').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='left').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='left').sub_(1)
            left_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) \
                              - torch.gather(Q_B, dim=-1, index=j_B) \
                              - torch.gather(Q_C, dim=-1, index=j_C)
            
            j_A = torch.searchsorted(A, y_div_x, side='right').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='right').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='right').sub_(1)
            right_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) \
                               - torch.gather(Q_B, dim=-1, index=j_B) \
                               - torch.gather(Q_C, dim=-1, index=j_C)

            # Local minima occur where derivative transitions from negative to positive
            is_extrema = (left_derivative < 0) & (right_derivative >= 0)
            is_extrema[..., 0] |= ~is_extrema.any(dim=-1)
            where_extrema_batch, where_extrema_index = torch.where(is_extrema)          

            extrema_a = y_div_x[where_extrema_batch, where_extrema_index]
            
            # Efficiently compute objective value for all candidate extrema
            MAX_ELEMENTS = 4096 ** 2
            SPLIT_SIZE = MAX_ELEMENTS // x.shape[-1]
            extrema_value = torch.cat([
                _compute_residual(
                    extrema_a_split[:, None], 
                    xyw[extrema_i_split, :, :], 
                    trunc
                )
                for extrema_a_split, extrema_i_split in zip(
                    extrema_a.split(SPLIT_SIZE), 
                    where_extrema_batch.split(SPLIT_SIZE)
                )
            ])
            
            # Aggregate the global minimum from the candidates
            minima, indices = scatter_min(
                size=batch_size, dim=0, index=where_extrema_batch, src=extrema_value
            )
            index = where_extrema_index[indices]

        # Gather final differentiable solution
        gather_idx = index[..., None]
        a = torch.gather(y, dim=-1, index=gather_idx) / \
            torch.gather(x, dim=-1, index=gather_idx).clamp_min(eps)
        
        a, loss, index = a.reshape(batch_shape), minima.reshape(batch_shape), index.reshape(batch_shape)

    return a, loss, index

def align_depth_scale(
    depth_src: torch.Tensor, 
    depth_tgt: torch.Tensor, 
    weight: Optional[torch.Tensor], 
    trunc: Optional[Union[float, torch.Tensor]] = None
):
    """
    Align `depth_src` to `depth_tgt` with given constant weights. 

    ### Parameters:
    - `depth_src: torch.Tensor` of shape (..., N)
    - `depth_tgt: torch.Tensor` of shape (..., N)

    """
    scale, _, _ = align(depth_src, depth_tgt, weight, trunc)

    return scale


def align_depth_affine(
    depth_src: torch.Tensor, 
    depth_tgt: torch.Tensor, 
    weight: torch.Tensor, 
    trunc: Optional[Union[float, torch.Tensor]] = None
):
    """
    Solves for scale (s) and shift (t) in: depth_tgt = s * depth_src + t.
    Uses an anchor-point strategy to reduce the 2D search space into 1D alignment problems.
    """
    batch_shape, n = depth_src.shape[:-1], depth_src.shape[-1]
    batch_size = math.prod(batch_shape)
    
    depth_src = depth_src.reshape(batch_size, n)
    depth_tgt = depth_tgt.reshape(batch_size, n)
    weight = weight.reshape(batch_size, n)

    # Use points with positive weights as potential anchors
    anchors_where_batch, anchors_where_n = torch.where(weight > 0)

    with torch.no_grad():
        src_anchor = depth_src[anchors_where_batch, anchors_where_n]
        tgt_anchor = depth_tgt[anchors_where_batch, anchors_where_n]

        # Shift the system so the anchor is at the origin (removing the 't' parameter)
        src_anchored = depth_src[anchors_where_batch, :] - src_anchor[..., None]
        tgt_anchored = depth_tgt[anchors_where_batch, :] - tgt_anchor[..., None]
        weight_anchored = weight[anchors_where_batch, :]

        # Solve for scale 's' for every possible anchor point
        MAX_ELEMENTS = 2 ** 20
        scale, loss, index = split_batch_fwd(
            align, MAX_ELEMENTS // n, src_anchored, tgt_anchored, weight_anchored, trunc
        )
        # Choose the anchor point that yields the overall minimum loss
        loss, index_anchor = scatter_min(
            size=batch_size, dim=0, index=anchors_where_batch, src=loss
        )

    # Recover the indices to calculate differentiable scale and shift
    idx1 = anchors_where_n[index_anchor]
    idx2 = index[index_anchor]

    tgt1 = torch.gather(depth_tgt, dim=1, index=idx1[..., None]).squeeze(-1)
    src1 = torch.gather(depth_src, dim=1, index=idx1[..., None]).squeeze(-1)
    tgt2 = torch.gather(depth_tgt, dim=1, index=idx2[..., None]).squeeze(-1)
    src2 = torch.gather(depth_src, dim=1, index=idx2[..., None]).squeeze(-1)

    # Final scale and shift calculation
    scale = (tgt2 - tgt1) / torch.where(src2 != src1, src2 - src1, 1e-7)
    shift = tgt1 - scale * src1

    return scale.reshape(batch_shape), shift.reshape(batch_shape)

def align_depth_affine_irls(
    depth_src: torch.Tensor, 
    depth_tgt: torch.Tensor, 
    weight: torch.Tensor, 
    max_iter: int = 100, 
    eps: float = 1e-12
):
    """
    Standard Iteratively Reweighted Least Squares (IRLS) for robust affine alignment.
    Slower but often more stable for non-truncated L1.
    """
    w = weight
    x = torch.stack([depth_src, torch.ones_like(depth_src)], dim=-1)
    y = depth_tgt

    for _ in range(max_iter):
        # Solve weighted least squares: (X^T W X)^-1 X^T W y
        xtw = x.transpose(-1, -2) * w[..., None, :]
        beta = torch.linalg.solve(xtw @ x, xtw @ y[..., None]).squeeze(-1)
        # Update weights: w = 1 / |residual|
        residual = (y - (x @ beta[..., None])[..., 0]).abs().clamp_min(eps)
        w = 1.0 / residual

    return beta[..., 0], beta[..., 1]


def align_points_scale(
    points_src: torch.Tensor, 
    points_tgt: torch.Tensor, 
    weight: Optional[torch.Tensor], 
    trunc: Optional[Union[float, torch.Tensor]] = None
):
    """
    Aligns points with a pure isotropic scale factor 's', assuming no shift (or pre-centered).
    """
    # Flatten all spatial and coordinate dimensions to solve as a 1D median problem
    src_flat = points_src.flatten(-2)
    tgt_flat = points_tgt.flatten(-2)
    weight_flat = weight[..., None].expand_as(points_src).flatten(-2)
    
    scale, _, _ = align(src_flat, tgt_flat, weight_flat, trunc)
    return scale


def align_points_scale_z_shift(
    points_src: torch.Tensor, 
    points_tgt: torch.Tensor, 
    weight: torch.Tensor, 
    trunc: Optional[Union[float, torch.Tensor]] = None
):
    """
    Aligns 3D point clouds with a global scale and a shift ONLY on the Z-axis.
    Common in depth estimation from monocular video with unknown focal/baseline.
    """
    batch_shape, n = points_src.shape[:-2], points_src.shape[-2]
    batch_size = math.prod(batch_shape)
    
    ps = points_src.reshape(batch_size, n, 3)
    pt = points_tgt.reshape(batch_size, n, 3)
    w = weight.reshape(batch_size, n)

    anchor_where_batch, anchor_where_n = torch.where(w > 0)
    with torch.no_grad():
        zeros = torch.zeros(anchor_where_batch.shape[0], device=ps.device, dtype=ps.dtype)
        # Create anchor points focusing only on Z-shift
        src_anchor = torch.stack([zeros, zeros, ps[anchor_where_batch, anchor_where_n, 2]], dim=-1)
        tgt_anchor = torch.stack([zeros, zeros, pt[anchor_where_batch, anchor_where_n, 2]], dim=-1)

        ps_anchored = ps[anchor_where_batch, :, :] - src_anchor[..., None, :]
        pt_anchored = pt[anchor_where_batch, :, :] - tgt_anchor[..., None, :]
        w_anchored = w[anchor_where_batch, :, None].expand(-1, -1, 3)

        MAX_ELEMENTS = 2 ** 15
        scale, loss, index = split_batch_fwd(
            align, MAX_ELEMENTS // n, 
            ps_anchored.flatten(-2), 
            pt_anchored.flatten(-2), 
            w_anchored.flatten(-2), 
            trunc
        )
        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchor_where_batch, src=loss)

    # Recover differentiable scale and Z-shift
    idx2 = index[index_anchor]
    idx1 = anchor_where_n[index_anchor] * 3 + idx2 % 3

    zeros_b = torch.zeros((batch_size, n), device=ps.device, dtype=ps.dtype)
    pt_z = torch.stack([zeros_b, zeros_b, pt[..., 2]], dim=-1)
    ps_z = torch.stack([zeros_b, zeros_b, ps[..., 2]], dim=-1)
    
    t1 = torch.gather(pt_z.flatten(-2), dim=1, index=idx1[..., None]).squeeze(-1)
    s1 = torch.gather(ps_z.flatten(-2), dim=1, index=idx1[..., None]).squeeze(-1)
    t2 = torch.gather(pt.flatten(-2), dim=1, index=idx2[..., None]).squeeze(-1)
    s2 = torch.gather(ps.flatten(-2), dim=1, index=idx2[..., None]).squeeze(-1)

    scale = (t2 - t1) / torch.where(s2 != s1, s2 - s1, 1.0)
    
    # Calculate shift for each of the 3 coordinates (X, Y will be 0)
    gather_idx = (idx1 // 3)[..., None, None].expand(-1, -1, 3)
    p_tgt_anc = torch.gather(pt_z, dim=1, index=gather_idx).squeeze(-2)
    p_src_anc = torch.gather(ps_z, dim=1, index=gather_idx).squeeze(-2)
    shift = p_tgt_anc - scale[..., None] * p_src_anc

    return scale.reshape(batch_shape), shift.reshape(*batch_shape, 3)


def align_points_scale_xyz_shift(
    points_src: torch.Tensor, 
    points_tgt: torch.Tensor, 
    weight: Optional[torch.Tensor], 
    trunc: Optional[Union[float, torch.Tensor]] = None, 
    max_iters: int = 30, 
    eps: float = 1e-6, 
    exp: int = 20
):
    """
    Align `points_src` to `points_tgt` with an isotropic scale and a 3D shift.
    
    The algorithm uses an 'anchor' strategy:
    1. Centering the clouds at every possible point pair (anchors).
    2. Solving for the optimal scale in that centered space.
    3. Selecting the anchor that yields the minimum global loss.
    """
    dtype, device = points_src.dtype, points_src.device

    # Flatten batch dimensions: (..., N, 3) -> (Batch, N, 3)
    batch_shape, n = points_src.shape[:-2], points_src.shape[-2]
    batch_size = math.prod(batch_shape)
    points_src = points_src.reshape(batch_size, n, 3)
    points_tgt = points_tgt.reshape(batch_size, n, 3)
    weight = weight.reshape(batch_size, n)

    # Find points with non-zero weights to serve as potential anchors
    anchor_where_batch, anchor_where_n = torch.where(weight > 0)

    with torch.no_grad():
        # Get 3D coordinates of anchors
        points_src_anchor = points_src[anchor_where_batch, anchor_where_n]
        points_tgt_anchor = points_tgt[anchor_where_batch, anchor_where_n]

        # Subtract anchor to remove translation: P' = P - P_anchor
        points_src_anchored = points_src[anchor_where_batch, :, :] - \
                              points_src_anchor[..., None, :]
        points_tgt_anchored = points_tgt[anchor_where_batch, :, :] - \
                              points_tgt_anchor[..., None, :]
        
        # Expand weights to match (Anchors, N, 3)
        weight_anchored = weight[anchor_where_batch, :, None].expand(-1, -1, 3)

        # Exhaustive search for optimal scale 's' across all anchors
        MAX_ELEMENTS = 2 ** exp
        scale, loss, index = split_batch_fwd(
            align, 
            MAX_ELEMENTS // 2, 
            points_src_anchored.flatten(-2), 
            points_tgt_anchored.flatten(-2), 
            weight_anchored.flatten(-2), 
            trunc
        )

        # Find the specific anchor that minimizes the objective function for each batch
        loss, index_anchor = scatter_min(
            size=batch_size, dim=0, index=anchor_where_batch, src=loss
        )

    # Reconstruction of differentiable results
    # index_2 is the index within the point cloud, index_1 is the anchor index
    index_2 = index[index_anchor]
    index_1 = anchor_where_n[index_anchor] * 3 + index_2 % 3

    # Gather the specific point pairs that defined the optimal scale
    src_1 = torch.gather(points_src.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1)
    tgt_1 = torch.gather(points_tgt.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1)
    src_2 = torch.gather(points_src.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1)
    tgt_2 = torch.gather(points_tgt.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1)

    # Compute scale: s = (y2 - y1) / (x2 - x1)
    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1.0)
    
    # Compute shift: t = y1 - s * x1
    anchor_idx = (index_1 // 3)[..., None, None].expand(-1, -1, 3)
    shift = torch.gather(points_tgt, dim=1, index=anchor_idx).squeeze(-2) - \
            scale[..., None] * torch.gather(points_src, dim=1, index=anchor_idx).squeeze(-2)

    return scale.reshape(batch_shape), shift.reshape(*batch_shape, 3)


def align_points_z_shift(
    points_src: torch.Tensor, 
    points_tgt: torch.Tensor, 
    weight: Optional[torch.Tensor], 
    trunc: Optional[Union[float, torch.Tensor]] = None, 
    **kwargs
):
    """
    Aligns points strictly along the Z-axis (translation only).
    Often used in depth alignment where XY are fixed image coordinates.
    """
    # Solves min sum |(P_src_z + shift) - P_tgt_z|
    shift, _, _ = align(
        torch.ones_like(points_src[..., 2]), 
        points_tgt[..., 2] - points_src[..., 2], 
        weight, 
        trunc
    )
    # Returns 3D vector with X and Y shifts as zero
    return torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)


def align_points_xyz_shift(
    points_src: torch.Tensor, 
    points_tgt: torch.Tensor, 
    weight: Optional[torch.Tensor], 
    trunc: Optional[Union[float, torch.Tensor]] = None, 
    **kwargs
):
    """
    Aligns points with a full 3D translation vector (X, Y, Z shift), but no scale.
    """
    # Swap axes to treat coordinates as a batch for the align function
    src_ones = torch.ones_like(points_src).swapaxes(-2, -1)
    diff = (points_tgt - points_src).swapaxes(-2, -1)
    
    shift, _, _ = align(src_ones, diff, weight[..., None, :], trunc)
    return shift


def align_affine_lstsq(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard Least Squares (L2) solver for y = ax + b.
    Fastest alignment but highly sensitive to outliers.
    """
    w_sqrt = torch.ones_like(x) if w is None else w.sqrt()
    # Construct design matrix A for [a, b]
    A = torch.stack([w_sqrt * x, torch.ones_like(x)], dim=-1)
    B = (w_sqrt * y)[..., None]
    # Solve (A^T A)x = A^T B
    sol = torch.linalg.lstsq(A, B)[0].squeeze(-1)
    a, b = sol.unbind(-1)
    return a, b