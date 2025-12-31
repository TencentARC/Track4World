import os
import torch
import numpy as np
from typing import Optional, Tuple, Union

EPS = 1e-6

def sub2ind(height: int, width: int, y: int, x: int) -> int:
    """Converts 2D subscripts to linear indices."""
    return y * width + x

def ind2sub(height: int, width: int, ind: int) -> Tuple[int, int]:
    """Converts linear indices to 2D subscripts."""
    y = ind // width
    x = ind % width
    return y, x
    
def get_lr_str(lr: float) -> str:
    """Formats learning rate to a compact string (e.g., 5e-4)."""
    return f"{lr:.1e}".replace('.0', '')
    
def strnum(x: float) -> str:
    """Formats a number to a short string representation."""
    s = f'{x:g}'
    if '.' in s:
        if x < 1.0:
            # Remove leading zero for decimals, e.g., 0.5 -> .5
            s = s[s.index('.'):]
        # Limit precision/length
        s = s[:min(len(s), 4)]
    return s

def assert_same_shape(t1: torch.Tensor, t2: torch.Tensor):
    """Asserts that two tensors have the same shape."""
    assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} vs {t2.shape}"

def mkdir(path: str):
    """Creates a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
        
def print_stats(name: str, tensor: torch.Tensor):
    """Prints basic statistics of a tensor."""
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:
        tensor_np = tensor
        
    print(f'{name} ({tensor_np.dtype}) '
          f'min = {np.min(tensor_np):.2f}, '
          f'mean = {np.mean(tensor_np):.2f}, '
          f'max = {np.max(tensor_np):.2f} '
          f'shape = {tensor_np.shape}')
    
def normalize_single(d: torch.Tensor) -> torch.Tensor:
    """Normalizes a single tensor to [0, 1]."""
    dmin = torch.min(d)
    dmax = torch.max(d)
    return (d - dmin) / (dmax - dmin + EPS)

def normalize(d: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a batch of tensors to [0, 1] independently per batch item.
    Input: (B, ...)
    """
    # Vectorized implementation (much faster than looping)
    B = d.shape[0]
    # Flatten all dimensions except batch
    d_flat = d.view(B, -1)
    
    # Calculate min/max per batch item
    dmin = d_flat.min(dim=1, keepdim=True)[0]
    dmax = d_flat.max(dim=1, keepdim=True)[0]
    
    # Reshape min/max to match d's dimensions for broadcasting
    view_shape = [B] + [1] * (d.ndim - 1)
    dmin = dmin.view(*view_shape)
    dmax = dmax.view(*view_shape)
    
    return (d - dmin) / (dmax - dmin + EPS)

def normalize_grid2d(grid_y: torch.Tensor, grid_x: torch.Tensor, Y: int, X: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalizes grid coordinates to [-1, 1] range."""
    # Maps [0, Y-1] -> [-1, 1]
    grid_y = 2.0 * (grid_y / (Y - 1 + EPS)) - 1.0
    grid_x = 2.0 * (grid_x / (X - 1 + EPS)) - 1.0
    return grid_y, grid_x

def meshgrid2d(B: int, Y: int, X: int, stack: bool = False, norm: bool = False, 
               device: Union[str, torch.device] = 'cuda', on_chans: bool = False):
    """
    Returns a meshgrid sized B x Y x X.
    """
    # Use torch.meshgrid with indexing='ij' to match original logic (Y varies along dim 0, X along dim 1)
    # grid_y: (Y, X), grid_x: (Y, X)
    y_range = torch.linspace(0.0, Y - 1, Y, device=device)
    x_range = torch.linspace(0.0, X - 1, X, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

    # Expand to batch size: (B, Y, X)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)

    if norm:
        grid_y, grid_x = normalize_grid2d(grid_y, grid_x, Y, X)

    if stack:
        # Note: grid_sample expects (x, y) in the last dimension, so we stack [grid_x, grid_y]
        if on_chans:
            # Shape: (B, 2, Y, X)
            grid = torch.stack([grid_x, grid_y], dim=1)
        else:
            # Shape: (B, Y, X, 2) -> Standard for grid_sample
            grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x

def gridcloud2d(B: int, Y: int, X: int, norm: bool = False, device: Union[str, torch.device] = 'cuda') -> torch.Tensor:
    """
    Returns a point cloud of grid coordinates.
    Output: (B, N, 2) where N = Y*X
    """
    grid_y, grid_x = meshgrid2d(B, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    # Stack as (x, y)
    xy = torch.stack([x, y], dim=2)
    return xy

def reduce_masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None, 
                       keepdim: bool = False, broadcast: bool = False) -> torch.Tensor:
    """Computes mean of x where mask is true."""
    if not broadcast:
        assert_same_shape(x, mask)
        
    prod = x * mask
    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask) + EPS
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim) + EPS
        
    mean = numer / denom
    return mean
        
def reduce_masked_median(x: torch.Tensor, mask: torch.Tensor, keep_batch: bool = False) -> torch.Tensor:
    """
    Computes median of x where mask is true.
    Note: Moves data to CPU to use numpy.median, as PyTorch masked median is non-trivial.
    """
    assert_same_shape(x, mask)
    device = x.device
    B = x.shape[0]

    # Detach and move to CPU
    x_np = x.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    if keep_batch:
        x_np = x_np.reshape(B, -1)
        mask_np = mask_np.reshape(B, -1)
        meds = np.zeros(B, dtype=np.float32)
        
        for b in range(B):
            xb = x_np[b]
            mb = mask_np[b]
            valid_data = xb[mb > 0]
            if valid_data.size > 0:
                meds[b] = np.median(valid_data)
            else:
                meds[b] = np.nan
        return torch.from_numpy(meds).float().to(device)
    else:
        x_flat = x_np.reshape(-1)
        mask_flat = mask_np.reshape(-1)
        valid_data = x_flat[mask_flat > 0]
        
        if valid_data.size > 0:
            med = np.median(valid_data)
        else:
            med = np.nan
            
        return torch.tensor([med], dtype=torch.float32, device=device)