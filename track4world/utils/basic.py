import os
import torch
import numpy as np
from typing import Optional, Tuple, Union

# Small constant to avoid numerical instability (e.g., division by zero).
# Used in normalization and statistical reduction.
EPS = 1e-6

def sub2ind(height: int, width: int, y: int, x: int) -> int:
    """
    Converts 2D subscripts (y, x) to linear indices (row-major order).
    
    Args:
        height: Image height (not used in calculation but kept for API consistency).
        width: Image width (stride).
        y: Row index.
        x: Column index.
    """
    return y * width + x

def ind2sub(height: int, width: int, ind: int) -> Tuple[int, int]:
    """
    Converts linear indices to 2D subscripts (y, x).
    
    Args:
        height: Image height.
        width: Image width.
        ind: Linear index.
        
    Returns:
        Tuple (y, x).
    """
    y = ind // width
    x = ind % width
    return y, x
    
def get_lr_str(lr: float) -> str:
    """
    Formats learning rate to a compact string (e.g., 0.0005 -> '5e-4').
    Removes trailing '.0' for cleaner logging.
    """
    return f"{lr:.1e}".replace('.0', '')
    
def strnum(x: float) -> str:
    """
    Formats a number to a short string representation for filenames or logs.
    
    Logic:
    1. Uses general format 'g'.
    2. Removes leading zero for decimals (e.g., 0.5 -> .5).
    3. Truncates string to max 4 characters to keep names short.
    """
    s = f'{x:g}'
    if '.' in s:
        if x < 1.0:
            # Remove leading zero for decimals, e.g., 0.5 -> .5
            s = s[s.index('.'):]
        # Limit precision/length to 4 characters
        s = s[:min(len(s), 4)]
    return s

def assert_same_shape(t1: torch.Tensor, t2: torch.Tensor):
    """
    Asserts that two tensors have the same shape.
    Useful for debugging element-wise operations.
    """
    assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} vs {t2.shape}"

def mkdir(path: str):
    """Creates a directory if it does not exist (equivalent to `mkdir -p`)."""
    os.makedirs(path, exist_ok=True)
        
def print_stats(name: str, tensor: torch.Tensor):
    """
    Prints basic statistics (min/mean/max/shape) of a tensor.
    Helpful for debugging gradient explosions or empty tensors.
    """
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
    """
    Normalizes a single tensor to the range [0, 1].
    Formula: (x - min) / (max - min)
    """
    dmin = torch.min(d)
    dmax = torch.max(d)
    return (d - dmin) / (dmax - dmin + EPS)

def normalize(d: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a batch of tensors to [0, 1] independently per batch element.
    
    This function handles arbitrary dimensions (B, C, H, W, etc.) by flattening
    everything after the batch dimension to compute min/max.

    Args:
        d: Tensor of shape (B, ...).

    Returns:
        Normalized tensor with the same shape as d.
    """
    # Vectorized implementation (much faster than looping over batch)
    B = d.shape[0]
    
    # Flatten all dimensions except batch to find min/max per sample
    # Shape becomes (B, N) where N is the product of remaining dims
    d_flat = d.view(B, -1)
    
    # Calculate min/max per batch item
    # keepdim=True ensures shape is (B, 1)
    dmin = d_flat.min(dim=1, keepdim=True)[0]
    dmax = d_flat.max(dim=1, keepdim=True)[0]
    
    # Reshape min/max to match d's dimensions for broadcasting
    # e.g., if d is (B, C, H, W), view_shape becomes (B, 1, 1, 1)
    view_shape = [B] + [1] * (d.ndim - 1)
    dmin = dmin.view(*view_shape)
    dmax = dmax.view(*view_shape)
    
    # Apply normalization with EPS to prevent division by zero
    return (d - dmin) / (dmax - dmin + EPS)

def normalize_grid2d(
    grid_y: torch.Tensor, 
    grid_x: torch.Tensor, 
    Y: int, 
    X: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes 2D grid coordinates to the range [-1, 1].
    
    This is specifically required by `torch.nn.functional.grid_sample`,
    where (-1, -1) represents the top-left pixel and (1, 1) represents 
    the bottom-right pixel.
    """
    # Maps [0, Y-1] -> [0, 1] -> [-1, 1]
    grid_y = 2.0 * (grid_y / (Y - 1 + EPS)) - 1.0
    grid_x = 2.0 * (grid_x / (X - 1 + EPS)) - 1.0
    return grid_y, grid_x

def meshgrid2d(
    B: int, 
    Y: int, 
    X: int, 
    stack: bool = False, 
    norm: bool = False, 
    device: Union[str, torch.device] = 'cuda', 
    on_chans: bool = False
):
    """
    Creates a 2D meshgrid and optionally expands it to a batch.

    Args:
        B: Batch size.
        Y: Height.
        X: Width.
        stack: If True, stacks coordinates into a single tensor.
        norm: If True, normalizes coordinates to [-1, 1].
        device: Device to create tensors on.
        on_chans: If True and stack=True, output shape is (B, 2, Y, X).
                  If False and stack=True, output shape is (B, Y, X, 2).

    Returns:
        grid_y, grid_x with shape (B, Y, X), or
        a stacked grid tensor.
    """
    # Create linear ranges for Y and X axes
    y_range = torch.linspace(0.0, Y - 1, Y, device=device)
    x_range = torch.linspace(0.0, X - 1, X, device=device)
    
    # Use torch.meshgrid with indexing='ij' 
    # 'ij' means the first dimension corresponds to y_range (rows),
    # and the second dimension corresponds to x_range (columns).
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

    # Expand to batch size: (B, Y, X)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)

    # Normalize to [-1, 1] if requested (for grid_sample)
    if norm:
        grid_y, grid_x = normalize_grid2d(grid_y, grid_x, Y, X)

    if stack:
        # Note: grid_sample expects (x, y) order in the last dimension
        if on_chans:
            # Shape: (B, 2, Y, X) 
            # Useful for predicting flow/displacement fields directly
            grid = torch.stack([grid_x, grid_y], dim=1)
        else:
            # Shape: (B, Y, X, 2) 
            # Standard format for F.grid_sample(input, grid)
            grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        # Return separate components
        return grid_y, grid_x

def gridcloud2d(
    B: int, 
    Y: int, 
    X: int, 
    norm: bool = False, 
    device: Union[str, torch.device] = 'cuda'
) -> torch.Tensor:
    """
    Converts a 2D grid into a point cloud representation (flattened spatial dims).

    Args:
        B, Y, X: Dimensions.
        norm: Whether to normalize to [-1, 1].

    Returns:
        Tensor of shape (B, Y*X, 2), storing (x, y) coordinates for every pixel.
    """
    # Get grid components (B, Y, X)
    grid_y, grid_x = meshgrid2d(B, Y, X, norm=norm, device=device)
    
    # Flatten spatial dimensions: (B, Y, X) -> (B, Y*X)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    
    # Stack as (x, y) points: (B, Y*X, 2)
    xy = torch.stack([x, y], dim=2)
    return xy

def reduce_masked_mean(
    x: torch.Tensor, 
    mask: torch.Tensor, 
    dim: Optional[int] = None, 
    keepdim: bool = False, 
    broadcast: bool = False
) -> torch.Tensor:
    """
    Computes the mean of tensor `x` only considering elements where `mask` is True.
    
    Formula: sum(x * mask) / sum(mask)
    """
    if not broadcast:
        assert_same_shape(x, mask)
        
    # Zero out masked-out values
    prod = x * mask
    
    if dim is None:
        # Global mean
        numer = torch.sum(prod)
        denom = torch.sum(mask) + EPS
    else:
        # Mean along specific dimension
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim) + EPS
        
    mean = numer / denom
    return mean
        
def reduce_masked_median(
    x: torch.Tensor, 
    mask: torch.Tensor, 
    keep_batch: bool = False
) -> torch.Tensor:
    """
    Computes the median of `x` over masked elements.

    Note:
        The computation is done on CPU using NumPy because PyTorch does not 
        natively support a masked median operation efficiently.
        
    Args:
        x: Data tensor.
        mask: Binary mask tensor (same shape as x).
        keep_batch: If True, returns a median per batch item (B,).
                    If False, returns a single global median.
    """
    assert_same_shape(x, mask)
    device = x.device
    B = x.shape[0]

    # Detach gradients and move to CPU for NumPy processing
    x_np = x.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    if keep_batch:
        # Flatten spatial dims but keep batch dim: (B, -1)
        x_np = x_np.reshape(B, -1)
        mask_np = mask_np.reshape(B, -1)
        meds = np.zeros(B, dtype=np.float32)
        
        # Iterate over batch to calculate median for each sample
        for b in range(B):
            xb = x_np[b]
            mb = mask_np[b]
            # Select only valid elements
            valid_data = xb[mb > 0]
            
            if valid_data.size > 0:
                meds[b] = np.median(valid_data)
            else:
                meds[b] = np.nan
        return torch.from_numpy(meds).float().to(device)
    else:
        # Global median: flatten everything
        x_flat = x_np.reshape(-1)
        mask_flat = mask_np.reshape(-1)
        valid_data = x_flat[mask_flat > 0]
        
        if valid_data.size > 0:
            med = np.median(valid_data)
        else:
            med = np.nan
            
        return torch.tensor([med], dtype=torch.float32, device=device)