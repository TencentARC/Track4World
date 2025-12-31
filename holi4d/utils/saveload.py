import pathlib
import os
import torch
import re
from typing import Optional, Union, Dict, Any

def save(ckpt_dir: str, 
         module: torch.nn.Module, 
         optimizer: torch.optim.Optimizer, 
         scheduler: Optional[object], 
         global_step: int, 
         keep_latest: int = 2, 
         model_name: str = 'model') -> bool:
    """
    Saves a checkpoint and maintains a limited number of recent checkpoints.
    """
    ckpt_path = pathlib.Path(ckpt_dir)
    ckpt_path.mkdir(exist_ok=True, parents=True)

    # Find existing checkpoints
    prev_ckpts = list(ckpt_path.glob(f'{model_name}-*pth'))
    # Sort by modification time (newest first)
    prev_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Remove older checkpoints if exceeding the limit
    if len(prev_ckpts) > keep_latest - 1:
        for f in prev_ckpts[keep_latest - 1:]:
            f.unlink()

    save_path = ckpt_path / f'{model_name}-{global_step:09d}.pth'
    
    save_dict = {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    print(f"saving {save_path}")
    torch.save(save_dict, save_path)
    return False

def _resolve_checkpoint_path(ckpt_path: str, model_name: str = 'model', verbose: bool = True) -> Optional[pathlib.Path]:
    """
    Helper function to resolve the actual file path from a directory or file path.
    """
    path_obj = pathlib.Path(ckpt_path)
    
    if not path_obj.exists():
        if verbose:
            print(f'...there is no checkpoint at {ckpt_path}')
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    if path_obj.is_file():
        if verbose:
            print(f'...found checkpoint {path_obj}')
        return path_obj
    else:
        # It's a directory, look for the latest file
        prev_ckpts = list(path_obj.glob(f'{model_name}-*pth'))
        prev_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        if prev_ckpts:
            latest_path = prev_ckpts[0]
            if verbose:
                # Try to parse step from filename (e.g., model-000050000.pth)
                try:
                    step = int(latest_path.stem.split('-')[-1])
                    print(f'...found checkpoint {latest_path}; (parsed step {step} from path)')
                except (ValueError, IndexError):
                    print(f'...found checkpoint {latest_path}')
            return latest_path
        else:
            if verbose:
                print('...there is no full checkpoint here!')
            return None

def _load_checkpoint_file(fabric, path: pathlib.Path, weights_only: bool = False) -> Dict[str, Any]:
    """Helper to load the dictionary from disk."""
    if fabric is not None:
        return fabric.load(path)
    else:
        return torch.load(path, weights_only=weights_only, map_location='cpu')

def load_unfreeze(fabric, 
                  ckpt_path: str, 
                  model: torch.nn.Module, 
                  optimizer: Optional[torch.optim.Optimizer] = None, 
                  scheduler: Optional[object] = None, 
                  model_ema=None, 
                  step: int = 0, 
                  model_name: str = 'model', 
                  ignore_load: Optional[list] = None, 
                  strict: bool = True, 
                  verbose: bool = True, 
                  weights_only: bool = False) -> int:
    """
    Loads a checkpoint but only updates parameters that are currently trainable (requires_grad=True).
    """
    if verbose:
        print(f'reading ckpt from {ckpt_path}')

    path = _resolve_checkpoint_path(ckpt_path, model_name, verbose)
    if path is None:
        return 0

    # Parse step from filename if possible
    try:
        step = int(path.stem.split('-')[-1])
    except (ValueError, IndexError):
        pass # Keep default step or passed step

    checkpoint = _load_checkpoint_file(fabric, path, weights_only)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    assert ignore_load is None, "ignore_load functionality is not implemented yet"

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Use the specialized loading function for unfrozen weights
    load_unfrozen_weights(model, state_dict)
    
    return step

def load(fabric, 
         ckpt_path: str, 
         model: torch.nn.Module, 
         optimizer: Optional[torch.optim.Optimizer] = None, 
         scheduler: Optional[object] = None, 
         model_ema=None, 
         step: int = 0, 
         model_name: str = 'model', 
         ignore_load: Optional[list] = None, 
         strict: bool = True, 
         verbose: bool = True, 
         weights_only: bool = False) -> int:
    """
    Standard checkpoint loading function.
    """
    if verbose:
        print(f'reading ckpt from {ckpt_path}')

    path = _resolve_checkpoint_path(ckpt_path, model_name, verbose)
    if path is None:
        return 0

    # Parse step from filename if possible
    try:
        step = int(path.stem.split('-')[-1])
    except (ValueError, IndexError):
        pass

    checkpoint = _load_checkpoint_file(fabric, path, weights_only)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    assert ignore_load is None, "ignore_load functionality is not implemented yet"

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    
    return step

def load_unfrozen_weights(model: torch.nn.Module, state_dict: Dict[str, Any]):
    """
    Loads weights from state_dict into the model, but only for parameters 
    that are marked as trainable (requires_grad=True) in the current model.
    """
    # Filter parameter names where requires_grad=True in the current model
    unfrozen_keys = [name for name, param in model.named_parameters() if param.requires_grad]

    # Create a new state_dict containing only the unfrozen parameters found in the loaded dict
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in unfrozen_keys}

    print(f"Loading {len(filtered_state_dict)} / {len(state_dict)} parameters (unfrozen only).")

    # Load weights with strict=False to allow missing keys (frozen weights)
    model.load_state_dict(filtered_state_dict, strict=False)