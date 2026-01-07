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
    Saves a training checkpoint and maintains a sliding window of the most recent files.
    
    Args:
        ckpt_dir: Directory where checkpoints will be stored.
        module: The PyTorch model to save.
        optimizer: The optimizer state to save.
        scheduler: Optional learning rate scheduler.
        global_step: Current training iteration (used for filename).
        keep_latest: Maximum number of recent checkpoints to retain.
        model_name: Prefix for the filename.
    """
    ckpt_path = pathlib.Path(ckpt_dir)
    ckpt_path.mkdir(exist_ok=True, parents=True)

    # Search for existing checkpoint files matching the naming pattern
    prev_ckpts = list(ckpt_path.glob(f'{model_name}-*pth'))
    # Sort files by modification time so that index 0 is the newest
    prev_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # FIFO (First-In-First-Out) logic: Delete the oldest files if we exceed keep_latest limit
    if len(prev_ckpts) > keep_latest - 1:
        for f in prev_ckpts[keep_latest - 1:]:
            f.unlink() # Delete file from disk

    # Construct filename with zero-padded global step for easy sorting (e.g., model-000001000.pth)
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
    Helper to determine the correct .pth path. 
    If a directory is provided, it automatically picks the latest checkpoint found inside.
    """
    path_obj = pathlib.Path(ckpt_path)
    
    if not path_obj.exists():
        if verbose:
            print(f'...there is no checkpoint at {ckpt_path}')
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    # Case 1: User provided a direct path to a file
    if path_obj.is_file():
        if verbose:
            print(f'...found checkpoint {path_obj}')
        return path_obj
    else:
        # Case 2: User provided a directory; search for the latest file inside
        prev_ckpts = list(path_obj.glob(f'{model_name}-*pth'))
        prev_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        if prev_ckpts:
            latest_path = prev_ckpts[0]
            if verbose:
                # Attempt to extract the step number from filename for logging
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
    """
    Unified loader helper. 
    Supports Lightning Fabric or standard torch.load.
    """
    if fabric is not None:
        return fabric.load(path)
    else:
        # map_location='cpu' ensures we can load models saved on GPU even on CPU-only machines
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
    Specialized loader that only updates parameters that are currently 'unfrozen' (trainable).
    Useful when you have a backbone frozen and only want to load weights into new heads.
    """
    if verbose:
        print(f'reading ckpt from {ckpt_path}')

    path = _resolve_checkpoint_path(ckpt_path, model_name, verbose)
    if path is None:
        return 0

    try:
        step = int(path.stem.split('-')[-1])
    except (ValueError, IndexError):
        pass 

    checkpoint = _load_checkpoint_file(fabric, path, weights_only)

    # Load states if they exist in the checkpoint
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    assert ignore_load is None, "ignore_load functionality is not implemented yet"

    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # Core logic: filter weights based on model's requires_grad status
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
    Standard full checkpoint loading function.
    """
    if verbose:
        print(f'reading ckpt from {ckpt_path}')

    path = _resolve_checkpoint_path(ckpt_path, model_name, verbose)
    if path is None:
        return 0

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

    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # Standard PyTorch loading. strict=False allows for partial architecture changes.
    model.load_state_dict(state_dict, strict=False)
    
    return step

def load_unfrozen_weights(model: torch.nn.Module, state_dict: Dict[str, Any]):
    """
    Selective parameter loading. 
    It checks which parameters in the current model have `requires_grad=True` 
    and only pulls those specific weights from the checkpoint's state_dict.
    """
    # Identify which keys in the current model are active/unfrozen
    unfrozen_keys = [name for name, param in model.named_parameters() if param.requires_grad]

    # Intersect unfrozen keys with keys actually present in the checkpoint
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in unfrozen_keys}

    print(f"Loading {len(filtered_state_dict)} / {len(state_dict)} parameters (unfrozen only).")

    # strict=False is required here because frozen weights will be missing from filtered_state_dict
    model.load_state_dict(filtered_state_dict, strict=False)