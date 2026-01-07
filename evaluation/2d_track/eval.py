import os
import sys
import time
import json
import random
import logging
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# ==============================================================================
# Path Setup
# ==============================================================================
# Add root directory to sys.path to allow imports from holi4d modules
ROOT = Path(__file__).resolve().parents[2]  # Holi4D/
sys.path.insert(0, str(ROOT))

import holi4d.utils.data
import holi4d.utils.basic
import holi4d.utils.improc
import holi4d.utils.misc
from holi4d.nets.model import Holi4D

# Optimize matrix multiplication precision for newer GPUs (Ampere+)
torch.set_float32_matmul_precision('medium')


# ==============================================================================
# Utility Functions
# ==============================================================================

def seed_everything(seed: int):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(dname, args):
    """
    Factory function to initialize datasets based on the name.
    
    Args:
        dname (str): Name of the dataset ('kin', 'rgb', 'rob').
        args (Namespace): Arguments containing dataset paths and config.
        
    Returns:
        dataset: The initialized dataset object.
        dataset_names: List containing the dataset name.
    """
    if dname == 'kin':
        from datasets import kineticsdataset
        dataset = kineticsdataset.KineticsDataset(
            data_root=os.path.join(args.dataset_root, 'tapvid_kinetics'),
            crop_size=args.image_size, 
            only_first=True, 
        )
        dataset_names = ['kin']
    elif dname == 'rgb':
        from datasets import rgbstackingdataset
        dataset = rgbstackingdataset.RGBStackingDataset(
            data_root=os.path.join(args.dataset_root, 'tapvid_rgb_stacking'),
            crop_size=args.image_size,
            only_first=False, 
        )
        dataset_names = ['rgb']
    elif dname == 'rob':
        from datasets import robotapdataset
        dataset = robotapdataset.RobotapDataset(
            data_root=os.path.join(args.dataset_root, 'robotap'),
            crop_size=args.image_size,
            only_first=True, 
        )
        dataset_names = ['rob']
    else:
        raise ValueError(f"Unknown dataset name: {dname}")
        
    return dataset, dataset_names


def create_pools(args, n_pool=10000, min_size=1): 
    """
    Creates metric accumulators (SimplePool) for evaluation statistics.
    
    Metrics:
    - d_x: Percentage of points within x pixels of ground truth.
    - jac_x: Jaccard index (IoU) at threshold x.
    - d_avg: Average distance.
    - aj: Average Jaccard.
    - oa: Occlusion Accuracy.
    """
    pools = {}
    n_pool = max(n_pool, 10)
    
    # Thresholds for distance metrics (1, 2, 4, 8, 16 pixels)
    thrs = [1, 2, 4, 8, 16]
    for thr in thrs:
        pools[f'd_{thr}'] = holi4d.utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
        pools[f'jac_{thr}'] = holi4d.utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
        
    pools['d_avg'] = holi4d.utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
    pools['aj'] = holi4d.utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
    pools['oa'] = holi4d.utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
    return pools


# ==============================================================================
# Core Inference Logic
# ==============================================================================

def forward_batch(batch, model, args, sw):
    """
    Runs inference on a single batch (video) and computes metrics.
    
    Args:
        batch: Data batch containing video, trajectories, visibility, etc.
        model: The Holi4D model instance.
        args: Configuration arguments.
        sw: SummaryWriter wrapper for logging.
        
    Returns:
        metrics: Dictionary of computed metrics for the batch.
    """
    # Unpack batch
    rgbs = batch.video      # [B, T, C, H, W]
    trajs_g = batch.trajs   # [B, T, N, 2] (Ground Truth Trajectories)
    vis_g = batch.visibs    # [B, T, N] (Ground Truth Visibility)
    valids = batch.valids   # [B, T, N]
    
    B, T, C, H, W = rgbs.shape
    B, T, N, D = trajs_g.shape
    assert C == 3, "Input video must have 3 channels (RGB)"
    assert B == 1, "Batch size must be 1 for evaluation"

    # Move data to GPU
    rgbs = rgbs.cuda()
    device = rgbs.device
    trajs_g = trajs_g.cuda()
    vis_g = vis_g.cuda()
    valids = valids.cuda()

    # Determine the first frame where each point is visible (query frame)
    # first_positive_inds: [B, N]
    __, first_positive_inds = torch.max(vis_g, dim=1)

    # Create a 2D grid for coordinate reference
    # grid_xy: [1, 1, 2, H, W]
    grid_xy = holi4d.utils.basic.gridcloud2d(1, H, W, norm=False, device=device).float()
    grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)

    # Initialize tensors for estimated trajectories and visibility confidence
    trajs_e = torch.zeros([B, T, N, 2], device=device)
    visconfs_e = torch.zeros([B, T, N, 2], device=device)
    
    query_points_all = []
    eval_dict = None
    pre_first_positive_ind = 0

    with torch.no_grad():
        # Iterate through unique starting frames (query frames)
        # This handles points that appear at different times in the video
        for first_positive_ind in torch.unique(first_positive_inds):
            
            # 1. Identify points starting at this specific frame
            chunk_pt_idxs = torch.nonzero(first_positive_inds[0] == first_positive_ind, as_tuple=False)[:, 0]  # Indices of points
            
            # Extract ground truth starting positions for these points
            # chunk_pts: [B, K, 2] where K is number of points in this chunk
            chunk_pts = trajs_g[:, first_positive_ind[None].repeat(chunk_pt_idxs.shape[0]), chunk_pt_idxs]
            
            # Store query points: [Frame_Index, X, Y]
            query_points_all.append(torch.cat([first_positive_inds[:, chunk_pt_idxs, None], chunk_pts], dim=2))
            
            # Initialize maps for this chunk
            traj_maps_e = grid_xy.repeat(1, T, 1, 1, 1) # [B, T, 2, H, W]
            visconf_maps_e = torch.zeros_like(traj_maps_e)

            # 2. Update evaluation dictionary (state) if continuing from previous chunks
            if eval_dict is not None:
                # Slice features to align with the current time window
                offset = first_positive_ind - pre_first_positive_ind
                for key in ['fmaps', 'ctxfeats', 'fmaps3d_detail', 'pms']:
                    if key in eval_dict:
                        eval_dict[key] = eval_dict[key][:, offset:]
                for key in ['points', 'masks']:
                    if key in eval_dict:
                        eval_dict[key] = eval_dict[key][offset:]

            # 3. Run Model Inference (Sliding Window)
            if first_positive_ind < T - 1:
                # forward_sliding handles long videos by processing in windows
                forward_flow_e, forward_visconf_e, _, _, eval_dict = model.forward_sliding(
                    rgbs[:, first_positive_ind:], 
                    iters=args.inference_iters, 
                    sw=sw, 
                    is_training=False, 
                    eval_dict=eval_dict
                )

                # Convert flow to absolute coordinates
                forward_traj_maps_e = forward_flow_e.to(device) + grid_xy
                
                # Update maps for the valid time range
                traj_maps_e[:, first_positive_ind:] = forward_traj_maps_e
                visconf_maps_e[:, first_positive_ind:] = forward_visconf_e

                # Clean up memory if not saving visualization
                if not sw.save_this:
                    del forward_flow_e, forward_visconf_e, forward_traj_maps_e

            pre_first_positive_ind = first_positive_ind    

            # 4. Sample the dense flow maps at the query point locations
            # Get starting coordinates (rounded to nearest integer for indexing)
            xyt = trajs_g[:, first_positive_ind].round().long()[0, chunk_pt_idxs] # [K, 2]
            
            # Sample trajectories: [B, T, 2, K] -> [B, T, K, 2]
            trajs_e_chunk = traj_maps_e[:, :, :, xyt[:, 1], xyt[:, 0]]
            trajs_e_chunk = trajs_e_chunk.permute(0, 1, 3, 2)
            
            # Scatter results back into the main storage tensor
            # We use scatter_add_ to place the K points into their correct indices N
            scatter_idx = chunk_pt_idxs[None, None, :, None].repeat(1, trajs_e_chunk.shape[1], 1, 2)
            trajs_e.scatter_add_(2, scatter_idx, trajs_e_chunk)

            # Sample visibility confidence
            visconfs_e_chunk = visconf_maps_e[:, :, :, xyt[:, 1], xyt[:, 0]]
            visconfs_e_chunk = visconfs_e_chunk.permute(0, 1, 3, 2)
            visconfs_e.scatter_add_(2, scatter_idx, visconfs_e_chunk)

    # Combine visibility confidence channels (if applicable)
    visconfs_e[..., 0] *= visconfs_e[..., 1]
    
    # Sanity check
    assert (torch.all(visconfs_e >= 0) and torch.all(visconfs_e <= 1))

    # 5. Compute Metrics (TAP-Vid style)
    vis_thr = 0.6
    query_points_all = torch.cat(query_points_all, dim=1)[..., [0, 2, 1]] # Reorder to [Frame, Y, X] or similar expected format
    
    gt_occluded = (vis_g < .5).bool().transpose(1, 2)
    gt_tracks = trajs_g.transpose(1, 2)
    pred_occluded = (visconfs_e[..., 0] < vis_thr).bool().transpose(1, 2)
    pred_tracks = trajs_e.transpose(1, 2)
    
    metrics = holi4d.utils.misc.compute_tapvid_metrics(
        query_points=query_points_all.cpu().numpy(),
        gt_occluded=gt_occluded.cpu().numpy(),
        gt_tracks=gt_tracks.cpu().numpy(),
        pred_occluded=pred_occluded.cpu().numpy(),
        pred_tracks=pred_tracks.cpu().numpy(),
        query_mode='first',
        crop_size=args.image_size
    )

    # Map raw metrics to readable keys
    for thr in [1, 2, 4, 8, 16]:
        metrics[f'd_{thr}'] = metrics[f'pts_within_{thr}']
        metrics[f'jac_{thr}'] = metrics[f'jaccard_{thr}']
    metrics['d_avg'] = metrics['average_pts_within_thresh']
    metrics['aj'] = metrics['average_jaccard']
    metrics['oa'] = metrics['occlusion_accuracy']
    
    return metrics


def run_evaluation(dname, model, args):
    """
    Main evaluation loop for a specific dataset.
    """
    seed = 42
    seed_everything(seed)
    
    # Generator for DataLoader reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed + worker_id)
        random.seed(worker_seed + worker_id)

    # Construct model name for logging
    model_name = f"{dname}_{int(args.image_size[0])}x{int(args.image_size[1])}"
    model_date = datetime.datetime.now().strftime('%M%S')
    model_name = f"{model_name}_{model_date}"

    # Initialize Dataset and DataLoader
    dataset, _ = get_dataset(dname, args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        drop_last=True,
        collate_fn=holi4d.utils.data.collate_fn_train,
    )
    
    print(f'Dataset: {dname}, Size: {len(dataloader)}')

    # Logging setup
    log_dir = 'evaluation/2d_track/logs_test_dense_on_sparse'
    overpools_t = create_pools(args)
    writer_t = SummaryWriter(
        os.path.join(log_dir, f"{args.model_type}-{model_name}", 't'), 
        max_queue=10, 
        flush_secs=60
    )
    
    global_step = 0
    max_steps = min(args.max_steps, len(dataloader))
    iterloader = iter(dataloader)

    # Evaluation Loop
    while global_step < max_steps:
        torch.cuda.empty_cache()
        iter_start_time = time.time()
        
        try:
            batch = next(iterloader)
        except StopIteration:
            # Should not happen if max_steps <= len(dataloader), but safe fallback
            iterloader = iter(dataloader)
            batch = next(iterloader)

        batch, gotit = batch
        if not all(gotit):
            print(f"Skipping batch {global_step} due to loading error.")
            continue
        
        # Setup summary writer for this step
        sw_t = holi4d.utils.improc.SummWriter(
            writer=writer_t,
            global_step=global_step,
            log_freq=args.log_freq,
            fps=8,
            scalar_freq=1,
            just_gif=True
        )
        if args.log_freq == 9999:
            sw_t.save_this = False

        rtime = time.time() - iter_start_time
        
        # Skip if no trajectories
        if batch.trajs.shape[2] == 0:
            global_step += 1
            continue

        # Run Inference
        metrics = forward_batch(batch, model, args, sw_t)

        # Update stats
        for key in list(overpools_t.keys()):
            if key in metrics:
                overpools_t[key].update([metrics[key]])
        
        # Log stats to tensorboard
        for key in list(overpools_t.keys()):
            sw_t.summ_scalar(f'_/{key}', overpools_t[key].mean())

        global_step += 1
        itime = time.time() - iter_start_time

        # Console Logging
        info_str = (
            f'{model_name}; step {global_step:06d}/{max_steps}; '
            f'rtime {rtime:.2f}; itime {itime:.2f}; '
            f'dname {dname}; '
            f'd_avg {overpools_t["d_avg"].mean()*100.0:.1f} '
            f'aj {overpools_t["aj"].mean()*100.0:.1f} '
            f'oa {overpools_t["oa"].mean()*100.0:.1f}'
        )
        
        if sw_t.save_this:
            print(f'Saving visualization for: {model_name}')

        if not args.print_less:
            print(info_str, flush=True)
            
    if args.print_less:
        print(info_str, flush=True)

    writer_t.close()
    
    # Return aggregated metrics
    return (
        overpools_t['d_avg'].mean() * 100.0, 
        overpools_t['aj'].mean() * 100.0, 
        overpools_t['oa'].mean() * 100.0
    )


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(args: argparse.Namespace, config) -> torch.nn.Module:
    """
    Initializes the Holi4D model and loads pretrained weights.
    """
    print("Initializing Holi4D model...")
    model = Holi4D(
        **config['model'],
        seqlen=16,
        use_3d=True,
    )

    if args.ckpt_init and os.path.exists(args.ckpt_init):
        print(f"Loading checkpoint from: {args.ckpt_init}")
        state_dict = torch.load(args.ckpt_init, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Checkpoint not found locally. Downloading from HuggingFace Hub...")
        url = "https://huggingface.co/cyun9286/holi4d/resolve/main/holi4d.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=False)
        model.load_state_dict(state_dict, strict=False)
    
    model.cuda()
    model.eval()
    
    # Freeze parameters for evaluation
    for p in model.parameters():
        p.requires_grad = False
    
    return model


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Holi4D Evaluation Script")
    
    # Experiment & Data
    parser.add_argument("--dname", type=str, nargs='+', default=None, 
                        help="Dataset names (e.g., 'kin', 'rgb', 'rob')")
    parser.add_argument("--dataset_root", type=str, 
                        default='./evaluation/2d_track',
                        help="Root directory for datasets")
    
    # Model & Checkpoint
    parser.add_argument("--ckpt_init", type=str, 
                        default="./checkpoints/holi4d.pth",
                        help="Path to model checkpoint file")
    parser.add_argument("--config_path", type=str, default="./holi4d/config/eval/v1.json",
                        help="Path to model configuration JSON file")
    parser.add_argument("--model_type", type=str, default="Holi4D", help="Model type name for logging")

    # Inference Parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (must be 1)")
    parser.add_argument("--num_workers", type=int, default=1, help="DataLoader workers")
    parser.add_argument("--max_steps", type=int, default=1500, help="Max evaluation steps")
    parser.add_argument("--inference_iters", type=int, default=4, help="Number of inference iterations")
    parser.add_argument("--window_len", type=int, default=16, help="Temporal window length")
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride")
    parser.add_argument("--image_size", nargs="+", default=[384, 512], help="Input resolution [H, W]")
    
    # Flags
    parser.add_argument("--log_freq", type=int, default=9999, help="Logging frequency")
    parser.add_argument("--print_less", action='store_true', default=False, help="Reduce console output")

    args = parser.parse_args()

    # Post-process arguments
    if args.dname is not None and len(args.dname) == 1 and ',' in args.dname[0]:
        args.dname = args.dname[0].split(',')
    if args.dname is None:
        args.dname = ['kin', 'rgb', 'rob']
    
    dataset_names = args.dname
    args.image_size = [int(args.image_size[0]), int(args.image_size[1])]
    
    full_start_time = time.time()

    # Load Configuration
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
        sys.exit(1)
        
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    # Load Model
    model = load_model(args, config)

    # Run Evaluation
    results_da, results_aj, results_oa = [], [], []
    
    print("\n" + "="*50)
    print("Starting Evaluation")
    print("="*50)

    for dname in dataset_names:
        da, aj, oa = run_evaluation(dname, model, args)
        results_da.append(da)
        results_aj.append(aj)
        results_oa.append(oa)

    # Print Final Summary
    print("\n" + "="*50)
    print("Final Results")
    print("="*50)
    
    summary_data = zip(dataset_names, results_da, results_aj, results_oa)
    print(f"{'Dataset':<10} | {'D_Avg':<10} | {'AJ':<10} | {'OA':<10}")
    print("-" * 46)
    for name, da, aj, oa in summary_data:
        print(f"{name:<10} | {da:<10.1f} | {aj:<10.1f} | {oa:<10.1f}")
    
    # Print CSV format for easy copying
    print("\nCSV Format:")
    print("Dataset, " + ", ".join(dataset_names))
    print("D_Avg, " + ", ".join([f"{x:.1f}" for x in results_da]))
    print("AJ,    " + ", ".join([f"{x:.1f}" for x in results_aj]))
    print("OA,    " + ", ".join([f"{x:.1f}" for x in results_oa]))

    full_time = time.time() - full_start_time
    print(f'\nTotal execution time: {full_time:.1f}s')

