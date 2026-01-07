import torch
import numpy as np
from tqdm import tqdm

###############################################################################
# 1) Utility constants / functions for "non-metric" thresholding.
###############################################################################
PIXEL_TO_FIXED_METRIC_THRESH = {
    1: 0.1,
    2: 0.3,
    4: 0.5,
    8: 1.0,
}


def get_visibility_from_depth(depth_maps, max_depth=5):
    # plt.imsave('./tmp.png', depth_maps[0] > 0)
    # depth_maps is a list of (H, W) depth maps
    # return a list of (H, W) visibility maps
    # visibility is 1 if the depth is not 0, otherwise 0
    visibility = np.ones_like(depth_maps).astype(np.bool_)
    visibility = visibility & (depth_maps > 0.1) & (depth_maps < max_depth)
    return visibility

def estimate_sim3(
    A, B, 
    ransac=True, 
    ransac_iterations=1000, 
    inlier_threshold=0.05, 
    refine_with_inliers=True
):
    """
    Estimate a global Sim(3) transformation that aligns A to B in a least-squares sense.
    That is, find s, R, t such that:
       B ~ s * R * A + t
    
    where:
      - s is a scalar scale factor
      - R is a 3x3 rotation matrix (orthonormal, det(R)=1)
      - t is a 3D translation vector

    Args:
      A: (N, 3) array of points (source).
      B: (N, 3) array of points (target).
      ransac (bool): If True, will run RANSAC to find a robust Sim(3).
      ransac_iterations (int): Number of RANSAC iterations.
      inlier_threshold (float): Distance threshold (in target space) to consider
                                a point an inlier during RANSAC.
      refine_with_inliers (bool): Once a best model is found with RANSAC, 
                                  re-estimate Sim(3) on all its inliers.

    Returns:
      s: scalar scale factor
      R: (3, 3) rotation matrix
      t: (3,) translation vector

    Notes:
      - If ransac=False, this simply runs the "closed-form" solution on all points.
      - If ransac=True, we use a minimal 3-point subset to solve for Sim(3) 
        (assuming they're not degenerate). Then we do a standard inlier check 
        against all points. The best solution is picked.
      - If refine_with_inliers=True, we'll do a final closed-form estimate 
        using all inliers from the best model.
    """
    # Basic checks
    A = np.asarray(A)
    B = np.asarray(B)
    assert A.shape == B.shape, "A and B must have the same shape, (N,3)."
    N = A.shape[0]
    if N < 3:
        # At least 3 non-collinear points needed for a stable Sim(3)
        raise ValueError("Not enough points to estimate Sim(3). Need >=3.")

    if not ransac:
        # Direct closed-form solution
        return _estimate_sim3_closed_form(A, B)

    # ---------------------------------------------------------
    # RANSAC-based approach
    # ---------------------------------------------------------
    best_inliers_count = -1
    best_model = None
    best_inliers_mask = None

    # We need random subsets of size 3:
    # Usually we do more checks for collinearity, degeneracy, etc.
    # but for simplicity, we'll just skip if singular or fail SVD
    rng = np.random.default_rng()

    for _ in tqdm(range(ransac_iterations)):
        # 1) Randomly pick 3 distinct indices
        subset_idx = rng.choice(N, size=3, replace=False)
        A_subset = A[subset_idx]
        B_subset = B[subset_idx]

        # 2) Estimate Sim(3) from this minimal subset
        try:
            s_m, R_m, t_m = _estimate_sim3_closed_form(A_subset, B_subset)
        except np.linalg.LinAlgError:
            # SVD might fail if the subset is degenerate
            continue

        # 3) Transform all A by this model
        A_transformed = s_m * (R_m @ A.T).T + t_m  # shape (N, 3)
        # 4) Compute errors and count inliers
        dists = np.linalg.norm(A_transformed - B, axis=1)
        inliers_mask = (dists < inlier_threshold)
        inliers_count = np.sum(inliers_mask)

        # 5) Keep track of the best model
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_inliers_mask = inliers_mask
            best_model = (s_m, R_m, t_m)

    if best_model is None:
        # Fall back: direct closed-form on all points, or raise an error
        print("[WARN] RANSAC failed to find a valid model. Using closed-form on all points.")
        return _estimate_sim3_closed_form(A, B)

    s_best, R_best, t_best = best_model

    # ---------------------------------------------------------
    # Optional refinement on inliers
    # ---------------------------------------------------------
    if refine_with_inliers and best_inliers_mask is not None:
        inlierA = A[best_inliers_mask]
        inlierB = B[best_inliers_mask]
        if len(inlierA) >= 3:  # re-check we have enough inliers
            s_refine, R_refine, t_refine = _estimate_sim3_closed_form(inlierA, inlierB)
            return s_refine, R_refine, t_refine
        else:
            # Not enough inliers to refine
            return s_best, R_best, t_best

    return s_best, R_best, t_best


def _estimate_sim3_closed_form(A, B):
    """
    Closed-form Sim(3) estimation via SVD, aligning A to B.

    Returns:
      s, R, t
    """
    # 1) Remove centroids

    centroidA = A.mean(axis=0, keepdims=True)
    centroidB = B.mean(axis=0, keepdims=True)
    A_ = A - centroidA
    B_ = B - centroidB

    # 2) Compute rotation via SVD
    H = A_.T @ B_
    U, S, Vt = np.linalg.svd(H)
    R_ = Vt.T @ U.T

    # Fix possible reflection (ensure det(R) = +1)
    if np.linalg.det(R_) < 0:
        # Flip the sign of the last column of Vt
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T

    # 3) Compute scale
    #    We want s that minimizes || B_ - s*R_* A_ ||^2
    #    => s = (sum of singular values) / (sum of squared norms of A_)
    varA = (A_ ** 2).sum()
    # sum(S) is the Frobenius norm of (R_ A_ vs B_) in the cross-cov way
    s_ = np.sum(S) / (varA + 1e-12)

    # 4) Compute translation
    t_ = centroidB[0] - s_ * R_ @ centroidA[0]

    return s_, R_, t_


def get_pointwise_threshold_multiplier(gt_tracks: np.ndarray, intrinsics_params: np.ndarray) -> np.ndarray:
    """
    Computes a per-point multiplier to turn a "pixel threshold" into a 
    "world-space threshold" based on camera intrinsics.

    - gt_tracks: (T, N, 3) ground truth (X, Y, Z).
    - intrinsics_params: [fx, fy, cx, cy] (but we only use fx, fy).
      We often do sqrt(fx * fy) as a single measure of focal length.

    Returns:
      A (T, N) array, each entry is (Z_gt / mean_focal_length).
    """
    fx, fy, cx, cy = intrinsics_params
    mean_focal_length = np.sqrt(fx * fy + 1e-12)  # single scalar
    # (T, N) => Z_gt / mean_focal_length
    multiplier = gt_tracks[..., 2] / mean_focal_length
    return multiplier

def _compute_scale_factor_global(gt_points, pred_points):
    # Flatten over time/points -> shape (T*N, 3)
    gt_flat = gt_points.reshape(-1, 3)
    pred_flat = pred_points.reshape(-1, 3)

    gt_norm = np.linalg.norm(gt_flat, axis=-1)
    pred_norm = np.linalg.norm(pred_flat, axis=-1)

    eps = 1e-12
    gt_norm = np.maximum(gt_norm, eps)
    pred_norm = np.maximum(pred_norm, eps)

    # Prevent division by zero: ensure denominator is at least eps
    scale = np.median(gt_norm) / max(np.median(pred_norm), eps)
    return scale

def _scale_per_trajectory(gt_points, pred_points):
    """
    Scales each track (N) independently by median norm ratio.
    """
    T, N, _ = gt_points.shape
    scaled_pred = np.zeros_like(pred_points)
    eps = 1e-12

    for i in range(N):
        gt_traj = gt_points[:, i]     # (T, 3)
        pred_traj = pred_points[:, i] # (T, 3)

        gt_norm = np.linalg.norm(gt_traj, axis=-1)
        pred_norm = np.linalg.norm(pred_traj, axis=-1)

        gt_norm = np.maximum(gt_norm, eps)
        pred_norm = np.maximum(pred_norm, eps)

        # Prevent division by zero: ensure denominator is at least eps
        scale_i = np.median(gt_norm) / max(np.median(pred_norm), eps)
        scaled_pred[:, i] = pred_traj * scale_i

    return scaled_pred

def compute_average_pts_within_thresh(
    gt_points, 
    pred_points, 
    scaling="global", 
    intrinsics_params=None, 
    use_fixed_metric_threshold=False, 
    pred_aligned=None, 
    compute_epe=True
):
    """
    For the "metric-based" approach, we do either a global or per-trajectory scale,
    or a global Sim(3) alignment, then measure fraction of points within thresholds [1,2,4,8,16].
    """
    if isinstance(gt_points, torch.Tensor):
        gt_points = gt_points.detach().cpu().numpy()
    if isinstance(pred_points, torch.Tensor):
        pred_points = pred_points.detach().cpu().numpy()

    s_, R_, t_ = None, None, None
    if pred_aligned is None:
        # 1. Scale/Align
        if scaling == "global":
            # Old median-based scale
            scale = _compute_scale_factor_global(gt_points, pred_points)
            pred_aligned = pred_points * scale
            s_, R_, t_ = scale, np.eye(3), np.zeros(3)

        elif scaling == "per_traj":
            pred_aligned = _scale_per_trajectory(gt_points, pred_points)
            s_, R_, t_ = 1, np.eye(3), np.zeros(3)
        elif scaling == "sim3":
            # Flatten all frames/tracks
            T, N, _ = gt_points.shape
            gt_flat = gt_points.reshape(-1, 3)   # (T*N, 3)
            pred_flat = pred_points.reshape(-1, 3)

            # Estimate global sim3
            # randomly sampled 5k points
            if pred_flat.shape[0] > 16384:
                print(f"Sampling 16384 points from {pred_flat.shape[0]}")
                random_idx = np.random.choice(pred_flat.shape[0], size=16384, replace=False)
                gt_flat_sampled = gt_flat[random_idx]
                pred_flat_sampled = pred_flat[random_idx]
            else:
                gt_flat_sampled = gt_flat
                pred_flat_sampled = pred_flat
            s_, R_, t_ = estimate_sim3(pred_flat_sampled, gt_flat_sampled, ransac=True)

            # Apply it to pred_flat
            pred_aligned_flat = s_ * (R_ @ pred_flat.T).T + t_
            pred_aligned = pred_aligned_flat.reshape(T, N, 3)

        elif scaling == "sim3_closed":
            # Flatten all frames/tracks
            T, N, _ = gt_points.shape
            gt_flat = gt_points.reshape(-1, 3)   # (T*N, 3)
            pred_flat = pred_points.reshape(-1, 3)

            # Estimate global sim3
            s_, R_, t_ = estimate_sim3(pred_flat, gt_flat, ransac=False)

            # Apply it to pred_flat
            pred_aligned_flat = s_ * (R_ @ pred_flat.T).T + t_
            pred_aligned = pred_aligned_flat.reshape(T, N, 3)

        else:
            raise ValueError(f"Unknown scaling: {scaling}")

    # 2. Compute per-point distances in 3D
    dists = np.linalg.norm(pred_aligned - gt_points, axis=-1)  # (T, N)

    # 3. Threshold logic
    thresholds = [1, 2, 4, 8, 16]
    total_points = dists.size

    # Decide original dimension based on cx
    cx_val = intrinsics_params[2]
    if cx_val < 260:     # ADT
        ori_dim = 512
    elif cx_val < 500:   # PStudio
        ori_dim = 360
    else:                # DriveTrack
        ori_dim = 1280

    # Scale intrinsics so net image dimension ~256
    scaled_intrinsics = intrinsics_params * (256.0 / ori_dim)
    multiplier_map = get_pointwise_threshold_multiplier(gt_points, scaled_intrinsics)
    
    # Store the results for each threshold
    fractions = {}
    for thr, fixed_threshold in PIXEL_TO_FIXED_METRIC_THRESH.items():
        # Use a fixed threshold value
        if use_fixed_metric_threshold:
            pointwise_thresh = fixed_threshold
        else:
            pointwise_thresh = np.array(thr * multiplier_map)
        within_dist = (dists <= pointwise_thresh)
        frac_within = np.sum(within_dist) / float(total_points)
        fractions[thr] = frac_within

    avg_pts_within = np.mean(list(fractions.values()))

    if compute_epe:
        # Compute EPE only for visible points
        valid_distances = dists[dists < float('inf')]
        epe = valid_distances.mean() if len(valid_distances) > 0 else float('inf')
        return avg_pts_within, pred_aligned, fractions, (s_, R_, t_), epe
    
    return avg_pts_within, pred_aligned, fractions, (s_, R_, t_)