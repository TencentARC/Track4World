from typing import List, Tuple, Union, Optional, Dict, Literal
from numbers import Number
import importlib
import warnings
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.version

# Custom modules
import utils3d
from holi4d.nets.blocks import (
    RelUpdateBlock, RelUpdate3D, InputPadder, CorrBlock, 
    CNBlockConfig, ConvNeXt, conv1x1
)
import holi4d.utils.basic
from holi4d.utils.geometry_torch import (
    normalized_view_plane_uv, recover_focal_shift, recover_global_focal, recover_global_focal_shift
)
from holi4d.utils import misc
from holi4d.nets.global_aggregator import Global_Aggregator
from holi4d.nets.external.pi3.models.pi3 import Pi3
from holi4d.nets.external.depth_anything_3.api import DepthAnything3
from holi4d.utils.geometry_torch import mask_aware_nearest_resize
from holi4d.utils.alignment import align_points_scale_xyz_shift

class ResidualConvBlock(nn.Module):
    """
    Standard Residual Convolutional Block with GroupNorm or LayerNorm.
    Consists of: Norm -> Activation -> Conv -> Norm -> Activation -> Conv.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: Optional[int] = None, 
                 hidden_channels: Optional[int] = None, 
                 padding_mode: str = 'replicate', 
                 activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu', 
                 norm: Literal['group_norm', 'layer_norm'] = 'group_norm'):
        super(ResidualConvBlock, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        # Select activation function
        if activation == 'relu':
            activation_cls = lambda: nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activation_cls = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'silu':
            activation_cls = lambda: nn.SiLU(inplace=True)
        elif activation == 'elu':
            activation_cls = lambda: nn.ELU(inplace=True)
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        # Main convolution path
        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activation_cls(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32 if norm == 'group_norm' else 1, hidden_channels),
            activation_cls(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        )

        # Skip connection (projection if dimensions change)
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x


def homogenize_points(points: torch.Tensor) -> torch.Tensor:
    """Convert batched points (xyz) to homogeneous coordinates (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def process_geometry(depth, extrinsics, intrinsics):
    """
    Args:
        depth: (B, T, H, W)
        extrinsics: (B, T, 3, 4), assumed to be OpenCV-style World-to-Camera (W2C)
        intrinsics: (B, T, 3, 3)
    Returns:
        world_points: (T, 3, H, W)
        camera_points: (T, 3, H, W)
        camera_poses: (T, 4, 4), Camera-to-World (C2W)
    """
    B, T, H, W = depth.shape
    device = depth.device

    # ==========================================
    # 1. Generate pixel grid
    # ==========================================
    # y: (H, W), x: (H, W)
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    # Flatten and expand dimensions: (1, 1, H*W)
    x = x.reshape(1, 1, -1).float()
    y = y.reshape(1, 1, -1).float()

    # Flatten depth map: (B, T, H*W)
    depth_flat = depth.reshape(B, T, -1)

    # ==========================================
    # 2. Compute camera-space points (back-projection)
    # ==========================================
    # Extract intrinsic parameters fx, fy, cx, cy
    # intrinsics: (B, T, 3, 3)
    fx = intrinsics[..., 0, 0].unsqueeze(-1)  # (B, T, 1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1)

    # Back-projection equations:
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    X_cam = (x - cx) * depth_flat / fx
    Y_cam = (y - cy) * depth_flat / fy
    Z_cam = depth_flat

    # Stack to obtain camera-space points: (B, T, 3, H*W)
    cam_points_flat = torch.stack([X_cam, Y_cam, Z_cam], dim=2)

    # Reshape for output: (B, T, 3, H, W) -> squeeze B -> (T, 3, H, W)
    camera_points = cam_points_flat.reshape(B, T, 3, H, W).squeeze(0)

    # ==========================================
    # 3. Compute camera poses (W2C -> C2W)
    # ==========================================
    # Input extrinsics are 3x4 matrices; convert them to 4x4
    # by appending the homogeneous row [0, 0, 0, 1]
    bottom_row = torch.tensor(
        [0, 0, 0, 1],
        device=device,
        dtype=extrinsics.dtype
    ).view(1, 1, 1, 4)
    bottom_row = bottom_row.expand(B, T, 1, 4)

    w2c_4x4 = torch.cat([extrinsics, bottom_row], dim=2)  # (B, T, 4, 4)

    # Invert to obtain Camera-to-World (C2W) poses
    c2w_4x4 = torch.inverse(w2c_4x4)

    # Output camera poses: (T, 4, 4)
    camera_poses = c2w_4x4.squeeze(0)

    # ==========================================
    # 4. Compute world-space points
    # ==========================================
    # P_world = R_c2w @ P_cam + t_c2w
    # Alternatively, using homogeneous coordinates:
    # P_world_homo = T_c2w @ P_cam_homo

    # Extract rotation R (B, T, 3, 3) and translation t (B, T, 3, 1)
    R_c2w = c2w_4x4[..., :3, :3]
    t_c2w = c2w_4x4[..., :3, 3:4]

    # Matrix multiplication:
    # (B, T, 3, 3) @ (B, T, 3, H*W) -> (B, T, 3, H*W)
    world_points_flat = torch.matmul(R_c2w, cam_points_flat) + t_c2w

    # Reshape to output format: (T, 3, H, W)
    world_points = world_points_flat.reshape(B, T, 3, H, W).squeeze(0)

    return world_points, camera_points, camera_poses

class Head(nn.Module):
    """
    Decoder Head for the Holi4D model. 
    Handles upsampling of features extracted from the backbone and projects them 
    to generate geometry outputs (points and masks).
    """
    def __init__(
        self,
        num_features: int,
        dim_in: int,
        dim_out: List[int],
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1
    ):
        super().__init__()

        # Project input features from backbone to a common dimension
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_proj, kernel_size=1, stride=1, padding=0) 
            for _ in range(num_features)
        ])

        # Upsampling blocks (ConvTranspose + Residual Blocks)
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2, out_ch), # +2 for UV coordinates
                *(ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm) 
                  for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        # Output prediction blocks (e.g., one for points, one for mask)
        self.output_block = nn.ModuleList([
            self._make_output_block(
                dim_upsample[-1] + 2, dim_out_, dim_times_res_block_hidden, last_res_blocks, 
                last_conv_channels, last_conv_size, res_block_norm,
            ) for dim_out_ in dim_out
        ])

    def _make_upsampler(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Creates a transposed convolution block for upsampling."""
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )
        # Initialize weights specifically for the first layer to behave like bilinear upsampling initially
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(self, dim_in, dim_out, dim_times_res_block_hidden, last_res_blocks, 
                           last_conv_channels, last_conv_size, res_block_norm) -> nn.Sequential:
        """Creates the final convolution block to project to output dimensions."""
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            *(ResidualConvBlock(last_conv_channels, last_conv_channels, dim_times_res_block_hidden * last_conv_channels, 
                                activation='relu', norm=res_block_norm) for _ in range(last_res_blocks)),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_conv_channels, dim_out, kernel_size=last_conv_size, stride=1, 
                      padding=last_conv_size // 2, padding_mode='replicate'),
        )

    def forward(self, hidden_states: List[Tuple[torch.Tensor, torch.Tensor]], image: torch.Tensor) -> List[torch.Tensor]:
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14

        # Combine projected features from different backbone layers
        # hidden_states contains (feature_map, class_token) tuples
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
                for proj, (feat, clstoken) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)

        # Upsampling pass with UV coordinate injection (CoordConv)
        for i, block in enumerate(self.upsample_blocks):
            # Generate UV coordinates
            uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, 
                                          dtype=x.dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            
            # Concatenate UV and process
            x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

        # Final interpolation to match image resolution
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        
        # Inject UV coordinates again before output
        uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, 
                                      dtype=x.dtype, device=x.device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        # Generate outputs (Points and Mask)
        if isinstance(self.output_block, nn.ModuleList):
            output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) for block in self.output_block]
        else:
            output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)

        return output

def get_aligned_scene_flow_temporal(flow2d_c, flow3d, points, visconf_maps_e, mode='align_direction'):
    """
    Args:
        flow2d_c: [B, T-1, H, W, 2]
        flow3d:   [B, T-1, H, W, 3] (Predicted scene flow to be aligned)
        points:   [B, T, H, W, 3]   (Point sequence)
        visconf_maps_e: [B, T-1, H, W, 2]
        align_func: function(pred, target) -> a, b (External function)

    Returns:
        s_aligned_final: [B, T-1, H, W, 3]
        valid_mask_final: [B, T-1, H, W]
    """
    B, T_minus_1, H, W, _ = flow2d_c.shape

    # -----------------------------------------------------------
    # 1. Data preparation: temporal slicing and flattening
    # -----------------------------------------------------------

    # points[:, t] is the source point cloud, corresponding to the start of flow2d_c[:, t]
    # points[:, t+1] is the target point cloud, corresponding to the end of flow2d_c[:, t]
    # We take the first T-1 frames as p_cur and the last T-1 frames as p_next

    p_cur = points[:, :-1]  # [B, T-1, H, W, 3]
    p_next = points[:, 1:]  # [B, T-1, H, W, 3]

    # Merge the B and T-1 dimensions and treat them as N independent samples
    # N = B * (T-1)
    p_cur_flat = p_cur.reshape(-1, H, W, 3)      # [N, H, W, 3]
    p_next_flat = p_next.reshape(-1, H, W, 3)    # [N, H, W, 3]
    flow2d_flat = flow2d_c.reshape(-1, H, W, 2)  # [N, H, W, 2]

    # Predicted scene flow (relative displacement)
    if mode == 'align_geo':
        s_pred_flat = flow3d.reshape(-1, H, W, 3)
    elif mode == 'align_dir':
        s_pred_flat = flow3d.reshape(-1, H, W, 3) - p_cur_flat  # [N, H, W, 3]
    else:
        raise ValueError(f"Unknown alignment mode: {mode}")

    visconf_flat = visconf_maps_e.reshape(-1, H, W, 2)  # [N, H, W, 2]

    N = p_cur_flat.shape[0]

    # -----------------------------------------------------------
    # 2. Geometric scene flow computation (optical flow + depth indexing)
    # -----------------------------------------------------------

    # Compute target sampling coordinates: pixel location + optical flow
    target_coords = flow2d_flat

    # Normalize coordinates to [-1, 1] for grid_sample
    norm_target_coords = target_coords.clone()
    norm_target_coords[..., 0] = 2.0 * target_coords[..., 0] / (W - 1) - 1.0
    norm_target_coords[..., 1] = 2.0 * target_coords[..., 1] / (H - 1) - 1.0

    # Sample P_next
    # grid_sample expects inputs of shape [N, C, H, W] and grid of shape [N, H, W, 2]
    p_next_permuted = p_next_flat.permute(0, 3, 1, 2)  # [N, 3, H, W]

    # Key step: sample corresponding geometric points from p_next
    p_next_sampled = F.grid_sample(
        p_next_permuted,
        norm_target_coords,
        mode='bilinear',
        align_corners=True,
        padding_mode='zeros'  # Out-of-bound samples are set to zero; handled later by masks
    )
    p_next_sampled = p_next_sampled.permute(0, 2, 3, 1)  # [N, H, W, 3]

    # Compute geometric scene flow: S_geo = P_next(u + flow) - P_cur(u)
    if mode == 'align_geo':
        # Use absolute 3D points in the next frame as geometric target
        s_geo_flat = p_next_sampled
    elif mode == 'align_dir':
        # Use relative displacement (scene flow) as geometric target
        s_geo_flat = p_next_sampled - p_cur_flat
    else:
        raise ValueError(f"Unknown alignment mode: {mode}")
    # -----------------------------------------------------------
    # 3. Valid mask computation
    # -----------------------------------------------------------

    # Base confidence mask
    conf_val = visconf_flat[..., 0] * visconf_flat[..., 1]
    valid_mask_flat = conf_val > 0.6  # [N, H, W]

    # Additional geometric validity checks (optional but recommended):
    # 1. Sampled points must lie within the image bounds
    in_bounds = (norm_target_coords.abs().max(dim=-1)[0] <= 1.0)

    # 2. Sampled points must have valid depth (assume zero indicates invalid points)
    has_depth = (
        (p_next_sampled.abs().sum(dim=-1) > 1e-4) &
        (p_cur_flat.abs().sum(dim=-1) > 1e-4)
    )
    # Combine all validity masks
    final_mask_flat = valid_mask_flat & in_bounds & has_depth

    # -----------------------------------------------------------
    # 4. Alignment
    # -----------------------------------------------------------

    # Initialize output container
    s_aligned_flat = s_pred_flat.clone()

    # Since align_func usually solves a least-squares problem over valid points,
    # and each sample has a different number of valid points,
    # full vectorization is difficult. We therefore iterate over the N samples.
    # N is typically small, so this is acceptable in practice.

    for i in range(N):
        mask_i = final_mask_flat[i]
        try:
            _, lr_mask, lr_index = mask_aware_nearest_resize(
                None, mask_i, (32, 32), return_index=True
            )
            s_pred_valid = s_pred_flat[i][lr_index][lr_mask]
            s_geo_valid = s_geo_flat[i][lr_index][lr_mask]

            scale, shift = align_points_scale_xyz_shift(
                s_pred_valid,
                s_geo_valid,
                1 / torch.ones_like(s_geo_valid.norm(dim=-1) + 1e-6),
                exp=20
            )

            # Apply the alignment parameters to the full-resolution scene flow
            # Broadcast scale and shift over the entire frame
            s_aligned_flat[i] = scale * s_pred_flat[i] + shift

        except Exception:
            # If alignment fails (e.g., due to a singular system),
            # fall back to the original prediction
            pass

    # -----------------------------------------------------------
    # 5. Restore original dimensions
    # -----------------------------------------------------------
    if mode == 'align_geo':
        s_aligned_final = s_aligned_flat.reshape(B, T_minus_1, H, W, 3)
    elif mode == 'align_dir':
        s_aligned_final = (s_aligned_flat + p_cur_flat).reshape(B, T_minus_1, H, W, 3)
    else:
        raise ValueError(f"Unknown alignment mode: {mode}")

    return s_aligned_final

class Holi4D(nn.Module):
    """
    Holi4D Model Implementation.
    Combines DINOv2 backbone for geometry estimation with a flow estimator (RAFT-like)
    for 4D (3D + Time) reconstruction and tracking.
    """
    image_mean: torch.Tensor
    image_std: torch.Tensor
    
    def __init__(self,
        encoder: str = 'dinov2_vitb14',
        intermediate_layers: Union[int, List[int]] = 4,
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        # --- Unified Parameters ---
        dim: int = 128,
        hdim: int = 128,
        seqlen: int = 8,
        corr_levels: int = 5,
        corr_radius: int = 4,
        num_blocks: int = 3,
        use_sinmotion: bool = True,
        # --- Original Model Parameters ---
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        remap_output: Literal[False, True, 'linear', 'sinh', 'exp', 'sinh_exp'] = 'linear',
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        num_tokens_range: Tuple[Number, Number] = [1200, 2500],
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        mask_threshold: float = 0.5,
        use_3d: bool = False,
        use_model: Literal['base', 'pi3', 'depthanythingv3'] = 'depthanythingv3',
        **deprecated_kwargs
    ):
        super(Holi4D, self).__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.encoder = encoder
        self.remap_output = remap_output
        self.intermediate_layers = intermediate_layers
        self.num_tokens_range = num_tokens_range
        self.mask_threshold = mask_threshold
        self.use_model = use_model
        # --- Backbone (DINOv2) and Feature Extractor ---
        # Dynamically load the DINOv2 backbone from the local hub
        hub_loader = getattr(importlib.import_module(".dinov2.hub.backbones", __package__), encoder)
        self.backbone = hub_loader(pretrained=False)
        self.dim_feature = self.backbone.blocks[0].attn.qkv.in_features
        if use_model == 'pi3':
            self.pi3 = Pi3.from_pretrained("yyfz233/Pi3")
            self.mask_threshold = 0.05
        elif use_model == 'depthanythingv3':
            self.dav3 = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
        # Token initialization (Register tokens and Camera tokens)
        num_register_tokens = 4
        register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dim_feature))
        nn.init.normal_(register_token, std=1e-6)
        self.register_buffer("register_token", register_token)
        
        num_camera_tokens = 1
        self.patch_start_idx = num_register_tokens + num_camera_tokens
        camera_token = nn.Parameter(torch.randn(1, 1, num_camera_tokens, self.dim_feature))
        nn.init.normal_(camera_token, std=1e-6)
        self.register_buffer("camera_token", camera_token)

        # Aggregator for global features
        self.aggregator = Global_Aggregator(patch_size=14, embed_dim=self.dim_feature, depth=8)

        # Geometry Head (Points and Mask prediction)
        self.head = Head(
            num_features=intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers), 
            dim_in=self.dim_feature, 
            dim_out=[3, 1], 
            dim_proj=dim_proj,
            dim_upsample=dim_upsample,
            dim_times_res_block_hidden=dim_times_res_block_hidden,
            num_res_blocks=num_res_blocks,
            res_block_norm=res_block_norm,
            last_res_blocks=last_res_blocks,
            last_conv_channels=last_conv_channels,
            last_conv_size=last_conv_size 
        )

        # --- Flow and Tracking Initialization ---
        self.dim = dim
        self.hdim = hdim
        self.seqlen = seqlen
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.pdim = 42 if use_sinmotion else 2
        self.corr_channel = self.corr_levels * (self.corr_radius * 2 + 1)**2
        self.flow_dim = 128
        self.flow3d_dim = 512
        self.use_3d = use_3d
        
        # Flow Aggregators
        self.flow_aggregator = Global_Aggregator(patch_size=14, embed_dim=self.flow_dim, depth=8)
        if use_3d:
            self.flow_aggregator3d = Global_Aggregator(patch_size=14, embed_dim=self.flow3d_dim, depth=4)

        # Context Encoders (ConvNeXt-based)
        block_setting = [
            CNBlockConfig(96, 192, 3, True),
            CNBlockConfig(192, 384, 3, False),
            CNBlockConfig(384, None, 9, False),
        ]
        self.ctx_encoder = ConvNeXt(block_setting, stochastic_depth_prob=0.0, init_weights=True)
        self.dot_conv = conv1x1(384, 128)
        self.dot_fc1 = nn.Linear(self.dim_feature, self.flow_dim)
        if use_3d:
            self.dot_fc2 = nn.Linear(self.dim_feature, self.flow3d_dim)

        # Update Blocks (RAFT-like iterative updates)
        self.flow_update_block = RelUpdateBlock(
            (self.corr_radius * 2 + 1)**2 * 5, 3, cdim=self.flow_dim, cdim1=128, 
            hdim=128, mdim=128, pdim2d=42, use_attn=True, use_layer_scale=True, 
            no_ctx=False, use_cxt_corr=True
        )
        if use_3d:
            self.flow3d_head = RelUpdate3D(
                (self.corr_radius * 2 + 1)**2 * 5, 2, tdim=self.flow3d_dim, cdim=self.flow_dim, 
                hdim=128, mdim=128, pdim3d=63, use_attn=True, use_layer_scale=True, 
                no_ctx=False, use_cxt_corr=True, use_prior=True
            )
            
        # Prediction Heads (Flow, Visibility Confidence, Upsampling weights)
        self.flow_2d_head = nn.Sequential(
            nn.Conv2d(self.flow_dim, 2 * self.flow_dim, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.flow_dim, 2, kernel_size=3, padding=1, padding_mode='replicate')
        )
        self.flow_visconf_head = nn.Sequential(
            nn.Conv2d(self.flow_dim, 2 * self.flow_dim, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.flow_dim, 2, kernel_size=3, padding=1, padding_mode='replicate')
        )
        self.flow_upsample_weight = nn.Sequential(
            nn.Conv2d(self.flow_dim, self.flow_dim * 2, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.flow_dim * 2, 64 * 9, 1, padding=0, padding_mode='replicate')
        )
        if use_3d:
            self.flow_3d_upsample_weight = nn.Sequential(
                nn.Conv2d(self.flow3d_dim, self.flow3d_dim * 2, 3, padding=1, padding_mode='replicate'),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.flow3d_dim * 2, 64 * 9, 1, padding=0, padding_mode='replicate')
            )

        # --- Positional Embeddings & Statistics ---
        time_line = torch.linspace(0, self.seqlen - 1, self.seqlen).reshape(1, self.seqlen, 1)
        self.register_buffer("time_emb", misc.get_1d_sincos_pos_embed_from_grid(self.flow_dim, time_line[0]))
        if use_3d:
            self.register_buffer("time_emb3d", misc.get_1d_sincos_pos_embed_from_grid(self.flow3d_dim, time_line[0]))
            
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
        
    def init_weights(self):
        """Initialize backbone weights from DINOv2 hub."""
        state_dict = torch.hub.load('facebookresearch/dinov2', self.encoder, pretrained=True).state_dict()
        self.backbone.load_state_dict(state_dict)

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        """Remap point coordinates based on configuration (e.g., for depth scaling)."""
        if self.remap_output == 'linear':
            pass
        elif self.remap_output == 'sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == 'sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def forward_point(self, image: torch.Tensor, current_batch_size=4, for_flow=False):
        """
        Process single frames to extract geometry (points, mask) and features.
        Used as the initial step before flow estimation.
        
        Args:
            image: Input images.
            current_batch_size: Batch size for processing.
            for_flow: Flag indicating if features are needed for flow estimation.
        """
        with torch.no_grad():
            original_height, original_width = image.shape[-2:]
            image = image.reshape(-1, 3, original_height, original_width)
            image_14 = F.interpolate(
                image, 
                (original_height // 14 * 14, original_width // 14 * 14), 
                mode="bilinear", align_corners=False, antialias=True
            )
            H_14, W_14 = image_14.shape[-2:]

            # Backbone Feature Extraction
            features = self.backbone.get_intermediate_layers(image_14, self.intermediate_layers, return_class_token=True)
            
            # Prepare tokens (Camera + Register + Patch tokens)
            camera_token = self.camera_token.expand(current_batch_size, image.shape[0]//current_batch_size, *self.camera_token.shape[2:])
            register_token = self.register_token.expand(current_batch_size, image.shape[0]//current_batch_size, *self.register_token.shape[2:])
            tokens = torch.cat([
                camera_token, 
                register_token, 
                features[-1][0].reshape(current_batch_size, -1, features[-1][0].shape[-2], features[-1][0].shape[-1])
            ], dim=2)
            
            # Aggregate global features
            global_features = self.aggregator(tokens, image, patch_start_idx=self.patch_start_idx)
            
            # Update features with aggregated information
            features = [list(f) for f in features]
            for i in range(len(features)):
                new_feat = global_features[2*i+1][:, :, self.patch_start_idx:].reshape(-1, features[-1][0].shape[-2], features[-1][0].shape[-1])
                features[i][0] = new_feat
            
            # Predict geometry (points and mask)
            points, mask = self.head(features, image)
            if self.use_model == 'pi3':
                
                results = self.pi3(image_14[None])
                points = results['local_points'][0].permute(0, -1, 1, 2)
                
                #pts_raw = F.interpolate(points.clone(), (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
                #pts_pi3 = F.interpolate(points.clone(), (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
                #norm_head = torch.norm(pts_raw, dim=1)
                #norm_pi3  = torch.norm(pts_pi3, dim=1)
                #valid = (norm_pi3 > 1e-6) & (norm_head > 1e-6)
                #scale = torch.median(norm_head[valid] / norm_pi3[valid])

                world_points = results['points'][0].permute(0, -1, 1, 2)
                camera_poses = results['camera_poses'][0]
                conf = results['conf'][0].permute(0, -1, 1, 2)
                mask = torch.sigmoid(conf)
                points = points / 10
                world_points = world_points / 10
                camera_poses[..., :3, 3] /= 10
            elif self.use_model == 'depthanythingv3':
                results = self.dav3.inference_v2(
                    image_14[None]
                )
                # pts_raw = F.interpolate(points.clone(), (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
                world_points, points, camera_poses = process_geometry(results["depth"], results["extrinsics"], results["intrinsics"])
                mask = results["depth_conf"].transpose(0, 1)
                self.mask_threshold = torch.quantile(mask, 0.05)
                # print(points.shape)
                # pts_dav3 = F.interpolate(points.clone(), (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
                # norm_head = torch.norm(pts_raw, dim=1)
                # norm_dav3  = torch.norm(pts_dav3, dim=1)
                # valid = (norm_dav3 > 1e-6) & (norm_head > 1e-6)
                # scale = torch.median(norm_dav3 / norm_head)
                # print(scale)
                points = points / 100
                world_points = world_points / 100
                camera_poses[..., :3, 3] /= 100
            else:
                world_points = torch.zeros_like(points)
                camera_poses = torch.zeros(points.shape[0], 4, 4).to(image_14.device)

        # Flow Feature extraction (in Autocast for efficiency)
        with torch.autocast(device_type=image_14.device.type, dtype=torch.float32):
            flow_features = self.dot_fc1(global_features[-1].detach())    
            flow_features = self.flow_aggregator(flow_features, image_14, patch_start_idx=self.patch_start_idx)
            
            if self.use_3d:
                flow3d_features = self.dot_fc2(global_features[-1].detach())    
                flow3d_features = self.flow_aggregator3d(flow3d_features, image_14, patch_start_idx=self.patch_start_idx)
            
            # Context extraction using ConvNeXt
            ctxfeat = self.ctx_encoder(image)
            ctxfeat = self.dot_conv(ctxfeat)
            
            flow_H, flow_W = original_height // 8, original_width // 8
            
            # Interpolate features to 1/8 resolution for flow estimation
            fmaps = F.interpolate(
                flow_features[-1][:, :, self.patch_start_idx:].reshape(-1, H_14//14, W_14//14, self.flow_dim).permute(0, -1, 1, 2), 
                (flow_H, flow_W), mode='area'
            ).reshape(-1, self.flow_dim, flow_H, flow_W)
            
            if self.use_3d:
                fmaps3d_detail = F.interpolate(
                    flow3d_features[-1][:, :, self.patch_start_idx:].reshape(-1, H_14//14, W_14//14, self.flow3d_dim).permute(0, -1, 1, 2), 
                    (flow_H, flow_W), mode='area'
                ).reshape(-1, self.flow3d_dim, flow_H, flow_W)
        
        # Process points and masks
        with torch.autocast(device_type=image_14.device.type, dtype=torch.float32):
            points = F.interpolate(points, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
            world_points = F.interpolate(world_points, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
            if self.use_model == 'base':
                points = points.permute(0, 2, 3, 1)
                points = self._remap_points(points).permute(0, -1, 1, 2)
            mask = F.interpolate(mask, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
        
        # Downsample points for flow correlation
        with torch.autocast(device_type=image_14.device.type, dtype=torch.float32):
            pm = F.interpolate(points, (flow_H, flow_W), mode='nearest')
            pm = pm.reshape(-1, pm.shape[-3], pm.shape[-2], pm.shape[-1])
        
        if self.use_3d:
            return fmaps.to(image.dtype), ctxfeat.to(image.dtype), fmaps3d_detail.to(image.dtype), pm.to(image.dtype), points, mask, world_points, camera_poses
        else:
            fmaps3d_detail = torch.zeros((pm.shape[0], self.flow3d_dim, flow_H, flow_W)).cuda()
            return fmaps.to(image.dtype), ctxfeat.to(image.dtype), fmaps3d_detail.to(image.dtype), pm.to(image.dtype), points, mask, world_points, camera_poses

    def forward(self, images, iters=4, sw=None, is_training=False, stride=None, tracking3d=False):
        """
        Main forward pass for video sequences.
        Estimates flow and geometry across frames using a sliding window approach if necessary.
        """
        B, T, C, H, W = images.shape
        S = self.seqlen
        device = images.device
        images = images.to(torch.float32)
        dtype = images.dtype

        # Normalize images
        images = images / 255.0
        images = (images - self.image_mean) / self.image_std

        T_bak = T
        if stride is not None:
            pad = False
        else:
            pad = True
        
        # Pad temporal dimension to fit window size
        images, T, indices = self.get_T_padded_images(images, T, S, is_training, stride=stride, pad=pad)

        images = images.contiguous()
        images_ = images.reshape(B * T, 3, H, W)
        padder = InputPadder(images_.shape)
        images_ = padder.pad(images_)[0]

        _, _, H_pad, W_pad = images_.shape
        H8, W8 = H_pad//8, W_pad//8
        C = self.flow_dim
        C1 = 128

        # Extract features (Geometry & Flow) for all frames
        fmaps, ctxfeats, fmaps3d_detail, pms, points, _, world_points, camera_poses = self.get_fmaps(images_, B, T, sw, is_training)
        
        fmaps = fmaps.to(dtype).reshape(B, T, C, H8, W8)
        fmaps3d_detail = fmaps3d_detail.to(dtype).reshape(B, T, self.flow3d_dim, H8, W8)
        ctxfeats = ctxfeats.to(dtype).reshape(B, T, C1, H8, W8)
        pms = pms.to(dtype).reshape(B, T, 3, H8, W8)
        
        # Anchor frame (t=0) features
        fmap_anchor = fmaps[:, 0]
        ctxfeat_anchor = ctxfeats[:, 0]
        pm_anchor = pms[:, 0]
        fmaps3d_detail_anchor = fmaps3d_detail[:, 0]

        # Containers for predictions
        if T <= 2 or is_training:
            all_flow_preds = []
            if tracking3d:
                all_flow3d_preds = []
            all_visconf_preds = []
        else:
            all_flow_preds = None
            if tracking3d:
                all_flow3d_preds = None
            all_visconf_preds = None

        if T > 2: # Multiframe tracking logic
            # Final output tensors
            full_flows = torch.zeros((B,T,2,H,W), dtype=dtype, device=device)
            if tracking3d:
                full_flows3d = torch.zeros((B,T,3,H,W), dtype=dtype, device=device)
            full_visconfs = torch.zeros((B,T,2,H,W), dtype=dtype, device=device)
            
            # Low-res output tensors for recurrent state
            full_flows8 = torch.zeros((B,T,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
            full_flow3ds8 = torch.zeros((B,T,3,H_pad//8,W_pad//8), dtype=dtype, device=device)
            full_visconfs8 = torch.zeros((B,T,2,H_pad//8,W_pad//8), dtype=dtype, device=device)

            visits = np.zeros((T))

            # Iterate over sliding windows
            for ii, ind in enumerate(indices):
                ara = np.arange(ind, ind+S)
                if ii < len(indices)-1:
                    next_ind = indices[ii+1]
                
                fmaps2 = fmaps[:, ara]
                fmaps3d_detail2 = fmaps3d_detail[:, ara]
                ctxfeats2 = ctxfeats[:, ara]
                pms2 = pms[:, ara]
                flows8 = full_flows8[:, ara].reshape(B*(S), 2, H_pad//8, W_pad//8).detach()
                flow3ds8 = full_flow3ds8[:, ara].reshape(B*(S), 3, H_pad//8, W_pad//8).detach()
                visconfs8 = full_visconfs8[:, ara].reshape(B*(S), 2, H_pad//8, W_pad//8).detach()
                
                feats8 = None

                # Core window processing (Iterative Flow Update)
                with torch.autocast(device_type=fmaps.device.type, dtype=torch.float32):
                    flow_predictions, flow3d_predictions, visconf_predictions, flows8, flow3ds8, feats8 = self.forward_window_unified(
                        fmap1_single=fmap_anchor, fmap2=fmaps2, fmaps3d_detail1_single=fmaps3d_detail_anchor, fmaps3d_detail2=fmaps3d_detail2,
                        visconfs8=visconfs8, iters=iters, flow2ds8=flows8, flow3ds8=flow3ds8,
                        cxt1_single=ctxfeat_anchor, cxt2=ctxfeats2,
                        pm1_single=pm_anchor.detach(), pm2=pms2.detach(),
                        is_training=is_training,
                        tracking3d=tracking3d
                    )
                
                # Unpadding and result collection
                unpad_flow_predictions = []
                if tracking3d:
                    unpad_flow3d_predictions = []
                unpad_visconf_predictions = []
                
                for i in range(len(flow_predictions)):
                    flow_predictions[i] = padder.unpad(flow_predictions[i])
                    unpad_flow_predictions.append(flow_predictions[i].reshape(B,S,2,H,W))
                    if tracking3d:
                        flow3d_predictions[i] = padder.unpad(flow3d_predictions[i])
                        unpad_flow3d_predictions.append(flow3d_predictions[i].reshape(B,S,3,H,W))
                    visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                    unpad_visconf_predictions.append(visconf_predictions[i].reshape(B,S,2,H,W))

                # Update full tensors
                full_flows[:, ara] = unpad_flow_predictions[-1].reshape(B,S,2,H,W)
                full_flows8[:, ara] = flows8.reshape(B,S,2,H_pad//8,W_pad//8)
                if tracking3d:
                    full_flows3d[:, ara] = unpad_flow3d_predictions[-1].reshape(B,S,3,H,W)
                    full_flow3ds8[:, ara] = flow3ds8.reshape(B,S,3,H_pad//8,W_pad//8)
                full_visconfs[:, ara] = unpad_visconf_predictions[-1].reshape(B,S,2,H,W)
                full_visconfs8[:, ara] = visconfs8.reshape(B,S,2,H_pad//8,W_pad//8)

                visits[ara] += 1

                if is_training:
                    all_flow_preds.append(unpad_flow_predictions)
                    if tracking3d:
                        all_flow3d_preds.append(unpad_flow3d_predictions)
                    all_visconf_preds.append(unpad_visconf_predictions)
                else:
                    del unpad_flow_predictions
                    if tracking3d:
                        del unpad_flow3d_predictions
                    del unpad_visconf_predictions

                # Fill gaps in sliding window (simple nearest neighbor for unvisited frames)
                invalid_idx = np.where(visits==0)[0]
                valid_idx = np.where(visits>0)[0]
                for idx in invalid_idx:
                    nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                    full_flows8[:, idx] = full_flows8[:, nearest]
                    if tracking3d:
                        full_flow3ds8[:, idx] = full_flow3ds8[:, nearest]
                    full_visconfs8[:, idx] = full_visconfs8[:, nearest]

        else: # Standard flow (2 frames)
            flows8 = torch.zeros((B, 2, H_pad//8, W_pad//8), dtype=dtype, device=device)
            flow3ds8 = torch.zeros((B, 3, H_pad//8, W_pad//8), dtype=dtype, device=device)
            visconfs8 = torch.zeros((B, 2, H_pad//8, W_pad//8), dtype=dtype, device=device)
            
            with torch.autocast(device_type=fmaps.device.type, dtype=torch.float32):
                flow_predictions, flow3d_predictions, visconf_predictions, flows8, flow3ds8, feats8 = self.forward_window_unified(
                        fmap1_single=fmap_anchor, fmap2=fmaps[:,1:2], fmaps3d_detail1_single=fmaps3d_detail_anchor, fmaps3d_detail2=fmaps3d_detail[:,1:2],
                        visconfs8=visconfs8, iters=iters, flow2ds8=flows8, flow3ds8=flow3ds8,
                        cxt1_single=ctxfeat_anchor, cxt2=ctxfeats[:,1:2],
                        pm1_single=pm_anchor.detach(), pm2=pms[:,1:2].detach(),
                        is_training=is_training,
                        tracking3d=tracking3d
                    )

            unpad_flow_predictions = []
            if tracking3d:
                unpad_flow3d_predictions = []
            unpad_visconf_predictions = []
            for i in range(len(flow_predictions)):
                flow_predictions[i] = padder.unpad(flow_predictions[i])
                all_flow_preds.append(flow_predictions[i].reshape(B,2,H,W))
                if tracking3d:
                    flow3d_predictions[i] = padder.unpad(flow3d_predictions[i])
                    all_flow3d_preds.append(flow3d_predictions[i].reshape(B,3,H,W))
                visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                all_visconf_preds.append(visconf_predictions[i].reshape(B,2,H,W))
            full_flows = all_flow_preds[-1].reshape(B,2,H,W)
            if tracking3d:
                full_flows3d = all_flow3d_preds[-1].reshape(B,3,H,W)
            full_visconfs = all_visconf_preds[-1].reshape(B,2,H,W)
                
        if (not is_training) and (T > 2):
            full_flows = full_flows[:, :T_bak]
            if tracking3d:
                full_flows3d = full_flows3d[:, :T_bak]
            full_visconfs = full_visconfs[:, :T_bak]
        
        points = padder.unpad(points)
        world_points = padder.unpad(world_points)
        if tracking3d:
            return full_flows, full_visconfs, all_flow_preds, all_visconf_preds, full_flows3d, all_flow3d_preds, points, world_points, camera_poses
        else:
            return full_flows, full_visconfs, all_flow_preds, all_visconf_preds

    def pairwise_concat(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts a time-series feature [B, T, C, H, W] to [B*(T-1), 2, C, H, W]
        by stacking adjacent frames (t, t+1).
        """
        B, T, C, H, W = tensor.shape
        first = tensor[:, :-1]   # [B, T-1, C, H, W]
        second = tensor[:, 1:]   # [B, T-1, C, H, W]
        
        # Stack into pairs [B, T-1, 2, C, H, W]
        pairs = torch.stack([first, second], dim=2)
        
        # Flatten batch and time dimensions
        pairs = pairs.reshape(B * (T - 1), 2, C, H, W)
        return pairs

    def forward_sliding1(self, images, iters=4, sw=None, is_training=False, window_len=None, stride=None, tracking3d=False):
        """
        Variant of forward pass using sliding window/pairwise estimation with memory optimizations.
        Splits processing into chunks along the batch dimension to save VRAM.
        """
        B, T, C, H, W = images.shape
        device = images.device
        images = images.to(torch.float32)
        dtype = images.dtype

        # Normalization
        images = images / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1,1,3,1,1).to(images.dtype)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1,1,3,1,1).to(images.dtype)
        images = (images - mean)/std

        images = images.contiguous()
        images_ = images.reshape(B*T,3,H,W)
        padder = InputPadder(images_.shape)
        images_ = padder.pad(images_)[0]
        
        _, _, H_pad, W_pad = images_.shape
        C, H8, W8 = self.flow_dim, H_pad//8, W_pad//8
        C1 = 128

        # Extract Features
        fmaps, ctxfeats, fmaps3d_detail, pms, points, masks, world_points, camera_poses = self.get_fmaps(images_, B, T, sw, is_training)
        
        fmaps = fmaps.to(dtype).reshape(B, T, C, H8, W8)
        fmaps3d_detail = fmaps3d_detail.to(dtype).reshape(B, T, self.flow3d_dim, H8, W8)
        ctxfeats = ctxfeats.to(dtype).reshape(B, T, C1, H8, W8)
        pms = pms.to(dtype).reshape(B, T, 3, H8, W8)
        
        # Convert to pairwise format for frame-to-frame estimation
        fmaps = self.pairwise_concat(fmaps)
        fmaps3d_detail = self.pairwise_concat(fmaps3d_detail)
        ctxfeats = self.pairwise_concat(ctxfeats)
        pms = self.pairwise_concat(pms)

        B_pair, T_pair = fmaps.shape[0], fmaps.shape[1]
        
        # --- Memory Optimization: Split and iterate along the Batch (B) dimension ---
        chunk_size = 12 
        num_chunks = (B_pair + chunk_size - 1) // chunk_size
        
        all_flow_preds_chunks = []
        all_visconf_preds_chunks = []
        all_flow3d_preds_chunks = [] if tracking3d else None
        
        full_flows_list = []
        full_visconfs_list = []
        full_flows3d_list = [] if tracking3d else None
        points_list = []
        masks_list = []
        world_points_list = []
        camera_poses_list = []
        
        # Split points/masks (note: points/masks correspond to original B,T structure, need care if reshaped)
        # Here points has shape related to original images, splitting based on original B dimension
        points_chunks = list(torch.split(points, chunk_size, dim=0))
        masks_chunks = list(torch.split(masks, chunk_size, dim=0))
        world_points_chunks = list(torch.split(world_points, chunk_size, dim=0))
        camera_poses_chunks = list(torch.split(camera_poses, chunk_size, dim=0)) 
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, B_pair)
            b_chunk = end_idx - start_idx
            
            print(f"Processing Batch Chunk {i+1}/{num_chunks} (Size: {b_chunk})")

            # 1. Slice input tensors for the current chunk
            fmaps_chunk = fmaps[start_idx:end_idx]
            ctxfeats_chunk = ctxfeats[start_idx:end_idx]
            pms_chunk = pms[start_idx:end_idx]
            fmaps3d_detail_chunk = fmaps3d_detail[start_idx:end_idx]
            
            # Note: points/masks splitting logic above might need adjustment depending on how they map to pairwise B
            # Assuming linear mapping for now based on context provided
            points_chunk = points_chunks[i] if i < len(points_chunks) else points_chunks[-1]
            masks_chunk = masks_chunks[i] if i < len(masks_chunks) else masks_chunks[-1]
            world_points_chunk = world_points_chunks[i] if i < len(world_points_chunks) else world_points_chunks[-1]
            camera_poses_chunk = camera_poses_chunks[i] if i < len(camera_poses_chunks) else camera_poses_chunks[-1]

            # 2. Define Anchor (t=0) for the pair
            fmap_anchor_chunk = fmaps_chunk[:, 0]
            ctxfeat_anchor_chunk = ctxfeats_chunk[:, 0]
            pm_anchor_chunk = pms_chunk[:, 0]
            fmaps3d_detail_anchor_chunk = fmaps3d_detail_chunk[:, 0]
            
            # 3. Initialize flow variables
            H8, W8 = fmap_anchor_chunk.shape[-2:]
            dtype = fmaps_chunk.dtype
            
            flows8_chunk = torch.zeros((b_chunk, 2, H8, W8), dtype=dtype, device=device)
            flow3ds8_chunk = torch.zeros((b_chunk, 3, H8, W8), dtype=dtype, device=device)
            visconfs8_chunk = torch.zeros((b_chunk, 2, H8, W8), dtype=dtype, device=device)
            
            # 4. Core Forward Pass (Flow Estimation)
            flow_predictions, flow3d_predictions, visconf_predictions, _, _, _ = self.forward_window_unified(
                fmap1_single=fmap_anchor_chunk, fmap2=fmaps_chunk[:, 1:2], 
                fmaps3d_detail1_single=fmaps3d_detail_anchor_chunk, fmaps3d_detail2=fmaps3d_detail_chunk[:, 1:2],
                visconfs8=visconfs8_chunk, iters=iters, 
                flow2ds8=flows8_chunk, flow3ds8=flow3ds8_chunk,
                cxt1_single=ctxfeat_anchor_chunk, cxt2=ctxfeats_chunk[:, 1:2],
                pm1_single=pm_anchor_chunk.detach(), pm2=pms_chunk[:, 1:2].detach(),
                is_training=is_training,
                tracking3d=tracking3d
            )
            
            # 5. Collect Chunk Results
            chunk_flow_preds = []
            chunk_visconf_preds = []
            chunk_flow3d_preds = [] if tracking3d else None
            
            for k in range(len(flow_predictions)):
                # Unpad and reshape
                flow_predictions[k] = padder.unpad(flow_predictions[k])
                chunk_flow_preds.append(flow_predictions[k].reshape(b_chunk, 2, H, W))
                
                visconf_predictions[k] = padder.unpad(torch.sigmoid(visconf_predictions[k]))
                chunk_visconf_preds.append(visconf_predictions[k].reshape(b_chunk, 2, H, W))
                
                if tracking3d:
                    flow3d_predictions[k] = padder.unpad(flow3d_predictions[k])
                    chunk_flow3d_preds.append(flow3d_predictions[k].reshape(b_chunk, 3, H, W))

            # 6. Store full resolution results
            full_flows_list.append(chunk_flow_preds[-1].detach().cpu())
            full_visconfs_list.append(chunk_visconf_preds[-1].detach().cpu())
            points_list.append(padder.unpad(points_chunk))
            masks_list.append(padder.unpad(masks_chunk))
            world_points_list.append(padder.unpad(world_points_chunk))
            camera_poses_list.append(camera_poses_chunk)
            

            if tracking3d:
                full_flows3d_list.append(chunk_flow3d_preds[-1].cpu())

            # 7. Store iteration history
            all_flow_preds_chunks.append(chunk_flow_preds)
            all_visconf_preds_chunks.append(chunk_visconf_preds)
            if tracking3d:
                all_flow3d_preds_chunks.append(chunk_flow3d_preds)

            # Clear Chunk Memory
            del fmaps_chunk, ctxfeats_chunk, pms_chunk, fmaps3d_detail_chunk
            del flow_predictions, visconf_predictions
            if tracking3d:
                del flow3d_predictions
            torch.cuda.empty_cache()

        # --- Aggregate Chunks ---
        
        # 1. Aggregate iterations
        all_flow_preds = []
        all_visconf_preds = []
        all_flow3d_preds = [] if tracking3d else None
        
        if all_flow_preds_chunks:
            num_iters = len(all_flow_preds_chunks[0])
            for k in range(num_iters):
                flows_k = torch.cat([preds_chunk[k] for preds_chunk in all_flow_preds_chunks], dim=0)
                all_flow_preds.append(flows_k)
                
                visconfs_k = torch.cat([preds_chunk[k] for preds_chunk in all_visconf_preds_chunks], dim=0)
                all_visconf_preds.append(visconfs_k)
                
                if tracking3d:
                    flows3d_k = torch.cat([preds_chunk[k] for preds_chunk in all_flow3d_preds_chunks], dim=0)
                    all_flow3d_preds.append(flows3d_k)

        # 2. Aggregate Final Results
        full_flows = torch.cat(full_flows_list, dim=0).reshape(B_pair, 2, H, W)
        full_visconfs = torch.cat(full_visconfs_list, dim=0).reshape(B_pair, 2, H, W)
        
        points = torch.cat(points_list, dim=0)
        masks = torch.cat(masks_list, dim=0)
        world_points = torch.cat(world_points_list, dim=0)
        camera_poses = torch.cat(camera_poses_list, dim=0)

        if tracking3d:
            full_flows3d = torch.cat(full_flows3d_list, dim=0).reshape(B_pair, 3, H, W)
            return full_flows, full_visconfs, all_flow_preds, all_visconf_preds, full_flows3d, all_flow3d_preds, points, masks, world_points, camera_poses
        else:
            return full_flows, full_visconfs, all_flow_preds, all_visconf_preds

    def forward_sliding(self, images, iters=4, sw=None, is_training=False, window_len=None, stride=None, tracking3d=False, eval_dict=None):
        """
        Sliding window inference for longer sequences.
        Optionally uses cached features (eval_dict) to speed up evaluation.
        """
        B, T, C, H, W = images.shape
        S = self.seqlen if window_len is None else window_len
        device = images.device
        images = images
        dtype = images.dtype
        stride = S // 2 if stride is None else stride

        images = images / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1,1,3,1,1).to(images.dtype)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1,1,3,1,1).to(images.dtype)
        images = (images - mean)/std

        T_bak = T
        images, T, indices = self.get_T_padded_images(images, T, S, is_training, stride)
        assert stride <= S // 2

        images = images.contiguous()
        images_ = images.reshape(B*T,3,H,W)
        padder = InputPadder(images_.shape)
        images_ = padder.pad(images_)[0]
        
        _, _, H_pad, W_pad = images_.shape
        C, H8, W8 = self.flow_dim, H_pad//8, W_pad//8
        C1 = 128
        
        # Feature Extraction or Retrieval from Cache
        if eval_dict is None:
            fmaps, ctxfeats, fmaps3d_detail, pms, points, masks, world_points, camera_poses = self.get_fmaps(images_, B, T, sw, is_training)
            fmaps = fmaps.to(dtype).reshape(B, T, C, H8, W8)
            fmaps3d_detail = fmaps3d_detail.to(dtype).reshape(B, T, self.flow3d_dim, H8, W8)
            ctxfeats = ctxfeats.to(dtype).reshape(B, T, C1, H8, W8)
            pms = pms.to(dtype).reshape(B, T, 3, H8, W8)
            
            dict1 = {
                'fmaps': fmaps, 'ctxfeats': ctxfeats, 
                'fmaps3d_detail': fmaps3d_detail, 'pms': pms, 
                'points': points, 'masks': masks
            }
        else:
            fmaps = eval_dict['fmaps']
            ctxfeats = eval_dict['ctxfeats']
            fmaps3d_detail = eval_dict['fmaps3d_detail']
            pms = eval_dict['pms']
            points = eval_dict['points']
            masks = eval_dict['masks']

            T_target = images_.shape[0]
            T_cur = fmaps.shape[1]

            if T_target <= T_cur:
                # Truncate if cached features are longer
                fmaps = fmaps[:, :T_target]
                ctxfeats = ctxfeats[:, :T_target]
                fmaps3d_detail = fmaps3d_detail[:, :T_target]
                pms = pms[:, :T_target]
                points = points[:T_target]
                masks = masks[:T_target]
            else:
                # Pad in time dimension if cached features are shorter
                pad_T = T_target - T_cur

                def pad_time(x, dim=1):
                    pad_shape = list(x.shape)
                    pad_shape[dim] = pad_T
                    pad_tensor = x[:, -1:].expand(*pad_shape)
                    return torch.cat([x, pad_tensor], dim=dim)

                fmaps = pad_time(fmaps)
                ctxfeats = pad_time(ctxfeats)
                fmaps3d_detail = pad_time(fmaps3d_detail)
                pms = pad_time(pms)
                points = torch.cat([points, points[-1:].expand(T_target - T_cur, *points.shape[1:])], dim=0)
                masks = torch.cat([masks, masks[-1:].expand(T_target - T_cur, *masks.shape[1:])], dim=0)
            
            dict1 = {
                'fmaps': fmaps, 'ctxfeats': ctxfeats, 
                'fmaps3d_detail': fmaps3d_detail, 'pms': pms, 
                'points': points, 'masks': masks
            }

        device = fmaps.device
        all_flow_preds = None
        all_visconf_preds = None
        all_flow3d_preds = None
        
        # --- Short Sequence (<= 2 frames) ---
        if T <= 2:
            all_flow_preds = []
            if tracking3d:
                all_flow3d_preds = []
            all_visconf_preds = []
            
            flows8 = torch.zeros((B, 2, H_pad//8, W_pad//8), dtype=dtype, device=device)
            flow3ds8 = torch.zeros((B, 3, H_pad//8, W_pad//8), dtype=dtype, device=device)
            visconfs8 = torch.zeros((B, 2, H_pad//8, W_pad//8), dtype=dtype, device=device)
                
            fmap_anchor = fmaps[:, 0]
            ctxfeat_anchor = ctxfeats[:, 0]
            pm_anchor = pms[:, 0]
            fmaps3d_detail_anchor = fmaps3d_detail[:, 0]
            
            flow_predictions, flow3d_predictions, visconf_predictions, flows8, flow3ds8, feats8 = self.forward_window_unified(
                fmap1_single=fmap_anchor, fmap2=fmaps[:,1:2], fmaps3d_detail1_single=fmaps3d_detail_anchor, fmaps3d_detail2=fmaps3d_detail[:,1:2],
                visconfs8=visconfs8, iters=iters, flow2ds8=flows8, flow3ds8=flow3ds8,
                cxt1_single=ctxfeat_anchor, cxt2=ctxfeats[:,1:2],
                pm1_single=pm_anchor.detach(), pm2=pms[:,1:2].detach(),
                is_training=is_training,
                tracking3d=tracking3d
            )
            
            unpad_flow_predictions = []
            if tracking3d:
                unpad_flow3d_predictions = []
            unpad_visconf_predictions = []
            
            for i in range(len(flow_predictions)):
                flow_predictions[i] = padder.unpad(flow_predictions[i])
                all_flow_preds.append(flow_predictions[i].reshape(B,2,H,W))
                if tracking3d:
                    flow3d_predictions[i] = padder.unpad(flow3d_predictions[i])
                    all_flow3d_preds.append(flow3d_predictions[i].reshape(B,3,H,W))
                visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                all_visconf_preds.append(visconf_predictions[i].reshape(B,2,H,W))
            
            full_flows = all_flow_preds[-1].reshape(B,2,H,W).detach().cpu()
            if tracking3d:
                full_flows3d = all_flow3d_preds[-1].reshape(B,3,H,W).cpu()
            full_visconfs = all_visconf_preds[-1].reshape(B,2,H,W).detach().cpu()
            points = padder.unpad(points)
            masks = padder.unpad(masks)
            world_points = padder.unpad(world_points)

            if tracking3d:
                return full_flows, full_visconfs, all_flow_preds, all_visconf_preds, full_flows3d, all_flow3d_preds, points, masks, dict1, world_points, camera_poses 
            else:
                return full_flows, full_visconfs, all_flow_preds, all_visconf_preds, dict1

        # --- Multiframe Tracking (T > 2) ---
        assert T > 2 
        
        if is_training:
            all_flow_preds = []
            if tracking3d:
                all_flow3d_preds = []
            all_visconf_preds = []
            
        # Store results in CPU to save GPU memory
        full_flows = torch.zeros((B,T,2,H,W), dtype=dtype, device='cpu')
        if tracking3d:
            full_flows3d = torch.zeros((B,T,3,H,W), dtype=dtype, device='cpu')
        full_visconfs = torch.zeros((B,T,2,H,W), dtype=dtype, device='cpu')

        fmap_anchor = fmaps[:, 0]
        ctxfeat_anchor = ctxfeats[:, 0]
        pm_anchor = pms[:, 0]
        fmaps3d_detail_anchor = fmaps3d_detail[:, 0]
        full_visited = torch.zeros((T,), dtype=torch.bool, device=device)

        for ii, ind in enumerate(indices):
            ara = np.arange(ind, ind+S)
            
            if ii == 0:
                flows8 = torch.zeros((B,S,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
                flow3ds8 = torch.zeros((B,S,3,H_pad//8,W_pad//8), dtype=dtype, device=device)
                visconfs8 = torch.zeros((B,S,2,H_pad//8,W_pad//8), dtype=dtype, device=device)

                fmaps2 = fmaps[:, ara]
                fmaps3d_detail2 = fmaps3d_detail[:, ara]
                ctxfeats2 = ctxfeats[:, ara]
                pms2 = pms[:, ara]
            else:
                # Reuse predictions from overlap region to maintain continuity
                flows8 = torch.cat([flows8[:,stride:stride+S//2], flows8[:,stride+S//2-1:stride+S//2].repeat(1,S//2,1,1,1)], dim=1)
                flow3ds8 = torch.cat([flow3ds8[:,stride:stride+S//2], flow3ds8[:,stride+S//2-1:stride+S//2].repeat(1,S//2,1,1,1)], dim=1)
                visconfs8 = torch.cat([visconfs8[:,stride:stride+S//2], visconfs8[:,stride+S//2-1:stride+S//2].repeat(1,S//2,1,1,1)], dim=1)
                
                fmaps2 = torch.cat([fmaps2[:,stride:stride+S//2], fmaps[:, ind+S//2:ind+S]], dim=1)
                fmaps3d_detail2 = torch.cat([fmaps3d_detail2[:,stride:stride+S//2], fmaps3d_detail[:, ind+S//2:ind+S]], dim=1)
                ctxfeats2 = torch.cat([ctxfeats2[:,stride:stride+S//2], ctxfeats[:, ind+S//2:ind+S]], dim=1)
                pms2 = torch.cat([pms2[:,stride:stride+S//2], pms[:, ind+S//2:ind+S]], dim=1)

            flows8 = flows8.reshape(B*S,2,H_pad//8,W_pad//8).detach()
            flow3ds8 = flow3ds8.reshape(B*S,3,H_pad//8,W_pad//8).detach()
            visconfs8 = visconfs8.reshape(B*S,2,H_pad//8,W_pad//8).detach()
            
            flow_predictions, flow3d_predictions, visconf_predictions, flows8, flow3ds8, feats8 = self.forward_window_unified(
                fmap1_single=fmap_anchor, fmap2=fmaps2, fmaps3d_detail1_single=fmaps3d_detail_anchor, fmaps3d_detail2=fmaps3d_detail2,
                visconfs8=visconfs8, iters=iters, flow2ds8=flows8, flow3ds8=flow3ds8,
                cxt1_single=ctxfeat_anchor, cxt2=ctxfeats2,
                pm1_single=pm_anchor.detach(), pm2=pms2.detach(),
                is_training=is_training,
                tracking3d=tracking3d
            )
            
            unpad_flow_predictions = []
            if tracking3d:
                unpad_flow3d_predictions = []
            unpad_visconf_predictions = []
            
            for i in range(len(flow_predictions)):
                flow_predictions[i] = padder.unpad(flow_predictions[i])
                unpad_flow_predictions.append(flow_predictions[i].reshape(B,S,2,H,W))
                if tracking3d:
                    flow3d_predictions[i] = padder.unpad(flow3d_predictions[i])
                    unpad_flow3d_predictions.append(flow3d_predictions[i].reshape(B,S,3,H,W))
                visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                unpad_visconf_predictions.append(visconf_predictions[i].reshape(B,S,2,H,W))

            current_visiting = torch.zeros((T,), dtype=torch.bool, device=device)
            current_visiting[ara] = True
            
            to_fill = current_visiting & (~full_visited)
            to_fill_sum = to_fill.sum().item()
            
            full_flows[:, to_fill] = unpad_flow_predictions[-1].reshape(B,S,2,H,W)[:, -to_fill_sum:].detach().cpu()
            if tracking3d:
                full_flows3d[:, to_fill] = unpad_flow3d_predictions[-1].reshape(B,S,3,H,W)[:, -to_fill_sum:].detach().cpu()
            full_visconfs[:, to_fill] = unpad_visconf_predictions[-1].reshape(B,S,2,H,W)[:, -to_fill_sum:].detach().cpu()
            full_visited |= current_visiting

            if is_training:
                all_flow_preds.append(unpad_flow_predictions)
                if tracking3d:
                    all_flow3d_preds.append(unpad_flow3d_predictions)
                all_visconf_preds.append(unpad_visconf_predictions)
            else:
                del unpad_flow_predictions
                if tracking3d:
                    del unpad_flow3d_predictions
                del unpad_visconf_predictions
                
            flows8 = flows8.reshape(B,S,2,H_pad//8,W_pad//8)
            flow3ds8 = flow3ds8.reshape(B,S,3,H_pad//8,W_pad//8)
            visconfs8 = visconfs8.reshape(B,S,2,H_pad//8,W_pad//8)
                
        if not is_training:
            full_flows = full_flows[:, :T_bak]
            if tracking3d:
                full_flows3d = full_flows3d[:, :T_bak]
            full_visconfs = full_visconfs[:, :T_bak]
        
        points = padder.unpad(points)    
        masks = padder.unpad(masks)    
        world_points = padder.unpad(world_points)    

        if tracking3d:
            return full_flows, full_visconfs, all_flow_preds, all_visconf_preds, full_flows3d, all_flow3d_preds, points, masks, dict1, world_points, camera_poses 
        else:
            return full_flows, full_visconfs, all_flow_preds, all_visconf_preds, dict1

    def forward_window_unified(self, fmap1_single, fmap2, visconfs8, iters=None, flow2ds8=None,
                                 flow3ds8=None, cxt1_single=None, cxt2=None, pm1_single=None,
                                 pm2=None, fmaps3d_detail1_single=None, fmaps3d_detail2=None, is_training=False, tracking3d=False):
        """
        Unified iterative flow update (RAFT-style).
        Handles both 2D and 3D flow updates using correlation volumes and recurrent units.

        Args:
            fmap1_single: Feature map of the anchor frame (t=0).
            fmap2: Feature maps of the target frames (t=1...S).
            visconfs8: Initial visibility confidence maps.
            iters: Number of GRU iterations.
            flow2ds8: Initial 2D flow estimates.
            flow3ds8: Initial 3D flow estimates.
            cxt1_single: Context features of the anchor frame.
            cxt2: Context features of the target frames.
            pm1_single: Point map (geometry) of the anchor frame.
            pm2: Point maps of the target frames.
            fmaps3d_detail*: 3D-specific feature maps.
            is_training: Training flag.
            tracking3d: Boolean flag to enable 3D tracking updates.
        """
        B, S, C_in, H8, W8 = fmap2.shape
        dtype, device = fmap2.dtype, fmap2.device
        
        # Expand anchor features to match the sequence length (B*S)
        fmap1 = fmap1_single.unsqueeze(1).repeat(1, S, 1, 1, 1).reshape(B * S, C_in, H8, W8).contiguous()
        fmap2_flat = fmap2.reshape(B * S, C_in, H8, W8).contiguous()

        pm1 = pm1_single.unsqueeze(1).repeat(1, S, 1, 1, 1).reshape(B * S, 3, H8, W8).contiguous()
        pm2 = pm2.reshape(B * S, 3, H8, W8).contiguous()

        cxt1 = cxt1_single.unsqueeze(1).repeat(1, S, 1, 1, 1).reshape(B * S, -1, H8, W8)
        cxt2 = cxt2.reshape(B * S, -1, H8, W8).contiguous()

        # Initialize Correlation Blocks (All-pairs correlation)
        fea_corr_fn = CorrBlock(fmap1, fmap2_flat, self.corr_levels, self.corr_radius)
        cxt_corr_fn = CorrBlock(cxt1, cxt2, self.corr_levels, self.corr_radius)
        flowfeat, ctxfeat = fmap1.clone(), cxt1.clone()
        
        # Generate coordinate grid
        coords1 = self.coords_grid(B * S, H8, W8, device=device, dtype=dtype)
        visconfs8 = visconfs8.reshape(B * S, 2, H8, W8).contiguous()

        # Add temporal embeddings to context features
        time_emb = self.fetch_time_embed(S, self.time_emb, ctxfeat.dtype, is_training).reshape(1, S, ctxfeat.shape[1], 1, 1).repeat(B, 1, 1, 1, 1)
        ctxfeat = ctxfeat + time_emb.reshape(B * S, -1, 1, 1)
        
        # Handle 3D specific features if tracking is enabled
        if tracking3d:
            fmaps3d_detail1 = fmaps3d_detail1_single.unsqueeze(1).repeat(1, S, 1, 1, 1).reshape(B * S, self.flow3d_dim, H8, W8).contiguous()
            time_emb3d = self.fetch_time_embed(S, self.time_emb3d, fmaps3d_detail1.dtype, is_training).reshape(1, S, fmaps3d_detail1.shape[1], 1, 1).repeat(B, 1, 1, 1, 1)
            fmaps3d_detail1 += time_emb3d.reshape(B * S, -1, 1, 1)
            fmaps3d_detail2 = fmaps3d_detail2.reshape(B * S, self.flow3d_dim, H8, W8).contiguous()

        flow2d_predictions, visconf_predictions = [], []
        flow3d_predictions = [] if tracking3d else None
        
        # --- Iterative Update Loop (GRU) ---
        for itr in range(iters):
            flow2ds8, flow3ds8 = flow2ds8.detach(), flow3ds8.detach()
            coords2 = (coords1 + flow2ds8).detach()
            
            # 1. Look up correlations using current flow estimates
            fea_corr = fea_corr_fn(coords2).to(dtype)
            cxt_corr = cxt_corr_fn(coords2).to(dtype)
            
            # Encode current flow motion
            motion2d = misc.posenc(flow2ds8.permute(0,2,3,1).reshape(B,S,-1,2),0,10).reshape(B*S,H8,W8,-1).permute(0,3,1,2).to(dtype)
            
            if tracking3d:
                # 3D Point correlation
                pm_corr_fn = CorrBlock((pm1 + flow3ds8).detach(), pm2, self.corr_levels, self.corr_radius, mode='nearest')
                pm_corr = pm_corr_fn(coords2).to(dtype)
                motion3d = misc.posenc(flow3ds8.permute(0,2,3,1).reshape(B,S,-1,3),0,10).reshape(B*S,H8,W8,-1).permute(0,3,1,2).to(dtype)
                
                # Sample features based on current 2D flow
                grid_pre = torch.stack((((coords1 + flow2ds8)[:,0]/(W8-1))*2-1, ((coords1 + flow2ds8)[:,1]/(H8-1))*2-1), -1).detach()
                sampled_feature_pre = F.grid_sample(fmaps3d_detail2, grid_pre, mode='bilinear', align_corners=True, padding_mode='border')
                sampled_pm_pre = F.grid_sample(pm2, grid_pre, mode='bilinear', align_corners=True, padding_mode='border')
 
            # 2. Update 2D Flow
            flowfeat = self.flow_update_block(flowfeat, ctxfeat, visconfs8, fea_corr, cxt_corr, motion2d, S)
            flow2d_update = self.flow_2d_head(flowfeat)
            visconf_update = self.flow_visconf_head(flowfeat)
            flow2ds8 = flow2ds8 + flow2d_update
            
            # 3. Update 3D Flow
            if tracking3d:
                grid_post = torch.stack((((coords1 + flow2ds8)[:,0]/(W8-1))*2-1, ((coords1 + flow2ds8)[:,1]/(H8-1))*2-1), -1).detach()
                sampled_feature = F.grid_sample(fmaps3d_detail2, grid_post, mode='bilinear', align_corners=True, padding_mode='border')
                sampled_pm = F.grid_sample(pm2, grid_post, mode='bilinear', align_corners=True, padding_mode='border')
                
                flow3d_update, fmaps3d_detail1 = self.flow3d_head(flowfeat.detach(), fmaps3d_detail1, sampled_feature_pre, sampled_feature, sampled_pm - sampled_pm_pre, pm_corr, motion3d, S)
                flow3ds8 = flow3ds8 + flow3d_update
            
            # 4. Predict Upsampling Weights and Upsample
            temperature = 0.25 # Scaling factor for updates
            weight_update = temperature * self.flow_upsample_weight(flowfeat)
            if tracking3d:
                weight_update_3d = temperature * self.flow_3d_upsample_weight(fmaps3d_detail1)
            visconfs8 = visconfs8 + visconf_update

            flow2d_predictions.append(self.upsample_data(flow2ds8, weight_update, dim1=2))
            if tracking3d:
                flow3d_predictions.append(self.upsample_3d_data(flow3ds8, weight_update_3d, dim1=3))
            visconf_predictions.append(self.upsample_data(visconfs8, weight_update, dim1=2))
            
            torch.cuda.empty_cache()

        return flow2d_predictions, flow3d_predictions, visconf_predictions, flow2ds8, flow3ds8, flowfeat

    # --- Helper Methods ---

    def fetch_time_embed(self, t, time_emb1, dtype, is_training=False):
        """Retrieves or interpolates temporal position embeddings."""
        S = time_emb1.shape[1]
        if t == S:
            return time_emb1.to(dtype)
        elif t == 1:
            # Randomly sample time embedding during training for robustness
            ind = np.random.choice(S) if is_training else 1
            return time_emb1[:, ind:ind + 1].to(dtype)
        else:
            # Interpolate embeddings if sequence length doesn't match
            time_emb = time_emb1.float()
            time_emb = F.interpolate(time_emb.permute(0, 2, 1), size=t, mode="linear").permute(0, 2, 1)
            return time_emb.to(dtype)

    def get_T_padded_images(self, images, T, S, is_training, stride=None, pad=True):
        """Calculates necessary padding for time dimension to fit window size S."""
        B, T, C, H, W = images.shape
        indices = None
        if T > 2:
            step = S // 2 if stride is None else stride
            indices = []
            start = 0
            # Create sliding window indices
            while start + S < T:
                indices.append(start)
                start += step
            indices.append(start)
            Tpad = indices[-1]+S-T
            
            if pad:
                if is_training:
                    assert Tpad == 0
                else:
                    # Pad the last frame to fill the window
                    images = images.reshape(B,1,T,C*H*W)
                    if Tpad > 0:
                        padding_tensor = images[:,:,-1:,:].expand(B,1,Tpad,C*H*W)
                        images = torch.cat([images, padding_tensor], dim=2)
                    images = images.reshape(B,T+Tpad,C,H,W)
                    T = T+Tpad
        else:
            assert T == 2
        return images, T, indices

    def get_fmaps(self, images_, B, T, sw, is_training):
        """
        Extract feature maps for all frames.
        Processes frames in chunks to manage memory usage.
        """
        _, _, H_pad, W_pad = images_.shape 

        H8, W8 = H_pad//8, W_pad//8
        C = self.flow_dim
        C1 = 128
        if self.use_model in ['pi3', 'depthanythingv3']:
            fmaps_chunk_size = 256
        else:
            fmaps_chunk_size = 64
        images = images_.reshape(B,T,3,H_pad,W_pad)
        fmaps = []
        fmaps3d_detail = []
        ctxfeats = []
        pms = []
        points = []
        masks = []
        world_points = []
        camera_poses = []
        # Iterate through chunks of frames
        for t in range(0, T, fmaps_chunk_size):
            images_chunk = images[:, t : t + fmaps_chunk_size]
            # Extract features and geometry (points/masks)
            fmaps_chunk, ctxfeats_chunk, fmaps3d_detail_chunk, pms_chunk, points_chunk, masks_chunk, world_points_chunk, camera_poses_chunk = self.forward_point(images_chunk, current_batch_size=B, for_flow=True)
            
            fmaps.append(fmaps_chunk.reshape(B, -1, C, H8, W8))
            ctxfeats.append(ctxfeats_chunk.reshape(B, -1, C1, H8, W8))
            pms.append(pms_chunk.reshape(B, -1, 3, H8, W8))
            points.append(points_chunk.reshape(B, -1, 3, H_pad, W_pad))
            masks.append(masks_chunk.reshape(B, -1, 1, H_pad, W_pad))
            world_points.append(world_points_chunk.reshape(B, -1, 3, H_pad, W_pad))
            camera_poses.append(camera_poses_chunk.reshape(B, -1, 4, 4))
            fmaps3d_detail.append(fmaps3d_detail_chunk.reshape(B, -1, self.flow3d_dim, H8, W8))
            
        # Concatenate all chunks
        fmaps = torch.cat(fmaps, dim=1).reshape(-1, C, H8, W8)
        fmaps3d_detail = torch.cat(fmaps3d_detail, dim=1).reshape(-1, self.flow3d_dim, H8, W8)
        ctxfeats = torch.cat(ctxfeats, dim=1).reshape(-1, C1, H8, W8)
        pms = torch.cat(pms, dim=1).reshape(-1, 3, H8, W8)
        points = torch.cat(points, dim=1).reshape(-1, 3, H_pad, W_pad)
        masks = torch.cat(masks, dim=1).reshape(-1, 1, H_pad, W_pad)
        world_points = torch.cat(world_points, dim=1).reshape(-1, 3, H_pad, W_pad)
        camera_poses = torch.cat(camera_poses, dim=1).reshape(-1, 4, 4)
        return fmaps, ctxfeats, fmaps3d_detail, pms, points, masks, world_points, camera_poses

    def upsample_data(self, flow, mask, dim1):
        """
        Upsample low-resolution flow using learned convex mask.
        Standard RAFT upsampling.
        """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        
        # Unfold flow to get neighbors
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, dim1, 9, 1, 1, H, W)
        
        # Weighted sum using the mask
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, dim1, 8 * H, 8 * W).to(flow.dtype)

    def upsample_3d_data(self, flow, mask, dim1):
        """
        Upsample low-resolution 3D flow using learned convex mask.
        Similar to 2D upsampling but for 3D vectors.
        """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, dim1, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, dim1, 8 * H, 8 * W).to(flow.dtype)

    def coords_grid(self, batch, ht, wd, device, dtype):
        """Generates a meshgrid of coordinates."""
        coords = torch.meshgrid(torch.arange(ht, device=device, dtype=dtype), torch.arange(wd, device=device, dtype=dtype), indexing='ij')
        coords = torch.stack(coords[::-1], dim=0)
        return coords[None].repeat(batch, 1, 1, 1)

    def forward_pure_point(
        self, 
        image: torch.Tensor, 
        num_tokens: int, 
        current_batch_size: int = 4
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to estimate point clouds from images.
        
        Args:
            image: Input tensor of shape (B*T, 3, H, W).
            num_tokens: Target number of tokens for the backbone.
            current_batch_size: The original batch size B (before flattening time T).
        """
        original_height, original_width = image.shape[-2:]
        
        # 1. Resize to expected resolution defined by num_tokens
        # We ensure the resolution is a multiple of the patch size (14)
        resize_factor = ((num_tokens * 14 ** 2) / (original_height * original_width)) ** 0.5
        resized_width = int(original_width * resize_factor)
        resized_height = int(original_height * resize_factor)
        
        # Initial resize
        image_resized = F.interpolate(
            image, (resized_height, resized_width), 
            mode="bicubic", align_corners=False, antialias=True
        )
    
        # Normalize image for DINOv2
        image_norm = (image_resized - self.image_mean) / self.image_std
        
        # Snap to patch grid (multiple of 14)
        image_14 = F.interpolate(
            image_norm, 
            (resized_height // 14 * 14, resized_width // 14 * 14), 
            mode="bilinear", align_corners=False, antialias=True
        )

        # 2. Extract features from backbone
        # features is a list of tuples: [(patch_tokens, cls_token), ...]
        features = self.backbone.get_intermediate_layers(
            image_14, self.intermediate_layers, return_class_token=True
        )
        
        # 3. Prepare tokens for Aggregator
        # Expand camera and register tokens: (B, T, ...)
        camera_token = self.camera_token.expand(current_batch_size, image.shape[0] // current_batch_size, *self.camera_token.shape[2:])
        register_token = self.register_token.expand(current_batch_size, image.shape[0] // current_batch_size, *self.register_token.shape[2:])
        
        # Reshape backbone features to (B, T, N, C)
        last_feat = features[-1][0]
        backbone_tokens = last_feat.reshape(
            current_batch_size, -1, last_feat.shape[-2], last_feat.shape[-1]
        )

        # Concatenate: [Camera, Register, Backbone]
        tokens = torch.cat([camera_token, register_token, backbone_tokens], dim=2)

        # 4. Aggregate Features
        global_features = self.aggregator(tokens, image_norm, patch_start_idx=self.patch_start_idx)
        
        # Inject aggregated features back into the backbone feature list for the head
        # Convert tuples to lists to allow modification
        features = [list(f) for f in features]  
        for i in range(len(features)):
            # Extract relevant tokens and reshape back to flattened batch (B*T, N, C)
            new_feat = global_features[2 * i + 1][:, :, self.patch_start_idx:]
            features[i][0] = new_feat.reshape(-1, features[-1][0].shape[-2], features[-1][0].shape[-1])

        # 5. Predict Points and Mask
        points, mask = self.head(features, image_norm)
        
        # 6. Post-process (Resize back to original resolution)
        with torch.autocast(device_type=image.device.type, dtype=torch.float32):
            points = F.interpolate(
                points, (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            mask = F.interpolate(
                mask, (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            
            # (B*T, 3, H, W) -> (B*T, H, W, 3)
            points = points.permute(0, 2, 3, 1)
            mask = mask.squeeze(1)
            
            # Remap points (e.g., for numerical stability)
            points = self._remap_points(points)

        # Reshape to (B, T, H, W, C)
        return {
            'points': points.reshape(current_batch_size, -1, points.shape[-3], points.shape[-2], points.shape[-1]), 
            'mask': mask.reshape(current_batch_size, -1, mask.shape[-2], mask.shape[-1]),
        }

    # --- Inference Wrappers ---
    @torch.inference_mode()
    def infer_pure_point(
        self, 
        image: torch.Tensor, 
        fov_x: Union[Number, torch.Tensor] = None,
        resolution_level: int = 9,
        num_tokens: Optional[int] = None,
        apply_mask: bool = False,
        force_projection: bool = True,
        use_fp16: bool = True,
        current_batch_size: int = 4,
        local: bool = False,
        no_shift: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        User-friendly inference function for point cloud estimation.

        Args:
            image: Input tensor of shape (B, 3, H, W) or (3, H, W).
            fov_x: Horizontal camera FoV in degrees. If None, inferred from prediction.
            resolution_level: Integer [0-9] controlling detail level (maps to num_tokens).
            num_tokens: Explicit number of tokens [1200, 2500]. Overrides resolution_level.
            apply_mask: If True, masks invalid points with inf.
            force_projection: If True, re-projects points using depth and intrinsics.
            use_fp16: Use mixed precision for inference.
            current_batch_size: Batch size for the temporal dimension grouping.
            local: If True, uses local focal length recovery logic.
            no_shift: If True, disables depth shifting during recovery.

        Returns:
            A list of dictionaries (one per batch item), each containing:
            - 'points': (T, H, W, 3) or (H, W, 3)
            - 'depth': (T, H, W) or (H, W)
            - 'intrinsics': (T, 3, 3) or (3, 3)
            - 'mask': (T, H, W) or (H, W)
        """
        # Handle input dimensions
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        
        image = image.to(dtype=self.dtype, device=self.device)
        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        # Determine token count
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
        
        # Run Forward Pass
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_fp16 and self.dtype != torch.float16):
            # Note: Calling forward_pure_point directly to match the provided snippet context
            output = self.forward_pure_point(image, num_tokens, current_batch_size=current_batch_size)
        
        points, mask = output['points'], output['mask']
        results_list = []

        # Post-process in FP32
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            # Ensure float32
            points = points.float()
            mask = mask.float()
            if isinstance(fov_x, torch.Tensor):
                fov_x = fov_x.float()

            mask_binary = mask > self.mask_threshold

            # --- Geometry Recovery ---
            if local:
                # Local Mode (Usually for single-frame or independent processing)
                if fov_x is None:
                    focal, shift = recover_focal_shift(points, mask_binary)
                else:
                    focal_val = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device)) / 2)
                    if focal_val.ndim == 0:
                        focal_val = focal_val[None].expand(points.shape[0])
                    focal = focal_val
                    _, shift = recover_focal_shift(points, mask_binary, focal=focal)
                
                # Construct Intrinsics
                fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
                fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
                intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
                
                depth = points[..., 2] + shift[..., None, None]

            else:
                # Global Mode (Usually for video sequences, consistent intrinsics)
                if fov_x is None:
                    if no_shift:
                        focal = recover_global_focal(points, mask_binary)
                        shift = torch.zeros_like(points[..., 0, 0, 0]) # Dummy shift
                    else:
                        focal, shift = recover_global_focal_shift(points, mask_binary)
                else:
                    focal_val = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device)) / 2)
                    if focal_val.ndim == 0:
                        focal_val = focal_val[None].expand(points.shape[0])
                    focal = focal_val
                    _, shift = recover_global_focal_shift(points, mask_binary, focal=focal)

                # Construct Intrinsics (Repeat for temporal dim)
                fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
                fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
                # Shape: (B, T, 3, 3)
                intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).repeat(1, points.shape[1], 1, 1)
                
                if no_shift:
                    depth = points[..., 2]
                else:
                    # Broadcast shift: (B, T, 1, 1)
                    depth = points[..., 2] + shift[..., None, None, None].repeat(1, points.shape[1], 1, 1)

            # --- Projection & Masking ---
            
            # Recompute points from depth to ensure geometric consistency
            if force_projection:
                points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)
            else:
                # Just apply the shift to Z
                if not no_shift:
                    # Create shift vector (0, 0, shift)
                    shift_vec = torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)
                    # Broadcast to (..., H, W, 3)
                    if local:
                         points = points + shift_vec[..., None, None, :]
                    else:
                         points = points + shift_vec[..., None, None, None, :].repeat(1, points.shape[1], 1, 1, 1)

            # Apply validity mask
            if apply_mask:
                points = torch.where(mask_binary[..., None], points, torch.tensor(float('inf'), device=points.device))
                depth = torch.where(mask_binary, depth, torch.tensor(float('inf'), device=depth.device))

            results_list.append({
                'points': points,
                'intrinsics': intrinsics,
                'depth': depth,
                'mask': mask_binary,
            })

        # Remove batch dimension if input was single image
        if omit_batch_dim:
            results_list = [{k: v.squeeze(0) for k, v in dic.items()} for dic in results_list]

        return results_list

    @torch.inference_mode()
    def infer(self, images, iters=4, sw=None, is_training=False, window_len=None, stride=None, tracking3d=False, apply_mask: bool = False,
        force_projection: bool = True, use_fp16: bool = True, current_batch_size: int = 4,
        local: bool = False, no_shift: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Standard inference method.
        Runs the model and post-processes outputs to return metric 3D points and camera parameters.
        
        Returns:
            Dict containing:
            - points: 3D point cloud.
            - intrinsics: Estimated camera intrinsics.
            - depth: Depth maps.
            - mask: Validity masks.
            - flow_2d: 2D optical flow.
            - flow_3d: 3D scene flow.
        """
        # Run forward pass
        flows_e, visconf_maps_e, _, _, flows3d_e, _, points, mask, _, world_points, camera_poses = self.forward_sliding(
            images, iters=iters, sw=sw, is_training=is_training, tracking3d=tracking3d
        )
        B, T, C, H, W = images.shape
        aspect_ratio = W / H
        
        # Create grid for coordinate conversion
        grid_xy = holi4d.utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float() # 1,H*W,2
        grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W) # 1,1,2,H,W
        
        # Restore FP32 for coordinate calculations and convert flow to absolute coordinates
        flow2d = flows_e.to(torch.float32).cuda() + grid_xy 
        visconf_maps_e = visconf_maps_e.to(torch.float32)
        flow3d = flows3d_e.to(torch.float32).cuda() + points[None, 0:1]
        flow3d = flow3d.permute(0, 1, 3, 4, 2)
        
        return_dict = []
        points = points[None].permute(0, 1, 3, 4, 2)[:, :T]
        mask = mask[None].permute(0, 1, 3, 4, 2)[..., 0][:, :T]
        world_points = world_points[None].permute(0, 1, 3, 4, 2)[:, :T]
        flow2d_c = flow2d.clone()
        flow2d_c = flow2d_c.permute(0, 1, 3, 4, 2)
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            points, mask = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [points, mask])
            mask_binary = mask > self.mask_threshold
            
            # Recover camera intrinsics and global scale/shift from points
            focal, shift = recover_global_focal_shift(points, mask_binary)

            fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
            fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).repeat(1, flow3d.shape[1], 1, 1)
            depth = points[..., 2] + shift[..., None, None, None].repeat(1, points.shape[1], 1, 1)

            # Project depth to 3D points
            if force_projection:
                points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics, use_ray=False)
            else:
                points = points + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

            if apply_mask:
                points = torch.where(mask_binary[..., None], points, torch.inf)
                world_points = torch.where(mask_binary[..., None], world_points, torch.inf)
                depth = torch.where(mask_binary, depth, torch.inf)

            return_dict.append({'points': points, 'intrinsics': intrinsics, 'depth': depth, 'mask': mask_binary, 'world_points': world_points, 'camera_poses': camera_poses})
            
            # Process 3D Flow / Forward Depth
            forward_depth = flow3d[..., 2] + shift[..., None, None, None].repeat(1, flow3d.shape[1], 1, 1)
            
            if force_projection:
                # Unproject 2D flow + depth to get 3D flow points
                flow2d_c[..., 0] /= flow2d_c.shape[-2]
                flow2d_c[..., 1] /= flow2d_c.shape[-3]
                points_f = utils3d.torch.unproject_cv(flow2d_c, forward_depth, intrinsics=intrinsics[..., None, :, :], use_ray=False)
            else:
                points_f = flow3d + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

            # Restore flow scaling
            flow2d_c[..., 0] *= flow2d_c.shape[-2]
            flow2d_c[..., 1] *= flow2d_c.shape[-3]
            
            return_dict.append({
                'flow_3d': points_f, 'flow_2d': flow2d, 'visconf_maps_e': visconf_maps_e, 
                'intrinsics': intrinsics, 'depth': forward_depth, 'mask': mask_binary
            })

        return return_dict

    @torch.inference_mode()
    def evaluation(self, images, iters=4, sw=None, is_training=False, window_len=None, stride=None, tracking3d=False, apply_mask: bool = False,
        force_projection: bool = True, use_fp16: bool = True, current_batch_size: int = 4,
        local: bool = False, no_shift: bool = False, eval_dict=None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluation method similar to infer but returns the eval_dict for caching features.
        Useful when evaluating on the same sequence multiple times.
        """
        flows_e, visconf_maps_e, _, _, flows3d_e, _, points, mask, eval_dict, _, _ = self.forward_sliding(
            images, iters=iters, sw=sw, is_training=is_training, tracking3d=tracking3d, eval_dict=eval_dict
        )
        B, T, C, H, W = images.shape
        aspect_ratio = W / H
        grid_xy = holi4d.utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float()
        grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W)
        flow2d = flows_e.to(torch.float32).cuda() + grid_xy 
        visconf_maps_e = visconf_maps_e.to(torch.float32)
        flow3d = flows3d_e.to(torch.float32).cuda() + points[None, 0:1]
        flow3d = flow3d.permute(0, 1, 3, 4, 2)
        
        return_dict = []
        points = points[None].permute(0, 1, 3, 4, 2)[:, :T]
        mask = mask[None].permute(0, 1, 3, 4, 2)[..., 0][:, :T]
        flow2d_c = flow2d.clone()
        flow2d_c = flow2d_c.permute(0, 1, 3, 4, 2)

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            points, mask = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [points, mask])
            mask_binary = mask > self.mask_threshold
            focal, shift = recover_global_focal_shift(points, mask_binary)

            fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
            fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).repeat(1, flow3d.shape[1], 1, 1)
            depth = points[..., 2] + shift[..., None, None, None].repeat(1, points.shape[1], 1, 1)

            if force_projection:
                points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics, use_ray=False)
            else:
                points = points + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

            if apply_mask:
                points = torch.where(mask_binary[..., None], points, torch.inf)
                depth = torch.where(mask_binary, depth, torch.inf)

            return_dict.append({'points': points, 'intrinsics': intrinsics, 'depth': depth, 'mask': mask_binary})
            
            forward_depth = flow3d[..., 2] + shift[..., None, None, None].repeat(1, flow3d.shape[1], 1, 1)
            if force_projection:
                flow2d_c[..., 0] /= flow2d_c.shape[-2]
                flow2d_c[..., 1] /= flow2d_c.shape[-3]
                points_f = utils3d.torch.unproject_cv(flow2d_c, forward_depth, intrinsics=intrinsics[..., None, :, :], use_ray=False)
            else:
                points_f = flow3d + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

            flow2d_c[..., 0] *= flow2d_c.shape[-2]
            flow2d_c[..., 1] *= flow2d_c.shape[-3]
            return_dict.append({
                'flow_3d': points_f, 'flow_2d': flow2d, 'visconf_maps_e': visconf_maps_e, 
                'intrinsics': intrinsics, 'depth': forward_depth, 'mask': mask_binary
            })

        return return_dict, eval_dict

    
    @torch.inference_mode()
    def infer_pair(self, images, iters=4, sw=None, is_training=False, window_len=None, stride=None, tracking3d=False, apply_mask: bool = False,
        force_projection: bool = True, use_fp16: bool = True, current_batch_size: int = 4,
        local: bool = False, no_shift: bool = False, aligned_scene_flow: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference optimized for pairwise estimation.
        Uses forward_sliding1 which implements memory optimization (chunking) for processing pairs.
        """
        # Run forward pass using the memory-optimized sliding window
        flows_e, visconf_maps_e, _, _, flows3d_e, _, points, mask, world_points, camera_poses = self.forward_sliding1(
            images, iters=iters, sw=sw, is_training=is_training, tracking3d=tracking3d
        )
        B, T, C, H, W = images.shape
        aspect_ratio = W / H
        grid_xy = holi4d.utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float()
        grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W)
        
        # Convert flow to absolute coordinates
        flow2d = flows_e[None].to(torch.float32).cuda() + grid_xy 
        visconf_maps_e = visconf_maps_e[None].to(torch.float32)
        flow3d = flows3d_e.to(torch.float32).cuda()[None] + points[None, 0:-1]
        flow3d = flow3d.permute(0, 1, 3, 4, 2)
        
        return_dict = []
        points = points[None].permute(0, 1, 3, 4, 2)[:, :T]
        world_points = world_points[None].permute(0, 1, 3, 4, 2)[:, :T]
        mask = mask[None].permute(0, 1, 3, 4, 2)[..., 0][:, :T]
        flow2d_c = flow2d.clone()
        flow2d_c = flow2d_c.permute(0, 1, 3, 4, 2)
        if aligned_scene_flow:
            flow3d = get_aligned_scene_flow_temporal(flow2d_c, flow3d, points, visconf_maps_e.cuda().permute(0, 1, 3, 4, 2), mode='align_dir')

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            points, mask = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [points, mask])
            mask_binary = mask > self.mask_threshold
            focal, shift = recover_global_focal_shift(points, mask_binary)

            fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
            fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).repeat(1, points.shape[1], 1, 1)
            depth = points[..., 2] + shift[..., None, None, None].repeat(1, points.shape[1], 1, 1)

            if force_projection:
                points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics, use_ray=False)
            else:
                points = points + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

            if apply_mask:
                points = torch.where(mask_binary[..., None], points, torch.inf)
                world_points = torch.where(mask_binary[..., None], world_points, torch.inf)
                depth = torch.where(mask_binary, depth, torch.inf)

            return_dict.append({'points': points, 'intrinsics': intrinsics, 'depth': depth, 'mask': mask_binary, 'world_points': world_points, 'camera_poses': camera_poses})
            
            forward_depth = flow3d[..., 2] + shift[..., None, None, None].repeat(1, flow3d.shape[1], 1, 1)
            
            if force_projection:
                flow2d_c[..., 0] /= flow2d_c.shape[-2]
                flow2d_c[..., 1] /= flow2d_c.shape[-3]
                points_f = utils3d.torch.unproject_cv(flow2d_c, forward_depth, intrinsics=intrinsics[:, :-1][..., None, :, :], use_ray=False)
            else:
                points_f = flow3d + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

            if apply_mask:
                # print(mask_binary.shape)
                points_f = torch.where(mask_binary[:, :-1, :, :, None], points_f, torch.inf)
                forward_depth = torch.where(mask_binary[:, :-1], forward_depth, torch.inf)
            
            flow2d_c[..., 0] *= flow2d_c.shape[-2]
            flow2d_c[..., 1] *= flow2d_c.shape[-3]
            return_dict.append({
                'flow_3d': points_f, 'flow_2d': flow2d, 'visconf_maps_e': visconf_maps_e, 
                'intrinsics': intrinsics, 'depth': forward_depth, 'mask': mask_binary
            })

        return return_dict
