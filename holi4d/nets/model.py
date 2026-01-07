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
    normalized_view_plane_uv, recover_focal_shift, 
    recover_global_focal, recover_global_focal_shift
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
            nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, 
                padding=1, padding_mode=padding_mode
            ),
            nn.GroupNorm(
                hidden_channels // 32 if norm == 'group_norm' else 1, 
                hidden_channels
            ),
            activation_cls(),
            nn.Conv2d(
                hidden_channels, out_channels, kernel_size=3, 
                padding=1, padding_mode=padding_mode
            )
        )

        # Skip connection (projection if dimensions change)
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )
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
    Convert depth maps and camera parameters to 3D points.

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
            nn.Conv2d(
                in_channels=dim_in, out_channels=dim_proj, 
                kernel_size=1, stride=1, padding=0
            ) 
            for _ in range(num_features)
        ])

        # Upsampling blocks (ConvTranspose + Residual Blocks)
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2, out_ch), # +2 for UV coordinates
                *(ResidualConvBlock(
                    out_ch, out_ch, dim_times_res_block_hidden * out_ch, 
                    activation="relu", norm=res_block_norm
                  ) for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        # Output prediction blocks (e.g., one for points, one for mask)
        self.output_block = nn.ModuleList([
            self._make_output_block(
                dim_upsample[-1] + 2, dim_out_, dim_times_res_block_hidden, 
                last_res_blocks, last_conv_channels, last_conv_size, res_block_norm,
            ) for dim_out_ in dim_out
        ])

    def _make_upsampler(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Creates a transposed convolution block for upsampling."""
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, 
                stride=1, padding=1, padding_mode='replicate'
            )
        )
        # Initialize weights specifically for the first layer to behave like 
        # bilinear upsampling initially
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(self, dim_in, dim_out, dim_times_res_block_hidden, 
                           last_res_blocks, last_conv_channels, last_conv_size, 
                           res_block_norm) -> nn.Sequential:
        """Creates the final convolution block to project to output dimensions."""
        return nn.Sequential(
            nn.Conv2d(
                dim_in, last_conv_channels, kernel_size=3, 
                stride=1, padding=1, padding_mode='replicate'
            ),
            *(ResidualConvBlock(
                last_conv_channels, last_conv_channels, 
                dim_times_res_block_hidden * last_conv_channels, 
                activation='relu', norm=res_block_norm
              ) for _ in range(last_res_blocks)),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                last_conv_channels, dim_out, kernel_size=last_conv_size, 
                stride=1, padding=last_conv_size // 2, padding_mode='replicate'
            ),
        )

    def forward(
        self, 
        hidden_states: List[Tuple[torch.Tensor, torch.Tensor]], 
        image: torch.Tensor
    ) -> List[torch.Tensor]:
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
            uv = normalized_view_plane_uv(
                width=x.shape[-1], height=x.shape[-2], 
                aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            
            # Concatenate UV and process
            x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

        # Final interpolation to match image resolution
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        
        # Inject UV coordinates again before output
        uv = normalized_view_plane_uv(
            width=x.shape[-1], height=x.shape[-2], 
            aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device
        )
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        # Generate outputs (Points and Mask)
        if isinstance(self.output_block, nn.ModuleList):
            output = [
                torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) 
                for block in self.output_block
            ]
        else:
            output = torch.utils.checkpoint.checkpoint(
                self.output_block, x, use_reentrant=False
            )

        return output


def get_aligned_scene_flow_temporal(
    flow2d_c, flow3d, points, visconf_maps_e, mode='align_direction'
):
    """
    Align predicted scene flow with geometric flow derived from optical flow.

    Args:
        flow2d_c: [B, T-1, H, W, 2]
        flow3d:   [B, T-1, H, W, 3] (Predicted scene flow to be aligned)
        points:   [B, T, H, W, 3]   (Point sequence)
        visconf_maps_e: [B, T-1, H, W, 2]
        mode:     'align_geo' or 'align_dir'

    Returns:
        s_aligned_final: [B, T-1, H, W, 3]
    """
    B, T_minus_1, H, W, _ = flow2d_c.shape

    # -----------------------------------------------------------
    # 1. Data preparation: temporal slicing and flattening
    # -----------------------------------------------------------

    # points[:, t] is the source point cloud
    # points[:, t+1] is the target point cloud
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
        padding_mode='zeros'
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

    # Additional geometric validity checks:
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

    # Iterate over the N samples to solve alignment per frame
    for i in range(N):
        mask_i = final_mask_flat[i]
        try:
            # Downsample mask for efficiency
            _, lr_mask, lr_index = mask_aware_nearest_resize(
                None, mask_i, (32, 32), return_index=True
            )
            s_pred_valid = s_pred_flat[i][lr_index][lr_mask]
            s_geo_valid = s_geo_flat[i][lr_index][lr_mask]

            # Solve for scale and shift to align predictions with geometric flow
            scale, shift = align_points_scale_xyz_shift(
                s_pred_valid,
                s_geo_valid,
                1 / torch.ones_like(s_geo_valid.norm(dim=-1) + 1e-6),
                exp=20
            )

            # Apply the alignment parameters to the full-resolution scene flow
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
        use_model: Literal['base', 'pi3', 'depthanythingv3'] = 'base',
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
            features = self.backbone.get_intermediate_layers(
                image_14, self.intermediate_layers, return_class_token=True
            )
            
            # Prepare tokens (Camera + Register + Patch tokens)
            camera_token = self.camera_token.expand(
                current_batch_size, image.shape[0]//current_batch_size, *self.camera_token.shape[2:]
            )
            register_token = self.register_token.expand(
                current_batch_size, image.shape[0]//current_batch_size, *self.register_token.shape[2:]
            )
            tokens = torch.cat([
                camera_token, 
                register_token, 
                features[-1][0].reshape(
                    current_batch_size, -1, features[-1][0].shape[-2], features[-1][0].shape[-1]
                )
            ], dim=2)
            
            # Aggregate global features
            global_features = self.aggregator(tokens, image, patch_start_idx=self.patch_start_idx)
            
            # Update features with aggregated information
            features = [list(f) for f in features]
            for i in range(len(features)):
                new_feat = global_features[2*i+1][:, :, self.patch_start_idx:].reshape(
                    -1, features[-1][0].shape[-2], features[-1][0].shape[-1]
                )
                features[i][0] = new_feat
            
            # Predict geometry (points and mask)
            points, mask = self.head(features, image)
            
            if self.use_model == 'pi3':
                results = self.pi3(image_14[None])
                points = results['local_points'][0].permute(0, -1, 1, 2)
                world_points = results['points'][0].permute(0, -1, 1, 2)
                camera_poses = results['camera_poses'][0]
                conf = results['conf'][0].permute(0, -1, 1, 2)
                mask = torch.sigmoid(conf)
                points = points / 10
                world_points = world_points / 10
                camera_poses[..., :3, 3] /= 10
                
            elif self.use_model == 'depthanythingv3':
                results = self.dav3.inference_v2(image_14[None])
                world_points, points, camera_poses = process_geometry(
                    results["depth"], results["extrinsics"], results["intrinsics"]
                )
                mask = results["depth_conf"].transpose(0, 1)
                max_elements = 1_000_000
                stride = max(1, mask.numel() // max_elements)

                # Use slicing to subsample
                sample = mask.view(-1)[::stride]

                self.mask_threshold = torch.quantile(sample.cpu(), 0.05)
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
                flow3d_features = self.flow_aggregator3d(
                    flow3d_features, 
                    image_14, 
                    patch_start_idx=self.patch_start_idx
                )
            
            # Context extraction using ConvNeXt
            ctxfeat = self.ctx_encoder(image)
            ctxfeat = self.dot_conv(ctxfeat)
            
            flow_H, flow_W = original_height // 8, original_width // 8
            
            # Interpolate features to 1/8 resolution for flow estimation
            fmaps = F.interpolate(
                flow_features[-1][:, :, self.patch_start_idx:].reshape(
                    -1, H_14//14, W_14//14, self.flow_dim
                ).permute(0, -1, 1, 2), 
                (flow_H, flow_W), mode='area'
            ).reshape(-1, self.flow_dim, flow_H, flow_W)
            
            if self.use_3d:
                fmaps3d_detail = F.interpolate(
                    flow3d_features[-1][:, :, self.patch_start_idx:].reshape(
                        -1, H_14//14, W_14//14, self.flow3d_dim
                    ).permute(0, -1, 1, 2), 
                    (flow_H, flow_W), mode='area'
                ).reshape(-1, self.flow3d_dim, flow_H, flow_W)
        
        # Process points and masks
        with torch.autocast(device_type=image_14.device.type, dtype=torch.float32):
            points = F.interpolate(
                points, (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            world_points = F.interpolate(
                world_points, (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            if self.use_model == 'base':
                points = points.permute(0, 2, 3, 1)
                points = self._remap_points(points).permute(0, -1, 1, 2)
            mask = F.interpolate(
                mask, (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
        
        # Downsample points for flow correlation
        # Use autocast for the interpolation operation to maintain precision
        with torch.autocast(device_type=image_14.device.type, dtype=torch.float32):
            pm = F.interpolate(points, (flow_H, flow_W), mode='nearest')
            pm = pm.reshape(-1, pm.shape[-3], pm.shape[-2], pm.shape[-1])
        
        # Prepare outputs with consistent data types
        if self.use_3d:
            return (
                fmaps.to(image.dtype), 
                ctxfeat.to(image.dtype), 
                fmaps3d_detail.to(image.dtype), 
                pm.to(image.dtype), 
                points, 
                mask, 
                world_points, 
                camera_poses
            )
        else:
            # Handle the 2D case by initializing a zero-filled 3D detail map
            fmaps3d_detail = torch.zeros(
                (pm.shape[0], self.flow3d_dim, flow_H, flow_W)
            ).cuda()
            
            return (
                fmaps.to(image.dtype), 
                ctxfeat.to(image.dtype), 
                fmaps3d_detail.to(image.dtype), 
                pm.to(image.dtype), 
                points, 
                mask, 
                world_points, 
                camera_poses
            )

    def forward(self, images, iters=4, sw=None, is_training=False, stride=None, tracking3d=False):
        """
        Main forward pass for video sequences.
        Estimates flow and geometry across frames using a sliding window approach.
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
        # Determine padding strategy
        pad = True if stride is None else False
        
        # Pad temporal dimension to fit window size
        images, T, indices = self.get_T_padded_images(
            images, T, S, is_training, stride=stride, pad=pad
        )

        images = images.contiguous()
        images_ = images.reshape(B * T, 3, H, W)
        padder = InputPadder(images_.shape)
        images_ = padder.pad(images_)[0]

        _, _, H_pad, W_pad = images_.shape
        H8, W8 = H_pad // 8, W_pad // 8
        C_flow = self.flow_dim
        C_ctx = 128

        # Extract features (Geometry & Flow) for all frames
        (fmaps, ctxfeats, fmaps3d_detail, pms, 
         points, _, world_points, camera_poses) = self.get_fmaps(
            images_, B, T, sw, is_training
        )
        
        # Reshape extracted features for sequence processing
        fmaps = fmaps.to(dtype).reshape(B, T, C_flow, H8, W8)
        fmaps3d_detail = fmaps3d_detail.to(dtype).reshape(B, T, self.flow3d_dim, H8, W8)
        ctxfeats = ctxfeats.to(dtype).reshape(B, T, C_ctx, H8, W8)
        pms = pms.to(dtype).reshape(B, T, 3, H8, W8)
        
        # Anchor frame (t=0) features used as reference for tracking
        fmap_anchor = fmaps[:, 0]
        ctxfeat_anchor = ctxfeats[:, 0]
        pm_anchor = pms[:, 0]
        fmaps3d_detail_anchor = fmaps3d_detail[:, 0]

        # Initialize containers for predictions
        all_flow_preds = [] if (T <= 2 or is_training) else None
        all_flow3d_preds = [] if (tracking3d and (T <= 2 or is_training)) else None
        all_visconf_preds = [] if (T <= 2 or is_training) else None

        if T > 2: # Multi-frame tracking logic
            # Final output tensors at full resolution
            full_flows = torch.zeros((B, T, 2, H, W), dtype=dtype, device=device)
            if tracking3d:
                full_flows3d = torch.zeros((B, T, 3, H, W), dtype=dtype, device=device)
            full_visconfs = torch.zeros((B, T, 2, H, W), dtype=dtype, device=device)
            
            # Low-res output tensors for internal state maintenance
            full_flows8 = torch.zeros((B, T, 2, H8, W8), dtype=dtype, device=device)
            full_flow3ds8 = torch.zeros((B, T, 3, H8, W8), dtype=dtype, device=device)
            full_visconfs8 = torch.zeros((B, T, 2, H8, W8), dtype=dtype, device=device)

            visits = np.zeros((T))

            # Iterate over sliding windows
            for ii, ind in enumerate(indices):
                ara = np.arange(ind, ind + S)
                
                fmaps2 = fmaps[:, ara]
                fmaps3d_detail2 = fmaps3d_detail[:, ara]
                ctxfeats2 = ctxfeats[:, ara]
                pms2 = pms[:, ara]
                
                # Fetch initial states from the global low-res tensors
                flows8 = full_flows8[:, ara].reshape(B * S, 2, H8, W8).detach()
                flow3ds8 = full_flow3ds8[:, ara].reshape(B * S, 3, H8, W8).detach()
                visconfs8 = full_visconfs8[:, ara].reshape(B * S, 2, H8, W8).detach()

                # Core window processing (Iterative Flow Update)
                with torch.autocast(device_type=fmaps.device.type, dtype=torch.float32):
                    (flow_predictions, flow3d_predictions, visconf_predictions, 
                     flows8, flow3ds8, feats8) = self.forward_window_unified(
                        fmap1_single=fmap_anchor, fmap2=fmaps2, 
                        fmaps3d_detail1_single=fmaps3d_detail_anchor, 
                        fmaps3d_detail2=fmaps3d_detail2,
                        visconfs8=visconfs8, iters=iters, 
                        flow2ds8=flows8, flow3ds8=flow3ds8,
                        cxt1_single=ctxfeat_anchor, cxt2=ctxfeats2,
                        pm1_single=pm_anchor.detach(), pm2=pms2.detach(),
                        is_training=is_training, tracking3d=tracking3d
                    )
                
                # Unpadding and result collection for current window
                unpad_flow_predictions = []
                unpad_flow3d_predictions = [] if tracking3d else None
                unpad_visconf_predictions = []
                
                for i in range(len(flow_predictions)):
                    flow_p = padder.unpad(flow_predictions[i])
                    unpad_flow_predictions.append(flow_p.reshape(B, S, 2, H, W))
                    
                    if tracking3d:
                        flow3dp = padder.unpad(flow3d_predictions[i])
                        unpad_flow3d_predictions.append(flow3dp.reshape(B, S, 3, H, W))
                    
                    vis_p = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                    unpad_visconf_predictions.append(vis_p.reshape(B, S, 2, H, W))

                # Update global tensors with the latest iterative prediction
                full_flows[:, ara] = unpad_flow_predictions[-1]
                full_flows8[:, ara] = flows8.reshape(B, S, 2, H8, W8)
                if tracking3d:
                    full_flows3d[:, ara] = unpad_flow3d_predictions[-1]
                    full_flow3ds8[:, ara] = flow3ds8.reshape(B, S, 3, H8, W8)
                full_visconfs[:, ara] = unpad_visconf_predictions[-1]
                full_visconfs8[:, ara] = visconfs8.reshape(B, S, 2, H8, W8)

                visits[ara] += 1

                if is_training:
                    all_flow_preds.append(unpad_flow_predictions)
                    if tracking3d:
                        all_flow3d_preds.append(unpad_flow3d_predictions)
                    all_visconf_preds.append(unpad_visconf_predictions)

                # Clean up memory if not training
                if not is_training:
                    del unpad_flow_predictions, unpad_visconf_predictions
                    if tracking3d: del unpad_flow3d_predictions

            # Fill gaps for frames never visited by the sliding window
            invalid_idx = np.where(visits == 0)[0]
            valid_idx = np.where(visits > 0)[0]
            for idx in invalid_idx:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                full_flows8[:, idx] = full_flows8[:, nearest]
                if tracking3d:
                    full_flow3ds8[:, idx] = full_flow3ds8[:, nearest]
                full_visconfs8[:, idx] = full_visconfs8[:, nearest]

        else: # Standard 2-frame flow logic
            init_f8 = torch.zeros((B, 2, H8, W8), dtype=dtype, device=device)
            init_f3d8 = torch.zeros((B, 3, H8, W8), dtype=dtype, device=device)
            init_v8 = torch.zeros((B, 2, H8, W8), dtype=dtype, device=device)
            
            with torch.autocast(device_type=fmaps.device.type, dtype=torch.float32):
                (flow_predictions, flow3d_predictions, visconf_predictions, 
                 flows8, flow3ds8, feats8) = self.forward_window_unified(
                        fmap1_single=fmap_anchor, fmap2=fmaps[:, 1:2], 
                        fmaps3d_detail1_single=fmaps3d_detail_anchor, 
                        fmaps3d_detail2=fmaps3d_detail[:, 1:2],
                        visconfs8=init_v8, iters=iters, 
                        flow2ds8=init_f8, flow3ds8=init_f3d8,
                        cxt1_single=ctxfeat_anchor, cxt2=ctxfeats[:, 1:2],
                        pm1_single=pm_anchor.detach(), pm2=pms[:, 1:2].detach(),
                        is_training=is_training, tracking3d=tracking3d
                    )

            # Process outputs for 2-frame case
            for i in range(len(flow_predictions)):
                f_unpad = padder.unpad(flow_predictions[i])
                all_flow_preds.append(f_unpad.reshape(B, 2, H, W))
                
                if tracking3d:
                    f3d_unpad = padder.unpad(flow3d_predictions[i])
                    all_flow3d_preds.append(f3d_unpad.reshape(B, 3, H, W))
                
                v_unpad = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                all_visconf_preds.append(v_unpad.reshape(B, 2, H, W))
                
            full_flows = all_flow_preds[-1]
            if tracking3d:
                full_flows3d = all_flow3d_preds[-1]
            full_visconfs = all_visconf_preds[-1]
                
        # Remove temporal padding for inference
        if (not is_training) and (T > 2):
            full_flows = full_flows[:, :T_bak]
            if tracking3d:
                full_flows3d = full_flows3d[:, :T_bak]
            full_visconfs = full_visconfs[:, :T_bak]
        
        # Unpad geometry related outputs
        points = padder.unpad(points)
        world_points = padder.unpad(world_points)
        
        if tracking3d:
            return (full_flows, full_visconfs, all_flow_preds, all_visconf_preds, 
                    full_flows3d, all_flow3d_preds, points, world_points, camera_poses)
        else:
            return (full_flows, full_visconfs, all_flow_preds, all_visconf_preds)

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

    def forward_sliding1(
        self, images, iters=4, sw=None, is_training=False, 
        window_len=None, stride=None, tracking3d=False
    ):
        """
        Variant of forward pass using sliding window/pairwise estimation 
        with memory optimizations by splitting the batch dimension.
        """
        B, T, C, H, W = images.shape
        device = images.device
        images = images.to(torch.float32)
        dtype = images.dtype

        # Normalization with broadcastable shapes
        images = images / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=device)
        mean = mean.reshape(1, 1, 3, 1, 1).to(images.dtype)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=device)
        std = std.reshape(1, 1, 3, 1, 1).to(images.dtype)
        images = (images - mean) / std

        images = images.contiguous()
        images_ = images.reshape(B * T, 3, H, W)
        padder = InputPadder(images_.shape)
        images_ = padder.pad(images_)[0]
        
        _, _, H_pad, W_pad = images_.shape
        C_flow, H8, W8 = self.flow_dim, H_pad // 8, W_pad // 8
        C_ctx = 128

        # Feature Extraction
        (fmaps, ctxfeats, fmaps3d_detail, pms, 
         points, masks, world_points, camera_poses) = self.get_fmaps(
            images_, B, T, sw, is_training
        )
        
        # Reshape to (B, T, C, H, W)
        fmaps = fmaps.to(dtype).reshape(B, T, C_flow, H8, W8)
        fmaps3d_detail = fmaps3d_detail.to(dtype).reshape(B, T, self.flow3d_dim, H8, W8)
        ctxfeats = ctxfeats.to(dtype).reshape(B, T, C_ctx, H8, W8)
        pms = pms.to(dtype).reshape(B, T, 3, H8, W8)
        
        # Convert to pairwise format for frame-to-frame estimation
        fmaps = self.pairwise_concat(fmaps)
        fmaps3d_detail = self.pairwise_concat(fmaps3d_detail)
        ctxfeats = self.pairwise_concat(ctxfeats)
        pms = self.pairwise_concat(pms)

        B_pair, T_pair = fmaps.shape[0], fmaps.shape[1]
        
        # --- Chunking along the Batch (B) dimension ---
        chunk_size = 12 
        num_chunks = (B_pair + chunk_size - 1) // chunk_size
        
        # Iteration history containers
        all_flow_preds_chunks, all_visconf_preds_chunks = [], []
        all_flow3d_preds_chunks = [] if tracking3d else None
        
        # Final result containers
        full_flows_list, full_visconfs_list = [], []
        full_flows3d_list = [] if tracking3d else None
        points_list, masks_list, world_points_list, camera_poses_list = [], [], [], []
        
        # Split point/pose data to match batch chunks
        points_chunks = list(torch.split(points, chunk_size, dim=0))
        masks_chunks = list(torch.split(masks, chunk_size, dim=0))
        world_points_chunks = list(torch.split(world_points, chunk_size, dim=0))
        camera_poses_chunks = list(torch.split(camera_poses, chunk_size, dim=0)) 
        
        for i in range(num_chunks):
            start_idx, end_idx = i * chunk_size, min((i + 1) * chunk_size, B_pair)
            b_chunk = end_idx - start_idx
            
            # 1. Slice chunk-specific inputs
            fmaps_chunk = fmaps[start_idx:end_idx]
            ctxfeats_chunk = ctxfeats[start_idx:end_idx]
            pms_chunk = pms[start_idx:end_idx]
            fmaps3d_detail_chunk = fmaps3d_detail[start_idx:end_idx]
            
            # 2. Get Chunk Anchor (Reference Frame)
            fmap_anchor_chunk = fmaps_chunk[:, 0]
            ctxfeat_anchor_chunk = ctxfeats_chunk[:, 0]
            pm_anchor_chunk = pms_chunk[:, 0]
            fmaps3d_detail_anchor_chunk = fmaps3d_detail_chunk[:, 0]
            
            # 3. Initialize flow states for current chunk
            v_init = torch.zeros((b_chunk, 2, H8, W8), dtype=dtype, device=device)
            f2d_init = torch.zeros((b_chunk, 2, H8, W8), dtype=dtype, device=device)
            f3d_init = torch.zeros((b_chunk, 3, H8, W8), dtype=dtype, device=device)
            
            # 4. Iterative Update
            (flow_preds, flow3d_preds, vis_preds, 
             _, _, _) = self.forward_window_unified(
                fmap1_single=fmap_anchor_chunk, fmap2=fmaps_chunk[:, 1:2], 
                fmaps3d_detail1_single=fmaps3d_detail_anchor_chunk, 
                fmaps3d_detail2=fmaps3d_detail_chunk[:, 1:2],
                visconfs8=v_init, iters=iters, 
                flow2ds8=f2d_init, flow3ds8=f3d_init,
                cxt1_single=ctxfeat_anchor_chunk, cxt2=ctxfeats_chunk[:, 1:2],
                pm1_single=pm_anchor_chunk.detach(), pm2=pms_chunk[:, 1:2].detach(),
                is_training=is_training, tracking3d=tracking3d
            )
            
            # 5. Result post-processing for current chunk
            chunk_flow_preds, chunk_visconf_preds = [], []
            chunk_flow3d_preds = [] if tracking3d else None
            
            for k in range(len(flow_preds)):
                flow_k = padder.unpad(flow_preds[k]).reshape(b_chunk, 2, H, W)
                chunk_flow_preds.append(flow_k)
                
                vis_k = torch.sigmoid(vis_preds[k])
                vis_k = padder.unpad(vis_k).reshape(b_chunk, 2, H, W)
                chunk_visconf_preds.append(vis_k)
                
                if tracking3d:
                    f3d_k = padder.unpad(flow3d_preds[k]).reshape(b_chunk, 3, H, W)
                    chunk_flow3d_preds.append(f3d_k)

            # 6. Store and detach to CPU to save VRAM
            full_flows_list.append(chunk_flow_preds[-1].detach().cpu())
            full_visconfs_list.append(chunk_visconf_preds[-1].detach().cpu())
            points_list.append(padder.unpad(points_chunks[i]))
            masks_list.append(padder.unpad(masks_chunks[i]))
            world_points_list.append(padder.unpad(world_points_chunks[i]))
            camera_poses_list.append(camera_poses_chunks[i])
            
            if tracking3d:
                full_flows3d_list.append(chunk_flow3d_preds[-1].cpu())

            # Clear memory for next chunk
            all_flow_preds_chunks.append(chunk_flow_preds)
            all_visconf_preds_chunks.append(chunk_visconf_preds)
            if tracking3d:
                all_flow3d_preds_chunks.append(chunk_flow3d_preds)

            del fmaps_chunk, ctxfeats_chunk, pms_chunk, fmaps3d_detail_chunk
            torch.cuda.empty_cache()

        # --- Aggregate Chunks into Final Tensors ---
        all_flow_preds, all_visconf_preds = [], []
        all_flow3d_preds = [] if tracking3d else None
        
        if all_flow_preds_chunks:
            for k in range(len(all_flow_preds_chunks[0])):
                all_flow_preds.append(torch.cat([c[k] for c in all_flow_preds_chunks], 0))
                all_visconf_preds.append(torch.cat([c[k] for c in all_visconf_preds_chunks], 0))
                if tracking3d:
                    all_flow3d_preds.append(torch.cat([c[k] for c in all_flow3d_preds_chunks], 0))

        full_flows = torch.cat(full_flows_list, 0).reshape(B_pair, 2, H, W)
        full_visconfs = torch.cat(full_visconfs_list, 0).reshape(B_pair, 2, H, W)
        
        if tracking3d:
            full_f3d = torch.cat(full_flows3d_list, 0).reshape(B_pair, 3, H, W)
            return (full_flows, full_visconfs, all_flow_preds, all_visconf_preds, 
                    full_f3d, all_flow3d_preds, torch.cat(points_list, 0), 
                    torch.cat(masks_list, 0), torch.cat(world_points_list, 0), 
                    torch.cat(camera_poses_list, 0))
        
        return (full_flows, full_visconfs, all_flow_preds, all_visconf_preds)

    def forward_sliding(
        self, images, iters=4, sw=None, is_training=False, 
        window_len=None, stride=None, tracking3d=False, eval_dict=None
    ):
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

        # Normalization
        images = images / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=device)
        mean = mean.reshape(1, 1, 3, 1, 1).to(images.dtype)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=device)
        std = std.reshape(1, 1, 3, 1, 1).to(images.dtype)
        images = (images - mean) / std

        T_bak = T
        # Pad temporal dimension for windowing
        images, T, indices = self.get_T_padded_images(
            images, T, S, is_training, stride
        )
        assert stride <= S // 2

        images = images.contiguous()
        images_ = images.reshape(B * T, 3, H, W)
        padder = InputPadder(images_.shape)
        images_ = padder.pad(images_)[0]
        
        _, _, H_pad, W_pad = images_.shape
        C_flow, H8, W8 = self.flow_dim, H_pad // 8, W_pad // 8
        C_ctx = 128
        
        # Feature Extraction or Retrieval from Cache
        if eval_dict is None:
            (fmaps, ctxfeats, fmaps3d_detail, pms, 
             points, masks, world_points, camera_poses) = self.get_fmaps(
                images_, B, T, sw, is_training
            )
            fmaps = fmaps.to(dtype).reshape(B, T, C_flow, H8, W8)
            fmaps3d_detail = fmaps3d_detail.to(dtype).reshape(B, T, self.flow3d_dim, H8, W8)
            ctxfeats = ctxfeats.to(dtype).reshape(B, T, C_ctx, H8, W8)
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
                
                # Manual padding for points and masks
                p_pad = points[-1:].expand(T_target - T_cur, *points.shape[1:])
                points = torch.cat([points, p_pad], dim=0)
                m_pad = masks[-1:].expand(T_target - T_cur, *masks.shape[1:])
                masks = torch.cat([masks, m_pad], dim=0)
            
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
            all_flow_preds, all_visconf_preds = [], []
            all_flow3d_preds = [] if tracking3d else None
            
            flows8 = torch.zeros((B, 2, H8, W8), dtype=dtype, device=device)
            flow3ds8 = torch.zeros((B, 3, H8, W8), dtype=dtype, device=device)
            visconfs8 = torch.zeros((B, 2, H8, W8), dtype=dtype, device=device)
                
            fmap_anchor = fmaps[:, 0]
            ctxfeat_anchor = ctxfeats[:, 0]
            pm_anchor = pms[:, 0]
            fmaps3d_detail_anchor = fmaps3d_detail[:, 0]
            
            (flow_predictions, flow3d_predictions, visconf_predictions, 
             flows8, flow3ds8, feats8) = self.forward_window_unified(
                fmap1_single=fmap_anchor, fmap2=fmaps[:, 1:2], 
                fmaps3d_detail1_single=fmaps3d_detail_anchor, 
                fmaps3d_detail2=fmaps3d_detail[:, 1:2],
                visconfs8=visconfs8, iters=iters, 
                flow2ds8=flows8, flow3ds8=flow3ds8,
                cxt1_single=ctxfeat_anchor, cxt2=ctxfeats[:, 1:2],
                pm1_single=pm_anchor.detach(), pm2=pms[:, 1:2].detach(),
                is_training=is_training, tracking3d=tracking3d
            )
            
            for i in range(len(flow_predictions)):
                # Unpad and store results
                flow_unpad = padder.unpad(flow_predictions[i])
                all_flow_preds.append(flow_unpad.reshape(B, 2, H, W))
                
                if tracking3d:
                    flow3d_unpad = padder.unpad(flow3d_predictions[i])
                    all_flow3d_preds.append(flow3d_unpad.reshape(B, 3, H, W))
                    
                vis_unpad = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                all_visconf_preds.append(vis_unpad.reshape(B, 2, H, W))
            
            full_flows = all_flow_preds[-1].reshape(B, 2, H, W).detach().cpu()
            full_visconfs = all_visconf_preds[-1].reshape(B, 2, H, W).detach().cpu()
            
            if tracking3d:
                full_flows3d = all_flow3d_preds[-1].reshape(B, 3, H, W).cpu()
                return (full_flows, full_visconfs, all_flow_preds, all_visconf_preds, 
                        full_flows3d, all_flow3d_preds, padder.unpad(points), 
                        padder.unpad(masks), dict1, padder.unpad(world_points), camera_poses)
            else:
                return (full_flows, full_visconfs, all_flow_preds, all_visconf_preds, dict1)

        # --- Multiframe Tracking (T > 2) ---
        assert T > 2 
        
        if is_training:
            all_flow_preds, all_visconf_preds = [], []
            all_flow3d_preds = [] if tracking3d else None
            
        # Initialize full tensors on CPU to manage VRAM
        full_flows = torch.zeros((B, T, 2, H, W), dtype=dtype, device='cpu')
        full_visconfs = torch.zeros((B, T, 2, H, W), dtype=dtype, device='cpu')
        full_flows3d = torch.zeros((B, T, 3, H, W), dtype=dtype, device='cpu') if tracking3d else None

        fmap_anchor = fmaps[:, 0]
        ctxfeat_anchor = ctxfeats[:, 0]
        pm_anchor = pms[:, 0]
        fmaps3d_detail_anchor = fmaps3d_detail[:, 0]
        full_visited = torch.zeros((T,), dtype=torch.bool, device=device)

        for ii, ind in enumerate(indices):
            ara = np.arange(ind, ind + S)
            
            if ii == 0:
                flows8 = torch.zeros((B, S, 2, H8, W8), dtype=dtype, device=device)
                flow3ds8 = torch.zeros((B, S, 3, H8, W8), dtype=dtype, device=device)
                visconfs8 = torch.zeros((B, S, 2, H8, W8), dtype=dtype, device=device)

                fmaps2, fmaps3d_detail2 = fmaps[:, ara], fmaps3d_detail[:, ara]
                ctxfeats2, pms2 = ctxfeats[:, ara], pms[:, ara]
            else:
                # Temporal overlap logic for continuity
                mid = stride + S // 2
                rep_count = S // 2
                
                flows8 = torch.cat([
                    flows8[:, stride:mid], 
                    flows8[:, mid-1:mid].repeat(1, rep_count, 1, 1, 1)
                ], dim=1)
                flow3ds8 = torch.cat([
                    flow3ds8[:, stride:mid], 
                    flow3ds8[:, mid-1:mid].repeat(1, rep_count, 1, 1, 1)
                ], dim=1)
                visconfs8 = torch.cat([
                    visconfs8[:, stride:mid], 
                    visconfs8[:, mid-1:mid].repeat(1, rep_count, 1, 1, 1)
                ], dim=1)
                
                fmaps2 = torch.cat([fmaps2[:, stride:mid], fmaps[:, ind+S//2:ind+S]], dim=1)
                fmaps3d_detail2 = torch.cat([
                    fmaps3d_detail2[:, stride:mid], 
                    fmaps3d_detail[:, ind+S//2:ind+S]
                ], dim=1)
                ctxfeats2 = torch.cat([ctxfeats2[:, stride:mid], ctxfeats[:, ind+S//2:ind+S]], dim=1)
                pms2 = torch.cat([pms2[:, stride:mid], pms[:, ind+S//2:ind+S]], dim=1)

            # Solve window
            (flow_predictions, flow3d_predictions, visconf_predictions, 
             flows8, flow3ds8, feats8) = self.forward_window_unified(
                fmap1_single=fmap_anchor, fmap2=fmaps2, 
                fmaps3d_detail1_single=fmaps3d_detail_anchor, 
                fmaps3d_detail2=fmaps3d_detail2,
                visconfs8=visconfs8.reshape(B*S, 2, H8, W8).detach(), 
                iters=iters, 
                flow2ds8=flows8.reshape(B*S, 2, H8, W8).detach(), 
                flow3ds8=flow3ds8.reshape(B*S, 3, H8, W8).detach(),
                cxt1_single=ctxfeat_anchor, cxt2=ctxfeats2,
                pm1_single=pm_anchor.detach(), pm2=pms2.detach(),
                is_training=is_training, tracking3d=tracking3d
            )
            
            unpad_flow_preds, unpad_vis_preds = [], []
            unpad_flow3d_preds = [] if tracking3d else None
            
            for i in range(len(flow_predictions)):
                f_p = padder.unpad(flow_predictions[i]).reshape(B, S, 2, H, W)
                unpad_flow_preds.append(f_p)
                
                if tracking3d:
                    f3_p = padder.unpad(flow3d_predictions[i]).reshape(B, S, 3, H, W)
                    unpad_flow3d_preds.append(f3_p)
                
                v_p = torch.sigmoid(visconf_predictions[i])
                v_p = padder.unpad(v_p).reshape(B, S, 2, H, W)
                unpad_vis_preds.append(v_p)

            # Fill unvisited frames in global result
            current_visiting = torch.zeros((T,), dtype=torch.bool, device=device)
            current_visiting[ara] = True
            to_fill = current_visiting & (~full_visited)
            to_fill_sum = to_fill.sum().item()
            
            full_flows[:, to_fill] = unpad_flow_preds[-1][:, -to_fill_sum:].detach().cpu()
            if tracking3d:
                full_flows3d[:, to_fill] = unpad_flow3d_preds[-1][:, -to_fill_sum:].detach().cpu()
            full_visconfs[:, to_fill] = unpad_vis_preds[-1][:, -to_fill_sum:].detach().cpu()
            full_visited |= current_visiting

            if is_training:
                all_flow_preds.append(unpad_flow_preds)
                if tracking3d:
                    all_flow3d_preds.append(unpad_flow3d_preds)
                all_visconf_preds.append(unpad_vis_preds)
            
            # Reshape states for next iteration
            flows8 = flows8.reshape(B, S, 2, H8, W8)
            flow3ds8 = flow3ds8.reshape(B, S, 3, H8, W8)
            visconfs8 = visconfs8.reshape(B, S, 2, H8, W8)
                
        if not is_training:
            full_flows = full_flows[:, :T_bak]
            full_visconfs = full_visconfs[:, :T_bak]
            if tracking3d:
                full_flows3d = full_flows3d[:, :T_bak]
        
        # Cleanup and Return
        if tracking3d:
            return (full_flows, full_visconfs, all_flow_preds, all_visconf_preds, 
                    full_flows3d, all_flow3d_preds, padder.unpad(points), 
                    padder.unpad(masks), dict1, padder.unpad(world_points), camera_poses)
        else:
            return (full_flows, full_visconfs, all_flow_preds, all_visconf_preds, dict1)

    def forward_window_unified(
        self, fmap1_single, fmap2, visconfs8, iters=None, flow2ds8=None,
        flow3ds8=None, cxt1_single=None, cxt2=None, pm1_single=None,
        pm2=None, fmaps3d_detail1_single=None, fmaps3d_detail2=None, 
        is_training=False, tracking3d=False
    ):
        """
        Unified iterative flow update (RAFT-style).
        Handles both 2D and 3D flow updates using correlation volumes and recurrent units.
        """
        B, S, C_in, H8, W8 = fmap2.shape
        dtype, device = fmap2.dtype, fmap2.device
        
        # 1. Expand anchor features to match the sequence length (B*S)
        fmap1 = fmap1_single.unsqueeze(1).repeat(1, S, 1, 1, 1)
        fmap1 = fmap1.reshape(B * S, C_in, H8, W8).contiguous()
        fmap2_flat = fmap2.reshape(B * S, C_in, H8, W8).contiguous()

        pm1 = pm1_single.unsqueeze(1).repeat(1, S, 1, 1, 1)
        pm1 = pm1.reshape(B * S, 3, H8, W8).contiguous()
        pm2 = pm2.reshape(B * S, 3, H8, W8).contiguous()

        cxt1 = cxt1_single.unsqueeze(1).repeat(1, S, 1, 1, 1)
        cxt1 = cxt1.reshape(B * S, -1, H8, W8)
        cxt2 = cxt2.reshape(B * S, -1, H8, W8).contiguous()

        # 2. Initialize Correlation Blocks (All-pairs correlation)
        fea_corr_fn = CorrBlock(fmap1, fmap2_flat, self.corr_levels, self.corr_radius)
        cxt_corr_fn = CorrBlock(cxt1, cxt2, self.corr_levels, self.corr_radius)
        flowfeat, ctxfeat = fmap1.clone(), cxt1.clone()
        
        # 3. Generate coordinate grid and handle temporal embeddings
        coords1 = self.coords_grid(B * S, H8, W8, device=device, dtype=dtype)
        visconfs8 = visconfs8.reshape(B * S, 2, H8, W8).contiguous()

        time_emb = self.fetch_time_embed(
            S, self.time_emb, ctxfeat.dtype, is_training
        ).reshape(1, S, ctxfeat.shape[1], 1, 1).repeat(B, 1, 1, 1, 1)
        ctxfeat = ctxfeat + time_emb.reshape(B * S, -1, 1, 1)
        
        # Handle 3D specific features if tracking is enabled
        if tracking3d:
            fmaps3d_detail1 = fmaps3d_detail1_single.unsqueeze(1).repeat(1, S, 1, 1, 1)
            fmaps3d_detail1 = fmaps3d_detail1.reshape(B * S, self.flow3d_dim, H8, W8).contiguous()
            
            time_emb3d = self.fetch_time_embed(
                S, self.time_emb3d, fmaps3d_detail1.dtype, is_training
            ).reshape(1, S, fmaps3d_detail1.shape[1], 1, 1).repeat(B, 1, 1, 1, 1)
            
            fmaps3d_detail1 += time_emb3d.reshape(B * S, -1, 1, 1)
            fmaps3d_detail2 = fmaps3d_detail2.reshape(B * S, self.flow3d_dim, H8, W8).contiguous()

        flow2d_predictions, visconf_predictions = [], []
        flow3d_predictions = [] if tracking3d else None
        
        # --- Iterative Update Loop (GRU) ---
        for itr in range(iters):
            flow2ds8, flow3ds8 = flow2ds8.detach(), flow3ds8.detach()
            coords2 = (coords1 + flow2ds8).detach()
            
            # Look up correlations using current flow estimates
            fea_corr = fea_corr_fn(coords2).to(dtype)
            cxt_corr = cxt_corr_fn(coords2).to(dtype)
            
            # Encode current 2D motion
            motion2d = misc.posenc(
                flow2ds8.permute(0, 2, 3, 1).reshape(B, S, -1, 2), 0, 10
            ).reshape(B * S, H8, W8, -1).permute(0, 3, 1, 2).to(dtype)
            
            if tracking3d:
                # 3D Point correlation and motion encoding
                pm_corr_fn = CorrBlock(
                    (pm1 + flow3ds8).detach(), pm2, 
                    self.corr_levels, self.corr_radius, mode='nearest'
                )
                pm_corr = pm_corr_fn(coords2).to(dtype)
                motion3d = misc.posenc(
                    flow3ds8.permute(0, 2, 3, 1).reshape(B, S, -1, 3), 0, 10
                ).reshape(B * S, H8, W8, -1).permute(0, 3, 1, 2).to(dtype)
                
                # Sample features and PM based on current 2D flow
                grid_pre = torch.stack((
                    ((coords1 + flow2ds8)[:, 0] / (W8 - 1)) * 2 - 1, 
                    ((coords1 + flow2ds8)[:, 1] / (H8 - 1)) * 2 - 1
                ), -1).detach()
                
                sampled_feature_pre = F.grid_sample(
                    fmaps3d_detail2, grid_pre, mode='bilinear', 
                    align_corners=True, padding_mode='border'
                )
                sampled_pm_pre = F.grid_sample(
                    pm2, grid_pre, mode='bilinear', 
                    align_corners=True, padding_mode='border'
                )

            # Update 2D Flow and Visibility
            flowfeat = self.flow_update_block(
                flowfeat, ctxfeat, visconfs8, fea_corr, cxt_corr, motion2d, S
            )
            flow2d_update = self.flow_2d_head(flowfeat)
            visconf_update = self.flow_visconf_head(flowfeat)
            flow2ds8 = flow2ds8 + flow2d_update
            
            # Update 3D Flow if tracking is enabled
            if tracking3d:
                grid_post = torch.stack((
                    ((coords1 + flow2ds8)[:, 0] / (W8 - 1)) * 2 - 1, 
                    ((coords1 + flow2ds8)[:, 1] / (H8 - 1)) * 2 - 1
                ), -1).detach()
                
                sampled_feature = F.grid_sample(
                    fmaps3d_detail2, grid_post, mode='bilinear', 
                    align_corners=True, padding_mode='border'
                )
                sampled_pm = F.grid_sample(
                    pm2, grid_post, mode='bilinear', 
                    align_corners=True, padding_mode='border'
                )
                
                flow3d_update, fmaps3d_detail1 = self.flow3d_head(
                    flowfeat.detach(), fmaps3d_detail1, sampled_feature_pre, 
                    sampled_feature, sampled_pm - sampled_pm_pre, 
                    pm_corr, motion3d, S
                )
                flow3ds8 = flow3ds8 + flow3d_update
            
            # Predict Upsampling Weights
            temperature = 0.25 
            weight_update = temperature * self.flow_upsample_weight(flowfeat)
            if tracking3d:
                weight_update_3d = temperature * self.flow_3d_upsample_weight(fmaps3d_detail1)
            
            visconfs8 = visconfs8 + visconf_update

            # Upsample and Collect Predictions
            flow2d_predictions.append(
                self.upsample_data(flow2ds8, weight_update, dim1=2)
            )
            if tracking3d:
                flow3d_predictions.append(
                    self.upsample_3d_data(flow3ds8, weight_update_3d, dim1=3)
                )
            visconf_predictions.append(
                self.upsample_data(visconfs8, weight_update, dim1=2)
            )
            
            torch.cuda.empty_cache()

        return (
            flow2d_predictions, flow3d_predictions, visconf_predictions, 
            flow2ds8, flow3ds8, flowfeat
        )

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
            # Reshape for interpolation: (B, C, S)
            time_emb = F.interpolate(
                time_emb.permute(0, 2, 1), 
                size=t, 
                mode="linear"
            ).permute(0, 2, 1)
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
            
            Tpad = indices[-1] + S - T
            
            if pad:
                if is_training:
                    assert Tpad == 0
                else:
                    # Pad the last frame to fill the window
                    images = images.reshape(B, 1, T, C * H * W)
                    if Tpad > 0:
                        padding_tensor = images[:, :, -1:, :].expand(B, 1, Tpad, C * H * W)
                        images = torch.cat([images, padding_tensor], dim=2)
                    
                    images = images.reshape(B, T + Tpad, C, H, W)
                    T = T + Tpad
        else:
            assert T == 2
            
        return images, T, indices

    def get_fmaps(self, images_, B, T, sw, is_training):
        """
        Extract feature maps for all frames.
        Processes frames in chunks to manage memory usage.
        """
        _, _, H_pad, W_pad = images_.shape 
        H8, W8 = H_pad // 8, W_pad // 8
        C, C1 = self.flow_dim, 128
        
        # Adjust chunk size based on the backbone model architecture
        if self.use_model in ['pi3', 'depthanythingv3']:
            fmaps_chunk_size = 256
        else:
            fmaps_chunk_size = 64
            
        images = images_.reshape(B, T, 3, H_pad, W_pad)
        
        # Initialize containers
        fmaps, fmaps3d_detail, ctxfeats = [], [], []
        pms, points, masks = [], [], []
        world_points, camera_poses = [], []

        # Iterate through chunks of frames along the time dimension
        for t in range(0, T, fmaps_chunk_size):
            images_chunk = images[:, t : t + fmaps_chunk_size]
            
            # Extract features and geometry via forward_point
            (f_c, c_c, f3d_c, pm_c, 
             pt_c, m_c, wp_c, cp_c) = self.forward_point(
                images_chunk, 
                current_batch_size=B, 
                for_flow=True
            )
            
            # Append reshaped outputs
            fmaps.append(f_c.reshape(B, -1, C, H8, W8))
            ctxfeats.append(c_c.reshape(B, -1, C1, H8, W8))
            pms.append(pm_c.reshape(B, -1, 3, H8, W8))
            points.append(pt_c.reshape(B, -1, 3, H_pad, W_pad))
            masks.append(m_c.reshape(B, -1, 1, H_pad, W_pad))
            world_points.append(wp_c.reshape(B, -1, 3, H_pad, W_pad))
            camera_poses.append(cp_c.reshape(B, -1, 4, 4))
            fmaps3d_detail.append(f3d_c.reshape(B, -1, self.flow3d_dim, H8, W8))
            
        # Final concatenation and flattening for downstream processing
        fmaps = torch.cat(fmaps, dim=1).reshape(-1, C, H8, W8)
        fmaps3d_detail = torch.cat(fmaps3d_detail, dim=1).reshape(-1, self.flow3d_dim, H8, W8)
        ctxfeats = torch.cat(ctxfeats, dim=1).reshape(-1, C1, H8, W8)
        pms = torch.cat(pms, dim=1).reshape(-1, 3, H8, W8)
        points = torch.cat(points, dim=1).reshape(-1, 3, H_pad, W_pad)
        masks = torch.cat(masks, dim=1).reshape(-1, 1, H_pad, W_pad)
        world_points = torch.cat(world_points, dim=1).reshape(-1, 3, H_pad, W_pad)
        camera_poses = torch.cat(camera_poses, dim=1).reshape(-1, 4, 4)
        
        return (fmaps, ctxfeats, fmaps3d_detail, pms, 
                points, masks, world_points, camera_poses)

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
        # Generate y and x range tensors
        y_range = torch.arange(ht, device=device, dtype=dtype)
        x_range = torch.arange(wd, device=device, dtype=dtype)
        
        # Create the meshgrid using 'ij' indexing (matrix-style)
        coords = torch.meshgrid(y_range, x_range, indexing='ij')
        
        # Stack and reverse order to get (x, y) format at dim 0
        coords = torch.stack(coords[::-1], dim=0)
        
        # Add batch dimension and repeat
        return coords[None].repeat(batch, 1, 1, 1)

    def forward_pure_point(
        self, 
        image: torch.Tensor, 
        num_tokens: int, 
        current_batch_size: int = 4
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to estimate point clouds and masks from images.
        """
        original_height, original_width = image.shape[-2:]
        
        # 1. Resolution Management
        # Scaling resolution based on target token count
        total_pixels = original_height * original_width
        resize_factor = ((num_tokens * 14 ** 2) / total_pixels) ** 0.5
        
        target_h = int(original_height * resize_factor)
        target_w = int(original_width * resize_factor)
        
        # Initial bicubic resize for anti-aliasing
        image_resized = F.interpolate(
            image, (target_h, target_w), 
            mode="bicubic", align_corners=False, antialias=True
        )
    
        # Normalization for DINOv2 backbone
        image_norm = (image_resized - self.image_mean) / self.image_std
        
        # Snap to 14x14 patch grid (backbone constraint)
        grid_h, grid_w = target_h // 14 * 14, target_w // 14 * 14
        image_14 = F.interpolate(
            image_norm, (grid_h, grid_w), 
            mode="bilinear", align_corners=False, antialias=True
        )

        # 2. Backbone Feature Extraction
        # Extracting multiple intermediate layers for hierarchical features
        features = self.backbone.get_intermediate_layers(
            image_14, self.intermediate_layers, return_class_token=True
        )
        
        # 3. Token Preparation & Concatenation
        # T: sequence length per batch (B*T total)
        T = image.shape[0] // current_batch_size
        
        # Expand specialized tokens: [Camera Token, Register Tokens]
        camera_token = self.camera_token.expand(current_batch_size, T, -1, -1)
        register_token = self.register_token.expand(current_batch_size, T, -1, -1)
        
        # Reshape backbone patches: (B*T, N, C) -> (B, T, N, C)
        last_feat = features[-1][0]
        n_patches, feat_dim = last_feat.shape[-2:]
        backbone_tokens = last_feat.reshape(current_batch_size, T, n_patches, feat_dim)

        # Concatenate tokens for the Aggregator
        tokens = torch.cat([camera_token, register_token, backbone_tokens], dim=2)

        # 4. Global Feature Aggregation
        # Fuses local backbone features with global context
        global_features = self.aggregator(
            tokens, image_norm, patch_start_idx=self.patch_start_idx
        )
        
        # Inject aggregated features back into the backbone feature list
        features = [list(f) for f in features]  
        for i in range(len(features)):
            # Slice only the patch tokens (ignoring Camera/Register tokens)
            new_feat = global_features[2 * i + 1][:, :, self.patch_start_idx:]
            # Reshape back to flattened batch (B*T, N, C)
            features[i][0] = new_feat.reshape(-1, n_patches, feat_dim)

        # 5. Point Cloud and Mask Prediction
        # The head uses hierarchical features to regress 3D geometry
        points, mask = self.head(features, image_norm)
        
        # 6. Post-processing and Upsampling
        with torch.autocast(device_type=image.device.type, dtype=torch.float32):
            # Upsample back to original image resolution
            points = F.interpolate(
                points, (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            mask = F.interpolate(
                mask, (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            
            # Reformat: (B*T, 3, H, W) -> (B*T, H, W, 3)
            points = points.permute(0, 2, 3, 1)
            mask = mask.squeeze(1)
            
            # Apply remapping (coordinate transformation / normalization)
            points = self._remap_points(points)

        # Final Reshape: return dictionary with (B, T, H, W, C)
        return {
            'points': points.reshape(current_batch_size, T, original_height, original_width, 3), 
            'mask': mask.reshape(current_batch_size, T, original_height, original_width),
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
        # 1. Input Preparation
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        
        image = image.to(dtype=self.dtype, device=self.device)
        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        # Determine token count based on resolution level [0-9]
        if num_tokens is None:
            min_t, max_t = self.num_tokens_range
            num_tokens = int(min_t + (resolution_level / 9) * (max_t - min_t))
        
        # 2. Model Execution (Mixed Precision)
        autocast_dtype = torch.float16 if self.dtype != torch.float16 else torch.float16
        with torch.autocast(
            device_type=self.device.type, 
            dtype=autocast_dtype, 
            enabled=use_fp16
        ):
            output = self.forward_pure_point(
                image, num_tokens, current_batch_size=current_batch_size
            )
        
        points, mask = output['points'], output['mask']
        results_list = []

        # 3. Geometric Post-processing (High Precision)
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            points, mask = points.float(), mask.float()
            mask_binary = mask > self.mask_threshold

            # --- Focal Length & Depth Shift Recovery ---
            # 'local' usually refers to frame-wise recovery; 
            # 'global' refers to sequence-wide consistency.
            if local:
                if fov_x is None:
                    focal, shift = recover_focal_shift(points, mask_binary)
                else:
                    # Calculate focal from FoV: f = 0.5 * w / tan(fov/2)
                    fov_rad = torch.deg2rad(torch.as_tensor(fov_x, device=points.device))
                    focal_val = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / \
                                torch.tan(fov_rad / 2)
                    
                    focal = focal_val[None].expand(points.shape[0]) if focal_val.ndim == 0 else focal_val
                    _, shift = recover_focal_shift(points, mask_binary, focal=focal)
                
                # Derive focal components
                norm_factor = 0.5 * (1 + aspect_ratio ** 2) ** 0.5
                fx, fy = focal * norm_factor / aspect_ratio, focal * norm_factor
                intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
                depth = points[..., 2] + shift[..., None, None]

            else:
                # Global Recovery (Optimized across temporal dimension)
                if fov_x is None:
                    if no_shift:
                        focal = recover_global_focal(points, mask_binary)
                        shift = torch.zeros_like(points[..., 0, 0, 0])
                    else:
                        focal, shift = recover_global_focal_shift(points, mask_binary)
                else:
                    fov_rad = torch.deg2rad(torch.as_tensor(fov_x, device=points.device))
                    focal_val = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / \
                                torch.tan(fov_rad / 2)
                    
                    focal = focal_val[None].expand(points.shape[0]) if focal_val.ndim == 0 else focal_val
                    _, shift = recover_global_focal_shift(points, mask_binary, focal=focal)

                norm_factor = 0.5 * (1 + aspect_ratio ** 2) ** 0.5
                fx, fy = focal * norm_factor / aspect_ratio, focal * norm_factor
                
                # Repeat intrinsics for sequence length
                intrinsics_base = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
                intrinsics = intrinsics_base.repeat(1, points.shape[1], 1, 1)
                
                if no_shift:
                    depth = points[..., 2]
                else:
                    depth = points[..., 2] + shift[..., None, None, None].repeat(1, points.shape[1], 1, 1)

            # --- Consistency Projection ---
            if force_projection:
                # Project 2D grid to 3D using recovered depth and intrinsics
                points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)
            else:
                # Simply apply the translation shift to the Z-axis
                shift_vec = torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)
                if local:
                    points = points + shift_vec[..., None, None, :]
                else:
                    points = points + shift_vec[..., None, None, None, :].repeat(1, points.shape[1], 1, 1, 1)

            # 4. Masking & Result Assembly
            if apply_mask:
                points = torch.where(mask_binary[..., None], points, torch.tensor(float('inf'), device=points.device))
                depth = torch.where(mask_binary, depth, torch.tensor(float('inf'), device=depth.device))

            results_list.append({
                'points': points,
                'intrinsics': intrinsics,
                'depth': depth,
                'mask': mask_binary,
            })

        # Squeeze batch dim if original input was a single image
        if omit_batch_dim:
            results_list = [{k: v.squeeze(0) for k, v in dic.items()} for dic in results_list]

        return results_list

    @torch.inference_mode()
    def infer(
        self, 
        images, 
        iters=4, 
        sw=None, 
        is_training=False, 
        window_len=None, 
        stride=None, 
        tracking3d=False, 
        apply_mask: bool = False,
        force_projection: bool = True, 
        use_fp16: bool = True, 
        current_batch_size: int = 4,
        local: bool = False, 
        no_shift: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform standard inference on a batch of video frames and output metric 3D geometry.

        Compared to the evaluation routine, this function focuses on a single forward
        inference pass without feature caching. It directly returns the reconstructed
        scene geometry, camera parameters, and motion estimates that are ready for
        downstream usage (e.g., visualization, metric evaluation, or 3D tracking).

        Inputs:
            images: Input video tensor of shape (B, T, C, H, W), where B is batch size,
                    T is the temporal length, and H/W are spatial dimensions.
            iters: Number of refinement iterations used inside the model.
            sw: Optional sliding-window configuration for long sequences.
            is_training: Whether the model runs in training mode (affects certain layers).
            window_len / stride: Parameters controlling temporal windowing.
            tracking3d: Enable dense 3D scene flow estimation.
            apply_mask: If True, invalid pixels are removed from outputs using infinite values.
            force_projection: If True, recompute 3D points via depth + intrinsics to enforce
                            geometric consistency.
            use_fp16: Whether mixed-precision inference is enabled.
            current_batch_size: Effective batch size used internally.
            local / no_shift: Reserved flags for alternative inference behaviors.

        Outputs:
            A dictionary containing:
                - points: Metric 3D points in camera coordinates.
                - intrinsics: Estimated camera intrinsics for each frame.
                - depth: Per-pixel depth maps.
                - mask: Binary validity masks.
                - flow_2d: Dense 2D optical flow fields.
                - flow_3d: Dense 3D scene flow in metric space.
                - world_points: 3D points in a global/world coordinate system.
                - camera_poses: Estimated camera poses for each frame.
        """
        # Run the model forward pass using a sliding-window strategy if needed.
        (
            flows_e, visconf_maps_e, _, _, flows3d_e, _, 
            points, mask, _, world_points, camera_poses
        ) = self.forward_sliding(
            images, iters=iters, sw=sw, is_training=is_training, tracking3d=tracking3d
        )
        
        B, T, C, H, W = images.shape
        aspect_ratio = W / H
        
        # Build a dense 2D pixel grid (in absolute pixel coordinates).
        # This grid is later added to optical flow to obtain absolute pixel locations.
        grid_xy = holi4d.utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float() 
        grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)
        
        # Convert 2D flow from displacement to absolute image coordinates.
        flow2d = flows_e.to(torch.float32).cuda() + grid_xy 
        visconf_maps_e = visconf_maps_e.to(torch.float32)
        
        # Convert relative 3D scene flow into absolute 3D positions
        # by adding the reference-frame point cloud.
        flow3d = flows3d_e.to(torch.float32).cuda() + points[None, 0:1]
        flow3d = flow3d.permute(0, 1, 3, 4, 2)
        
        return_dict = []
        
        # Reorder tensors into (B, T, H, W, C) format for geometry processing.
        points = points[None].permute(0, 1, 3, 4, 2)[:, :T]
        mask = mask[None].permute(0, 1, 3, 4, 2)[..., 0][:, :T]
        world_points = world_points[None].permute(0, 1, 3, 4, 2)[:, :T]
        
        # Prepare a copy of 2D flow in a convenient layout for projection/unprojection.
        flow2d_c = flow2d.clone()
        flow2d_c = flow2d_c.permute(0, 1, 3, 4, 2)
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            # Ensure that all tensors participating in geometric computation
            # are promoted to float32 for numerical stability.
            points, mask = map(
                lambda x: x.float() if isinstance(x, torch.Tensor) else x, 
                [points, mask]
            )
            
            # Threshold the soft mask to obtain a binary validity mask.
            mask_binary = mask > self.mask_threshold
            
            # Estimate global focal length and depth shift from valid 3D points.
            # This step recovers metric scale and camera intrinsics.
            focal, shift = recover_global_focal_shift(points, mask_binary)

            # Construct camera intrinsics assuming a normalized principal point.
            fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
            fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            intrinsics = utils3d.torch.intrinsics_from_focal_center(
                fx, fy, 0.5, 0.5
            ).repeat(1, flow3d.shape[1], 1, 1)
            
            # Apply the estimated global shift to convert relative depth to metric depth.
            depth = points[..., 2] + shift[..., None, None, None].repeat(
                1, points.shape[1], 1, 1
            )

            if force_projection:
                # Recompute 3D points from depth and intrinsics to enforce
                # consistency between depth and camera parameters.
                points = utils3d.torch.depth_to_points(
                    depth, intrinsics=intrinsics, use_ray=False
                )
            else:
                # Simply translate the existing points along the Z-axis.
                shift_vec = torch.stack(
                    [torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1
                )
                points = points + shift_vec[..., None, None, None, :].repeat(
                    1, points.shape[1], 1, 1, 1
                )

            # Optionally mask out invalid regions by assigning infinite values.
            if apply_mask:
                points = torch.where(mask_binary[..., None], points, torch.inf)
                world_points = torch.where(mask_binary[..., None], world_points, torch.inf)
                depth = torch.where(mask_binary, depth, torch.inf)

            # Store reconstructed scene geometry and camera information.
            return_dict.append({
                'points': points, 
                'intrinsics': intrinsics, 
                'depth': depth, 
                'mask': mask_binary, 
                'world_points': world_points, 
                'camera_poses': camera_poses
            })
            
            # Compute forward (next-frame) depth from the predicted 3D scene flow.
            forward_depth = flow3d[..., 2] + shift[..., None, None, None].repeat(
                1, flow3d.shape[1], 1, 1
            )
            
            if force_projection:
                # Normalize pixel coordinates before unprojection.
                flow2d_c[..., 0] /= flow2d_c.shape[-2]
                flow2d_c[..., 1] /= flow2d_c.shape[-3]
                
                # Unproject 2D flow + depth to obtain metric 3D flow points.
                points_f = utils3d.torch.unproject_cv(
                    flow2d_c, forward_depth, 
                    intrinsics=intrinsics[..., None, :, :], use_ray=False
                )
            else:
                # Apply the depth shift directly in 3D space.
                shift_vec = torch.stack(
                    [torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1
                )
                points_f = flow3d + shift_vec[..., None, None, None, :]

            # Restore pixel-scale coordinates after unprojection.
            flow2d_c[..., 0] *= flow2d_c.shape[-2]
            flow2d_c[..., 1] *= flow2d_c.shape[-3]
            
            # Store motion-related outputs.
            return_dict.append({
                'flow_3d': points_f, 
                'flow_2d': flow2d, 
                'visconf_maps_e': visconf_maps_e, 
                'intrinsics': intrinsics, 
                'depth': forward_depth, 
                'mask': mask_binary
            })

        return return_dict


    @torch.inference_mode()
    def evaluation(
        self, 
        images, 
        iters=4, 
        sw=None, 
        is_training=False, 
        window_len=None, 
        stride=None, 
        tracking3d=False, 
        apply_mask: bool = False,
        force_projection: bool = True, 
        use_fp16: bool = True, 
        current_batch_size: int = 4,
        local: bool = False, 
        no_shift: bool = False, 
        eval_dict=None
    ) -> Dict[str, torch.Tensor]:
        """
        Run evaluation on an input video batch and return structured geometric outputs.

        This function is conceptually similar to `infer`, but is designed for evaluation
        scenarios where intermediate features (stored in `eval_dict`) may be reused
        across multiple runs on the same sequence to avoid redundant computation.

        Inputs:
            images: Tensor of shape (B, T, C, H, W), representing a batch of video clips.
            iters: Number of refinement iterations for the underlying model.
            sw: Optional sliding-window configuration.
            is_training: Flag indicating whether the model is in training mode.
            window_len / stride: Sliding window parameters (if applicable).
            tracking3d: Whether to additionally estimate dense 3D motion.
            apply_mask: If True, invalid regions are masked out using infinite values.
            force_projection: If True, explicitly re-project depth to 3D points using intrinsics.
            use_fp16: Whether mixed precision is enabled.
            current_batch_size: Effective batch size during evaluation.
            local / no_shift: Flags controlling evaluation behavior (kept for compatibility).
            eval_dict: Optional cache for reusing computed features.

        Outputs:
            return_dict: A list of dictionaries containing reconstructed geometry,
                        motion, camera intrinsics, and validity masks.
            eval_dict: Updated feature cache for future reuse.
        """
        (
            flows_e, visconf_maps_e, _, _, flows3d_e, _, 
            points, mask, eval_dict, _, _
        ) = self.forward_sliding(
            images, iters=iters, sw=sw, is_training=is_training, 
            tracking3d=tracking3d, eval_dict=eval_dict
        )
        
        B, T, C, H, W = images.shape
        aspect_ratio = W / H
        
        # Generate a dense 2D pixel grid in image coordinates (unnormalized),
        # which will later be used to convert optical flow into absolute positions.
        grid_xy = holi4d.utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float()
        grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)
        
        # Convert predicted 2D flow from displacement form to absolute pixel locations.
        flow2d = flows_e.to(torch.float32).cuda() + grid_xy 
        visconf_maps_e = visconf_maps_e.to(torch.float32)
        
        # Convert predicted 3D flow into absolute 3D positions by adding reference points.
        flow3d = flows3d_e.to(torch.float32).cuda() + points[None, 0:1]
        flow3d = flow3d.permute(0, 1, 3, 4, 2)
        
        return_dict = []
        
        # Reformat point cloud and mask tensors to (B, T, H, W, C) layout
        # for more convenient downstream processing.
        points = points[None].permute(0, 1, 3, 4, 2)[:, :T]
        mask = mask[None].permute(0, 1, 3, 4, 2)[..., 0][:, :T]
        
        # Keep a copy of 2D flow in (B, T, H, W, 2) format for projection steps.
        flow2d_c = flow2d.clone()
        flow2d_c = flow2d_c.permute(0, 1, 3, 4, 2)

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            # Ensure all tensors are in float32 for numerically stable geometry computation.
            points, mask = map(
                lambda x: x.float() if isinstance(x, torch.Tensor) else x, 
                [points, mask]
            )
            
            # Convert soft mask into a binary validity mask.
            mask_binary = mask > self.mask_threshold
            
            # Estimate global focal length and depth shift from visible 3D points.
            focal, shift = recover_global_focal_shift(points, mask_binary)

            # Recover camera intrinsics (fx, fy) assuming normalized principal point (0.5, 0.5).
            fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
            fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            intrinsics = utils3d.torch.intrinsics_from_focal_center(
                fx, fy, 0.5, 0.5
            ).repeat(1, flow3d.shape[1], 1, 1)
            
            # Apply the estimated global depth shift to obtain absolute depth values.
            depth = points[..., 2] + shift[..., None, None, None].repeat(
                1, points.shape[1], 1, 1
            )

            if force_projection:
                # Reconstruct 3D points explicitly from depth and camera intrinsics.
                points = utils3d.torch.depth_to_points(
                    depth, intrinsics=intrinsics, use_ray=False
                )
            else:
                # Alternatively, directly shift the existing 3D points along the z-axis.
                shift_vec = torch.stack(
                    [torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1
                )
                points = points + shift_vec[..., None, None, None, :]

            if apply_mask:
                # Mask out invalid regions by assigning infinite values,
                # making them easy to ignore in later processing.
                points = torch.where(mask_binary[..., None], points, torch.inf)
                depth = torch.where(mask_binary, depth, torch.inf)

            # Store reconstructed geometry for the reference frame(s).
            return_dict.append({
                'points': points, 
                'intrinsics': intrinsics, 
                'depth': depth, 
                'mask': mask_binary
            })
            
            # Compute forward depth by applying the same global shift to the predicted 3D flow.
            forward_depth = flow3d[..., 2] + shift[..., None, None, None].repeat(
                1, flow3d.shape[1], 1, 1
            )
            
            if force_projection:
                # Normalize 2D coordinates before unprojection.
                flow2d_c[..., 0] /= flow2d_c.shape[-2]
                flow2d_c[..., 1] /= flow2d_c.shape[-3]
                
                # Unproject 2D positions with depth into 3D space.
                points_f = utils3d.torch.unproject_cv(
                    flow2d_c, forward_depth, 
                    intrinsics=intrinsics[..., None, :, :], use_ray=False
                )
            else:
                # Directly apply the depth shift to the 3D flow representation.
                shift_vec = torch.stack(
                    [torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1
                )
                points_f = flow3d + shift_vec[..., None, None, None, :]

            # Restore pixel-scale coordinates after unprojection.
            flow2d_c[..., 0] *= flow2d_c.shape[-2]
            flow2d_c[..., 1] *= flow2d_c.shape[-3]
            
            # Store motion-related outputs for evaluation.
            return_dict.append({
                'flow_3d': points_f, 
                'flow_2d': flow2d, 
                'visconf_maps_e': visconf_maps_e, 
                'intrinsics': intrinsics, 
                'depth': forward_depth, 
                'mask': mask_binary
            })

        return return_dict, eval_dict


    @torch.inference_mode()
    def infer_pair(
        self, 
        images, 
        iters=4, 
        sw=None, 
        is_training=False, 
        window_len=None, 
        stride=None, 
        tracking3d=False, 
        apply_mask: bool = False,
        force_projection: bool = True, 
        use_fp16: bool = True, 
        current_batch_size: int = 4,
        local: bool = False, 
        no_shift: bool = False, 
        aligned_scene_flow: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Pairwise inference function designed for consecutive-frame processing.

        In contrast to the standard `infer` interface (which focuses on full-sequence
        reconstruction), this function explicitly targets *pairwise* relationships
        between neighboring frames. It is therefore well-suited for:
            - frame-to-frame motion estimation
            - short-term 3D scene flow analysis
            - memory-efficient inference on long sequences

        The function internally uses a memory-optimized sliding-window forward pass
        and produces geometry and motion outputs aligned to frame pairs (t → t+1).

        Inputs:
            images: Input video tensor of shape (B, T, C, H, W).
            iters: Number of iterative refinement steps in the network.
            sw: Sliding-window configuration for temporal chunking.
            is_training: Whether to enable training-time behaviors.
            window_len / stride: Temporal windowing parameters (reserved).
            tracking3d: Enable dense 3D scene flow prediction.
            apply_mask: Whether to invalidate unreliable regions using the predicted mask.
            force_projection: If True, recompute 3D points via depth + intrinsics
                            to enforce geometric consistency.
            use_fp16: Enable mixed-precision inference.
            current_batch_size: Effective batch size used by the model.
            local / no_shift: Optional flags for alternative inference modes.
            aligned_scene_flow: If True, temporally align 3D flow directions to
                                reduce frame-to-frame inconsistencies.

        Outputs:
            A list of dictionaries containing:
                (1) Per-frame geometry:
                    - points: Metric 3D points in camera coordinates.
                    - world_points: Metric 3D points in a global coordinate frame.
                    - depth: Depth maps after global shift correction.
                    - intrinsics: Estimated camera intrinsics.
                    - mask: Binary validity mask.
                    - camera_poses: Estimated camera poses.
                (2) Pairwise motion:
                    - flow_2d: Absolute 2D optical flow.
                    - flow_3d: Metric 3D scene flow (t → t+1).
                    - visconf_maps_e: Visibility / confidence maps.
        """

        # -------------------------------------------------------------
        # Forward pass using a memory-efficient pairwise sliding window
        # -------------------------------------------------------------
        (
            flows_e,                 # (T-1, 2, H, W): 2D optical flow
            visconf_maps_e,          # (T-1, 1, H, W): visibility / confidence
            _, _, 
            flows3d_e,               # (T-1, 3, H, W): 3D scene flow
            _, 
            points,                  # (T, 3, H, W): per-frame 3D point maps
            mask,                    # (T, 1, H, W): validity mask
            world_points,            # (T, 3, H, W): world-coordinate points
            camera_poses             # (T, ...): camera poses
        ) = self.forward_sliding1(
            images, iters=iters, sw=sw, is_training=is_training, tracking3d=tracking3d
        )
        
        # -------------------------------------------------------------
        # Basic shape bookkeeping
        # -------------------------------------------------------------
        B, T, C, H, W = images.shape
        aspect_ratio = W / H
        
        # -------------------------------------------------------------
        # Build an absolute pixel grid for converting flow offsets
        # into absolute image coordinates
        # -------------------------------------------------------------
        grid_xy = holi4d.utils.basic.gridcloud2d(
            1, H, W, norm=False, device='cuda:0'
        ).float()
        grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)
        
        # -------------------------------------------------------------
        # Convert 2D flow from displacement to absolute pixel positions
        # -------------------------------------------------------------
        flow2d = flows_e[None].to(torch.float32).cuda() + grid_xy
        visconf_maps_e = visconf_maps_e[None].to(torch.float32)

        # -------------------------------------------------------------
        # Convert relative 3D flow to absolute 3D positions
        # (reference frame + scene flow)
        # -------------------------------------------------------------
        flow3d = flows3d_e.to(torch.float32).cuda()[None] + points[None, 0:-1]
        flow3d = flow3d.permute(0, 1, 3, 4, 2)  # (B, T-1, H, W, 3)
        
        return_dict = []

        # -------------------------------------------------------------
        # Reformat tensors to a geometry-friendly layout
        # -------------------------------------------------------------
        points = points[None].permute(0, 1, 3, 4, 2)[:, :T]
        world_points = world_points[None].permute(0, 1, 3, 4, 2)[:, :T]
        mask = mask[None].permute(0, 1, 3, 4, 2)[..., 0][:, :T]

        flow2d_c = flow2d.clone().permute(0, 1, 3, 4, 2)

        # -------------------------------------------------------------
        # Optional temporal alignment of scene flow
        # This step reduces directional jitter across time
        # -------------------------------------------------------------
        if aligned_scene_flow:
            flow3d = get_aligned_scene_flow_temporal(
                flow2d_c,
                flow3d,
                points,
                visconf_maps_e.cuda().permute(0, 1, 3, 4, 2),
                mode='align_dir'
            )

        # -------------------------------------------------------------
        # Geometry processing in float32 for numerical stability
        # -------------------------------------------------------------
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):

            # Ensure float precision for all geometric tensors
            points, mask = map(
                lambda x: x.float() if isinstance(x, torch.Tensor) else x, 
                [points, mask]
            )

            # Threshold soft mask to obtain binary validity mask
            mask_binary = mask > self.mask_threshold
            
            # ---------------------------------------------------------
            # Estimate global focal length and depth shift
            # from the predicted point cloud
            # ---------------------------------------------------------
            focal, shift = recover_global_focal_shift(points, mask_binary)

            # Convert normalized focal to pixel-space fx / fy
            fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
            fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5

            # Construct camera intrinsics for all frames
            intrinsics = utils3d.torch.intrinsics_from_focal_center(
                fx, fy, 0.5, 0.5
            ).repeat(1, points.shape[1], 1, 1)
            
            # Recover metric depth by applying the global shift
            depth = points[..., 2] + shift[..., None, None, None].repeat(
                1, points.shape[1], 1, 1
            )

            # ---------------------------------------------------------
            # Enforce geometric consistency via reprojection if required
            # ---------------------------------------------------------
            if force_projection:
                points = utils3d.torch.depth_to_points(
                    depth, intrinsics=intrinsics, use_ray=False
                )
            else:
                shift_vec = torch.stack(
                    [torch.zeros_like(shift),
                    torch.zeros_like(shift),
                    shift],
                    dim=-1
                )
                points = points + shift_vec[..., None, None, :]

            # Optionally mask invalid regions
            if apply_mask:
                points = torch.where(mask_binary[..., None], points, torch.inf)
                world_points = torch.where(mask_binary[..., None], world_points, torch.inf)
                depth = torch.where(mask_binary, depth, torch.inf)

            # Store per-frame reconstructed geometry
            return_dict.append({
                'points': points,
                'intrinsics': intrinsics,
                'depth': depth,
                'mask': mask_binary,
                'world_points': world_points,
                'camera_poses': camera_poses
            })
            
            # ---------------------------------------------------------
            # Compute forward (t → t+1) depth and 3D flow
            # ---------------------------------------------------------
            forward_depth = flow3d[..., 2] + shift[..., None, None, None].repeat(
                1, flow3d.shape[1], 1, 1
            )
            
            if force_projection:
                # Normalize pixel coordinates before unprojection
                flow2d_c[..., 0] /= flow2d_c.shape[-2]
                flow2d_c[..., 1] /= flow2d_c.shape[-3]

                points_f = utils3d.torch.unproject_cv(
                    flow2d_c,
                    forward_depth,
                    intrinsics=intrinsics[:, :-1][..., None, :, :],
                    use_ray=False
                )
            else:
                shift_vec = torch.stack(
                    [torch.zeros_like(shift),
                    torch.zeros_like(shift),
                    shift],
                    dim=-1
                )
                points_f = flow3d + shift_vec[..., None, None, :]

            # Apply mask to forward flow if requested
            if apply_mask:
                points_f = torch.where(
                    mask_binary[:, :-1, :, :, None], points_f, torch.inf
                )
                forward_depth = torch.where(
                    mask_binary[:, :-1], forward_depth, torch.inf
                )
            
            # Restore pixel-scale flow coordinates
            flow2d_c[..., 0] *= flow2d_c.shape[-2]
            flow2d_c[..., 1] *= flow2d_c.shape[-3]
            
            # Store pairwise motion results
            return_dict.append({
                'flow_3d': points_f,
                'flow_2d': flow2d,
                'visconf_maps_e': visconf_maps_e,
                'intrinsics': intrinsics,
                'depth': forward_depth,
                'mask': mask_binary
            })

        return return_dict
