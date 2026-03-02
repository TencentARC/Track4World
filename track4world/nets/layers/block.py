# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from ..dinov2.layers.drop_path import DropPath
from ..dinov2.layers.layer_scale import LayerScale
from ..dinov2.layers.mlp import Mlp
from ..dinov2.layers.block import *
from .attention import Attention
# Flag indicating whether xFormers (memory-efficient attention) is available
XFORMERS_AVAILABLE = False


class Block(nn.Module):
    """
    Standard Transformer block consisting of:
        - LayerNorm + Multi-Head Self-Attention
        - LayerNorm + Feed-Forward Network (MLP)
        - Optional LayerScale and Stochastic Depth (DropPath)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__()

        # 1. Attention Branch
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 2. MLP Branch
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 3. Drop Path Configuration
        self.sample_drop_ratio = drop_path
        # Pre-calculate condition to avoid repeated checks in forward
        self.use_optimized_drop = (drop_path > 0.1)

    def _forward_attn(self, x: Tensor, pos=None) -> Tensor:
        """Helper to compute the attention residual content."""
        return self.ls1(self.attn(self.norm1(x), pos=pos))

    def _forward_mlp(self, x: Tensor) -> Tensor:
        """Helper to compute the MLP residual content."""
        return self.ls2(self.mlp(self.norm2(x)))

    def forward(self, hidden_states: Tensor, pos=None) -> Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (B, N, C)
            pos: Optional positional encoding
        """
        
        # --- Attention Block ---
        if self.training and self.use_optimized_drop:
            # Optimized path for high drop rates (fused drop+add)
            hidden_states = drop_add_residual_stochastic_depth(
                hidden_states,
                pos=pos,
                residual_func=self._forward_attn,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        else:
            # Standard path
            hidden_states = hidden_states + self.drop_path1(self._forward_attn(hidden_states, pos=pos))

        # --- MLP Block ---
        if self.training and self.use_optimized_drop:
            # Optimized path
            hidden_states = drop_add_residual_stochastic_depth(
                hidden_states,
                residual_func=self._forward_mlp,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        else:
            # Standard path
            hidden_states = hidden_states + self.drop_path2(self._forward_mlp(hidden_states))

        return hidden_states

def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
    pos=None,
) -> Tensor:
    """
    Applies stochastic depth by computing residuals only on a subset
    of batch samples and scaling them accordingly.

    This reduces computation while preserving expectation.
    """

    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)