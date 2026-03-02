import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from itertools import repeat
import collections
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from functools import partial
import einops
import math
from torchvision.ops.misc import Conv2dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth

# --- Utility Functions ---

def _ntuple(n):
    """
    Creates a function that repeats a value n times if it's not already 
    an iterable. Used for handling kernel sizes, strides, etc.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

def exists(val):
    """Checks if a value is not None."""
    return val is not None

def default(val, d):
    """Returns val if it exists, otherwise returns default d."""
    return val if exists(val) else d

to_2tuple = _ntuple(2)

class InputPadder:
    """ 
    Pads images such that dimensions are divisible by a certain stride (default 64).
    Useful for architectures that require specific input resolutions (e.g., RAFT).
    """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        
        # Calculate padding to make height and width divisible by 64
        pad_ht = (((self.ht // 64) + 1) * 64 - self.ht) % 64
        pad_wd = (((self.wd // 64) + 1) * 64 - self.wd) % 64
        
        if mode == 'sintel':
            self._pad = [
                pad_wd // 2, pad_wd - pad_wd // 2, 
                pad_ht // 2, pad_ht - pad_ht // 2
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        """Pads a list of input tensors."""
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        """Crops the padded tensor back to original dimensions."""
        ht, wd = x.shape[-2:]
        c = [
            self._pad[2], ht - self._pad[3], 
            self._pad[0], wd - self._pad[1]
        ]
        return x[..., c[0]:c[1], c[2]:c[3]]

def bilinear_sampler(
        input: torch.Tensor, 
        coords: torch.Tensor,
        align_corners: bool = True,
        padding_mode: str = "border",
        normalize_coords: bool = True) -> torch.Tensor:
    """
    Wrapper around grid_sample with coordinate normalization.
    
    Args:
        input: Input feature map (N, C, H, W) or (N, C, D, H, W).
        coords: Coordinates for sampling.
        normalize_coords: If True, normalizes coords from [0, W] to [-1, 1].
    """
    if input.ndim not in [4, 5]:
        raise ValueError("input must be 4D or 5D.")
    
    if input.ndim == 4 and not coords.ndim == 4:
        raise ValueError("input is 4D, but coords is not 4D.")

    if input.ndim == 5 and not coords.ndim == 5:
        raise ValueError("input is 5D, but coords is not 5D.")

    # If 5D, rearrange coords to match grid_sample expectation (x, y, t)
    if coords.ndim == 5:
        coords = coords[..., [1, 2, 0]] 

    if normalize_coords:
        # Calculate normalization factors based on input shape
        shape_rev = reversed(input.shape[2:])
        device = coords.device
        
        if align_corners:
            # Normalize coordinates from [0, W/H - 1] to [-1, 1].
            size_tensor = torch.tensor(
                [2 / max(size - 1, 1) for size in shape_rev], 
                device=device
            )
            coords = coords * size_tensor - 1
        else:
            # Normalize coordinates from [0, W/H] to [-1, 1].
            size_tensor = torch.tensor(
                [2 / size for size in shape_rev], 
                device=device
            )
            coords = coords * size_tensor - 1
            
    return F.grid_sample(
        input, coords, 
        align_corners=align_corners, 
        padding_mode=padding_mode
    )


class CorrBlock:
    """
    Correlation Block for RAFT-like architectures.
    Computes all-pairs correlation between fmap1 and fmap2 and builds a pyramid.
    Allows looking up correlation values based on current flow estimates.
    """
    def __init__(
        self, fmap1, fmap2, corr_levels, corr_radius, mode='area', save_fmap=False
    ):
        self.num_levels = corr_levels
        self.radius = corr_radius
        self.save_fmap = save_fmap
        self.corr_pyramid = []
        if save_fmap:
            self.fmap_pyramid = []
            
        # Compute all-pairs correlation
        for i in range(self.num_levels):
            corr = CorrBlock.corr(fmap1, fmap2, 1)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
            
            if save_fmap:
                self.fmap_pyramid.append(fmap2)
            
            # Downsample fmap2 for the next level of the pyramid
            fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode=mode)
            self.corr_pyramid.append(corr)

    def __call__(self, coords, dilation=None):
        """
        Sample from the correlation pyramid at the given coordinates.
        """
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        if dilation is None:
            dilation = torch.ones(batch, 1, h1, w1, device=coords.device)
            
        if self.save_fmap:
            out_fmap_pyramid = []
        out_pyramid = []
        
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            if self.save_fmap:
                fmap = self.fmap_pyramid[i]
                fmap = fmap.repeat(h1 * w1, 1, 1, 1)
                
            device = coords.device
            # Create a grid around the target coordinate
            dx = torch.linspace(-r, r, 2 * r + 1, device=device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)
            
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            delta_lvl = delta_lvl * dilation.view(batch * h1 * w1, 1, 1, 1)
            
            # Scale coordinates for the current pyramid level
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + delta_lvl.to(centroid_lvl.dtype)
            
            # Sample correlation
            corr = bilinear_sampler(corr, coords_lvl)
            
            if self.save_fmap:
                fmap = bilinear_sampler(fmap, coords_lvl)
                fmap = fmap.view(batch, h1, w1, -1, (2 * r + 1) * (2 * r + 1))
                out_fmap_pyramid.append(fmap)
                
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous()
        
        if self.save_fmap:
            out_fmap = torch.cat(out_fmap_pyramid, dim=-1).contiguous()
            return out, out_fmap
        else:
            return out

    @staticmethod
    def corr(fmap1, fmap2, num_head):
        """Computes correlation between two feature maps."""
        batch, dim, h1, w1 = fmap1.shape
        h2, w2 = fmap2.shape[2:]
        fmap1 = fmap1.view(batch, num_head, dim // num_head, h1 * w1)
        fmap2 = fmap2.view(batch, num_head, dim // num_head, h2 * w2) 
        
        # Matrix multiplication to get correlation
        corr = fmap1.transpose(2, 3) @ fmap2
        corr = corr.reshape(batch, num_head, h1, w1, h2, w2)
        corr = corr.permute(0, 2, 3, 1, 4, 5)
        return corr / torch.sqrt(torch.tensor(dim)).to(corr.dtype)
    
# --- Basic Layers ---

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0
    )

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1
    )

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for 2D images (N, C, H, W)."""
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x
    
class CNBlock1d(nn.Module):
    """
    ConvNeXt-style block adapted for 1D sequences (Temporal).
    Can use Attention, MLP-Mixer, or 1D Convolutions.
    """
    def __init__(
        self,
        dim,
        output_dim,
        layer_scale: float = 1e-6,
        stochastic_depth_prob: float = 0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dense=True,
        use_attn=True,
        use_mixer=False,
        use_conv=False,
        use_convb=False,
        use_layer_scale=True,
    ) -> None:
        super().__init__()
        self.dense = dense
        self.use_attn = use_attn
        self.use_mixer = use_mixer
        self.use_conv = use_conv
        self.use_layer_scale = use_layer_scale

        # Ensure only one mode is selected
        if use_attn:
            assert not use_mixer and not use_conv and not use_convb
        
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Initialize the core block based on selected mode
        if use_attn:
            num_heads = 8
            self.block = AttnBlock(
                hidden_size=dim,
                num_heads=num_heads,
                mlp_ratio=4,
                attn_class=Attention,
            )
        elif use_mixer:
            self.block = MLPMixerBlock(
                S=16,
                dim=dim,
                depth=1,
                expansion_factor=2,
            )
        elif use_conv:
            self.block = nn.Sequential(
                nn.Conv1d(
                    dim, dim, kernel_size=7, padding=3, 
                    groups=dim, bias=True, padding_mode='zeros'
                ),
                Permute([0, 2, 1]),
                norm_layer(dim),
                nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
                nn.GELU(),
                nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
                Permute([0, 2, 1]),
            )
        elif use_convb:
            self.block = nn.Sequential(
                nn.Conv1d(
                    dim, dim, kernel_size=3, padding=1, 
                    bias=True, padding_mode='zeros'
                ),
                Permute([0, 2, 1]),
                norm_layer(dim),
                nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
                nn.GELU(),
                nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
                Permute([0, 2, 1]),
            )
        else:
            raise ValueError("Must choose attn, mixer, or conv.")

        if self.use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim, 1) * layer_scale)
        else:
            self.layer_scale = 1.0
            
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        if output_dim != dim:
            self.final = nn.Conv1d(dim, output_dim, kernel_size=1, padding=0)
        else:
            self.final = nn.Identity()

    def forward(self, input, S=None):
        if self.dense:
            # Reshape (B*S, C, H, W) -> (B*H*W, C, S) for temporal processing
            assert S is not None
            BS, C, H, W = input.shape
            B = BS // S

            input = einops.rearrange(
                input, '(b s) c h w -> (b h w) c s', 
                b=B, s=S, c=C, h=H, w=W
            )

            if self.use_mixer or self.use_attn:
                # mixer/transformer blocks expect (Batch, Seq, Channels)
                result = self.block(input.permute(0, 2, 1)).permute(0, 2, 1)
                result = self.layer_scale * result
            else:
                result = self.layer_scale * self.block(input)
            
            result = self.stochastic_depth(result)
            result += input
            result = self.final(result)

            result = einops.rearrange(
                result, '(b h w) c s -> (b s) c h w', 
                b=B, s=S, c=C, h=H, w=W
            )
        else:
            # Standard 1D processing
            B, S, C = input.shape

            if S < 7:
                return input

            input = einops.rearrange(input, 'b s c -> b c s', b=B, s=S, c=C)

            result = self.layer_scale * self.block(input)
            result = self.stochastic_depth(result)
            result += input

            result = self.final(result)

            result = einops.rearrange(result, 'b c s -> b s c', b=B, s=S, c=C)
        
        return result
    
class CNBlock2d(nn.Module):
    """
    Standard ConvNeXt Block for 2D spatial processing.
    """
    def __init__(
        self,
        dim,
        output_dim,
        layer_scale: float = 1e-6,
        stochastic_depth_prob: float = 0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_layer_scale=True,
    ) -> None:
        super().__init__()
        self.use_layer_scale = use_layer_scale
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(
                dim, dim, kernel_size=7, padding=3, 
                groups=dim, bias=True, padding_mode='zeros'
            ),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        if self.use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        else:
            self.layer_scale = 1.0
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        
        if output_dim != dim:
            self.final = nn.Conv2d(dim, output_dim, kernel_size=1, padding=0)
        else:
            self.final = nn.Identity()

    def forward(self, input, S=None):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        result = self.final(result)
        return result

class CNBlockConfig:
    """Stores configuration for ConvNeXt blocks."""
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
        downsample: bool,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.downsample = downsample

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ", downsample={downsample}"
        s += ")"
        return s.format(**self.__dict__)
    
class ConvNeXt(nn.Module):
    """
    ConvNeXt Backbone implementation.
    Can initialize weights from torchvision's pre-trained models.
    """
    def __init__(
            self,
            block_setting: List[CNBlockConfig],
            stochastic_depth_prob: float = 0.0,
            layer_scale: float = 1e-6,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            init_weights=True):
        super().__init__()

        self.init_weights = init_weights
        
        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock2d

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem Layer
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # Adjust stochastic depth probability based on depth
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            
            if cnf.out_channels is not None:
                if cnf.downsample:
                    layers.append(
                        nn.Sequential(
                            norm_layer(cnf.input_channels),
                            nn.Conv2d(
                                cnf.input_channels, cnf.out_channels, 
                                kernel_size=2, stride=2
                            ),
                        )
                    )
                else:
                    # Convert 2x2 downsampling to 3x3 with dilation 
                    # to maintain resolution but increase receptive field
                    layers.append(
                        nn.Sequential(
                            norm_layer(cnf.input_channels),
                            nn.Conv2d(
                                cnf.input_channels, cnf.out_channels, 
                                kernel_size=3, stride=1, padding=2, 
                                dilation=2, padding_mode='zeros'
                            ),
                        )
                    )

        self.features = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.init_weights:
            # Load pre-trained weights from torchvision
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            pretrained_dict = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            for k, v in pretrained_dict.items():
                if k == 'features.4.1.weight': # This layer is normally 2x2 downsampling
                    # Convert to 3x3 filter for the dilated convolution case
                    pretrained_dict[k] = F.interpolate(
                        v, (3, 3), mode='bicubic', align_corners=True
                    ) * (4 / 9.0)
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
        

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# --- Transformer Components ---

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    """Standard Multi-Head Attention."""
    def __init__(
        self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = num_heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_bias=None):
        B, N1, C = x.shape
        H = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        # Rearrange for multi-head attention
        q, k, v = map(
            lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.heads), 
            (q, k, v)
        )
        
        # Use PyTorch's optimized scaled dot product attention
        x = F.scaled_dot_product_attention(q, k, v) 
        x = einops.rearrange(x, 'b h n d -> b n (h d)')
        return self.to_out(x)
    
class CrossAttnBlock(nn.Module):
    """Block containing Cross-Attention and MLP."""
    def __init__(
        self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context, mask=None):
        attn_bias = None
        if mask is not None:
            # Handle mask expansion for multi-head attention
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(
                    -1, self.cross_attn.heads, x.shape[1], -1
                )

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
            
        x = x + self.cross_attn(
            self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias
        )
        x = x + self.mlp(self.norm2(x))
        return x
    
class AttnBlock(nn.Module):
    """Block containing Self-Attention and MLP."""
    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_class: Callable[..., nn.Module] = Attention,
        mlp_ratio=4.0,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_class(
            hidden_size, num_heads=num_heads, qkv_bias=True, 
            dim_head=hidden_size // num_heads
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, mask=None):
        attn_bias = mask
        if mask is not None:
            mask = (
                (mask[:, None] * mask[:, :, None])
                .unsqueeze(1)
                .expand(-1, self.attn.num_heads, -1, -1)
            )
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value

        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x
    
# --- ResNet Components ---

class ResidualBlock(nn.Module):
    """Standard ResNet Residual Block."""
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), 
                self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    """Basic ResNet-style encoder for feature extraction."""
    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4,
            output_dim * 2,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        def _bilinear_intepolate(x):
            return F.interpolate(
                x,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )

        a = _bilinear_intepolate(a)
        b = _bilinear_intepolate(b)
        c = _bilinear_intepolate(c)
        d = _bilinear_intepolate(d)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x

# --- Update Formers (Tracking) ---

class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates using virtual tracks and attention.
    """
    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        num_virtual_tracks=64,
        add_space_attn=True,
        linear_layer_for_vis_conf=False,
        use_time_conv=False,
        use_time_mixer=False,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        
        if linear_layer_for_vis_conf:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim - 2, bias=True)
            self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
            
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(
            torch.randn(1, num_virtual_tracks, 1, hidden_size)
        )
        self.add_space_attn = add_space_attn
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf

        # Initialize Time Blocks
        if use_time_conv:
            self.time_blocks = nn.ModuleList(
                [CNBlock1d(hidden_size, hidden_size, dense=False) 
                 for _ in range(time_depth)]
            )
        elif use_time_mixer:
            self.time_blocks = nn.ModuleList(
                [MLPMixerBlock(S=16, dim=hidden_size, depth=1) 
                 for _ in range(time_depth)]
            )
        else:
            self.time_blocks = nn.ModuleList(
                [AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=Attention) 
                 for _ in range(time_depth)]
            )

        # Initialize Space Blocks
        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=Attention) 
                 for _ in range(space_depth)]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) 
                 for _ in range(space_depth)]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) 
                 for _ in range(space_depth)]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            if self.linear_layer_for_vis_conf:
                torch.nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None, add_space_attn=True):
        tokens = self.input_transform(input_tensor)

        B, _, T, _ = tokens.shape
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape
        j = 0
        
        for i in range(len(self.time_blocks)):
            # Time processing
            # B N T C -> (B N) T C
            time_tokens = tokens.contiguous().view(B * N, T, -1)  
            time_tokens = self.time_blocks[i](time_tokens)
            # (B N) T C -> B N T C
            tokens = time_tokens.view(B, N, T, -1)  
            
            # Space processing (interleaved)
            if (
                add_space_attn
                and hasattr(self, "space_virtual_blocks")
                and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0)
            ):
                # B N T C -> (B T) N C
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  

                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                # Cross attention between points and virtual tracks
                virtual_tokens = self.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, mask=mask
                )

                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens, mask=mask
                )

                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                # (B T) N C -> B N T C
                tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)  
                j += 1
                
        tokens = tokens[:, : N - self.num_virtual_tracks]

        flow = self.flow_head(tokens)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        return flow


class MMPreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x
    
def MMFeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout) 
    )

def MLPMixerBlock(S, dim, depth=1, expansion_factor=4, dropout=0., do_reduce=False):
    """Single block of MLP-Mixer."""
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
    return nn.Sequential(
        *[nn.Sequential(
            MMPreNormResidual(
                dim, MMFeedForward(S, expansion_factor, dropout, chan_first)
            ),
            MMPreNormResidual(
                dim, MMFeedForward(dim, expansion_factor, dropout, chan_last)
            )
        ) for _ in range(depth)],
    )
# --- Flow Update Blocks (RAFT-style) ---
# These blocks iteratively update flow features using correlation,
# motion encoding, and temporal/spatial refinement.

class BasicMotionEncoder(nn.Module):
    """
    Encodes motion information by jointly processing:
      - Correlation volume (matching cost)
      - Current flow estimate (2D / 3D / high-dim parameterization)

    Output features are concatenated with the original flow
    to preserve low-level motion cues (RAFT-style design).
    """
    def __init__(self, corr_channel, dim=128, pdim=2):
        super(BasicMotionEncoder, self).__init__()
        self.pdim = pdim

        # Encode correlation volume
        self.convc1 = nn.Conv2d(corr_channel, dim * 4, 1, padding=0)
        self.convc2 = nn.Conv2d(dim * 4, dim + dim // 2, 3, padding=1)

        if pdim == 2 or pdim == 4:
            # Explicit encoding for low-dimensional flow (e.g., 2D or 4D)
            self.convf1 = nn.Conv2d(pdim, dim * 2, 5, padding=2)
            self.convf2 = nn.Conv2d(dim * 2, dim // 2, 3, padding=1)
            self.conv = nn.Conv2d(dim * 2, dim - pdim, 3, padding=1)
        else:
            # Flow is already high-dimensional (e.g., 3D or factorized params)
            self.conv = nn.Conv2d(dim + dim // 2 + pdim, dim, 3, padding=1)

    def forward(self, flow, corr):
        """
        Args:
            flow: current flow / motion estimate (B, pdim, H, W)
            corr: correlation features (B, C_corr, H, W)

        Returns:
            Motion features concatenated with raw flow.
        """
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))

        if self.pdim == 2 or self.pdim == 4:
            # Encode flow explicitly and fuse with correlation
            flo = F.relu(self.convf1(flow))
            flo = F.relu(self.convf2(flo))
            cor_flo = torch.cat([cor, flo], dim=1)
            out = F.relu(self.conv(cor_flo))
            return torch.cat([out, flow], dim=1)
        else:
            # Flow already embedded: concatenate directly
            cor_flo = torch.cat([cor, flow], dim=1)
            return F.relu(self.conv(cor_flo))

    
def conv133_encoder(input_dim, dim, expansion_factor=4):
    """
    Lightweight 3-layer CNN encoder (1x1 -> 3x3 -> 3x3),
    used to compress correlation features.
    """
    return nn.Sequential(
        nn.Conv2d(input_dim, dim * expansion_factor, kernel_size=1),
        nn.GELU(),
        nn.Conv2d(dim * expansion_factor, dim * expansion_factor, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(dim * expansion_factor, dim, kernel_size=3, padding=1),
    )     
    
class BasicUpdateBlock(nn.Module):
    """
    GRU-like update block used in RAFT-style iterative flow refinement.

    Combines:
      - Previous flow features
      - Context features
      - Motion-encoded correlation + flow
    Followed by alternating temporal (1D) and spatial (2D) refinement.
    """
    def __init__(self, corr_channel, num_blocks, hdim=128, cdim=128):
        super(BasicUpdateBlock, self).__init__()

        self.encoder = BasicMotionEncoder(corr_channel, dim=cdim)
        self.compressor = conv1x1(2 * cdim + hdim, hdim)

        # Alternating temporal and spatial refinement blocks
        self.refine = []
        for _ in range(num_blocks):
            self.refine.append(CNBlock1d(hdim, hdim))  # temporal / sequence dimension
            self.refine.append(CNBlock2d(hdim, hdim))  # spatial dimension
        self.refine = nn.ModuleList(self.refine)

    def forward(self, flowfeat, ctxfeat, corr, flow, S, upsample=True):
        """
        Args:
            flowfeat: hidden flow features (B*S, C, H, W)
            ctxfeat: context features
            corr: correlation volume
            flow: current flow estimate
            S: temporal length (sequence factor)
        """
        BS, C, H, W = flowfeat.shape

        motion_features = self.encoder(flow, corr)

        # Fuse all information into hidden state
        flowfeat = self.compressor(
            torch.cat([flowfeat, ctxfeat, motion_features], dim=1)
        )

        for blk in self.refine:
            flowfeat = blk(flowfeat, S)

        return flowfeat

    
class FullUpdateBlock(nn.Module):
    """
    Update block that additionally models visibility confidence.
    Useful for occlusion-aware flow estimation.
    """
    def __init__(self, corr_channel, num_blocks, hdim=128, cdim=128,
                 pdim=2, use_attn=False):
        super(FullUpdateBlock, self).__init__()

        self.encoder = BasicMotionEncoder(corr_channel, dim=cdim, pdim=pdim)

        # Include visconf and (optionally) raw flow in fusion
        if pdim == 2:
            self.compressor = conv1x1(2 * cdim + hdim + 2, hdim)
        else:
            self.compressor = conv1x1(2 * cdim + hdim + 2 + pdim, hdim)

        self.refine = []
        for _ in range(num_blocks):
            self.refine.append(CNBlock1d(hdim, hdim, use_attn=use_attn))
            self.refine.append(CNBlock2d(hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, flowfeat, ctxfeat, visconf, corr, flow, S, upsample=True):
        motion_features = self.encoder(flow, corr)
        flowfeat = self.compressor(
            torch.cat([flowfeat, ctxfeat, motion_features, visconf], dim=1)
        )

        for blk in self.refine:
            flowfeat = blk(flowfeat, S)

        return flowfeat


class MixerUpdateBlock(nn.Module):
    """
    Update block that replaces temporal attention / convolution
    with an MLP-Mixer style token mixing module.

    Design motivation:
        - Treat each frame as a token in the temporal dimension
        - Use channel-wise MLPs to mix temporal information efficiently
        - Better scalability than attention for long sequences

    Architecture:
        Motion Encoding → Feature Compression →
        [Temporal Mixer (1D) → Spatial Refinement (2D)] × N
    """
    def __init__(self, corr_channel, num_blocks, hdim=128, cdim=128):
        super(MixerUpdateBlock, self).__init__()

        # Encode correlation + flow into motion-aware features
        self.encoder = BasicMotionEncoder(corr_channel, dim=cdim)

        # Fuse flow features, context features, and motion features
        self.compressor = conv1x1(2 * cdim + hdim, hdim)
        
        # Alternating temporal (MLP-Mixer) and spatial refinement blocks
        self.refine = []
        for i in range(num_blocks):
            # Temporal mixing across sequence dimension
            self.refine.append(CNBlock1d(hdim, hdim, use_mixer=True))
            # Spatial refinement within each frame
            self.refine.append(CNBlock2d(hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, flowfeat, ctxfeat, corr, flow, S, upsample=True):
        """
        Args:
            flowfeat: hidden flow features, shape (B*S, C, H, W)
            ctxfeat: context features
            corr: correlation volume
            flow: current flow estimate
            S: sequence length (temporal dimension)
        """
        BS, C, H, W = flowfeat.shape
        B = BS // S  # batch size (unused but kept for clarity)

        # Encode motion from correlation and current flow
        motion_features = self.encoder(flow, corr)

        # Feature fusion
        flowfeat = self.compressor(
            torch.cat([flowfeat, ctxfeat, motion_features], dim=1)
        )
            
        # Iterative temporal + spatial refinement
        for blk in self.refine:
            flowfeat = blk(flowfeat, S)

        return flowfeat

    
class FacUpdateBlock(nn.Module):
    """
    Factorized update block designed for high-dimensional motion
    representations (e.g., 3D flow, deformation parameters).

    Key idea:
        - Do NOT explicitly encode flow in the motion encoder
        - Treat flow as a factorized latent representation
        - Correlation is compressed separately and fused later

    This design avoids over-constraining complex motion parameterizations.
    """
    def __init__(self, corr_channel, num_blocks,
                 hdim=128, cdim=128, pdim=84, use_attn=False):
        super(FacUpdateBlock, self).__init__()

        # Compress correlation volume only (no flow encoding here)
        self.corr_encoder = conv133_encoder(corr_channel, cdim)

        # Fuse:
        #   - flow hidden features
        #   - context features
        #   - compressed correlation
        #   - visibility confidence
        #   - raw factorized flow parameters
        self.compressor = conv1x1(2 * cdim + hdim + 2 + pdim, hdim)

        # Alternating temporal and spatial refinement blocks
        self.refine = []
        for i in range(num_blocks):
            self.refine.append(CNBlock1d(hdim, hdim, use_attn=use_attn))
            self.refine.append(CNBlock2d(hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, flowfeat, ctxfeat, visconf, corr, flow, S, upsample=True):
        """
        Args:
            flowfeat: hidden flow features (B*S, C, H, W)
            ctxfeat: context features
            visconf: visibility / confidence map
            corr: correlation volume
            flow: factorized motion parameters
            S: sequence length
        """
        BS, C, H, W = flowfeat.shape
        B = BS // S

        # Correlation compression
        corr = self.corr_encoder(corr)

        # Feature fusion
        flowfeat = self.compressor(
            torch.cat([flowfeat, ctxfeat, corr, visconf, flow], dim=1)
        )

        # Temporal + spatial refinement
        for blk in self.refine:
            flowfeat = blk(flowfeat, S)

        return flowfeat

class CleanUpdateBlock(nn.Module):
    """
    Clean and stabilized update block with LayerScale.

    Design goals:
        - Minimal, symmetric architecture
        - Stable deep refinement using LayerScale
        - Decouple motion compression and refinement depth

    This block is especially suitable for:
        - Deep iterative refinement
        - Large-scale training
        - 3D / deformation-based motion fields
    """
    def __init__(self, corr_channel, num_blocks,
                 cdim=128, hdim=256, pdim=84,
                 use_attn=False, use_layer_scale=True):
        super(CleanUpdateBlock, self).__init__()

        # Correlation-only encoder
        self.corr_encoder = conv133_encoder(corr_channel, cdim)

        # Initial feature fusion
        self.compressor = conv1x1(3 * cdim + pdim + 2, hdim)

        # Deep refinement blocks with optional LayerScale
        self.refine = []
        for i in range(num_blocks):
            self.refine.append(
                CNBlock1d(
                    hdim, hdim,
                    use_attn=use_attn,
                    use_layer_scale=use_layer_scale
                )
            )
            self.refine.append(
                CNBlock2d(
                    hdim, hdim,
                    use_layer_scale=use_layer_scale
                )
            )
        self.refine = nn.ModuleList(self.refine)

        # Project back to compact feature dimension
        self.final_conv = conv1x1(hdim, cdim)

    def forward(self, flowfeat, ctxfeat, visconf, corr, flow, S, upsample=True):
        """
        Args:
            flowfeat: hidden flow features
            ctxfeat: context features
            visconf: visibility confidence
            corr: correlation volume
            flow: motion parameters
            S: temporal length
        """
        BS, C, H, W = flowfeat.shape
        B = BS // S

        # Encode correlation
        corrfeat = self.corr_encoder(corr)

        # Initial fusion
        flowfeat = self.compressor(
            torch.cat([flowfeat, ctxfeat, corrfeat, flow, visconf], dim=1)
        )

        # Deep temporal-spatial refinement
        for blk in self.refine:
            flowfeat = blk(flowfeat, S)

        # Channel reduction for next iteration / output
        flowfeat = self.final_conv(flowfeat)

        return flowfeat


class RelUpdateBlock(nn.Module):
    """
    Relative Update Block for 2D flow refinement.

    Core idea:
        - Update flow in a *relative* manner rather than absolute prediction
        - Separately encode motion w.r.t. feature correlation and context correlation
        - Fuse motion, context, and confidence before iterative refinement

    This design is particularly effective when:
        - Context and feature streams carry different motion cues
        - Relative displacement is easier to regress than absolute flow
    """
    def __init__(
        self,
        corr_channel,
        num_blocks,
        cdim=128,
        cdim1=128,
        mdim=128,
        hdim=128,
        pdim2d=4,
        use_attn=True,
        use_mixer=False,
        use_conv=False,
        use_convb=False,
        use_layer_scale=True,
        no_time=False,
        no_space=False,
        no_ctx=False,
        use_cxt_corr=False,
    ):
        super(RelUpdateBlock, self).__init__()

        # Motion encoder using feature correlation
        self.motion_encoder2d = BasicMotionEncoder(
            corr_channel, dim=hdim, pdim=pdim2d
        )  # (B, hdim, H, W)

        # Motion encoder using context correlation
        self.motion_encoder2d_ctx = BasicMotionEncoder(
            corr_channel, dim=hdim, pdim=pdim2d
        )  # (B, hdim, H, W)

        # Project flow and context features to a shared motion dimension
        self.compressor_f = conv1x1(cdim, mdim)
        self.compressor_c = conv1x1(cdim1, mdim)

        self.no_ctx = no_ctx
        self.use_cxt_corr = use_cxt_corr

        # Final fusion of:
        #   flow features + context features +
        #   motion from feature corr + motion from context corr +
        #   visibility confidence
        self.compressor = conv1x1(2 * mdim + 2 * hdim + 2, hdim)

        # Iterative refinement blocks
        self.refine = []
        for i in range(num_blocks):
            if not no_time:
                # Temporal refinement across sequence dimension
                self.refine.append(
                    CNBlock1d(
                        hdim,
                        hdim,
                        use_attn=use_attn,
                        use_mixer=use_mixer,
                        use_conv=use_conv,
                        use_convb=use_convb,
                        use_layer_scale=use_layer_scale,
                    )
                )
            if not no_space:
                # Spatial refinement within each frame
                self.refine.append(
                    CNBlock2d(hdim, hdim, use_layer_scale=use_layer_scale)
                )
        self.refine = nn.ModuleList(self.refine)

        # Project back to original flow feature dimension
        self.final_conv = conv1x1(hdim, cdim)

    def forward(self, flowfeat, ctxfeat, visconf, fea_corr, cxt_corr, motion2d, S):
        """
        Args:
            flowfeat: flow hidden features (B*S, C, H, W)
            ctxfeat: context features
            visconf: visibility / confidence map
            fea_corr: correlation from feature stream
            cxt_corr: correlation from context stream
            motion2d: current 2D motion estimate
            S: temporal sequence length
        """
        BS, C, H, W = flowfeat.shape

        # Encode relative motion from two correlation sources
        motion_features2d = self.motion_encoder2d(motion2d, fea_corr)
        motion_features2d_ctx = self.motion_encoder2d_ctx(motion2d, cxt_corr)

        # Feature projection
        flowfeat = self.compressor_f(flowfeat)
        ctxfeat = self.compressor_c(ctxfeat)

        # Fuse all cues
        flowfeat = self.compressor(
            torch.cat(
                [
                    flowfeat,
                    ctxfeat,
                    motion_features2d,
                    motion_features2d_ctx,
                    visconf,
                ],
                dim=1,
            )
        )

        # Iterative temporal-spatial refinement
        for blk in self.refine:
            flowfeat = blk(flowfeat, S)

        # Output refined flow features
        flowfeat = self.final_conv(flowfeat)
        return flowfeat

class RelUpdate3D(nn.Module):
    """
    Relative Update Block for 3D flow / scene motion.

    Key characteristics:
        - Relative refinement of 3D motion fields
        - Multi-source feature fusion:
            * 2D flow features
            * 3D detail features
            * sampled temporal features
            * motion correlation features
        - Optional prior-guided prediction head

    This block bridges 2D observations and 3D motion estimation.
    """
    def __init__(
        self,
        corr_channel,
        num_blocks,
        tdim=512,
        cdim=128,
        mdim=128,
        hdim=128,
        pdim3d=6,
        use_attn=True,
        use_mixer=False,
        use_conv=False,
        use_convb=False,
        use_layer_scale=True,
        no_time=False,
        no_space=False,
        no_ctx=False,
        use_cxt_corr=False,
        use_prior=False,
    ):
        super(RelUpdate3D, self).__init__()

        # Project different feature sources into a common motion space
        self.compressor_f = conv1x1(cdim, mdim)
        self.compressor_d = conv1x1(tdim, mdim)
        self.compressor_c = conv1x1(tdim, mdim // 2)
        self.compressor_c1 = conv1x1(tdim, mdim // 2)

        # Encode relative 3D motion from correlation volume
        self.motion_encoder3d = BasicMotionEncoder(
            corr_channel, dim=mdim, pdim=pdim3d
        )

        # Fuse all motion-related features
        self.compressor = conv1x1(4 * mdim, hdim)

        self.use_prior = use_prior

        # Temporal-spatial refinement blocks
        self.refine = []
        for i in range(num_blocks):
            if not no_time:
                self.refine.append(
                    CNBlock1d(
                        hdim,
                        hdim,
                        use_attn=use_attn,
                        use_mixer=use_mixer,
                        use_conv=use_conv,
                        use_convb=use_convb,
                        use_layer_scale=use_layer_scale,
                    )
                )
            if not no_space:
                self.refine.append(
                    CNBlock2d(hdim, hdim, use_layer_scale=use_layer_scale)
                )
        self.refine = nn.ModuleList(self.refine)

        # Project refined features back to 3D feature space
        self.final_conv = conv1x1(hdim, tdim)

        # Prediction head
        if use_prior:
            # Prior-guided residual prediction
            self.head = nn.Sequential(
                conv1x1(tdim + 3, cdim),
                nn.Conv2d(
                    cdim,
                    2 * cdim,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.GELU(),
                nn.Conv2d(
                    2 * cdim,
                    3,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
        else:
            # Direct prediction without prior
            self.head = nn.Sequential(
                conv1x1(hdim, cdim),
                nn.Conv2d(
                    cdim,
                    2 * cdim,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    2 * cdim,
                    3,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
            )

    def forward(
        self,
        flowfeat,
        fmaps3d_detail,
        sampled_feature_pre,
        sampled_feature,
        flow_3d_prior,
        pm_corr,
        motion3d,
        S,
    ):
        """
        Args:
            flowfeat: 2D flow features
            fmaps3d_detail: high-resolution 3D features
            sampled_feature_pre: previous-frame sampled features
            sampled_feature: current-frame sampled features
            flow_3d_prior: prior 3D flow (optional)
            pm_corr: point-matching correlation
            motion3d: current 3D motion estimate
            S: temporal length
        """
        BS, C, H, W = flowfeat.shape

        # Feature projection
        flowfeat = self.compressor_f(flowfeat)
        sampled_feature = self.compressor_c(sampled_feature)
        fmaps3d_detail = self.compressor_d(fmaps3d_detail)
        sampled_feature_pre = self.compressor_c1(sampled_feature_pre)

        # Encode relative 3D motion
        motion_features2d_3d = self.motion_encoder3d(motion3d, pm_corr)

        # Feature fusion
        feat = self.compressor(
            torch.cat(
                [
                    flowfeat,
                    fmaps3d_detail,
                    sampled_feature,
                    sampled_feature_pre,
                    motion_features2d_3d,
                ],
                dim=1,
            )
        )

        # Temporal-spatial refinement
        for blk in self.refine:
            feat = blk(feat, S)

        # Refined 3D features
        feat = self.final_conv(feat)

        # Predict relative 3D flow
        flow3d = self.head(torch.cat([feat, flow_3d_prior], dim=1))

        return flow3d, feat

class RelUpdate3D34(nn.Module):
    def __init__(self, corr_channel, num_blocks, tdim=512, cdim=128, mdim=128, hdim=128, pdim3d=6, use_attn=True, use_mixer=False, use_conv=False, use_convb=False, use_layer_scale=True, no_time=False, no_space=False, no_ctx=False, use_cxt_corr=False, use_prior=False):
        super(RelUpdate3D34, self).__init__()

        self.compressor_f = conv1x1(cdim, mdim//2)
        self.compressor_d = conv1x1(tdim, mdim//2)
        self.compressor_c = conv1x1(tdim, mdim//2)
        self.compressor_c1 = conv1x1(tdim, mdim//2)
        self.motion_encoder3d = BasicMotionEncoder(corr_channel, dim=mdim//2, pdim=pdim3d) # B,hdim,H,W
        self.compressor = conv1x1(mdim + mdim//2, hdim)
        self.use_prior = use_prior
        self.refine = []
        for i in range(num_blocks):
            if not no_time:
                self.refine.append(CNBlock1d(hdim, hdim, use_attn=use_attn, use_mixer=use_mixer, use_conv=use_conv, use_convb=use_convb, use_layer_scale=use_layer_scale))
            if not no_space:
                self.refine.append(CNBlock2d(hdim, hdim, use_layer_scale=use_layer_scale))
        self.refine = nn.ModuleList(self.refine)
        self.final_conv = conv1x1(hdim, tdim)

        self.head = nn.Sequential(
            conv1x1(tdim, cdim),
            nn.Conv2d(cdim, 2*cdim, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(2*cdim, 3, kernel_size=3, padding=1, padding_mode='replicate')
        )

    def forward(self, flowfeat, fmaps3d_detail, sampled_feature_pre, sampled_feature, flow_3d_prior, pm_corr, motion3d, S):
        BS, C, H, W = flowfeat.shape
        flowfeat = self.compressor_f(flowfeat)
        # sampled_feature = self.compressor_c(sampled_feature)
        
        fmaps3d_detail = self.compressor_d(fmaps3d_detail)
        # sampled_feature_pre = self.compressor_c1(sampled_feature_pre)
        motion_features2d_3d = self.motion_encoder3d(motion3d, pm_corr)
        feat = self.compressor(torch.cat([flowfeat, fmaps3d_detail, motion_features2d_3d], dim=1))
        for blk in self.refine:
            feat = blk(feat, S)
        feat = self.final_conv(feat)
        flow3d = self.head(feat)

        return flow3d, feat
