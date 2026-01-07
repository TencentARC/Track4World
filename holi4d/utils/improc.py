import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from skimage.color import hsv2rgb

import holi4d.utils.basic
import holi4d.utils.py

EPS = 1e-6
COLORMAP_FILE = "holi4d/utils/bremm.png"

# ==============================================================================
# Helper Functions: Color & Transform
# ==============================================================================

def _convert(input_, type_):
    """Helper to convert tensor types."""
    return {
        'float': input_.float(),
        'double': input_.double(),
    }.get(type_, input_)

def _generic_transform_sk_3d(transform, in_type='', out_type=''):
    """Applies a scikit-image 3D transform to a PyTorch tensor."""
    def apply_transform_individual(input_):
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)
        
        # CHW -> HWC for skimage processing
        input_np = input_.permute(1, 2, 0).detach().numpy()
        transformed = transform(input_np)
        
        # Back to Tensor CHW
        output = torch.from_numpy(transformed).float().permute(2, 0, 1)
        output = _convert(output, out_type)
        return output.to(device)

    def apply_transform(input_):
        # Stack processing for batch dimension
        to_stack = [apply_transform_individual(img) for img in input_]
        return torch.stack(to_stack)
    
    return apply_transform

def get_2d_colors(xys, H, W):
    """
    Maps 2D coordinates to colors using a 2D colormap.
    """
    N, D = xys.shape
    assert(D == 2)
    bremm = ColorMap2d()
    
    # Normalize coordinates to [0, 1]
    xys[:, 0] /= float(W - 1)
    xys[:, 1] /= float(H - 1)
    
    colors = bremm(xys)
    return colors

# Create specific transform function for HSV to RGB conversion
hsv_to_rgb_torch = _generic_transform_sk_3d(hsv2rgb)

def colorize(d, cmap_name='inferno'):
    """
    Maps a 1-channel tensor to a 3-channel RGB tensor using a matplotlib colormap.
    Input: (1, H, W) or (H, W)
    Output: (3, H, W)
    """
    if d.ndim == 2:
        d = d.unsqueeze(dim=0)
    else:
        assert d.ndim == 3
        
    # Get colormap (handle matplotlib deprecation if needed)
    try:
        color_map = plt.get_cmap(cmap_name)
    except AttributeError:
        color_map = cm.get_cmap(cmap_name)

    C, H, W = d.shape
    assert C == 1
    
    d_flat = d.reshape(-1).detach().cpu().numpy()
    
    # Apply colormap: returns (N, 4) RGBA, we take RGB * 255
    color = np.array(color_map(d_flat)) * 255
    color = color[:, :3] # Drop Alpha channel
    
    # Reshape back to image dimensions
    color = color.reshape(H, W, 3)
    color = torch.from_numpy(color).permute(2, 0, 1) # HWC -> CHW
    
    return color

def oned2inferno(d, norm=True, do_colorize=False):
    """
    Converts a 1-channel tensor to a 3-channel heatmap (Inferno) or grayscale.
    Input: (B, 1, H, W) or (B, H, W)
    Output: (B, 3, H, W) uint8 tensor
    """
    if d.ndim == 3:
        d = d.unsqueeze(dim=1)
    
    B, C, H, W = d.shape
    assert C == 1

    if norm:
        d = holi4d.utils.basic.normalize(d)
        
    if do_colorize:
        rgb = torch.zeros(B, 3, H, W)
        for b in range(B):
            rgb[b] = colorize(d[b])
    else:
        # Grayscale repeated to 3 channels
        rgb = d.repeat(1, 3, 1, 1) * 255.0
        
    return rgb.to(torch.uint8)

def flow2color(flow, clip=0.0):
    """
    Converts optical flow to HSV-based RGB visualization.
    Input: (B, 2, H, W)
    Output: (1, 3, H, W) uint8 tensor (Visualizes the first item in batch)
    """
    B, C, H, W = flow.size()
    assert C == 2
    
    flow = flow[0:1].detach() # Take first element
    
    if clip == 0:
        clip = torch.max(torch.abs(flow)).item()
        
    # Normalize flow values
    flow = torch.clamp(flow, -clip, clip) / (clip + EPS)
    
    # Compute magnitude and angle
    radius = torch.sqrt(torch.sum(flow**2, dim=1, keepdim=True)) 
    radius_clipped = torch.clamp(radius, 0.0, 1.0)
    
    angle = torch.atan2(-flow[:, 1:2], -flow[:, 0:1]) / np.pi
    
    # Map to HSV
    hue = torch.clamp((angle + 1.0) / 2.0, 0.0, 1.0)
    saturation = torch.ones_like(hue) * 0.75
    value = radius_clipped
    
    hsv = torch.cat([hue, saturation, value], dim=1) # (1, 3, H, W)
    
    rgb = hsv_to_rgb_torch(hsv)
    return (rgb * 255.0).to(torch.uint8)

def preprocess_color(x):
    """Converts [0, 255] uint8/float to [-0.5, 0.5] float."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32) * (1./255) - 0.5
    else:
        return x.float() * (1./255) - 0.5
    
def back2color(i, blacken_zeros=False):
    """Converts [-0.5, 0.5] float to [0, 255] uint8."""
    if blacken_zeros:
        const = torch.tensor([-0.5], device=i.device)
        i = torch.where(i == 0.0, const, i)
        return back2color(i, blacken_zeros=False)
    else:
        return ((i + 0.5) * 255).to(torch.uint8)

# ==============================================================================
# Helper Functions: PCA & Embeddings
# ==============================================================================

def pca_embed(emb, keep, valid=None):
    """
    Reduces channel dimension using PCA per image in batch.
    Input: (B, C, H, W)
    Output: (B, keep, H, W)
    """
    H, W = emb.shape[-2], emb.shape[-1]
    emb = emb + EPS
    # Convert to numpy: B x H x W x C
    emb_np = emb.permute(0, 2, 3, 1).cpu().detach().numpy() 

    if valid is not None:
        valid_np = valid.cpu().detach().numpy().reshape((H * W))

    emb_reduced = []
    for img in emb_np:
        if np.isnan(img).any():
            emb_reduced.append(np.zeros([H, W, keep], dtype=np.float32))
            continue

        pixels = img.reshape(H * W, -1)
        
        if valid is not None:
            pixels_fit = pixels[valid_np]
        else:
            pixels_fit = pixels

        # Handle case where valid pixels are too few for PCA
        if pixels_fit.shape[0] < keep:
            emb_reduced.append(np.zeros([H, W, keep], dtype=np.float32))
            continue

        pca = PCA(n_components=keep)
        pca.fit(pixels_fit)
        
        # Transform all pixels (masked ones will be zeroed later if needed)
        pixels_3d = pca.transform(pixels)
        
        if valid is not None:
            # Re-apply mask logic if strictly needed
            pass 

        out_img = pixels_3d.reshape(H, W, keep).astype(np.float32)
        
        if np.isnan(out_img).any():
            emb_reduced.append(np.zeros([H, W, keep], dtype=np.float32))
        else:
            emb_reduced.append(out_img)

    emb_reduced = np.stack(emb_reduced, axis=0) # B, H, W, keep
    return torch.from_numpy(emb_reduced).permute(0, 3, 1, 2) # B, keep, H, W

def pca_embed_together(emb, keep):
    """
    Reduces channel dimension using PCA across the entire batch.
    """
    emb = emb + EPS
    # B x H x W x C
    emb_np = emb.permute(0, 2, 3, 1).cpu().detach().float().numpy()
    B, H, W, C = emb_np.shape
    
    if np.isnan(emb_np).any():
        return torch.zeros(B, keep, H, W)
    
    pixels = emb_np.reshape(B * H * W, C)
    
    pca = PCA(n_components=keep)
    pca.fit(pixels)
    pixels_3d = pca.transform(pixels)
    
    out_img = pixels_3d.reshape(B, H, W, keep).astype(np.float32)
        
    if np.isnan(out_img).any():
        return torch.zeros(B, keep, H, W)
    
    return torch.from_numpy(out_img).permute(0, 3, 1, 2)

def get_feat_pca(feat, valid=None):
    """Wrapper to perform PCA on features and normalize."""
    # feat: B, C, D, W (or similar)
    keep = 4
    
    # Use the "together" version for batch consistency
    reduced_emb = pca_embed_together(feat, keep)
    
    # Drop first component (often intensity) and normalize
    reduced_emb = reduced_emb[:, 1:] 
    reduced_emb = holi4d.utils.basic.normalize(reduced_emb) - 0.5
    
    return reduced_emb

# ==============================================================================
# Helper Functions: Visualization & Drawing
# ==============================================================================

class ColorMap2d:
    def __init__(self, filename=None):
        self._colormap_file = filename or COLORMAP_FILE
        if os.path.exists(self._colormap_file):
            self._img = (plt.imread(self._colormap_file) * 255).astype(np.uint8)
            self._height, self._width = self._img.shape[:2]
        else:
            # Fallback if file missing
            print(f"Warning: Colormap file {self._colormap_file} not found. "
                  "Using random noise.")
            self._img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            self._height, self._width = 256, 256

    def __call__(self, X):
        # X: (N, 2) floats in [0, 1]
        assert len(X.shape) == 2
        output = np.zeros((X.shape[0], 3), dtype=np.uint8)
        
        # Lookup colors
        for i in range(X.shape[0]):
            x, y = X[i, :]
            xp = int((self._width - 1) * x)
            yp = int((self._height - 1) * y)
            xp = np.clip(xp, 0, self._width - 1)
            yp = np.clip(yp, 0, self._height - 1)
            output[i, :] = self._img[yp, xp]
        return output

def gif_and_tile(ims, just_gif=False):
    """Combines a sequence of images into a GIF (left) and Tiled view (right)."""
    S = len(ims) 
    # ims: list of (B, C, H, W)
    gif = torch.stack(ims, dim=1) # (B, S, C, H, W)
    
    if just_gif:
        return gif
        
    til = torch.cat(ims, dim=3) # Concatenate along Width: (B, C, H, S*W)
    til = til.unsqueeze(dim=1).repeat(1, S, 1, 1, 1) # Repeat for time dimension
    
    # Concatenate GIF and Tile along Width
    im = torch.cat([gif, til], dim=4) 
    return im

def draw_text_on_vis(vis, text, scale=0.5, left=5, top=20, shadow=True):
    """Generic function to draw text on a tensor image."""
    rgb = vis.detach().cpu().numpy()[0]
    rgb = np.transpose(rgb, [1, 2, 0]) # CHW -> HWC
    
    # Ensure contiguous for OpenCV
    rgb = np.ascontiguousarray(rgb)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
    
    color = (255, 255, 255)
    text_color_bg = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    text_size, _ = cv2.getTextSize(text, font, scale, 1)
    text_w, text_h = text_size
    
    if shadow:
        cv2.rectangle(
            rgb, (left, top - text_h), (left + text_w, top + 1), 
            text_color_bg, -1
        )
    
    cv2.putText(rgb, text, (left, top), font, scale, color, 1)
    
    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    vis = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return vis

def get_n_colors(N, sequential=False):
    label_colors = []
    for ii in range(N):
        if sequential:
            rgb = cm.winter(ii / (N - 1))
            rgb = (np.array(rgb) * 255).astype(np.uint8)[:3]
        else:
            rgb = np.zeros(3)
            while np.sum(rgb) < 128: # ensure min brightness
                rgb = np.random.randint(0, 256, 3)
        label_colors.append(rgb)
    return label_colors

# ==============================================================================
# Main Class: SummWriter
# ==============================================================================

class SummWriter(object):
    def __init__(
        self, writer, global_step, log_freq=10, fps=8, 
        scalar_freq=100, just_gif=False
    ):
        self.writer = writer
        self.global_step = global_step
        self.log_freq = log_freq
        self.scalar_freq = max(scalar_freq, 1)
        self.fps = fps
        self.just_gif = just_gif
        self.maxwidth = 10000
        
        self.save_this = (self.global_step % self.log_freq == 0)
        self.save_scalar = (self.global_step % self.scalar_freq == 0)
        
        if self.save_this:
            self.save_scalar = True

    def summ_gif(self, name, tensor, blacken_zeros=False):
        """Writes a video/gif to TensorBoard."""
        # tensor: (B, S, C, H, W)
        if tensor.dtype == torch.float32:
            tensor = back2color(tensor, blacken_zeros=blacken_zeros)

        video_to_write = tensor[0:1] # Take first batch item: (1, S, C, H, W)
        S = video_to_write.shape[1]
        
        if S == 1:
            self.writer.add_image(
                name, video_to_write[0, 0], global_step=self.global_step
            )
        else:
            self.writer.add_video(
                name, video_to_write, fps=self.fps, global_step=self.global_step
            )
            
        return video_to_write

    def summ_rgbs(
        self, name, ims, frame_ids=None, frame_strs=None, 
        blacken_zeros=False, only_return=False
    ):
        """Summarizes a list of images as a GIF/Tile."""
        if not self.save_this:
            return

        vis = gif_and_tile(ims, just_gif=self.just_gif)
        
        if vis.dtype == torch.float32:
            vis = back2color(vis, blacken_zeros)           

        B, S, C, H, W = vis.shape

        # Draw annotations
        if frame_ids is not None:
            for s in range(S):
                vis[:, s] = draw_text_on_vis(
                    vis[:, s], holi4d.utils.basic.strnum(frame_ids[s]), top=20
                )
                
        if frame_strs is not None:
            for s in range(S):
                vis[:, s] = draw_text_on_vis(vis[:, s], frame_strs[s], top=40)

        # Crop width if too large
        if int(W) > self.maxwidth:
            vis = vis[:, :, :, :self.maxwidth]

        if only_return:
            return vis
        else:
            return self.summ_gif(name, vis, blacken_zeros)

    def summ_rgb(
        self, name, ims, blacken_zeros=False, frame_id=None, 
        frame_str=None, only_return=False, halfres=False, shadow=True
    ):
        """Summarizes a single image."""
        if not self.save_this:
            return

        if ims.dtype == torch.float32:
            ims = back2color(ims, blacken_zeros)

        vis = ims[0:1] # (1, C, H, W)
        
        if halfres:
            vis = F.interpolate(vis, scale_factor=0.5)

        if frame_id is not None:
            vis = draw_text_on_vis(
                vis, holi4d.utils.basic.strnum(frame_id), top=20, shadow=shadow
            )
            
        if frame_str is not None:
            vis = draw_text_on_vis(vis, frame_str, top=40, shadow=shadow)

        if vis.shape[-1] > self.maxwidth:
            vis = vis[..., :self.maxwidth]

        if only_return:
            return vis
        else:
            # Add fake time dimension for summ_gif
            return self.summ_gif(name, vis.unsqueeze(1), blacken_zeros)

    def summ_flow(
        self, name, im, clip=0.0, only_return=False, 
        frame_id=None, frame_str=None, shadow=True
    ):
        """Summarizes optical flow."""
        if self.save_this:
            rgb_flow = flow2color(im, clip=clip)
            return self.summ_rgb(
                name, rgb_flow, only_return=only_return, 
                frame_id=frame_id, frame_str=frame_str, shadow=shadow
            )
        return None

    def summ_oneds(
        self, name, ims, frame_ids=None, frame_strs=None, bev=False, fro=False, 
        logvis=False, reduce_max=False, max_val=0.0, norm=True, 
        only_return=False, do_colorize=False
    ):
        """Summarizes 1D features (heatmaps) over time."""
        if not self.save_this:
            return

        # Reduce dimensions based on view (Bird's Eye View or Front View)
        processed_ims = []
        for im in ims:
            if bev: # B, C, H, D, W -> reduce D (dim 3)
                processed_ims.append(
                    torch.max(im, dim=3)[0] if reduce_max else torch.mean(im, dim=3)
                )
            elif fro: # B, C, D, H, W -> reduce D (dim 2)
                processed_ims.append(
                    torch.max(im, dim=2)[0] if reduce_max else torch.mean(im, dim=2)
                )
            else:
                processed_ims.append(im)
        
        ims = processed_ims

        if len(ims) != 1:
            im = gif_and_tile(ims, just_gif=self.just_gif)
        else:
            im = torch.stack(ims, dim=1)

        B, S, C, H, W = im.shape
        
        # Log scaling
        if logvis and max_val:
            max_val = np.log(max_val)
            im = torch.log(torch.clamp(im, 0) + 1.0)
            im = torch.clamp(im, 0, max_val) / max_val
            norm = False
        elif max_val:
            im = torch.clamp(im, 0, max_val) / max_val
            norm = False
            
        if norm:
            im = holi4d.utils.basic.normalize(im)

        # Colorize
        im_flat = im.view(B * S, C, H, W)
        vis = oned2inferno(im_flat, norm=norm, do_colorize=do_colorize)
        vis = vis.view(B, S, 3, H, W)

        # Annotations
        if frame_ids is not None:
            for s in range(S):
                vis[:, s] = draw_text_on_vis(
                    vis[:, s], holi4d.utils.basic.strnum(frame_ids[s]), top=20
                )

        if frame_strs is not None:
            for s in range(S):
                vis[:, s] = draw_text_on_vis(vis[:, s], frame_strs[s], top=40)

        if W > self.maxwidth:
            vis = vis[..., :self.maxwidth]

        if only_return:
            return vis
        else:
            self.summ_gif(name, vis)

    def summ_oned(
        self, name, im, bev=False, fro=False, logvis=False, max_val=0, 
        max_along_y=False, norm=True, frame_id=None, frame_str=None, 
        only_return=False, shadow=True
    ):
        """Summarizes a single 1D feature map."""
        if not self.save_this:
            return

        if bev: 
            im = torch.max(im, dim=3)[0] if max_along_y else torch.mean(im, dim=3)
        elif fro:
            im = torch.max(im, dim=2)[0] if max_along_y else torch.mean(im, dim=2)
            
        im = im[0:1] # Take first
        
        if logvis and max_val:
            max_val = np.log(max_val)
            im = torch.log(im)
            im = torch.clamp(im, 0, max_val) / max_val
            norm = False
        elif max_val:
            im = torch.clamp(im, 0, max_val) / max_val
            norm = False

        vis = oned2inferno(im, norm=norm)
        
        if vis.shape[-1] > self.maxwidth:
            vis = vis[..., :self.maxwidth]
            
        return self.summ_rgb(
            name, vis, blacken_zeros=False, frame_id=frame_id, 
            frame_str=frame_str, only_return=only_return, shadow=shadow
        )

    def summ_feats(
        self, name, feats, valids=None, pca=True, fro=False, 
        only_return=False, frame_ids=None, frame_strs=None
    ):
        """Summarizes high-dimensional features using PCA or magnitude."""
        if not self.save_this:
            return

        if valids is not None:
            valids = torch.stack(valids, dim=1)
        
        feats = torch.stack(feats, dim=1) # B, S, C, ...

        if feats.ndim == 6: # B, S, C, D, H, W
            reduce_dim = 3 if fro else 4
            if valids is None:
                feats = torch.mean(feats, dim=reduce_dim)
            else: 
                valids = valids.repeat(1, 1, feats.size()[2], 1, 1, 1)
                feats = holi4d.utils.basic.reduce_masked_mean(
                    feats, valids, dim=reduce_dim
                )

        B, S, C, H, W = feats.size()

        if not pca:
            # Magnitude
            feats_mag = torch.mean(torch.abs(feats), dim=2, keepdims=True)
            feats_list = torch.unbind(feats_mag, dim=1)
            return self.summ_oneds(
                name=name, ims=feats_list, norm=True, only_return=only_return, 
                frame_ids=frame_ids, frame_strs=frame_strs
            )
        else:
            # PCA
            feats_packed = holi4d.utils.basic.pack_seqdim(feats, B)
            if valids is None:
                feats_pca_packed = get_feat_pca(feats_packed)
            else:
                # Note: get_feat_pca logic might need adjustment to handle valids
                feats_pca_packed = get_feat_pca(feats_packed)

            feats_pca = holi4d.utils.basic.unpack_seqdim(feats_pca_packed, B)
            return self.summ_rgbs(
                name=name, ims=torch.unbind(feats_pca, dim=1), 
                only_return=only_return, frame_ids=frame_ids, frame_strs=frame_strs
            )

    def summ_scalar(self, name, value):
        """Logs a scalar value."""
        if hasattr(value, 'detach'):
            value = value.detach().cpu().item()
        elif isinstance(value, np.ndarray):
            value = value.item()
            
        if not np.isnan(value):
            if self.log_freq == 1 or self.save_this or self.save_scalar:
                self.writer.add_scalar(name, value, global_step=self.global_step)

    # ==========================================================================
    # Trajectory & Point Drawing Methods
    # ==========================================================================

    def _prepare_traj_data(self, trajs, rgbs, valids, visibs, max_show, W):
        """Helper to prepare data for trajectory drawing."""
        rgbs = rgbs[0] # S, C, H, W
        trajs = trajs[0] # S, N, 2
        
        valids = valids[0] if valids is not None else torch.ones_like(trajs[:, :, 0])
        visibs = visibs[0] if visibs is not None else torch.ones_like(trajs[:, :, 0])

        N = trajs.shape[1]
        if N > max_show:
            inds = np.random.choice(N, max_show, replace=False)
            trajs = trajs[:, inds]
            valids = valids[:, inds]
            visibs = visibs[:, inds]
            N = max_show

        trajs = trajs.clamp(-16, W + 16)
        
        # Convert to numpy color images (HWC)
        rgbs_color = []
        for rgb in rgbs:
            rgb_np = back2color(rgb).detach().cpu().numpy()
            rgb_np = np.transpose(rgb_np, [1, 2, 0])
            rgbs_color.append(np.ascontiguousarray(rgb_np))

        return rgbs_color, trajs, valids, visibs, N

    def summ_traj2ds_on_rgbs(
        self, name, trajs, rgbs, visibs=None, valids=None, frame_ids=None, 
        frame_strs=None, only_return=False, show_dots=True, cmap='coolwarm', 
        vals=None, linewidth=1, max_show=1024
    ):
        """Draws trajectories on a sequence of RGB images."""
        if not self.save_this: return

        B, S, C, H, W = rgbs.shape
        rgbs_color, trajs, valids, visibs, N = self._prepare_traj_data(
            trajs, rgbs, valids, visibs, max_show, W
        )
        
        if vals is not None:
            vals = vals[0] # N (assuming B=1 for vals too)
            if vals.shape[0] > N: 
                 pass 

        trajs_np = trajs.long().detach().cpu().numpy()
        valids_np = valids.long().detach().cpu().numpy()
        visibs_np = visibs.round().detach().cpu().numpy()

        for i in range(N):
            if cmap == 'onediff':
                cmap_ = 'spring' if i == 0 else 'winter'
            else:
                cmap_ = cmap
            
            traj = trajs_np[:, i]
            valid = valids_np[:, i]
            vis = visibs_np[:, i]
            
            # Draw lines up to current frame
            for t in range(S):
                if valid[t]:
                    rgbs_color[t] = self.draw_traj_on_image_py(
                        rgbs_color[t], traj[:t+1], S=S, show_dots=show_dots, 
                        cmap=cmap_, linewidth=linewidth
                    )

            # Draw current point
            rgbs_color = self.draw_circ_on_images_py(
                rgbs_color, traj, vis, S=S, show_dots=show_dots, 
                cmap=cmap_, linewidth=linewidth
            )

        # Convert back to tensor
        out_rgbs = []
        for rgb in rgbs_color:
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            out_rgbs.append(preprocess_color(rgb_t))

        return self.summ_rgbs(
            name, out_rgbs, only_return=only_return, 
            frame_ids=frame_ids, frame_strs=frame_strs
        )

    def draw_traj_on_image_py(
        self, rgb, traj, S=50, linewidth=1, show_dots=False, 
        show_lines=True, cmap='coolwarm', val=None, maxdist=None
    ):
        """Draws a single trajectory on a single image (numpy)."""
        rgb = rgb.copy() # Ensure we don't modify original if shared
        S1, D = traj.shape
        
        try:
            color_map = plt.get_cmap(cmap)
        except:
            color_map = cm.get_cmap(cmap)

        for s in range(S1):
            if val is not None:
                c_val = val
            else:
                c_val = (s) / max(1, float(S - 2))
            
            color = np.array(color_map(c_val)[:3]) * 255

            if show_lines and s < (S1 - 1):
                cv2.line(
                    rgb,
                    (int(traj[s, 0]), int(traj[s, 1])),
                    (int(traj[s+1, 0]), int(traj[s+1, 1])),
                    color,
                    linewidth,
                    cv2.LINE_AA
                )
            if show_dots:
                cv2.circle(
                    rgb, (int(traj[s, 0]), int(traj[s, 1])), 
                    linewidth, color, -1
                )
        return rgb

    def draw_circ_on_images_py(
        self, rgbs, traj, vis, S=50, linewidth=1, 
        show_dots=False, cmap=None, maxdist=None
    ):
        """Draws the current position circle on a list of images."""
        # rgbs: list of HWC numpy arrays
        H, W, _ = rgbs[0].shape
        
        # Determine color strategy
        fixed_color = None
        if cmap is None:
            bremm = ColorMap2d()
            traj_norm = traj[0:1].astype(np.float32).copy()
            traj_norm[:, 0] /= float(W)
            traj_norm[:, 1] /= float(H)
            c = bremm(traj_norm)[0]
            fixed_color = (int(c[0]), int(c[1]), int(c[2]))

        for s in range(len(rgbs)):
            if cmap is not None:
                try:
                    color_map = plt.get_cmap(cmap)
                except:
                    color_map = cm.get_cmap(cmap)
                c_val = s / max(1, float(S - 2))
                color = np.array(color_map(c_val)[:3]) * 255
            else:
                color = fixed_color

            pt = (int(traj[s, 0]), int(traj[s, 1]))
            
            # Outer circle (Color)
            cv2.circle(rgbs[s], pt, linewidth + 2, color, -1)
            
            # Inner circle (Visibility indicator: white or black)
            vis_val = int(np.squeeze(vis[s]) * 255)
            vis_color = (vis_val, vis_val, vis_val)
            cv2.circle(rgbs[s], pt, linewidth + 1, vis_color, -1)
                
        return rgbs

    def summ_pts_on_rgb(
        self, name, trajs, rgb, visibs=None, valids=None, frame_id=None, 
        frame_str=None, only_return=False, show_dots=True, colors=None, 
        cmap='coolwarm', linewidth=1, max_show=1024, already_sorted=False
    ):
        """
        Draws points or trajectories on a single RGB image for visualization.
        
        Args:
            trajs: (B, S, N, 2) tensor of point coordinates.
            visibs: (B, S, N) visibility flags (visible vs occluded).
            valids: (B, S, N) validity flags (existing vs non-existing points).
            linewidth: Thickness of the points/circles.
            max_show: Maximum number of points to draw to avoid clutter.
        """
        if not self.save_this: 
            return

        B, C, H, W = rgb.shape
        # Take the first item in the batch
        rgb = rgb[0]
        trajs = trajs[0] # S, N, 2
        
        # Default valids/visibs to 1s if not provided
        valids = valids[0] if valids is not None else torch.ones_like(trajs[:, :, 0])
        visibs = visibs[0] if visibs is not None else torch.ones_like(trajs[:, :, 0])

        # Clamp trajectories slightly outside image boundaries for smooth drawing
        trajs = trajs.clamp(-16, W + 16)
        N = trajs.shape[1]

        # 1. Memory Management: Subsample points if they exceed max_show
        if N > max_show:
            inds = np.random.choice(N, max_show, replace=False)
            trajs, valids, visibs = trajs[:, inds], valids[:, inds], visibs[:, inds]
            N = max_show

        # 2. Depth Ordering: Sort by Y-coordinate to simulate 3D occlusion
        if not already_sorted:
            # Sort points based on their average vertical position
            inds = torch.argsort(torch.mean(trajs[:, :, 1], dim=0))
            trajs, valids, visibs = trajs[:, inds], valids[:, inds], visibs[:, inds]

        # 3. Preparation: Convert tensors to CPU numpy arrays
        rgb_np = back2color(rgb).detach().cpu().numpy()
        rgb_np = np.transpose(rgb_np, [1, 2, 0])
        rgb_np = np.ascontiguousarray(rgb_np)

        trajs_np = trajs.long().detach().cpu().numpy()
        valids_np = valids.long().detach().cpu().numpy()
        visibs_np = visibs.long().detach().cpu().numpy()
        S = trajs_np.shape[0]

        try:
            color_map = plt.get_cmap(cmap)
        except:
            color_map = cm.get_cmap(cmap)

        # 4. Drawing Loop
        for i in range(N):
            # Special case for 'onediff' colormap to highlight the first point
            cmap_ = 'spring' if (cmap == 'onediff' and i == 0) else \
                    ('winter' if cmap == 'onediff' else cmap)
            
            traj, valid, visib = trajs_np[:, i], valids_np[:, i], visibs_np[:, i]

            # Assign colors based on index or provided list
            if colors is None:
                ii = i / (1e-4 + N - 1.0)
                color = np.array(color_map(ii)[:3]) * 255
            else:
                color = np.array(colors[i]).astype(np.int64)
            
            color = (int(color[0]), int(color[1]), int(color[2]))

            # Draw circles for each timestep in the trajectory
            for s in range(S):
                if valid[s]:
                    # Visual Encoding: 
                    # Filled circle (-1) = Visible, Outlined circle (2) = Occluded
                    thickness = -1 if visib[s] else 2
                    cv2.circle(
                        rgb_np, 
                        (int(traj[s, 0]), int(traj[s, 1])), 
                        linewidth, color, thickness
                    )

        # 5. Output: Preprocess back to tensor format
        rgb_out = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)
        rgb_out = preprocess_color(rgb_out)
        
        return self.summ_rgb(
            name, rgb_out, only_return=only_return, 
            frame_id=frame_id, frame_str=frame_str
        )