"""MIT2025 ZacharyVarley. Selected explicit numerical kernels extracted from
pcadi commit88f676e6f44812900b3c873692aa1a1c9e48b746.
Only the bound cubic-Ni mathematics is retained; no domain-library import,
quantization framework, IO package, or generic vendor class is included.
"""
import math
import torch
from torch import Tensor
from typing import Tuple,Optional
def clahe_grayscale(x: torch.Tensor, clip_limit: float=40.0, n_bins: int=64, grid_shape: Tuple[int, int]=(4, 4)) -> Tensor:
    """
    CLAHE on batches of 2D tensor.

    Args:
        x (Tensor): Input grayscale images of shape (B, 1, H, W).
        clip_limit (float): The clip limit for the histogram equalization.
        n_bins (int): The number of bins to use for the histogram equalization.
        grid_shape (Tuple[int, int]): The shape of the grid to divide the image into.

    Returns:
        Tensor: Output images of shape (B, 1, H, W).


    """
    h_img, w_img = x.shape[-2:]
    signal_shape = x.shape[:-2]
    B = int(torch.prod(torch.tensor(signal_shape)).item())
    h_grid, w_grid = grid_shape
    n_tiles = h_grid * w_grid
    h_tile = math.ceil(h_img / h_grid)
    w_tile = math.ceil(w_img / w_grid)
    voxels_per_tile = h_tile * w_tile
    pad_w = w_grid - w_img % w_grid if w_img % w_grid != 0 else 0
    pad_h = h_grid - h_img % h_grid if h_img % h_grid != 0 else 0
    paddings = (0, pad_w, 0, pad_h)
    x_padded = torch.nn.functional.pad(x, paddings, mode='reflect')
    tiles = torch.nn.functional.unfold(x_padded, kernel_size=(h_tile, w_tile), stride=(h_tile, w_tile))
    tiles = tiles.view((B, voxels_per_tile, n_tiles))
    tiles = tiles.swapdims(-1, -2)
    pdf = torch.zeros((B, n_tiles, n_bins), device=x.device, dtype=torch.float32)
    x_discrete = (tiles * (n_bins - 1)).to(torch.int64)
    pdf.scatter_(dim=-1, index=x_discrete, value=1, reduce='add')
    pdf = pdf / pdf.sum(dim=-1, keepdim=True)
    histos = (pdf * voxels_per_tile).view(-1, n_bins)
    if clip_limit > 0:
        limit = max(clip_limit * voxels_per_tile // n_bins, 1)
        histos.clamp_(max=limit)
        clipped = voxels_per_tile - histos.sum(dim=-1)
        residual = torch.remainder(clipped, n_bins)
        redist = (clipped - residual).div(n_bins)
        histos += redist[..., None]
        v_range = torch.arange(n_bins, device=histos.device)
        mat_range = v_range.repeat(histos.shape[0], 1)
        histos += mat_range < residual[None].transpose(0, 1)
    cdfs = torch.cumsum(histos, dim=-1) / voxels_per_tile
    cdfs = cdfs.clamp_(min=0.0, max=1.0)
    cdfs = cdfs.view(B, h_grid, w_grid, n_bins)
    coords = torch.meshgrid([torch.linspace(-1.0, 1.0, w_img, device=x.device), torch.linspace(-1.0, 1.0, h_img, device=x.device)], indexing='xy')
    coords = torch.stack(coords, dim=-1)
    coords_into_cdfs = torch.cat([x[..., None].view(B, 1, h_img, w_img, 1) * 2.0 - 1.0, coords[None, None].repeat(B, 1, 1, 1, 1)], dim=-1)
    x = torch.nn.functional.grid_sample(cdfs[:, None, :, :, :], coords_into_cdfs, mode='bilinear', padding_mode='border', align_corners=False)
    return x[:, 0].reshape(signal_shape + (h_img, w_img))
import numpy as np
def load_data(path):
    """Return uint8 raw(40,50,60,60),float32 combined Lambert master(401,802)."""
    with np.load(path,allow_pickle=False) as z:p=z['patterns'];m=z['master_pattern']
    if p.shape!=(1,40,50,60,60) or p.dtype!=np.uint8:raise ValueError('patterns must be uint8(1,40,50,60,60)')
    if m.shape!=(1,2,401,401) or m.dtype!=np.uint8:raise ValueError('master_pattern must be uint8(1,2,401,401)')
    return p[0],torch.cat([torch.from_numpy(m[0,0].copy()).float(),torch.from_numpy(m[0,1].copy()).float()],-1)
def minmax(patterns):
    """Original per-pattern normalization preserving input dtype and1e-4 epsilon."""
    shape=patterns.shape;a=patterns.reshape(len(patterns),-1);lo=a.min(-1).values;hi=a.max(-1).values
    a=a-lo[:,None];a=a/(1e-4+hi[:,None]-lo[:,None]);return a.reshape(shape)
def prepare_patterns(raw,return_stages=False):
    """Author detector60->30 bilinear, static background, minmax, CLAHE, minmax.

    This resamples detector pixels only; native40x50 scan coordinates are intact.
    FP16 is retained until source CLAHE returns FP32, exactly as original.
    """
    p=torch.nn.functional.interpolate(torch.from_numpy(raw.copy()).half().reshape(-1,1,60,60),size=(30,30),mode='bilinear',align_corners=False).squeeze(1)
    stages={'downsampled':p.clone()};p-=p.mean(0,keepdim=True);stages['background_subtracted']=p.clone();p=minmax(p);stages['first_normalized']=p.clone();p=clahe_grayscale(p[:,None],clip_limit=40.,n_bins=64,grid_shape=(4,4)).squeeze(1);stages['clahe']=p.clone();p=minmax(p)
    return (p,stages) if return_stages else p
def prepare_master(master,return_stages=False):
    """Author minmax+8x8tile CLAHE of the independently simulated calibration."""
    master=master.clone();lo=master.min();hi=master.max();master-=lo;master/=1e-4+hi-lo
    initial=master.clone();master=clahe_grayscale(master[None,None],clip_limit=40.,n_bins=64,grid_shape=(8,8))[0,0]
    return (master,initial) if return_stages else master
