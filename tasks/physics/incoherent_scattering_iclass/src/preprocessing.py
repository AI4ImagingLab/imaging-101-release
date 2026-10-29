"""Camera data and Fourier matrix convention; attribution: test/validation.md."""
from pathlib import Path
import numpy as np
import torch

SHAPE = (1, 150, 400, 400)


def load_observations(path='data/raw_data.npz'):
    """Return owned float32 (150,400,400) Fourier-resampled camera DN."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if 'camera_frames' not in archive:
            raise ValueError('camera_frames is required')
        frames = archive['camera_frames']
    if frames.shape != SHAPE or frames.dtype != np.float32 or not np.isfinite(frames).all():
        raise ValueError('Expected finite float32 camera_frames with shape '+str(SHAPE))
    return np.ascontiguousarray(frames[0])


def prepare_matrix(frames, device='cpu'):
    """Hann, temporal centering, spatial FFT; return R and uncorrected RMS.

    R has shape (H*W,M), row index ix*H+iy. Upstream fftshift rotates all
    three axes, including frame order; preserving this gives stage parity.
    Frame rotation cannot change the covariance or RMS. No input mutation.
    """
    x = torch.as_tensor(frames, dtype=torch.float32, device=device).clone()
    if x.ndim != 3 or not torch.isfinite(x).all():
        raise ValueError('Expected finite (frames,height,width) values')
    height, width = x.shape[-2:]
    wy = torch.hann_window(height, periodic=False, device=x.device)
    wx = torch.hann_window(width, periodic=False, device=x.device)
    x = x*wy[None,:,None]*wx[None,None,:]
    x -= x.mean(0)
    raw_rms = torch.sqrt(torch.mean(x*x, 0))
    spectra = torch.fft.fftshift(torch.fft.fft2(x))
    matrix = spectra.permute(2,1,0).reshape(height*width, x.shape[0])
    return matrix.contiguous(), raw_rms
