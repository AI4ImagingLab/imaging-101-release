"""Explicit blur and structural similarity mathematics."""
import math
import torch
from torch.nn import functional as F

def structural_similarity(a,b):
    """Mean 11x11 Gaussian SSIM, zero-padded moments, unit luminance scale."""
    if a.shape != b.shape or a.ndim != 4 or a.shape[1] != 1:
        raise ValueError('SSIM requires equal BCHW arrays with one channel')
    g=torch.tensor([math.exp(-(j-5)**2/(2*1.5**2)) for j in range(11)],dtype=torch.float32)
    g=g/g.sum()
    window=(g[:,None]*g[None,:]).reshape(1,1,11,11).to(device=a.device,dtype=a.dtype)
    smooth=lambda x:F.conv2d(x,window,padding=5)
    ma=smooth(a);mb=smooth(b)
    ma2=ma.pow(2);mb2=mb.pow(2);product=ma*mb
    va=smooth(a*a)-ma2;vb=smooth(b*b)-mb2;cov=smooth(a*b)-product
    return (((2*product+.0001)*(2*cov+.0009))/((ma2+mb2+.0001)*(va+vb+.0009))).mean()


def blur_image(latent, kernel):
    """Valid cross-correlation; kernel weights index positive row/column offsets."""
    if latent.ndim != 4 or kernel.ndim != 4 or latent.shape[:2] != (1,1):
        raise ValueError('latent and kernel must be single-channel BCHW tensors')
    if kernel.shape[:2] != (1,1):
        raise ValueError('kernel must have batch/channel shape (1,1)')
    return F.conv2d(latent,kernel)


def loss_for_step(prediction, observation, step):
    """MSE through zero-based iteration 499, then 1-SSIM, without TV penalty."""
    if step < 0:
        raise ValueError('step must be nonnegative')
    if step < 500:
        return F.mse_loss(prediction,observation)
    return 1-structural_similarity(prediction,observation)
