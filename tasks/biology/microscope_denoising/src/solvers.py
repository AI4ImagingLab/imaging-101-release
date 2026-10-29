"""
Two-stage ZS-DeconvNet: denoising (Stage 1) + deconvolution (Stage 2).

Architecture: two U-Nets trained jointly.
  Stage 1 (denoiser): noisy input  →  denoised output
  Stage 2 (deconvolver): denoised  →  PSF-deconvolved output

Combined training loss (Eq. 3 of Qiao et al., Nat. Commun. 2024):
  L = μ · L_den + (1 − μ) · L_dec
  L_den = MSE(Stage1(ŷ),  ȳ)                                  (Eq. 4)
  L_dec = MSE(Stage2(Stage1(ŷ)) * PSF,  ȳ)
        + λ · R_Hessian(Stage2(Stage1(ŷ)))                     (Eq. 5)

where (ŷ, ȳ) are recorrupted training pairs, PSF is the optical system's
point-spread function, and R_Hessian penalises second-order intensity
gradients to suppress high-frequency noise amplification during deconvolution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ── Building blocks ─────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv2d(3×3) + ReLU blocks with same-padding."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ── U-Net ────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for fluorescence image denoising / deconvolution.

    Matches the Stage-I/II backbone of ZS-DeconvNet (Qiao et al. 2024):
    4 encoder blocks + bottleneck + 4 decoder blocks, base channel count = 32.

    Parameters
    ----------
    base : int
        Number of feature channels in the first encoder block (default 32).
    """

    def __init__(self, base=32):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(1, base)            # (B,  32, H,    W   )
        self.enc2 = DoubleConv(base, base * 2)     # (B,  64, H/2,  W/2 )
        self.enc3 = DoubleConv(base * 2, base * 4) # (B, 128, H/4,  W/4 )
        self.enc4 = DoubleConv(base * 4, base * 8) # (B, 256, H/8,  W/8 )
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DoubleConv(base * 8, base * 8)  # (B, 256, H/16, W/16)

        # Decoder
        self.dec4 = DoubleConv(base * 8 + base * 8, base * 4)  # 512 → 128
        self.dec3 = DoubleConv(base * 4 + base * 4, base * 2)  # 256 → 64
        self.dec2 = DoubleConv(base * 2 + base * 2, base)      # 128 → 32
        self.dec1 = DoubleConv(base + base, base)               # 64  → 32

        # Output: single channel with no activation
        self.out_conv = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([F.interpolate(b, size=e4.shape[2:],
                                                mode='bilinear', align_corners=True), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=e3.shape[2:],
                                                mode='bilinear', align_corners=True), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[2:],
                                                mode='bilinear', align_corners=True), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:],
                                                mode='bilinear', align_corners=True), e1], dim=1))

        return self.out_conv(d1)


# ── Loss functions ────────────────────────────────────────────────────────────

def hessian_loss(x):
    """
    Hessian regularisation: L2 norm of all second-order finite differences.

    Penalises high-frequency oscillations introduced by deconvolution,
    equivalent to R_Hessian in Eq. 5 of Qiao et al. 2024.

    Parameters
    ----------
    x : torch.Tensor, shape (B, 1, H, W)

    Returns
    -------
    loss : torch.Tensor, scalar
    """
    # First-order differences
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]   # (B, 1, H, W-1)
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]   # (B, 1, H-1, W)

    # Second-order differences (Hessian components)
    dxx = dx[:, :, :, 1:] - dx[:, :, :, :-1]   # (B, 1, H, W-2)
    dyy = dy[:, :, 1:, :] - dy[:, :, :-1, :]   # (B, 1, H-2, W)
    dxy = dy[:, :, :, 1:] - dy[:, :, :, :-1]   # (B, 1, H-1, W-1)
    dyx = dx[:, :, 1:, :] - dx[:, :, :-1, :]   # (B, 1, H-1, W-1)

    return (dxx.pow(2).mean() + dyy.pow(2).mean() +
            dxy.pow(2).mean() + dyx.pow(2).mean())


# ── Training ─────────────────────────────────────────────────────────────────

def train_zs_deconvnet(y_hat, y_bar, psf,
                       n_iters=30000, batch_size=4,
                       lr=5e-4, lr_decay_steps=10000, lr_decay_factor=0.5,
                       mu=0.5, hess_weight=0.02,
                       base=32, device=None, verbose=True):
    """
    Jointly train Stage 1 (denoiser) and Stage 2 (deconvolver) U-Nets.

    Training follows the ZS-DeconvNet dual-stage scheme:
      L = μ · L_den + (1 − μ) · L_dec
    where
      L_den = MSE(f_den(ŷ), ȳ)
      L_dec = MSE(f_dec(f_den(ŷ)) * PSF, ȳ)  +  λ · Hessian(f_dec(f_den(ŷ)))

    Parameters
    ----------
    y_hat : np.ndarray, shape (N, 1, H, W), float32
        More-noisy recorrupted patches (network inputs), normalised to [0, 1].
    y_bar : np.ndarray, shape (N, 1, H, W), float32
        Less-noisy recorrupted patches (training targets), normalised to [0, 1].
    psf : np.ndarray, shape (kH, kW), float32
        Normalised PSF kernel (sum = 1).
    n_iters : int
        Total gradient steps.
    batch_size : int
        Mini-batch size.
    lr : float
        Initial Adam learning rate.
    lr_decay_steps : int
        Halve lr every this many iterations.
    lr_decay_factor : float
        Multiplicative decay factor.
    mu : float
        Weight of the denoising loss (1 − μ for deconvolution loss).
    hess_weight : float
        Weight of Hessian regularisation in Stage 2 loss (λ in Eq. 5).
    base : int
        U-Net base channel count.
    device : str or torch.device or None
    verbose : bool

    Returns
    -------
    model_den : UNet  (Stage 1, on CPU)
    model_dec : UNet  (Stage 2, on CPU)
    loss_history : list of (loss_total, loss_den, loss_dec) tuples, every 100 iters
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model_den = UNet(base=base).to(device)
    model_dec = UNet(base=base).to(device)

    params = list(model_den.parameters()) + list(model_dec.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_steps, gamma=lr_decay_factor)

    # PSF as a (1, 1, kH, kW) convolution kernel
    kH, kW = psf.shape
    psf_t = torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0).to(device)
    # Asymmetric padding so output matches input size for even kernel sizes
    ph_lo = (kH - 1) // 2;  ph_hi = kH - 1 - ph_lo
    pw_lo = (kW - 1) // 2;  pw_hi = kW - 1 - pw_lo

    dataset = TensorDataset(
        torch.from_numpy(y_hat).float(),
        torch.from_numpy(y_bar).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True, pin_memory=True)
    loader_iter = iter(loader)

    loss_history = []
    model_den.train()
    model_dec.train()

    for step in range(n_iters):
        try:
            xb, yb = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            xb, yb = next(loader_iter)

        xb, yb = xb.to(device), yb.to(device)

        # Stage 1: denoise
        out_den = model_den(xb)
        loss_den = F.mse_loss(out_den, yb)

        # Stage 2: deconvolve (input = Stage 1 output, end-to-end)
        out_dec = model_dec(out_den)
        # PSF consistency: PSF-convolved SR output should match the observed y_bar
        out_dec_padded = F.pad(out_dec, (pw_lo, pw_hi, ph_lo, ph_hi), mode='reflect')
        out_dec_conv = F.conv2d(out_dec_padded, psf_t, padding=0)
        loss_dec = (F.mse_loss(out_dec_conv, yb) +
                    hess_weight * hessian_loss(out_dec))

        loss = mu * loss_den + (1.0 - mu) * loss_dec

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            loss_history.append((loss.item(), loss_den.item(), loss_dec.item()))
        if verbose and step % 5000 == 0:
            print(f'  iter {step:6d}/{n_iters}  '
                  f'loss={loss.item():.6f}  '
                  f'(den={loss_den.item():.6f}  dec={loss_dec.item():.6f})  '
                  f'lr={optimizer.param_groups[0]["lr"]:.2e}')

    return model_den.cpu(), model_dec.cpu(), loss_history


# ── Inference ────────────────────────────────────────────────────────────────

def _sliding_window_inference(model, y_norm, patch_size, overlap, batch_size, device):
    """
    Shared sliding-window patch inference with Hann-taper blending.

    Parameters
    ----------
    model : UNet
        Trained model (already on CPU; moved to device temporarily).
    y_norm : np.ndarray, shape (H, W), float32
        Input image normalised to [0, 1].
    patch_size, overlap, batch_size, device : see public wrappers.

    Returns
    -------
    output : np.ndarray, shape (H, W), float64
        Blended prediction (in normalised [0, 1]-ish range).
    """
    H, W = y_norm.shape
    stride = patch_size - overlap

    # Collect patches
    patches, positions = [], []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patches.append(y_norm[i:i+patch_size, j:j+patch_size][np.newaxis, np.newaxis])
            positions.append((i, j))
    if H > patch_size:
        for j in range(0, W - patch_size + 1, stride):
            i = H - patch_size
            patches.append(y_norm[i:i+patch_size, j:j+patch_size][np.newaxis, np.newaxis])
            positions.append((i, j))
    if W > patch_size:
        for i in range(0, H - patch_size + 1, stride):
            j = W - patch_size
            patches.append(y_norm[i:i+patch_size, j:j+patch_size][np.newaxis, np.newaxis])
            positions.append((i, j))
    if H > patch_size and W > patch_size:
        patches.append(y_norm[H-patch_size:H, W-patch_size:W][np.newaxis, np.newaxis])
        positions.append((H - patch_size, W - patch_size))

    patches_t = torch.from_numpy(np.concatenate(patches, axis=0)).float()

    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for b in range(0, len(patches_t), batch_size):
            out = model(patches_t[b:b+batch_size].to(device))
            results.append(out.cpu().numpy())
    model.cpu()

    pred_patches = np.concatenate(results, axis=0)[:, 0]  # (N, H_p, W_p)

    output = np.zeros((H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)
    h_win = np.hanning(patch_size).astype(np.float64)
    h_win = 0.1 + 0.9 * h_win   # floor at 0.1 so corner pixels always have weight
    blend = np.outer(h_win, h_win)

    for patch, (i, j) in zip(pred_patches, positions):
        output[i:i+patch_size, j:j+patch_size] += patch * blend
        weight[i:i+patch_size, j:j+patch_size] += blend

    weight = np.maximum(weight, 1e-6)
    return output / weight


def denoise_image(model_den, y, patch_size=128, overlap=32, batch_size=8, device=None):
    """
    Stage 1 inference: denoise a full image via sliding-window patch processing.

    Parameters
    ----------
    model_den : UNet
        Trained Stage 1 denoiser.
    y : np.ndarray, shape (H, W), float64
        Noisy input image (raw ADU values, not normalised).
    patch_size : int
    overlap : int
    batch_size : int
    device : str or torch.device or None

    Returns
    -------
    denoised : np.ndarray, shape (H, W), float32
        Denoised image, rescaled to match the input intensity range.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    lo = float(np.percentile(y, 0))
    hi = float(np.percentile(y, 100))
    y_norm = np.clip((y - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)

    output = _sliding_window_inference(model_den, y_norm, patch_size, overlap, batch_size, device)
    return (output * (hi - lo) + lo).astype(np.float32)


def deconvolve_image(model_den, model_dec, y, patch_size=128, overlap=32,
                     batch_size=8, device=None):
    """
    Two-stage inference: denoise (Stage 1) then deconvolve (Stage 2).

    Parameters
    ----------
    model_den : UNet
        Trained Stage 1 denoiser.
    model_dec : UNet
        Trained Stage 2 deconvolver.
    y : np.ndarray, shape (H, W), float64
        Noisy input image (raw ADU values).
    patch_size, overlap, batch_size, device : see denoise_image.

    Returns
    -------
    denoised : np.ndarray, shape (H, W), float32
        Stage 1 output (denoised, still PSF-blurred).
    deconvolved : np.ndarray, shape (H, W), float32
        Stage 2 output (denoised + PSF-deconvolved).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    lo = float(np.percentile(y, 0))
    hi = float(np.percentile(y, 100))
    y_norm = np.clip((y - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)

    # Stage 1: denoise
    den_norm = _sliding_window_inference(
        model_den, y_norm, patch_size, overlap, batch_size, device)
    denoised = (den_norm * (hi - lo) + lo).astype(np.float32)

    # Stage 2: deconvolve (applied to Stage 1 normalised output)
    den_norm_f32 = np.clip(den_norm, 0.0, 1.0).astype(np.float32)
    dec_norm = _sliding_window_inference(
        model_dec, den_norm_f32, patch_size, overlap, batch_size, device)
    deconvolved = (dec_norm * (hi - lo) + lo).astype(np.float32)

    return denoised, deconvolved
