# Approach: PnP-ADMM CS-MRI with RealSN-DnCNN

## Problem Statement

Reconstruct a 256x256 brain MRI image from 30% randomly sampled k-space using ADMM where the regularization proximal operator is replaced by a pretrained spectrally-normalized denoiser.

## Algorithm: PnP-ADMM

ADMM splits the reconstruction into a data fidelity step and a regularization step. In PnP-ADMM, the regularization step is replaced by a denoiser.

### Variables
- `x`: reconstructed image (m, n)
- `v`: auxiliary variable (m, n)
- `u`: scaled dual variable (m, n)

### Update Rules

**v-update** (data fidelity proximal, closed-form in Fourier domain):
```
vtilde = x + u
vf = FFT2(vtilde)
La2 = 1 / (2 * alpha)
vf[sampled] = (La2 * vf[sampled] + y[sampled]) / (1 + La2)
v = real(IFFT2(vf))
```

**x-update** (denoising step):
```
xtilde = 2*v - x_old - u_old
# Normalize to [0, 1]
xtilde_norm = (xtilde - min) / (max - min)
# Scale/shift for denoiser training distribution
scale_range = 1 + sigma/255/2
scale_shift = (1 - scale_range) / 2
xtilde_scaled = xtilde_norm * scale_range + scale_shift
# Denoise (residual denoiser)
r = DnCNN(xtilde_scaled)
x_denoised = xtilde_scaled - r
# Invert normalization
x = (x_denoised - scale_shift) / scale_range * (max - min) + min
```

**u-update** (dual variable):
```
u = u_old + x_old - v
```

## Denoiser: RealSN-DnCNN

- 17-layer DnCNN (Conv-BN-ReLU architecture)
- Real spectral normalization on every Conv layer via convolution-based power iteration
- Per-layer scaling by 0.3^(1/17) targeting network Lipschitz constant ~0.3
- Residual output: predicts noise, not clean image
- Trained at sigma=15 for Gaussian denoising

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| alpha | 2.0 | ADMM penalty parameter (La2 = 0.25) |
| sigma | 15 | Denoiser noise level matching pretrained weights |
| maxitr | 100 | Sufficient for convergence (PSNR plateaus by ~20 iters) |
| mask | Random 30% | Standard CS-MRI acceleration factor |
| noise_scale | 3.0 | Complex noise amplitude scaling |

## Expected Results

- PnP-ADMM: PSNR ~19.91 dB, NCC ~0.995, NRMSE ~0.033
- Zero-fill baseline: PSNR ~11.21 dB, NCC ~0.963, NRMSE ~0.090
- Convergence within ~20 iterations (PSNR plateaus)
