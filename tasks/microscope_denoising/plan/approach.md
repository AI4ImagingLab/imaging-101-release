# Approach: Zero-Shot Fluorescence Image Denoising and Deconvolution (ZS-DeconvNet)

## Problem Summary

Jointly remove noise and reverse PSF blurring from a single low-SNR fluorescence microscopy
image, without any clean reference, by exploiting the recorruption self-supervised training
scheme on the noisy images themselves.

## Key Insight: Recorruption as Self-Supervised Supervision

Given a noisy image $y = x + n$ where $n \sim \mathcal{N}(0, \sigma^2(x))$,
any additional noise vector $g \sim \mathcal{N}(0, \sigma^2(y))$ generates two
new observations:

- $\hat{y} = y + \alpha g$: more noisy version (network input)
- $\bar{y} = y - g/\alpha$: less noisy version (training target)

Because $E[g] = 0$, both $\hat{y}$ and $\bar{y}$ are unbiased estimates of $y$.
By Stein's identity, minimising $\text{MSE}(f(\hat{y}), \bar{y})$ over many random draws of $g$
is equivalent to minimising $\text{MSE}(f(y), x)$ — without ever seeing the clean image $x$.

## Algorithm

### Stage 0: Data preparation

1. Load 55 noisy Microtubule TIRF images (502×502 px, background ≈100 ADU).
2. Estimate noise parameters $(\beta_1, \beta_2)$ from frame 0:
   - Divide image into 32×32 blocks.
   - For each block: compute $\mu = \text{mean}(H(y-b))$, $v = \text{Var}(y - H(y))$.
   - Linear regression: $v = \beta_1 \mu + \beta_2$.
3. Set $\alpha = 1.5$ (noise strength; the paper shows results are robust for $\alpha \in [1, 2]$).

### Stage 1: Training data generation

1. Sample $N \approx 10\,000$ random 128×128 patches from all 55 noisy frames.
2. For each patch $p$:
   a. Augment: random rotation ($k \times 90°$) and horizontal flip.
   b. Draw $g \sim \mathcal{N}(0,\, (\beta_1 H(p-b) + \beta_2) \cdot I)$.
   c. Compute $\hat{p} = p + \alpha g$, $\bar{p} = p - g/\alpha$.
   d. Normalise jointly to $[0, 1]$ using percentile (0–100) of $\hat{p}$.

### Stage 2: Joint U-Net training (two stages)

**Architecture** — two identical U-Nets:
- Encoder: $1 \to 32 \to 64 \to 128 \to 256$ channels, MaxPool downsampling.
- Bottleneck: $256 \to 256$.
- Decoder: bilinear upsampling + skip concatenation, symmetric channel reduction.
- Output: single-channel, no activation.

**Joint loss** — Stage 1 and Stage 2 are trained end-to-end simultaneously:

$$\mathcal{L} = \mu \mathcal{L}_\text{den} + (1-\mu) \mathcal{L}_\text{dec}$$

$$\mathcal{L}_\text{den} = \|f_\text{den}(\hat{p}) - \bar{p}\|_2^2$$

$$\mathcal{L}_\text{dec} = \|f_\text{dec}(f_\text{den}(\hat{p})) * \text{PSF} - \bar{p}\|_2^2
+ \lambda \mathcal{R}_\text{Hessian}(f_\text{dec}(f_\text{den}(\hat{p})))$$

where $\mu = 0.5$, $\lambda = 0.02$.

**Hessian regulariser** — sum of squared second-order finite differences
(dxx, dyy, dxy, dyx), penalises high-frequency oscillations from deconvolution.

**Optimiser**: Adam, $\text{lr} = 5 \times 10^{-4}$, decay $\times 0.5$ every 10 000 steps,
batch size 4, 30 000 total iterations.

### Stage 3: Inference

1. Normalise the full noisy image to $[0, 1]$ (percentile 0–100).
2. Stage 1: sliding-window patch inference (128×128, stride 96, Hann-taper blending) → denoised.
3. Stage 2: sliding-window inference on the normalised Stage 1 output → deconvolved.
4. Rescale both outputs back to original intensity range.

## Expected Performance (Microtubule TIRF dataset)

- Background noise: ~10× reduction (Stage 1)
- SNR: ~10× improvement (Stage 1)
- Laplacian sharpness: ~8× improvement (Stage 2)
- PSF residual: < 0.002 (Stage 2 consistency check)

Visual improvement: individual microtubule filaments (~200 nm diameter) become clearly
resolved from background; deconvolution restores the sub-diffraction-limited sharpness.
