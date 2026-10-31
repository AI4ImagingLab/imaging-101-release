# Approach: PnP-MSSN MRI Reconstruction

## Problem statement

Reconstruct a 320x320 MRI knee image from 36-line radially subsampled k-space measurements. The forward model is a masked 2D Fourier transform. The problem is severely underdetermined: only ~11% of k-space is sampled, so naive IFFT produces strong streaking artifacts.

## Solution strategy

### Step 1: Data preprocessing
- Load the MRI image from `raw_data.npz`
- Normalize to [0, 1] range
- Generate 36-line radial sampling mask
- Compute subsampled k-space measurements: y = mask * FFT(x) / sqrt(N*M)

### Step 2: Forward model
- **Forward:** `fmult(x, mask)` — masked, centered FFT with 1/sqrt(N*M) normalization
- **Adjoint:** `ftran(z, mask)` — masked inverse centered FFT with sqrt(N*M) normalization
- **Gradient:** `grad(x) = ftran(fmult(x, mask) - y, mask)` — gradient of data fidelity

### Step 3: PnP-PGM reconstruction
Proximal Gradient Method with MSSN denoiser:
```
for k = 1, ..., 200:
    g = grad(x_k)                      # data fidelity gradient
    s = clip(x_k - step * g, 0, inf)   # gradient step + positivity
    x_{k+1} = MSSN_denoise(s)          # denoising proximal step
```

The MSSN denoiser:
1. Extracts overlapping 42x42 patches with stride 7
2. Scales input to [0, 255] range
3. Runs each patch through the pre-trained MSSN network (8 recurrent states, 2-head attention)
4. Averages overlapping patch outputs
5. Adds denoised residual back to input (residual learning)
6. Scales output back to [0, 1]

### Step 4: Evaluation
- **SNR:** 20 * log10(||x_true|| / ||x_true - x_hat||) — signal-to-noise ratio in dB
- **IFFT baseline:** naive zero-filled inverse FFT reconstruction
- Track SNR and fixed-point residual ||x - Prox(x)||^2 over iterations

### Step 5: Visualization
- Ground truth vs IFFT vs PnP-MSSN reconstruction
- Convergence curves (SNR vs iteration, residual vs iteration)
- Error maps
- Sampling mask visualization

## Expected results

| Method | SNR (dB) |
|--------|----------|
| IFFT (zero-filled) | ~13.5 |
| PnP-MSSN (200 iter) | ~17.1 |

The PnP-MSSN reconstruction achieves ~3.6 dB improvement over the IFFT baseline, with stable convergence (SNR plateaus around iteration 100).
