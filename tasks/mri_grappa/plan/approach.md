# Approach: MRI GRAPPA Reconstruction

## Problem Statement

Reconstruct a 128x128 brain MRI image from 2x-accelerated multi-coil k-space using GRAPPA k-space interpolation calibrated from a 20-line auto-calibration signal region.

## Algorithm: GRAPPA

### Step 1: Data Preparation
- Generate/load synthetic 8-coil k-space (128x128x8, complex128)
- Undersample: keep every 2nd phase-encode line + 20-line ACS region (57.8% sampling)
- Extract calibration data from the fully-sampled ACS region

### Step 2: Kernel Calibration
For each unique sampling geometry within the kernel window:
1. Extract all overlapping (5, 5, 8) patches from ACS data
2. Identify source samples (acquired) and target samples (centre)
3. Solve: $W = (S^H S + \lambda_0 I)^{-1} S^H T$
   where $\lambda_0 = \lambda \cdot \|S^H S\| / \text{dim}$

### Step 3: Interpolation
For each missing k-space location:
1. Find the matching sampling geometry
2. Collect source samples from the local neighbourhood
3. Apply: $\hat{k}[\text{hole}] = W \cdot s$

### Step 4: Image Reconstruction
1. Centered inverse FFT: $\text{img}_c = \sqrt{N^2} \cdot \text{fftshift}(\text{IFFT2}(\text{ifftshift}(k_c)))$
2. RSS coil combination: $\text{img} = \sqrt{\sum_c |\text{img}_c|^2}$

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| R | 2 | Standard acceleration for GRAPPA |
| acs_width | 20 | Sufficient ACS for 5x5 kernel calibration |
| kernel_size | (5, 5) | Standard GRAPPA kernel |
| lambda | 0.01 | Tikhonov regularisation (adaptive scaling applied) |
| n_coils | 8 | Provides sufficient parallel imaging capability |

## Expected Results

- GRAPPA: SSIM=0.985, NCC=0.9999, NRMSE=0.004
- Zero-fill: SSIM=0.648, NCC=0.951, NRMSE=0.075
