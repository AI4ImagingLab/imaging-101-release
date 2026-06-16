# Approach: MRI End-to-End VarNet Reconstruction

## Problem Statement

Reconstruct 320x320 knee MRI images from 4x-accelerated 15-coil k-space using a pretrained End-to-End Variational Network.

## Algorithm: End-to-End VarNet

### Architecture
- 12 cascades of unrolled proximal gradient descent
- Each cascade: data consistency gradient + U-Net regularizer (18 channels, 4 pools)
- Jointly estimated sensitivity maps via SensitivityModel (8 channels, 4 pools)
- Trained end-to-end with SSIM loss on fastMRI knee training data

### Inference Pipeline
1. Apply equispaced undersampling mask (4x, 8% center fraction) to fully-sampled k-space
2. Convert to real-valued tensor format (Nc, H, W, 2)
3. Forward pass through 12 VarNet cascades
4. Center-crop output to 320x320

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_cascades | 12 |
| chans | 18 |
| pools | 4 |
| sens_chans | 8 |
| sens_pools | 4 |
| acceleration | 4 |
| center_fraction | 0.08 |

## Expected Results

- VarNet avg: SSIM=0.943, NCC=0.999, NRMSE=0.013
- Zero-fill avg: SSIM=0.335, NCC=0.824, NRMSE=0.268
