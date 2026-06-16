# MRI End-to-End Variational Network (VarNet) Reconstruction

> Reconstruct 320x320 knee MRI images from 4x-accelerated multi-coil k-space using an End-to-End Variational Network — a 12-cascade unrolled optimization network that jointly estimates coil sensitivity maps and applies learned U-Net regularizers with data consistency, trained end-to-end with SSIM loss on the fastMRI dataset.

> Domain: Medicine | Keywords: deep learning reconstruction, unrolled network | Difficulty: Hard

## Background

Traditional MRI reconstruction solves an optimization problem with hand-crafted regularizers (TV, wavelets) or plug-in denoisers. **End-to-End Variational Networks (VarNet)** replace this entire pipeline with a learned unrolled network: each cascade performs sensitivity-weighted data consistency followed by a U-Net regularizer, and the entire architecture is trained end-to-end on paired data.

VarNet is the top-performing method on the fastMRI leaderboard and represents the current state-of-the-art for accelerated multi-coil MRI reconstruction.

## Problem Description

The multi-coil MRI forward model is:

$$y_c = M \cdot \mathcal{F}(S_c \cdot x) + \eta_c \quad \text{for each coil } c = 1, \ldots, N_c$$

where $x$ is the unknown image, $S_c$ are coil sensitivity maps, $\mathcal{F}$ is the 2D DFT, $M$ is a binary equispaced undersampling mask (4x acceleration), and $\eta_c$ is measurement noise.

**What makes this hard**: The reconstruction requires a deep unrolled network with complex-valued multi-coil operations, joint sensitivity map estimation, and data consistency enforcement. The pretrained weights encode learned priors from thousands of training volumes, and the model must exactly match the original architecture to load them correctly.

**Input**: Undersampled multi-coil k-space $(N_c, H, W)$ complex64, undersampling mask.

**Output**: Reconstructed magnitude image $(320, 320)$ float32.

## Data Description

**Source**: fastMRI knee multicoil validation dataset (file1001077.h5). 1 slice extracted from a CORPD_FBK acquisition with 15 receiver coils and 640x368 k-space matrix.

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `kspace_real` | (1, 15, 640, 368) | float32 | Real part of fully-sampled multi-coil k-space |
| `kspace_imag` | (1, 15, 640, 368) | float32 | Imaginary part of fully-sampled multi-coil k-space |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `image` | (1, 320, 320) | float32 | RSS ground truth from fully-sampled data |

### data/varnet_knee_state_dict.pt

Pretrained End-to-End VarNet weights (~120 MB). Architecture: 12 cascades, 18 U-Net channels, 4 pooling layers, 8 sensitivity channels, 4 sensitivity pooling layers. Trained on the fastMRI knee training set with SSIM loss.

### data/meta_data.json

Contains imaging parameters: number of slices, coils, k-space shape, image shape, acceleration factor, center fraction, acquisition protocol.

## Method Hints

VarNet is an end-to-end variational network that unrolls proximal gradient descent into a deep neural network with multiple cascades. Each cascade applies a data consistency step (using the multi-coil forward model) followed by a learned CNN regularizer (U-Net). Sensitivity maps are estimated jointly by a separate sub-network from the ACS region. The pretrained model performs inference on undersampled k-space with equispaced sampling, and the output magnitude image is center-cropped to match the ground truth resolution.

## References

1. Sriram, A. et al. (2020). End-to-End Variational Networks for Accelerated MRI Reconstruction. MICCAI 2020.
2. Zbontar, J. et al. (2018). fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. arXiv:1811.08839.
3. fastMRI GitHub: https://github.com/facebookresearch/fastMRI
4. fastMRI Dataset: https://fastmri.med.nyu.edu/
