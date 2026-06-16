# MRI CG-SENSE Parallel Imaging Reconstruction

> Reconstruct a 128x128 brain MRI image from 4x-accelerated multi-coil Cartesian k-space using Conjugate Gradient SENSE (CG-SENSE), an image-domain parallel imaging method that solves the SENSE encoding equation directly via conjugate gradient on the normal equations.

> Domain: Medicine | Keywords: parallel imaging, image-domain reconstruction | Difficulty: Medium

## Background

SENSE (SENSitivity Encoding) is one of the two foundational parallel imaging methods for MRI, alongside GRAPPA. While GRAPPA operates in k-space by interpolating missing samples, SENSE works in the image domain by directly solving the linear system that relates the unknown image to the acquired multi-coil k-space data through known coil sensitivity maps.

CG-SENSE extends SENSE to handle arbitrary Cartesian undersampling patterns by formulating the reconstruction as a least-squares problem and solving it iteratively with conjugate gradient.

## Problem Description

The multi-coil MRI encoding model is:

$$y_c = M \cdot \mathcal{F}(S_c \cdot x) \quad \text{for each coil } c = 1, \ldots, N_c$$

where:
- $x \in \mathbb{C}^{N \times N}$ is the unknown single-coil image
- $S_c \in \mathbb{C}^{N \times N}$ is the sensitivity map for coil $c$
- $\mathcal{F}$ is the 2D centered DFT
- $M$ is a binary undersampling mask
- $y_c$ is the acquired k-space for coil $c$

In matrix form: $y = Ax$ where $A$ stacks the per-coil encoding operators.

**What makes this hard**: With R=4 acceleration and 8 coils, the problem is underdetermined in each local voxel group. The condition number of the normal operator $A^H A$ depends on the coil geometry — poor coil sensitivity variation leads to noise amplification (high g-factor). The system must be solved iteratively since $A^H A$ is too large to form explicitly.

**Input**: Undersampled multi-coil k-space $(N, N, N_c)$, coil sensitivity maps $(N, N, N_c)$.

**Output**: Reconstructed single-coil magnitude image $(N, N)$.

## Data Description

**Source**: Synthetic Shepp-Logan phantom with 8 Gaussian coil sensitivity maps, generated via `phantominator` and pygrappa's `gaussian_csm`. Complex Gaussian noise (sigma=0.005) added to k-space. 4x uniform undersampling in phase-encode with 16-line ACS region.

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `kspace_full_real` | (1, 128, 128, 8) | float32 | Real part of noisy fully-sampled multi-coil k-space |
| `kspace_full_imag` | (1, 128, 128, 8) | float32 | Imaginary part of noisy fully-sampled multi-coil k-space |
| `sensitivity_maps_real` | (1, 128, 128, 8) | float32 | Real part of coil sensitivity maps |
| `sensitivity_maps_imag` | (1, 128, 128, 8) | float32 | Imaginary part of coil sensitivity maps |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `image` | (1, 128, 128) | float32 | Shepp-Logan phantom magnitude, [0, 1] |

### data/meta_data.json

Contains imaging parameters: image shape, number of coils, phantom type, sensitivity model, acceleration factor, ACS width.

## Method Hints

CG-SENSE solves the normal equations $A^H A \hat{x} = A^H y$ using the Conjugate Gradient algorithm applied to the matrix-free normal operator. The SENSE encoding operator combines coil sensitivity modulation, Fourier encoding, and undersampling masking. The adjoint combines coil images via conjugate sensitivity weighting. The output is a complex image; take the magnitude for evaluation.

## References

1. Pruessmann, K.P. et al. (1999). SENSE: Sensitivity Encoding for Fast MRI. MRM, 42(5), 952-962.
2. Pruessmann, K.P. et al. (2001). Advances in sensitivity encoding with arbitrary k-space trajectories. MRM, 46(4), 638-651.
3. pygrappa: https://github.com/mckib2/pygrappa
