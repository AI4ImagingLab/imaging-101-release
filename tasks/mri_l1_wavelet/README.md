# Multi-Coil MRI Reconstruction with L1-Wavelet Regularization

> Reconstruct a Shepp-Logan phantom from 8x undersampled multi-coil k-space measurements using L1-wavelet regularization, where the wavelet sparsity prior captures multi-scale image features (edges and textures) more effectively than gradient-based priors like total variation.

> Domain: medicine | Keywords: compressed sensing, MRI reconstruction | Difficulty: Easy

## Background

Magnetic Resonance Imaging (MRI) acquires data in the spatial frequency domain (k-space). Each receive coil in a multi-coil array measures a modulated version of the full k-space, weighted by its spatially varying sensitivity profile. Fully sampling k-space is time-consuming, so **compressed sensing MRI** accelerates acquisition by deliberately undersampling k-space and using regularization to suppress the resulting aliasing artifacts during reconstruction.

Multi-coil MRI exploits redundancy across coil channels. Each coil sees the same underlying image but weighted by a different sensitivity map, providing complementary spatial information that helps constrain the reconstruction.

**L1-Wavelet regularization** promotes sparsity in the wavelet domain rather than the gradient domain (as in Total Variation). The discrete wavelet transform decomposes the image into multi-resolution subbands: coarse approximation coefficients and detail coefficients at each scale. Natural and medical images are approximately sparse in the wavelet domain because most energy concentrates in a few large coefficients. Penalizing the L1 norm of wavelet coefficients encourages sparse reconstructions that preserve edges and fine details at multiple scales, making wavelets particularly effective for images with texture and multi-scale structure.

## Problem Description

The multi-coil MRI forward model maps a complex image $x \in \mathbb{C}^{H \times W}$ to undersampled k-space measurements:

$$y_c = M \cdot \mathcal{F} \cdot S_c \cdot x + \epsilon_c, \quad c = 1, \ldots, C$$

where:
- $S_c \in \mathbb{C}^{H \times W}$ is the sensitivity map for coil $c$
- $\mathcal{F}$ is the 2D discrete Fourier transform (ortho-normalized)
- $M \in \{0, 1\}^W$ is a 1-D binary undersampling mask applied along the phase-encode direction
- $\epsilon_c$ is measurement noise

The inverse problem is ill-posed because the mask $M$ retains only 12.5% of k-space lines (8x acceleration). The retained lines include a fully-sampled auto-calibration signal (ACS) region at the center of k-space (8% of lines) plus randomly selected outer lines.

**Input:** Undersampled multi-coil k-space $\{y_c\}_{c=1}^C$, sensitivity maps $\{S_c\}_{c=1}^C$, undersampling mask $M$.

**Output:** Reconstructed complex image $\hat{x} \in \mathbb{C}^{H \times W}$.

## Data Description

The data is synthetically generated using a Shepp-Logan phantom with Gaussian coil sensitivity maps.

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `masked_kspace` | (1, 8, 128, 128) | complex128 | Undersampled multi-coil k-space measurements |
| `sensitivity_maps` | (1, 8, 128, 128) | complex128 | Gaussian coil sensitivity maps (analytically generated, RSS-normalized) |
| `undersampling_mask` | (128,) | float32 | 1-D binary phase-encode undersampling mask (16/128 lines sampled at 8x acceleration) |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `phantom` | (1, 1, 128, 128) | complex128 | Shepp-Logan phantom normalized to [0, 1] (arbitrary units) |

### data/wavelet_filters.npz

Daubechies-4 and Haar wavelet filter coefficients for decomposition and reconstruction. Provided as data so that agents do not need to recall precise floating-point filter values from memory.

| Key | Shape | Description |
|-----|-------|-------------|
| `db4_dec_lo` | (8,) | DB4 low-pass decomposition filter |
| `db4_dec_hi` | (8,) | DB4 high-pass decomposition filter |
| `db4_rec_lo` | (8,) | DB4 low-pass reconstruction filter |
| `db4_rec_hi` | (8,) | DB4 high-pass reconstruction filter |
| `haar_dec_lo` | (2,) | Haar low-pass decomposition filter |
| `haar_dec_hi` | (2,) | Haar high-pass decomposition filter |
| `haar_rec_lo` | (2,) | Haar low-pass reconstruction filter |
| `haar_rec_hi` | (2,) | Haar high-pass reconstruction filter |

### data/meta_data.json

Contains imaging parameters: image size (128x128), number of coils (8), acceleration ratio (8), mask pattern (random, seed 0), ACS fraction (8%), FFT normalization (ortho), and data source (synthetic Shepp-Logan with Gaussian coil maps).

## Method Hints

L1-Wavelet reconstruction solves the optimization problem:

$$\hat{x} = \arg\min_x \; \frac{1}{2} \sum_c \|M \cdot \mathcal{F}(S_c \cdot x) - y_c\|_2^2 + \lambda \|\Psi x\|_1$$

where $\Psi$ is the discrete wavelet transform (e.g., Daubechies-4 wavelets) and $\|\cdot\|_1$ is the L1 norm applied to all wavelet coefficients. The regularization parameter $\lambda$ controls the trade-off between data fidelity and wavelet sparsity.

The key advantage over Total Variation (which penalizes $\|\nabla x\|_1$) is that wavelets provide a multi-resolution sparse representation. TV enforces piecewise smoothness and can produce "staircase" artifacts, while wavelets capture features at multiple scales and better preserve fine textures.

This convex problem can be solved using proximal gradient methods (e.g., ISTA/FISTA), which alternate between gradient steps on the data fidelity and soft-thresholding of wavelet coefficients to enforce sparsity.

## References

- Lustig, M., Donoho, D., & Pauly, J. M. (2007). Sparse MRI: The application of compressed sensing for rapid MR imaging. *Magnetic Resonance in Medicine*, 58(6), 1182-1195.
- Mallat, S. (2009). *A Wavelet Tour of Signal Processing: The Sparse Way*. Academic Press.
- Beck, A. & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202.
- Ong, F. & Lustig, M. (2019). SigPy: A Python package for high performance iterative reconstruction. *Proceedings of the ISMRM*. (Algorithm ported from SigPy's L1WaveletRecon.)
