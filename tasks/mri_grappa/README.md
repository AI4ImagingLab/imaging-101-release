# MRI GRAPPA Reconstruction

> Reconstruct a 128x128 multi-coil brain MRI image from 2x-accelerated Cartesian k-space using GRAPPA, a k-space interpolation method that estimates missing samples from acquired neighbours using linear kernels calibrated on a fully-sampled auto-calibration signal region.

> Domain: Medicine | Keywords: parallel imaging, k-space interpolation | Difficulty: Medium

## Background

Parallel imaging exploits spatial encoding from multiple receiver coils to reconstruct images from undersampled k-space. Unlike compressed sensing methods that solve an optimization problem with explicit regularization, **GRAPPA (GeneRalized Autocalibrating Partially Parallel Acquisitions)** directly interpolates missing k-space samples using linear kernels learned from a fully-sampled calibration region.

The key insight is that linear relationships between k-space samples across coils are shift-invariant: weights learned from the auto-calibration signal (ACS) region at the centre of k-space generalise to the entire k-space.

## Problem Description

The multi-coil MRI forward model is:

$$y_c = M \cdot \mathcal{F}(S_c \cdot x) \quad \text{for each coil } c = 1, \ldots, N_c$$

where:
- $x \in \mathbb{C}^{N \times N}$ is the unknown image
- $S_c \in \mathbb{C}^{N \times N}$ is the sensitivity map for coil $c$
- $\mathcal{F}$ is the 2D centered DFT with $1/\sqrt{N^2}$ normalization
- $M$ is a binary undersampling mask (phase-encode lines)
- $y_c$ is the acquired k-space for coil $c$

**What makes this hard**: The missing k-space samples must be inferred from the acquired neighbours using inter-coil correlations. A limited ACS region provides insufficient calibration data, leading to noise amplification (high g-factor) and aliasing artefacts. The reconstruction accuracy depends critically on the kernel size and regularisation strength.

**Input**: Undersampled multi-coil k-space $(N, N, N_c)$, ACS calibration data.

**Output**: Reconstructed magnitude image $(N, N)$.

## Data Description

**Source**: Synthetic Shepp-Logan phantom with Gaussian coil sensitivity maps (8 coils), generated via `phantominator` and `pygrappa.utils.gaussian_csm`. The phantom provides a known ground truth for quantitative evaluation.

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `kspace_full_real` | (1, 128, 128, 8) | float32 | Real part of fully-sampled multi-coil k-space |
| `kspace_full_imag` | (1, 128, 128, 8) | float32 | Imaginary part of fully-sampled multi-coil k-space |
| `sensitivity_maps_real` | (1, 128, 128, 8) | float32 | Real part of coil sensitivity maps |
| `sensitivity_maps_imag` | (1, 128, 128, 8) | float32 | Imaginary part of coil sensitivity maps |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `image` | (1, 128, 128) | float32 | Shepp-Logan phantom (magnitude, [0, 1]) |

### data/meta_data.json

Contains imaging parameters: image shape, number of coils, phantom type, sensitivity model, FFT normalization convention.

## Method Hints

GRAPPA is a k-space interpolation method that estimates missing samples from acquired neighbours using linear kernels. The kernels are calibrated from a fully-sampled auto-calibration signal (ACS) region at k-space centre using regularised least-squares fitting. Each unique undersampling pattern within the kernel window requires a separate set of weights. After filling all missing k-space entries, the image is obtained by inverse FFT and root sum-of-squares (RSS) coil combination.

## References

1. Griswold, M.A. et al. (2002). Generalized Autocalibrating Partially Parallel Acquisitions (GRAPPA). Magnetic Resonance in Medicine, 47(6), 1202-1210.
2. pygrappa: https://github.com/mckib2/pygrappa
