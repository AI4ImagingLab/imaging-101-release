# Multi-Coil MRI Reconstruction with Total Variation

> Reconstruct knee MRI images from 8× undersampled multi-coil k-space measurements using Total Variation regularization, where the severe undersampling creates aliasing artifacts that must be suppressed while preserving anatomical detail.

> Domain: medicine | Keywords: compressed sensing, MRI reconstruction | Difficulty: Easy

## Background

Magnetic Resonance Imaging (MRI) acquires data in the spatial frequency domain (k-space). Each receive coil in a multi-coil array measures a modulated version of the full k-space, weighted by its spatially varying sensitivity profile. Fully sampling k-space is time-consuming, so **compressed sensing MRI** accelerates acquisition by deliberately undersampling k-space and using regularization to suppress the resulting aliasing artifacts during reconstruction.

Multi-coil MRI exploits redundancy across coil channels. Each coil sees the same underlying image but weighted by a different sensitivity map, providing complementary spatial information that helps constrain the reconstruction.

Total Variation (TV) regularization is a classical sparsity-promoting prior that penalizes the L1 norm of the image gradient. It is well suited for MRI because anatomical images are approximately piecewise smooth — organ boundaries produce sparse gradients while interior regions are relatively uniform.

## Problem Description

The multi-coil MRI forward model maps a complex image $x \in \mathbb{C}^{H \times W}$ to undersampled k-space measurements:

$$y_c = M \cdot \mathcal{F} \cdot S_c \cdot x + \epsilon_c, \quad c = 1, \ldots, C$$

where:
- $S_c \in \mathbb{C}^{H \times W}$ is the sensitivity map for coil $c$
- $\mathcal{F}$ is the 2D discrete Fourier transform (ortho-normalized)
- $M \in \{0, 1\}^W$ is a 1-D binary undersampling mask applied along the phase-encode direction
- $\epsilon_c$ is measurement noise

The inverse problem is ill-posed because the mask $M$ retains only 12.5% of k-space lines (8× acceleration). The retained lines include a small fully-sampled auto-calibration signal (ACS) region at the center of k-space (4% of lines) plus randomly selected outer lines.

**Input:** Undersampled multi-coil k-space $\{y_c\}_{c=1}^C$, sensitivity maps $\{S_c\}_{c=1}^C$, undersampling mask $M$.

**Output:** Reconstructed complex image $\hat{x} \in \mathbb{C}^{H \times W}$.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `masked_kspace` | (1, 15, 320, 320) | complex64 | Undersampled multi-coil k-space measurements (normalized by per-sample 99th-percentile MVUE magnitude) |
| `sensitivity_maps` | (1, 15, 320, 320) | complex64 | Coil sensitivity maps estimated via ESPIRiT |
| `undersampling_mask` | (320,) | float32 | 1-D binary phase-encode undersampling mask (40/320 lines sampled at 8× acceleration) |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `mvue` | (1, 1, 320, 320) | complex64 | Minimum Variance Unbiased Estimate from fully-sampled k-space (same per-sample normalization as masked_kspace) |

The ground truth is the MVUE image, computed as:

$$x_{\text{mvue}} = \frac{\sum_c \bar{S}_c \cdot \mathcal{F}^{-1}(y_c^{\text{full}})}{\sqrt{\sum_c |S_c|^2}}$$

### data/meta_data.json

Contains imaging parameters: image size (320×320), number of coils (15), acceleration ratio (8), mask pattern (random, seed 0), ACS fraction (4%), FFT normalization (ortho), and data source (fastMRI knee validation).

## Method Hints

Total Variation reconstruction solves the optimization problem:

$$\hat{x} = \arg\min_x \; \frac{1}{2} \sum_c \|M \cdot \mathcal{F}(S_c \cdot x) - y_c\|_2^2 + \lambda \cdot \text{TV}(x)$$

where $\text{TV}(x) = \sum_{i,j} \sqrt{|\nabla_h x_{i,j}|^2 + |\nabla_v x_{i,j}|^2}$ is the isotropic total variation. The regularization parameter $\lambda$ controls the trade-off between data fidelity and smoothness.

This convex problem can be solved efficiently using primal-dual splitting methods that handle the non-smooth TV term directly. TV regularization promotes piecewise-constant images and is particularly effective for phantoms and anatomical structures with sharp boundaries.

## References

- Lustig, M., Donoho, D., & Pauly, J. M. (2007). Sparse MRI: The application of compressed sensing for rapid MR imaging. *Magnetic Resonance in Medicine*, 58(6), 1182-1195.
- Block, K. T., Uecker, M., & Frahm, J. (2007). Undersampled radial MRI with multiple coils. Iterative image reconstruction using a total variation constraint. *Magnetic Resonance in Medicine*, 57(6), 1086-1098.
- Chambolle, A. & Pock, T. (2011). A first-order primal-dual algorithm for convex problems with applications to imaging. *Journal of Mathematical Imaging and Vision*, 40(1), 120-145.
- Ong, F. & Lustig, M. (2019). SigPy: A Python package for high performance iterative reconstruction. *Proceedings of the ISMRM*. (Algorithm ported from SigPy's TotalVariationRecon.)
- Wu, Z., Sun, H., & Bouman, K. L. (2025). InverseBench: A Comprehensive Benchmark for Scientific Inverse Problems.
