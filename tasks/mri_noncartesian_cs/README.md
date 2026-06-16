# Non-Cartesian MRI Reconstruction with L1-Wavelet Compressed Sensing

> Reconstruct a complex MRI image from radially sampled multi-coil k-space data using Non-Uniform FFT and L1-wavelet regularization, where the non-Cartesian sampling pattern creates spatially varying aliasing artifacts that standard FFT-based methods cannot handle.

> Domain: medicine | Keywords: compressed sensing, non-Cartesian MRI | Difficulty: Medium

## Background

Magnetic Resonance Imaging (MRI) acquires data in the spatial frequency domain (k-space). While conventional MRI samples k-space on a Cartesian grid (enabling reconstruction via standard FFT), many acquisition strategies sample k-space along non-Cartesian trajectories such as radial spokes, spirals, or random patterns. These non-Cartesian trajectories offer advantages in motion robustness, SNR efficiency, and ultra-short echo time imaging, but require specialized reconstruction algorithms.

In radial sampling, k-space is acquired along spokes emanating from the center, typically following a golden-angle increment to ensure approximately uniform angular coverage. The golden angle ($\pi(\sqrt{5}-1)/2 \approx 111.25^\circ$) between successive spokes provides near-optimal azimuthal uniformity for any number of acquired spokes, enabling flexible retrospective undersampling.

Multi-coil MRI exploits redundancy across receive coils: each coil measures the same underlying image modulated by its spatially varying sensitivity profile, providing complementary spatial encoding that helps constrain the reconstruction.

L1-wavelet regularization promotes sparsity in the wavelet domain, which is effective for MRI because natural images have approximately sparse wavelet representations. Combined with the NUFFT forward model, this yields a compressed sensing reconstruction that can recover high-quality images from undersampled non-Cartesian data.

## Problem Description

The multi-coil non-Cartesian MRI forward model maps a complex image $x \in \mathbb{C}^{H \times W}$ to non-Cartesian k-space measurements:

$$y_c = \mathcal{F}_\text{NU}(S_c \cdot x) + \eta_c, \quad c = 1, \ldots, C$$

where:
- $S_c \in \mathbb{C}^{H \times W}$ is the sensitivity map for coil $c$
- $\mathcal{F}_\text{NU}$ is the Non-Uniform FFT (NUFFT) evaluated at the non-Cartesian k-space locations
- $\eta_c$ is complex Gaussian measurement noise

The NUFFT computes:

$$(\mathcal{F}_\text{NU} f)(\mathbf{k}_m) = \sum_{n_1, n_2} f[n_1, n_2] \, e^{-i \, \mathbf{k}_m \cdot \mathbf{r}_n}$$

where $\mathbf{k}_m$ are the non-Cartesian k-space sample locations and $\mathbf{r}_n$ are the image pixel positions. Unlike the standard FFT which requires $\mathbf{k}_m$ on a regular grid, the NUFFT handles arbitrary sample positions via interpolation (gridding) in k-space.

The problem is ill-posed because 64 radial spokes with 128 readout points yield 8,192 k-space samples for a 128x128 = 16,384-pixel image, providing only 50% sampling coverage. The radial sampling pattern concentrates samples near k-space center (low frequencies are oversampled) while undersampling high frequencies, creating spatially varying aliasing that manifests as streaking artifacts.

**Input:** Multi-coil non-Cartesian k-space $\{y_c\}_{c=1}^C$, k-space trajectory coordinates $\{\mathbf{k}_m\}$, coil sensitivity maps $\{S_c\}_{c=1}^C$.

**Output:** Reconstructed complex image $\hat{x} \in \mathbb{C}^{H \times W}$.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `kdata` | (1, 4, 8192) | complex64 | Multi-coil non-Cartesian k-space samples (4 coils, 64 spokes x 128 readout points) |
| `coord` | (1, 8192, 2) | float32 | Non-Cartesian k-space trajectory coordinates in units of cycles/pixel, range [-N/2, N/2] |
| `coil_maps` | (1, 4, 128, 128) | complex64 | Birdcage coil sensitivity maps |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `phantom` | (1, 128, 128) | complex64 | Shepp-Logan phantom (complex-valued, real positive, arbitrary intensity units) |

### data/meta_data.json

Contains imaging parameters: image size (128x128), number of coils (4), number of spokes (64), number of readout points per spoke (128), trajectory type (golden-angle radial), noise standard deviation (0.005), and data source (synthetic Shepp-Logan).

## Method Hints

The reconstruction solves the L1-wavelet regularized problem:

$$\hat{x} = \arg\min_x \; \frac{1}{2} \sum_{c=1}^{C} \|\mathcal{F}_\text{NU}(S_c \cdot x) - y_c\|_2^2 + \lambda \|\Psi x\|_1$$

where $\Psi$ is a wavelet transform (e.g., Daubechies wavelets) and $\lambda$ controls the trade-off between data fidelity and wavelet sparsity.

Two reconstruction approaches are relevant:

1. **Gridding reconstruction** (fast baseline): Apply density compensation weights to correct for non-uniform sampling density, then compute the adjoint NUFFT and combine coils. Fast but produces blurry results with residual streaking artifacts.

2. **L1-wavelet compressed sensing** (iterative): Solve the regularized problem using proximal gradient methods, where each iteration involves NUFFT forward/adjoint operations and soft-thresholding in the wavelet domain to enforce sparsity.

## References

- Lustig, M., Donoho, D., & Pauly, J. M. (2007). Sparse MRI: The application of compressed sensing for rapid MR imaging. *Magnetic Resonance in Medicine*, 58(6), 1182-1195.
- Ong, F. & Lustig, M. (2019). SigPy: A Python package for high performance iterative reconstruction. *Proceedings of the ISMRM*.
- Pipe, J. G. & Menon, P. (1999). Sampling density compensation in MRI: Rationale and an iterative numerical solution. *Magnetic Resonance in Medicine*, 41(1), 179-186.
