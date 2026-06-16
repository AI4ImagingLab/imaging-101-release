# Spectral Snapshot Compressive Imaging (PnP-CASSI)

> Reconstruct a full 3D hyperspectral cube (256 × 256 × 31 spectral bands) from a single 2D compressed measurement captured by a coded aperture snapshot spectral imager.

> Domain: Physics | Keywords: compressed sensing, hyperspectral imaging | Difficulty: Medium

## Background

**Coded Aperture Snapshot Spectral Imaging (CASSI)** captures an entire 3D hyperspectral data cube in a single detector exposure. A binary coded aperture mask spatially modulates the scene, and a dispersive prism shifts each spectral channel by a wavelength-dependent displacement before all channels are integrated onto a 2D detector array. The system trades spectral resolution for the ability to capture all bands simultaneously — enabling single-shot acquisition of dynamic scenes.

The inverse problem is to recover the 256 × 256 × 31 spectral cube (400–700 nm at 10 nm intervals) from this single 2D measurement, which is highly underdetermined: 31 spectral images must be inferred from a single compressed snapshot.

## Problem Description

The SD-CASSI (Single Disperser) forward model integrates all shifted, masked spectral channels onto the detector:

$$y(m, n) = \sum_{k=1}^{N_c} \Phi(m, n-d_k)\, x(m, n-d_k, k)$$

where:
- $x(m, n, k)$ is the spectral scene intensity at pixel $(m, n)$ and wavelength channel $k$
- $\Phi(m, n)$ is the binary coded aperture mask (0 or 1 at each pixel)
- $d_k = (k-1) \cdot \Delta d$ is the dispersion shift for channel $k$ (integer pixels, proportional to wavelength offset)
- $y(m, n)$ is the measured intensity at detector pixel $(m, n)$

In matrix form: $\mathbf{y} = \mathbf{\Phi}\mathbf{x}$, where $\mathbf{\Phi} \in \mathbb{R}^{M \times (M \cdot N_c)}$ is the block-diagonal sensing matrix formed by stacking shifted, masked copies of the aperture. The measurement $\mathbf{y}$ has size $H \times (W + \Delta d \cdot (N_c-1))$; the full cube $\mathbf{x}$ has size $H \times W \times N_c$. Recovery is severely underdetermined — the ratio of unknowns to measurements is approximately $N_c = 31$.

## Data Description

### `data/raw_data.npz`

Single-disperser CASSI measurement using a KAIST color-checker crop as the scene, with a binary random coded aperture mask.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `measurement` | (1, 256, 286) | float32 | Compressed 2D CASSI detector measurement (arbitrary units) |
| `mask` | (1, 256, 256) | float32 | Binary coded aperture mask (0 or 1 at each pixel) |

### `data/ground_truth.npz`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `hyperspectral_cube` | (1, 256, 256, 31) | float32 | Ground truth spectral data cube (normalised intensity, range ≈ [0, 1]) |

### `data/meta_data.json`

| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` | 256 | Image height (pixels) |
| `c` | 256 | Image width (pixels) |
| `nC` | 31 | Number of spectral channels |
| `step` | 1 | Dispersion shift per channel (pixels) |
| `wavelength_start_nm` | 400 | Shortest wavelength (nm) |
| `wavelength_end_nm` | 700 | Longest wavelength (nm) |
| `wavelength_step_nm` | 10 | Spectral channel spacing (nm) |

## Method Hints

**Algorithm family:** Plug-and-Play (PnP) priors within a GAP (Generalized Alternating Projection) optimization framework.

**Key idea:** GAP enforces data consistency by projecting onto the affine constraint $\{\mathbf{x} : \mathbf{\Phi}\mathbf{x} = \mathbf{y}\}$ in closed form, while a learned CNN denoiser (HSI-SDeCNN, trained on hyperspectral patches) implicitly enforces the spectral image prior at each iteration. Replacing the proximal step in a classical ADMM with a denoiser decouples the optimization structure from the prior specification, allowing powerful learned priors without retraining for each new sensing matrix. A hybrid strategy starts with TV denoising (stable at high noise) and switches to the CNN denoiser (higher quality) partway through iteration.

**Why it suits the problem:** The sensing matrix $\mathbf{\Phi}$ has a highly structured block-diagonal form, making the data-consistency projection analytic and cheap. The main difficulty is the prior: hyperspectral images have strong cross-band correlations not captured by TV. Operating the denoiser on sliding windows of 7 adjacent bands exploits local spectral correlations without requiring a full 31-band network.

## References

- Zheng, Liu, Meng, Qiao, Tong, Yang, Han, Yuan. "Deep plug-and-play priors for spectral snapshot compressive imaging." *Photon. Res.* 9(2), B18–B29 (2021). https://doi.org/10.1364/PRJ.411745
- Yuan. "Generalized Alternating Projection Based Total Variation Minimization for Compressive Sensing." *ICIP* 2016.
