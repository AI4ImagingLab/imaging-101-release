# Shack-Hartmann Wavefront Sensing

> Reconstruct the full-pupil wavefront phase from raw Shack-Hartmann detector images
> at four aberration levels (WFE = 50–400 nm RMS), using weighted-centroid slope
> estimation followed by Tikhonov reconstruction, and evaluate accuracy with NCC and
> NRMSE against ground-truth phase maps.

> Domain: Astronomy | Keywords: adaptive optics, wavefront sensing, inverse problems | Difficulty: Medium

## Background

A **Shack-Hartmann wavefront sensor (SH-WFS)** measures the phase distortion of
an incoming optical wavefront by dividing the telescope pupil into an array of
small lenslets. Each lenslet focuses its sub-aperture onto a detector; the centroid
displacement of the focused spot is proportional to the **local wavefront gradient**.
The full reconstruction pipeline has two stages:

1. **Centroid estimation** — for each subaperture, compute the intensity-weighted
   centroid of the spot image and subtract the reference (flat-wavefront) centroid
   to obtain a slope measurement (local gradient).
2. **Wavefront reconstruction** — invert the linear slope-to-mode map using the
   calibrated response matrix to recover the full-pupil phase map.

The reconstruction is challenging because:
- The SH-WFS measures **gradients**, not the phase itself; centroids are extracted
  from noisy detector images before inversion.
- The response matrix is ill-conditioned: high-order modes couple weakly to
  measurable slopes, requiring regularisation.
- At large WFE (≥ 200 nm), spot displacements can approach the subaperture size,
  violating the linear slope model (nonlinear regime).
- Poisson noise from finite photon counts limits signal-to-noise ratio at small
  wavefront amplitudes.

## Problem Description

### Detector image formation

Each detector pixel at focal-plane position $(x, y)$ belongs to one subaperture.
The raw detector image is a mosaic of $N_{\text{subaps}}$ focused spots on a
$128 \times 128$ pixel grid (the same spatial dimensions as the pupil).

### Forward model

The deformable mirror (DM) shapes the wavefront as a linear combination of
$N_\text{modes}$ disk-harmonic basis functions $\{\psi_j\}$:

$$\varphi(x,y) = \frac{4\pi}{\lambda}\sum_j a_j\,\psi_j(x,y)$$

For subaperture $i$, the weighted centroid of the spot image gives the local
wavefront slope:

$$\text{slope}_{x,i} = \frac{\sum_k x_k I_k}{\sum_k I_k} - \frac{\sum_k x_k I^{\mathrm{ref}}_k}{\sum_k I^{\mathrm{ref}}_k}$$

where the sum is over pixels $k$ in subaperture $i$, $I_k$ is the measured intensity,
$I_k$<sup>ref</sup> is the flat-wavefront reference intensity, and ($x_k$, $y_k$) are the
focal-plane coordinates. Collecting all $N$<sub>subaps</sub> subapertures gives:

$$\mathbf{s} = R\mathbf{a} + \mathbf{n}$$

where $R \in \mathbb{R}^{N_\text{slopes} \times N_\text{modes}}$ is the calibrated
response matrix and $\mathbf{n}$ is noise.

### Inverse problem

Given raw SH-WFS images and a reference image (flat wavefront), reconstruct the
wavefront phase map $\varphi(x,y)$ by:

1. Extracting slope differences via weighted-centroid estimation.
2. Inverting $\mathbf{s} = R\mathbf{a}$ with Tikhonov regularisation.

**Input**: `wfs_images` (N_levels × N_det), `ref_image`, `detector_coords_x/y`,
`subap_map`, `response_matrix`, `dm_modes`, `aperture`.

**Output**: reconstructed phase maps (N_levels × N_pupil_px) compared against
`wavefront_phases` in `ground_truth.npz`.

## Data Description

### data/raw_data.npz

| Key               | Shape                      | Dtype   | Description |
|-------------------|----------------------------|---------|-------------|
| response_matrix   | (1, N_slopes, 150)         | float32 | Push-pull calibrated DM→slope response matrix [slope/m] |
| wfs_images        | (1, 4, 128, 128)           | float32 | Raw SH-WFS detector images at each WFE level [photons] |
| ref_image         | (1, 128, 128)              | float32 | Reference (flat wavefront) SH-WFS image [photons] |
| detector_coords_x | (1, 128, 128)              | float32 | Focal-plane x coordinate per pixel [m] |
| detector_coords_y | (1, 128, 128)              | float32 | Focal-plane y coordinate per pixel [m] |
| subap_map         | (1, 128, 128)              | int32   | Subaperture rank per pixel (−1 = invalid, 0..N_valid−1) |
| dm_modes          | (1, 150, N_pupil_px)       | float32 | DM mode shapes on the 128×128 pupil grid (disk harmonics, ptp-normalised) |
| aperture          | (1, N_pupil_px)            | float32 | Pupil transmission mask (128×128 px) |

N_slopes = 592 (296 valid subapertures × 2).
N_pupil_px = 16384 (128×128 pupil grid).
The 4 WFE levels (from `meta_data.json`) are 50, 100, 200, 400 nm RMS.

### data/ground_truth.npz

| Key              | Shape             | Dtype   | Description |
|------------------|-------------------|---------|-------------|
| wavefront_phases | (1, 4, N_pupil_px) | float32 | True wavefront phase at each WFE level [rad at λ_WFS = 700 nm] |

The ground-truth phase is a random DM mode combination scaled to the target RMS.
Because the ground truth lies in the DM mode subspace, perfect reconstruction is
in principle achievable in the noise-free limit.

### data/baseline_reference.npz

Reference PSF images produced by the HCIPy simulation, provided for qualitative
comparison of AO-corrected optical quality. Not used for NCC/NRMSE evaluation
(superseded by `ground_truth.npz`). All images are on a 240×240 focal-plane grid
(57 600 pixels), stored in row-major order.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `long_exposure_psf` | `(1, 57600)` | float32 | Long-exposure PSF with AO correction, averaged over 150-mode DM correction; peak-normalised [dimensionless] |
| `long_exposure_no_ao` | `(1, 57600)` | float32 | Long-exposure PSF without AO correction (turbulence only); peak-normalised [dimensionless] |
| `unaberrated_psf` | `(1, 57600)` | float32 | Diffraction-limited PSF (no aberrations, no turbulence); peak-normalised [dimensionless] |
| `focal_grid_coords` | `(1, 57600, 2)` | float32 | Focal-plane (x, y) coordinates for each PSF pixel [λ/D] |

### data/meta_data.json

Key parameters:

| Field                               | Value      | Description |
|-------------------------------------|------------|-------------|
| telescope.diameter_m                | 8.0        | Primary mirror diameter [m] |
| wavefront_sensor.n_lenslets         | 20         | Lenslets across diameter |
| wavefront_sensor.n_valid_subaps     | 296        | Valid (illuminated) subapertures |
| wavefront_sensor.n_det_pixels       | 16384      | Detector pixels (128×128) |
| wavefront_sensor.det_image_shape    | [128, 128] | Detector image shape [H, W] |
| wavefront_sensor.wavelength_wfs_m   | 7e-7       | Sensing wavelength [m] |
| deformable_mirror.n_modes           | 150        | Number of DM correction modes |
| simulation.photons_per_frame        | 390000000  | WFS photon count per frame (mag 5 star) |
| wfe_levels_nm                       | [50,100,200,400] | Target RMS WFE at each level [nm] |

## Method Hints

The standard approach combines **weighted-centroid slope estimation** with
**Tikhonov (truncated-SVD) wavefront reconstruction**:

**Stage 1 — Centroid estimation** (per subaperture $j$):

$$\text{slope}_{x,j} = \frac{\sum_{k \in j} x_k\,I_k}{\sum_{k \in j} I_k} - \frac{\sum_{k \in j} x_k\,I^{\mathrm{ref}}_k}{\sum_{k \in j} I^{\mathrm{ref}}_k}$$

The `subap_map` array identifies which pixels belong to each subaperture.
Output ordering: all x-slopes first ($j = 0, \ldots, N-1$), then all y-slopes.

**Stage 2 — Tikhonov reconstruction**:

1. Decompose $R = U\Sigma V^\top$.
2. Invert: $M = V\Sigma^+ U^\top$, zeroing singular values below `rcond × σ_max`.
3. Reconstruct mode amplitudes: $\hat{\mathbf{a}} = M\mathbf{s}$.
4. Convert to phase: $\varphi = (4\pi/\lambda)\Psi^\top\hat{\mathbf{a}}$.

The choice of `rcond` controls the noise–accuracy trade-off.  A value around 1e-3
is typical. Modes with singular values below the threshold are noise-dominated and
should be zeroed.

**Timing constraint**: The reconstruction pipeline (centroid extraction + wavefront
reconstruction for all 4 WFE levels) must complete in under 100 ms.

Alternative methods include zonal reconstruction (direct wavefront integration),
Bayesian estimation (MAP with Kolmogorov prior), and neural-network reconstructors.

## References

1. Por, E. H., Haffert, S. Y., et al. (2018). *HCIPy: an open-source adaptive
   optics simulation framework.* SPIE 10703.
2. Hardy, J. W. (1998). *Adaptive Optics for Astronomical Telescopes.* OUP.
3. Roddier, F. (1999). *Adaptive Optics in Astronomy.* Cambridge UP.
