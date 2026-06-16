# Exoplanet Direct Imaging

> Reconstruct a faint planetary companion from a temporal sequence of high-contrast coronagraphic images by building a data-driven PSF model and subtracting it while the field of view rotates.

> Domain: astronomy | Keywords: high-contrast imaging, PSF subtraction | Difficulty: Medium

## Background

Direct detection of exoplanets requires suppressing stellar light by a factor of $10^6$–$10^{10}$.  After adaptive optics and a coronagraph reduce the coherent stellar peak, the dominant residual is **quasi-static speckle noise** — wavefront errors that produce PSF artefacts resembling faint point sources.  Distinguishing a real companion from a speckle requires exploiting diversity between frames.

**Angular Differential Imaging (ADI)** is the standard observational strategy: the telescope pupil is kept fixed while the sky (and any companions) rotates as the target transits.  Over a $\sim 1$–$2$ hour observation the companion arc sweeps by tens of degrees while the speckle pattern remains nearly constant.  Post-processing algorithms exploit this differential rotation to separate companion signal from speckle noise.

## Problem Description

### Forward model

Each observed frame is

$$T_k(\mathbf{n}) = I_\psi(\mathbf{n}) + A(\mathcal{R}_{-\theta_k} \mathbf{n})$$

where:
- $T_k$ is the $k$-th detector image (counts),
- $I_\psi$ is the quasi-static PSF speckle pattern (approximately constant across frames),
- $A$ is the astrophysical sky scene (star + companions),
- $\mathcal{R}_{-\theta_k}$ is a 2-D rotation by the negative parallactic angle $\theta_k$.

A companion at sky separation $r$ and position angle $\text{PA}$ appears at detector pixel

$$\mathbf{x}_k = \mathcal{R}_{\theta_k}\,(r\sin\text{PA},\; r\cos\text{PA})^T + \mathbf{c}$$

where $\mathbf{c}$ is the stellar centroid.  The companion traces an arc of
$\Delta\theta = \theta_{\max} - \theta_{\min}$ degrees across the observation.

### Inverse problem

Given $N$ frames $\{T_k\}$ and the corresponding parallactic angles $\{\theta_k\}$, recover the astrophysical signal $A$.  The problem is ill-conditioned because $I_\psi$ is unknown and varies slowly but not perfectly between frames, and the companion signal is $\sim 10^{-4}$ times fainter than the stellar halo.

The **input** is the ADI cube $(N, H, W)$ and the parallactic angle array $(N,)$.  The **output** is a 2-D detection map $(H, W)$ in which companions appear as positive point sources above the residual speckle background.

## Data Description

### data/raw_data.npz

| Key    | Shape         | Dtype   | Description                                      |
|--------|---------------|---------|--------------------------------------------------|
| cube   | (1, 61, 100, 100) | float32 | ADI image cube; 61 frames, 100×100 px, VLT/NACO L'-band (3.8 µm) |
| angles | (1, 61)       | float32 | Parallactic angles in degrees, range −14.3° to +67.1° |
| psf    | (1, 38, 38)   | float32 | Normalized off-axis PSF used for FWHM estimation |

**Target**: Beta Pictoris (A6V, 19.44 pc).  **Companion**: Beta Pic b — a $\sim 13 M_J$ planet at $\approx 0.44$" separation (PA $\approx 211.5°$).

The data were acquired with the VLT/NACO instrument in pupil-tracking mode on 2013-02-01, using an AGPM vector vortex coronagraph.  The total field rotation is 81.4°.  Cube was preprocessed (sky subtraction, flat-fielding, bad-pixel correction, frame centring) using the Vortex Image Processing (VIP) pipeline.  Source: [carlos-gg/VIP_extras](https://github.com/carlos-gg/VIP_extras/tree/master/datasets).

### data/meta_data.json

Instrument and observation parameters only (no algorithm parameters):

| Key                   | Value       | Description                            |
|-----------------------|-------------|----------------------------------------|
| instrument            | VLT/NACO    | Telescope/instrument                   |
| filter                | L'          | Photometric band                       |
| wavelength_um         | 3.8         | Central wavelength in µm               |
| pixel_scale_arcsec    | 0.02719     | Plate scale in arcsec/pixel            |
| fwhm_px               | 4.80        | Measured PSF FWHM in pixels            |
| n_frames              | 61          | Number of ADI frames                   |
| frame_size            | [100, 100]  | Frame height × width in pixels         |
| center_px             | [50, 50]    | Stellar centroid position (x, y)       |
| iwa_px                | 4           | Inner working angle in pixels          |
| total_field_rotation_deg | 81.4     | Total parallactic angle range          |
| known_companion.separation_px | 16.2 | Beta Pic b separation in pixels   |
| known_companion.pa_deg | 211.5      | Beta Pic b position angle in degrees   |

No `ground_truth.npz` is provided for this task; this is real observational data and the companion is the known planet Beta Pic b.  The reference solution is the KLIP detection map stored in `evaluation/reference_outputs/klip_result.npz`.

## Method Hints

**Algorithm family**: Principal Component Analysis / Karhunen-Loève Image Processing (KLIP).

KLIP models the stellar PSF as a low-dimensional subspace spanned by the $K$ dominant eigenimages of the frame covariance matrix.  Projecting each science frame onto this subspace and subtracting the reconstruction removes the quasi-static speckle pattern while preserving signal components orthogonal to the PSF subspace — in particular the companion, whose position moves frame-to-frame due to field rotation.

The key design choices are:
1. **Reference library**: In ADI, use all frames (possibly excluding those where the companion would overlap the target position); full-frame KLIP is the simplest variant.
2. **Truncation level $K$**: Too small → under-fitting (speckle residuals); too large → over-fitting (companion self-subtraction). Typical optimal range: $K \sim 5$–$20$ for $N \approx 60$ frames.
3. **Combination statistic**: Mean is standard; median is more robust to outlier frames.

After subtraction, derotate residuals by $-\theta_k$ and average to build the detection map.  SNR is measured via the Mawet et al. (2014) aperture t-test.

## References

1. Soummer, R., Pueyo, L. & Larkin, J. (2012). *Detection and Characterization of Exoplanets and Disks Using Projections on Karhunen-Loève Eigenimages*. ApJL 755, L28. arXiv:1207.4197
2. Ko, C.-L., Douglas, E. S. & Hom, J. (2024). *A PyTorch benchmark for high-contrast imaging post processing (torchKLIP)*. SPIE 13138, 1313811. arXiv:2409.16466
3. Mawet, D. et al. (2014). *Fundamental Limitations of High Contrast Imaging Set by Small Sample Statistics*. ApJ 792, 97.
4. Gomez Gonzalez, C. A. et al. (2017). *VIP: Vortex Image Processing package for high-contrast direct imaging*. AJ 154, 7.
