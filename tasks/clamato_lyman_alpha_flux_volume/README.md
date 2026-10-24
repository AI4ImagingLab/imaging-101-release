# CLAMATO Lyman-alpha Flux Tomography

> Recover a three-dimensional intergalactic transmitted-flux fluctuation field from sparse, noisy absorption spectra whose sightlines leave most of the volume unobserved.
> Domain: astronomy | Keywords: absorption tomography, statistical reconstruction, cosmic structure | Difficulty: Hard

## Background

Neutral hydrogen between galaxies absorbs radiation at the Lyman-alpha transition. Its absorption lines appear at different observed wavelengths according to the absorber's redshift. A spectrum of a distant background galaxy or quasar therefore samples intervening hydrogen along one narrow line of sight. Many such spectra provide sparse probes of a common three-dimensional volume.

Absorption tomography fills the gaps between these sightlines using spatial correlations. It can reveal coherent intergalactic structures in regions where faint galaxies are difficult to survey directly. Its resolution depends on sightline spacing, spectral resolution and measurement noise. Transmitted-flux contrast traces the absorption field; converting it into matter density or temperature requires additional physical assumptions.

## Problem Description

Reconstruct the transmitted-flux fluctuation field in the COSMOS survey volume. Each supplied spectral pixel is a real, reduced and continuum-normalized observation. The observational model is

```math
d_i=s(\mathbf r_i)+\epsilon_i,\qquad
\mathbb E[\epsilon_i]=0,\qquad
\mathbb E[\epsilon_i\epsilon_j]=\sigma_i^2\delta_{ij}.
```

Here $d_i$ is the dimensionless measured flux contrast, $s$ is the unknown continuous contrast field, $\mathbf r_i$ is the pixel's comoving position, $\sigma_i$ is its released noise standard deviation, and $\delta_{ij}$ is the Kronecker delta. The contrast convention is $d=f/(C\overline F)-1$, where $f$ is spectral flux, $C$ is the estimated background continuum and $\overline F$ is the mean transmitted fraction. Negative contrast means stronger absorption. Continuum estimation has already been performed; its systematic uncertainty is not fully included in the supplied independent pixel noise.

There are 64,332 observations from 240 background sources. Recover the field on every cell of the declared Cartesian grid, including cells far from a sightline. The sparse sampling makes the field between observations underdetermined; the target is the conditional Wiener reconstruction defined in Method Hints, rather than independently known cosmic truth.

Save `output/reconstruction.npz` relative to the task workspace, with required key `reconstruction`, shape `(1, 60, 48, 876)`, real `float32` or `float64`, dimensionless flux contrast. Axes are batch, increasing right-ascension comoving coordinate, increasing declination comoving coordinate, and increasing line-of-sight comoving distance. Samples are cell centers at `(index + 0.5) × 0.5` Mpc/h on every spatial axis. Supply the direct map, before any extra visualization smoothing. No alignment, rescaling, cropping or masking is applied. Evaluation measures cosine NCC and dynamic-range-normalized RMS error over the complete field against a validated standard-method reconstruction. Reference agreement measures reproducibility of this conditional inverse, not physical truth.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| positions | `(1, 64332, 3)` | float64 | Pixel positions, coordinate order x/y/z, in comoving Mpc/h |
| noise_std | `(1, 64332)` | float64 | Released dimensionless flux-contrast noise standard deviations |
| flux_contrast | `(1, 64332)` | float64 | Real dimensionless continuum-normalized absorption contrasts |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Grid dimensions | 60 × 48 × 876 | cells | x/right ascension, y/declination, z/line of sight |
| Volume extent | 30 × 24 × 438 | comoving Mpc/h | Grid edge-to-edge dimensions |
| Cell spacing | 0.5 | comoving Mpc/h | All three axes; cell-centered samples |
| Transverse origin | RA 149.9500, Dec 2.1500 | degrees, J2000 | Origin used in released positions |
| Line-of-sight origin | 2.05 | redshift | z coordinate is relative comoving distance from this origin |
| Coordinate conversion epoch | 2.30 | redshift | Fixed conversion throughout the volume |
| Comoving distance at conversion epoch | 3874.867 | Mpc/h | Adopted distance for angular conversions |
| Redshift distance derivative | 871.627 | Mpc/h per redshift | Fixed line-of-sight conversion |
| Hubble parameter | 70 | km/s/Mpc | h = 0.7 |
| Matter / cosmological-constant fractions | 0.31 / 0.69 | dimensionless | Flat cosmology used for coordinates |
| Absorption redshift interval | 2.05–2.55 | redshift | Released foreground volume |
| Rest-frame Lyman-alpha wavelength | 1215.67 | angstrom | Hydrogen absorption transition |
| Spectral resolving power | approximately 1100 | dimensionless | Keck-I LRIS blue-channel acquisition |
| Released noise floor | 0.2 (stored minimum 0.20000000298023224) | dimensionless contrast | Author single-precision floor preserved in float64 storage; do not add a second floor variance |
| Boundary convention | nonperiodic | — | Retain all supplied observations, including sightlines just outside transverse grid edges |

The data are author-released processed measurements, not raw detector frames. Use their supplied positions directly; the fixed cosmological conversion is already applied. Real noise can produce contrast values outside the noiseless physical range, so clipping observations would change the inverse problem.

## Method Hints

A correlated Gaussian prior leads to the Wiener conditional mean

```math
\mathbf m=K_{GD}(K_{DD}+N)^{-1}\mathbf d,
\qquad N=\operatorname{diag}(\sigma_i^2),
\qquad K(\mathbf a,\mathbf b)=0.05\,g(q),
\qquad q=\frac{(a_x-b_x)^2+(a_y-b_y)^2}{2(2.5)^2}
             +\frac{(a_z-b_z)^2}{2(2.0)^2}.
```

The vector $\mathbf m$ contains grid contrasts, $\mathbf d$ contains observed contrasts, $K_{DD}$ is the observation covariance and $K_{GD}$ is the grid-to-observation covariance. Coordinates and correlation lengths in $q$ are in Mpc/h. To define the selected discrete target, $g$ linearly interpolates $\exp(-q)$ at 200 equally spaced samples from 0 to 8 and is zero for $q\geq8$. This finite-support lookup approximates a Gaussian correlation; it is part of the selected method's discretization. The prior mean is zero and its variance is 0.05.

Noise whitening and a diagonal preconditioner make conjugate gradients suitable for the linear solve. The finite correlation support permits sparse or matrix-free products. The standard reconstruction stops at a whitened residual norm relative to its right-hand side of $10^{-3}$. More accurate solutions of the same conditional model are also meaningful. Additional Gaussian smoothing can help visualization, but should be reported separately and excluded from the submitted map.

## References

- Lee et al., *First Data Release of the COSMOS Lyman-Alpha Mapping and Tomography Observations: 3D Lyman-alpha Forest Tomography at 2.05 < z < 2.55*, ApJS (2018), [doi:10.3847/1538-4365/aace58](https://doi.org/10.3847/1538-4365/aace58); [arXiv:1710.02894v3](https://arxiv.org/abs/1710.02894v3).
- CLAMATO DR1 real observations, [doi:10.5281/zenodo.1292459](https://doi.org/10.5281/zenodo.1292459), Creative Commons Attribution 4.0.
