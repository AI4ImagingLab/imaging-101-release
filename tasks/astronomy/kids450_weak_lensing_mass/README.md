# KiDS-450 Weak-Lensing Mass Mapping

> Reconstruct a projected cosmic mass map from noisy galaxy ellipticities whose weak shear signal is blurred by intrinsic shape noise and irregular sampling.
> Domain: astronomy | Keywords: weak gravitational lensing, mass mapping, inverse problems | Difficulty: Medium

## Background

Gravity bends light from distant galaxies. Across a small patch of sky the tidal component of this deflection stretches the background galaxy population coherently, while the isotropic part traces projected mass. This provides a way to map matter, including dark matter, through its gravitational influence rather than its emitted light.

Individual galaxy shapes carry much larger intrinsic variations than the gravitational signal. Combining many galaxies and independent instrumental shape calibrations makes weak lensing useful for mapping large-scale structure, studying galaxy clusters and testing the growth of cosmic structure. Irregular sampling and the inability to observe an overall uniform mass sheet limit the reconstruction.

## Problem Description

The observations are reliable KiDS-450 DR3.1 galaxy shapes in a one-degree patch. A calibrated ensemble shear in a spatial cell is $\gamma_j=\sum_i w_i e_{ij}/\sum_i w_i(1+m_i)$, where $i$ indexes galaxies, $j=1,2$ identifies ellipticity component, $e$ is the lensfit shape, $w$ its inverse-variance weight and $m$ the independent multiplicative shear calibration. Empty cells provide no measured shear and are represented by zero. Shape measurements already account for the point-spread function through lensfit; the independent PSF ellipticities remain available for systematic diagnostics. No additional division by two is applied to these lensfit ellipticities.

In the flat-sky weak-shear approximation the dimensionless E-mode convergence $\kappa_E$ and B-mode diagnostic $\kappa_B$ satisfy


```math
\widehat\gamma_1=P_1\widehat\kappa_E-P_2\widehat\kappa_B,\qquad
\widehat\gamma_2=P_2\widehat\kappa_E+P_1\widehat\kappa_B,
\qquad P_1={k_x^2-k_y^2\over k_x^2+k_y^2},\quad P_2={2k_xk_y\over k_x^2+k_y^2}.
```


A hat is the two-dimensional Fourier transform; $k_x,k_y$ are spatial frequencies along the declared east and north pixel axes. Both kernels are zero at zero frequency. The unobservable constant mass sheet is fixed by requiring zero-mean convergence. Finite coverage, intrinsic galaxy shapes and missing cells make fine structure uncertain; the ellipticity-to-shear approximation does not include a nonlinear reduced-shear correction.

Save `output/reconstruction.npz`, relative to the task workspace, containing `reconstruction` (E convergence) and `bmode` (B convergence). Both represent the requested Gaussian output resolution with standard deviation2.0 pixels, using periodic convolution; this fixes the spatial scale of the target map. Both must be real `float64` arrays with batch-first shape `(1, 64, 64)`, axis order `(batch, north/declination row, east/right-ascension column)`, dimensionless physical scale $\Sigma/\Sigma_{\mathrm{crit}}$ with an undetermined constant fixed to zero. Here $\Sigma$ is effective projected mass density and $\Sigma_{\mathrm{crit}}$ is the geometry-dependent critical density of the source ensemble; absolute mass-density conversion is not requested. B is dimensionless and represents the parity-odd shear component, rather than an additional positive mass density. No absolute surface-density units or positivity constraint are imposed. Evaluation uses cosine normalized cross-correlation (NCC, without subtracting means) and dynamic-range normalized RMS error (NRMSE). For flattened real estimate $a$ and comparison image $b$, NCC is $a^Tb/(\|a\|_2\|b\|_2)$ and NRMSE is $\sqrt{\mathrm{mean}[(a-b)^2]}/[\max(b)-\min(b)]$. Every required array must pass both criteria. Forward shear from the two output maps is also compared with the ensemble-calibrated shear at the declared Gaussian resolution after removing its mean: the metric is sqrt(sum squared shear residual / sum squared observed shear) over both components and all cells. Evaluation compares the exact declared grid and physical scale, with no crop, fitted rescaling, translation or other alignment. The comparison establishes agreement with a validated standard reconstruction; these real observations have no independently known target truth.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| positions | (1, 16625, 2) | float64 | Right ascension and declination in degrees, J2000. |
| ellipticity | (1, 16625, 2) | float64 | Dimensionless lensfit e1/e2 shape measurements, in supplied sky basis. |
| weight | (1, 16625) | float64 | Dimensionless inverse-variance lensfit weights. |
| multiplicative_bias | (1, 16625) | float64 | Dimensionless independently calibrated ensemble shear bias m. |
| psf_ellipticity | (1, 16625, 2) | float64 | Dimensionless independent PSF e1/e2 model diagnostics at galaxy positions. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Survey | KiDS-450 / ESO DR3.1 | dimensionless | Real VST/OmegaCAM r-band lensfit catalogue. |
| RA centre | 129.0 | degree | Tangent-plane origin. |
| Declination centre | -0.5 | degree | Tangent-plane origin. |
| RA range | [128.5, 129.5) | degree | Fixed outer observation boundaries. |
| Declination range | [-1, 0) | degree | Fixed outer observation boundaries. |
| Target grid | 64 by 64 | pixel | North rows and east columns, increasing coordinates. |
| Coordinate conversion | x=(RA-129.0)cos(-0.5 degrees), y=Dec+0.5 | degree | Constant-declination small-field projection. |
| Grid extent | [-0.5cos(-0.5 degrees),0.5cos(-0.5 degrees)] by [-0.5,0.5] | degree | Gridding uses projected boundaries; FFT uses equal pixel spacings (flat-sky aspect approximation of 0.0038%). |
| Sampling | weighted cell mean | dimensionless | Upper boundary excluded by catalogue selection; empty cell has shear zero. |
| Source selection | 0.1<=zbest<=0.9; Mask=0; FitClass=0; weight>0 | dimensionless | Reliable source shapes and photometric-redshift range, already applied. |
| Shear calibration | ensemble division by sum w(1+m) | dimensionless | Never divide each individual galaxy by its own 1+m. |
| Uncertainty | intrinsic shapes and measurement noise | dimensionless | No homogeneous Gaussian pixel-noise variance is asserted. |
| Fourier convention | exp(-2 pi i k dot x); inverse 1/(64*64) | dimensionless | NumPy DFT, periodic boundary, zero DC, real-part projection. |

## Method Hints

Fourier weak-lensing inversion separates the even and odd parity components through the spin-2 kernels. Regularizing the resulting map with spatial smoothing can suppress the intrinsic-shape contribution while retaining larger mass structures. Weighted ensemble calibration and a B-mode/PSF diagnostic are essential for interpreting a finite noisy field. The zero-frequency mass-sheet ambiguity cannot be recovered from shear alone. The declared finite Gaussian resolution uses separable normalized weights g(j)=exp(-j*j/8)/sum(exp(-k*k/8), k=-8,...,8), for integer offsets j=-8,...,8, with periodic wrapping. This finite sampled kernel, rather than a continuous Fourier Gaussian or an unspecified infinite kernel, fixes the output resolution.

## References

- Kaiser and Squires (1993), *Mapping the dark matter with weak gravitational lensing*, ApJ 404, 441–450. [DOI](https://doi.org/10.1086/172297).
- Hildebrandt et al. (2017), *KiDS-450: cosmological parameter constraints from tomographic weak gravitational lensing*. [DOI](https://doi.org/10.1093/mnras/stw2805).
- KiDS shear observations, [VizieR II/384](https://cdsarc.cds.unistra.fr/viz-bin/cat/II/384), CC BY 4.0. Based on data obtained from the ESO Science Archive Facility with [data DOI](https://doi.org/10.18727/archive/37).
