# Quantitative Birefringent-Sample Polarization Imaging

> Reconstruct retardance, slow-axis orientation, relative transmission, and polarization retention from real four-state microscope images despite instrumental polarization and weak anisotropy.
> Domain: physics | Keywords: polarization microscopy, birefringence, quantitative imaging | Difficulty: Hard

## Background

Anisotropic structures delay orthogonal components of an optical field by different amounts. Polarization microscopy uses this birefringence to image molecular alignment without fluorescent labels. Retardance depends on optical path difference, while the slow-axis angle describes the orientation of the anisotropic structure in the image plane.

Quantitative polarization measurements are useful for studying ordered fibers, tissue architecture, and engineered anisotropic materials. Several analyzer states are required to distinguish intensity changes from polarization changes. Instrumental birefringence, incomplete polarization, and weak anisotropy create uncertainty; an independently measured empty-field background helps separate the specimen from the optical system.

## Problem Description

The real observations image a deposited Kazansky birefringent calibration specimen at three stage positions and sixteen defocus planes. This is an experimental optical specimen, not a biological-cell image and not a simulation. Reconstruct its effective in-plane attenuating, depolarizing retarder parameters at every recorded pixel. Defocus planes are independent effective images, not a claim of a tomographic three-dimensional local material tensor.

Let delta be retardance in radians, theta the slow-axis angle in radians, T the intensity relative to the independent background illumination, and P the retained degree of polarization. For circular incident light the effective specimen Stokes vector is

```math
\mathbf s=T\begin{pmatrix}1\\P\sin\delta\sin(2\theta)\\-P\sin\delta\cos(2\theta)\\P\cos\delta\end{pmatrix},
\qquad \mathbf I=\mathbf A\mathbf M_{\rm bg}\mathbf s+d\mathbf 1+\boldsymbol\eta.
```

I contains the four recorded analyzer-state intensities in ADU; A is the supplied instrument matrix, Mbg is the spatially varying instrumental Mueller matrix constrained by the independent no-sample calibration under the same attenuating-retarder model, d is the known camera dark offset, 1 is a vector of ones, and eta denotes measurement noise. The independent background records A times Mbg applied to incident circular Stokes vector (1,0,0,1). Subscript bg denotes the empty-field optical background. Use the weak-retardance branch 0 through pi/2, and theta modulo pi. The recorded microscope plane determines angle convention; no flip or rotation is permitted. T is explicitly a background-relative intensity, because absolute optical power and background exposure are not deposited. P is a noisy effective retention estimate; preserve values above one rather than hiding their model/noise limitation.

Save **output/reconstruction.npz** with all required real float32 arrays. Each shape is (1, 3, 16, 128, 256), with axes batch, ordered stage position, defocus plane, detector row, detector column:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 3, 16, 128, 256) | float32 | Effective retardance delta, radian, weak branch [0, pi/2] |
| orientation | (1, 3, 16, 128, 256) | float32 | Slow-axis theta, radian in [0, pi); 0 along increasing detector column, increasing toward detector row |
| transmittance | (1, 3, 16, 128, 256) | float32 | T, unitless intensity relative to independently observed background, not absolute transmission calibration |
| depolarization | (1, 3, 16, 128, 256) | float32 | P, unitless retained polarization fraction, without clipping noisy values |

All recorded pixels must be reconstructed; do not normalize maps, crop, smooth, align angles, or suppress exceptional pixels. Retardance, relative transmission, and P are each evaluated by relative Euclidean error against an independently reproduced calibrated-method result. Orientation is evaluated by pi-periodic angular RMSE: delta-angle is half atan2(sin(2(theta-estimate minus theta-reference)),cos(2(theta-estimate minus theta-reference))). Its squared error is continuously weighted by sin(reference retardance) squared and normalized by the sum of those weights. This fixed physical convention reduces the influence of nearly isotropic pixels whose axis is unidentifiable; no pixel is removed. There is no registration or fit of an intensity scale.

## Data Description

The first acquired time point at every deposited stage position is retained, with all native defocus planes and the complete detector region. Empty-field images and the instrument-response settings were calibrated independently of the specimen.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| intensities | (1, 3, 4, 16, 128, 256) | uint16 | Raw analyzer intensities in ADU; axes batch, position, analyzer state, defocus, row, column |
| background | (1, 4, 128, 256) | float64 | Independent no-sample analyzer images, ADU; axes batch, analyzer state, row, column |
| instrument_matrix | (1, 4, 4) | float64 | Analyzer response A, dimensionless; rows State0 through State3, columns Stokes s0 through s3 |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Wavelength | 532 | nm | Incident light wavelength |
| Analyzer scheme | 4-State | dimensionless | State0, State1, State2, State3 order |
| Liquid-crystal swing | 0.1 | wave | Calibration retardance swing |
| Dark offset | 100.58 | ADU | Measured camera black level, subtract from specimen and background |
| Extinction ratio | 164.47 | dimensionless | Independently measured polarimeter extinction |
| Exposure | 10.0 | ms | Specimen page metadata; background exposure not deposited |
| Objective | 20x, NA 0.55, dry | dimensionless | Actual microscope page hardware metadata |
| Defocus spacing | 2.0 | micrometer | Native z step, planes 0 through 15 |
| First recorded z | 197.5 | micrometer | Stage-z coordinate in specimen page metadata |
| Detector region | 128 by 256 | pixel | Complete native recorded region |
| Lateral object-plane pitch | Not reliably deposited | unavailable | Page PixelSizeUm is zero; use detector pixels rather than inventing micrometer distances |
| Stage positions | 0, 1, 2 | index | Original deposited position order |
| Included time point | 0 | zero-based index | First complete acquired time point |
| Coordinates | x increases right, y increases down | detector pixel | No orientation rotation or flip |
| Spatial boundary | Per-pixel retarder model | dimensionless | No spatial convolution or wraparound |
| Saturation | Zero samples at 65535 | sample | No detector clipping in the bound acquisition |
| Noise variance | Not calibrated | unavailable | Preserve noisy P estimates, including values above one |

## Method Hints

Use calibrated Stokes polarimetry and an attenuating-depolarizing-retarder model. The four-state instrument inverse separates Stokes components. A measured-background Mueller correction is preferable to treating each channel as an independent scalar flat field, since the background can rotate polarization. The corrected Stokes vector determines retardance from linear-polarization magnitude, slow-axis direction from its angular phase, intensity from s0, and retention from the polarized-vector norm. Distinguish angle periodicity from isotropic-pixel uncertainty.

## References

Guo, S.-M., et al. *Revealing architectural order with quantitative label-free imaging and deep learning*. eLife 9, e55502 (2020). [Publication](https://doi.org/10.7554/eLife.55502).

Chandler, T., et al. *waveOrder: generalist framework for label-agnostic computational microscopy*. [Preprint](https://doi.org/10.48550/arXiv.2412.09775).

*recOrder example and test datasets*, experimental calibration specimen and independent empty-field measurements. [Data deposit](https://doi.org/10.5281/zenodo.8386856), CC BY 4.0.
