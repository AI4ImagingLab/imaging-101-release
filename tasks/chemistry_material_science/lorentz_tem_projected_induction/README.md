# Lorentz TEM Projected Induction

> Reconstruct magnetic and electrostatic electron phase images and projected magnetic induction from two experimentally acquired, sample-flipped Lorentz TEM focal series with mixed contrast and uncertain boundary flux.
> Domain: chemistry & material science | Keywords: electron imaging, magnetic imaging, phase retrieval | Difficulty: Hard

## Background

Electrons crossing a thin specimen accumulate phase from its electrostatic potential and magnetic vector potential. Defocusing the image converts transverse phase variations into intensity contrast. Lorentz transmission electron microscopy can therefore reveal magnetic domains and patterned magnetic structures at spatial scales that are difficult to access with optical microscopy.

Electrostatic thickness contrast overlaps the magnetic signal. Turning the specimen over reverses its magnetic phase while preserving its electrostatic contribution, allowing the two effects to be separated after the images are aligned. The phase gradient measures magnetic induction integrated along the electron path, including stray fields. It does not determine local magnetization or a depth-resolved magnetic field.

## Problem Description

The observations are two aligned focal series of an artificial spin-ice island array, acquired before and after specimen inversion. At the exit surface the wave is

```math
\psi_\pm(x,y)=\sqrt{I_0(x,y)}\exp\bigl(i[\phi_E(x,y)\pm\phi_B(x,y)]\bigr).
```

Here x is the image column coordinate, y is the row coordinate, I0 is in-focus intensity, and the real phases φE and φB are respectively electrostatic and magnetic contributions in radians. The plus sign denotes the unflipped series. In the lens-defocus convention used by these observations, the coherent transfer at spatial frequency k is exp(iπλdf|k|²), where λ is electron wavelength and df is signed lens defocus. Its near-focus transport relation is

```math
\frac{\partial I_\pm}{\partial d_f}
=\frac{\lambda}{2\pi}\nabla\cdot
\left[I_0\nabla(\phi_E\pm\phi_B)\right].
```

Spatial derivatives here use physical image coordinates. Positive lens defocus is opposite positive free-space propagation distance. Illumination gain can differ between focal planes, and finite focus separation, noise, specimen edges and unobserved exterior fields limit the transport approximation. Both phases have an unobservable additive constant. Use the fixed convention that each phase has minimum zero over the complete supplied ROI, with no subsequent fit or alignment.

Save **output/reconstruction.npz** relative to the task workspace, containing every array below. Arrays must be real float64, finite and batch first, with axes (batch, row y, column x).

| Key | Shape | Dtype | Description |
|---|---|---|---|
| phase_B | (1, 512, 512) | float64 | Magnetic electron phase, radians, minimum-zero gauge |
| phase_E | (1, 512, 512) | float64 | Electrostatic electron phase, radians, minimum-zero gauge |
| Bx | (1, 512, 512) | float64 | Column component of path-integrated magnetic induction, T·nm |
| By | (1, 512, 512) | float64 | Row component of path-integrated magnetic induction, T·nm |

The induction convention is

```math
\mathcal B_x=\frac{\hbar}{e}\partial_y\phi_B,
\qquad \mathcal B_y=-\frac{\hbar}{e}\partial_x\phi_B,
\qquad \mathcal B=\int B\,dz.
```

Here e is positive elementary charge and ℏ is reduced Planck constant. Convert the physical derivative consistently to T·nm. No independent specimen thickness is supplied; do not report local B in tesla. The declared induction discretization uses centered image derivatives and second-order one-sided derivatives at the outermost pixels.

All four arrays are evaluated with cosine NCC (without mean subtraction) and RMS error divided by reference dynamic range, without cropping, rescaling or alignment. Independent checks evaluate consistency of induction with magnetic phase and the spectral transport equation against the observed nearest-focus differences. Evaluation measures agreement with a validated standard-method estimate and observation consistency; no independent phase or induction truth is available.

## Data Description

These are existing real experimental LTEM intensity observations. The supplied crop preserves native pixels, with no spatial resizing, and both specimen orientations are already aligned into the same image coordinates.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| unflipped | (1, 7, 512, 512) | float32 | Aligned unflipped focal series, arbitrary detector intensity units |
| flipped | (1, 7, 512, 512) | float32 | Aligned inverted-specimen focal series, arbitrary detector intensity units |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Pixel pitch | 5.1066 | nm/pixel | Same pitch along rows and columns |
| Accelerating voltage | 200000 | V | Relativistic electron wavelength calibration |
| Defocus planes, in array order | -928972, -398131, -199065, 0, 199065, 398131, 928972 | nm | Signed lens defocus, common to both series |
| Native source image size | 2048 × 2048 | pixel | Original aligned acquisition field |
| Crop | rows 768:1280, columns 768:1280 | pixel | Zero-based, stop-exclusive native central ROI |
| Coordinate origin | upper-left pixel center | pixel | Columns increase rightward, rows downward |
| Physical field width | 2614.5792 | nm | 512 pixel pitches |
| Specimen inversion | magnetic phase changes sign | dimensionless | Flipped images are already spatially registered |
| Boundary convention | periodic finite ROI | dimensionless | Unknown exterior is discarded; boundary choice limits quantitative accuracy |
| Aperture bandwidth | circular radius N/3 for N=512 | cycles per field | Fixed reciprocal-space support for transport reconstruction |
| Noise | measured, no calibrated variance supplied | arbitrary intensity units | No noiseless observation or independently known phase is supplied |

## Method Hints

Transport-of-intensity inversion with two inverse Laplacians can separate magnetic and electrostatic phase by taking the difference and sum of the two focus derivatives. Equalizing integrated image intensity handles illumination gain variation. The zero spatial frequency cannot determine the phase constant. Fourier regularization and a physically stated boundary convention help control amplification of low-frequency uncertainty. Different focus separations and mirrored boundary extensions provide useful sensitivity checks rather than independent truth.

## References

1. A. R. C. McCray et al., “Understanding complex magnetic spin textures with simulation-assisted Lorentz transmission electron microscopy,” Physical Review Applied 15, 044025 (2021). https://doi.org/10.1103/PhysRevApplied.15.044025
2. A. McCray and C. Phatak, “Example data for PyLorentz LTEM code,” version 0.1.0 (2024), CC BY 4.0. https://doi.org/10.5281/zenodo.13147848
