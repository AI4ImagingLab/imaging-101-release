# HL60 Single-Shot Off-Axis Wavefield Reconstruction

> Reconstruct the relative complex optical wavefield of a real HL60 cell from a single spatial-carrier hologram despite overlapping diffraction orders and reference aberrations.
> Domain: biology | Keywords: quantitative phase microscopy, optical interferometry, wavefield reconstruction | Difficulty: Hard

## Background

Transparent biological cells alter optical phase through their integrated refractive-index contrast, while often producing little ordinary absorption contrast. Interference with a reference wave converts both optical quadratures into intensity variations that a camera can record. The complex wavefield reveals cellular optical structure without a fluorescent label.

Off-axis microscopy tilts the reference wave so that object information occupies displaced Fourier sidebands in a single exposure. This enables imaging of rapidly changing suspended cells. Finite aperture, imperfect separation of diffraction orders and reference-wave aberration limit the resolution and accuracy of quantitative phase and amplitude.

## Problem Description

The measurement is a complete200×210 pixel hologram of an HL60 myeloid leukemia cell and a separately recorded empty-field hologram with the same optical configuration. For camera coordinatesr=(y,x), the model is

```math
I(r)=|O(r)+R(r)|^2+\eta(r),\qquad
R(r)=A(r)e^{-2\pi i f\cdot r},\qquad
I_b(r)=|O_b(r)+R_b(r)|^2+\eta_b(r).
```

I andI_b are measured object and empty intensities inADU; O and O_b are transmitted optical waves, R andR_breference waves, A is reference amplitude, f spatial carrier incycles/pixel, i the imaginary unit, eta and eta_bnoise/model errors. O,R have amplitude units sqrt(ADU) under this digital intensity convention. Subscript b denotes an independent empty observation. The two displaced cross-term sidebands encode complex quadratures; separating one sideband is imperfect because the central and sideband spectral supports partly overlap. Recover the full bandlimited relative complex transmission, using the same sideband/aperture for object and empty fields. Absolute cell thickness and refractive index are not determined by this one projection.

Save `output/reconstruction.npz` with:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 200, 210) | complex128 | Dimensionless bandlimited complex transmission relative to independent empty field; axes batch, detector row, detector column |

Keep the native coordinate orientation, y downward andxrightward. Retain complex quadratures: amplitude isabs(reconstruction), phase isangle(reconstruction)in rad modulo2pi. The empty-field ratio fixes both phase origin and amplitude scale; no additional plane, piston, intensity normalization, unwrapping or global phase fit is permitted in the submission. Select the upper spectral sideband, use the same integer carrier translation for both observations, and retain the finite-aperture convention of a smoothed circular aperture with radius half the selected carrier-to-origin distance, Gaussian smoothing sigma=radius/5. This fixes the output's finite spatial bandwidth. Mean subtraction and top-left zero extension to512×512define the finite detector boundary convention. Relative complex Euclidean error is evaluated on the full array, without rescaling/alignment or magnitude conversion.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| hologram | (1, 200, 210) | uint8 | Original deposited cell interference intensity inADU |
| empty_hologram | (1, 200, 210) | uint8 | Independent empty-field interference intensity inADU |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Specimen | HL60 myeloid leukemia cell | identifier | Actual measured single-cell specimen |
| Exposure count | one cell and one empty | exposure | Spatial rather than temporal phase multiplexing |
| Native region | 200 rows ×210 columns | pixel | Complete deposited region, no ROI selection |
| Detector encoding | unsigned 8-bit | ADU | Original deposited digital precision |
| Coordinates | x increases right, y increases down | pixel | Camera-plane orientation retained |
| Spatial carrier | determined from observed upper sideband | cycles/pixel | Same carrier used for independent reference |
| Reference observation | separate empty field | ADU | Independent optical baseline, not a clean cell answer |
| Boundaries | finite observation, top-left zero extension | dimensionless | Native pixels are not reflected or spatially wrapped |
| Physical pitch/wavelength | not deposited with this example | unavailable | Report native detector pixels and relative optical wavefield |
| Noise variance | not independently calibrated | unavailable | No invented variance or absolute power calibration |

## Method Hints

Use spatial-carrier Fourier demodulation with a finite sideband aperture. Apply the identical selected aperture and integer Fourier translation to the independently measured empty hologram, then form the complex object/empty ratio. Compare aperture widths to expose order overlap and resolution tradeoffs. A review-only unwrapped phase can aid interpretation, but the scientific output is the complete complex transmission with the fixed reference convention.

## References

Schürmann et al., *Refractive index measurements of single,spherical cells using digital holographic microscopy*, Methods in Cell Biology125,143–159(2015). [Paper](https://doi.org/10.1016/bs.mcb.2014.10.016). The present task recovers the spatial wavefield, without assuming spherical-cell geometry for scalar refractive-index fitting.

qpretrieve author implementation and measuredHL60example, copyright2022Refractive Index Imaging, MIT license.
