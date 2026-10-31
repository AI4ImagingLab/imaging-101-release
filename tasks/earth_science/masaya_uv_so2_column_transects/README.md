# Masaya UV SO2 Column Transects

> Recover volcanic sulfur-dioxide column profiles along two measured car traverses from ultraviolet spectra despite overlapping atmospheric absorption and uncertain spectrometer response.
> Domain: earth science | Keywords: atmospheric spectroscopy, volcanic remote sensing, nonlinear inversion | Difficulty: Hard

## Background

Volcanic sulfur dioxide absorbs ultraviolet light in narrow spectral bands. Measuring scattered sunlight through a plume gives a noncontact estimate of the amount of gas along each viewing ray. When a zenith-pointing spectrometer moves underneath a plume, successive measurements form a profile of integrated gas concentration across it.

Such profiles inform monitoring of magma degassing and volcanic hazards. The absorption signature overlaps ozone and atmospheric rotational Raman scattering, while finite spectral resolution blends sharp solar and molecular features. Accurate retrieval therefore requires radiative transfer and the spectrometer response to be considered together.

## Problem Description

Recover the SO2 slant-column profile for all 360 measured spectra from two car traverses at Masaya volcano on 14 January 2018. The observations are the complete published traverse sequence. Each spectrum records a different position/time; no spatial interpolation or smoothing between acquisitions is assumed.

```math
I_j(\lambda)=\mathcal{S}_{\delta_j}\left[G_j*\left(F_\odot(\lambda)B_j(\lambda)
\exp[-a_j\sigma_{\mathrm{SO2}}(\lambda)-b_j\sigma_{\mathrm{O3}}(\lambda)-r_j\sigma_{\mathrm{Ring}}(\lambda)]\right)\right]+d(\lambda)+q_j.
```

Here j is acquisition index, lambda is wavelength in nm, I is measured intensity in instrument counts, F is the supplied high-resolution solar spectrum, B is a smooth broadband response, a is the unknown SO2 column in molecules/cm², b is an ozone column in the same units, and r is a dimensionless Ring-effect amplitude. The supplied sigma arrays are molecular absorption cross sections (cm²/molecule) or a dimensionless Ring pseudo-cross-section. G is the normalized instrument line shape, * denotes convolution, and S denotes sampling after wavelength shift/stretch delta. The measured dark spectrum d and scalar stray signal q are additive. Atmospheric ozone suppresses solar flux in the supplied short-wavelength stray-light band.

Unknown broadband response, ozone, Ring amplitude, wavelength drift and line shape are nuisance variables. The requested physical unknown is the SO2 slant column. The coupled spectral model is nonlinear and ill-conditioned, particularly at small plume columns. The published no-flat-field variant is used because no measured pixel-gain calibration accompanies the acquisition; its residual instrumental error remains visible.

Save `output/reconstruction.npz` relative to the task workspace with key `reconstruction`, shape (1, 360), dtype float32 or float64. Axes are batch then chronological acquisition index; units are molecules/cm², with no logarithm, scaling, clipping or magnitude transform. Small negative noisy estimates near plume-free measurements are permitted. Evaluate the complete profile using cosine NCC (no mean subtraction) and RMS error normalized by the reference profile's dynamic range. The numerical reference is a standard-method estimate, not independent physical truth. No alignment, crop or scale fitting is applied.

## Data Description

`data/raw_data.npz` contains all observations and legitimate external calibrations:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| wavelength | (1,2048) | float64 | Native detector wavelength calibration, nm |
| spectra | (1,360,2048) | float64 | Actual measured intensities, instrument counts; chronological spectra then wavelength |
| dark | (1,2048) | float64 | Measured dark spectrum for subtraction, instrument counts |
| model_wavelength | (1,1201) | float64 | Supplied high-resolution calibration grid, nm |
| solar | (1,1201) | float64 | External high-resolution solar spectral intensity, native arbitrary intensity units; broadband response supplies count conversion |
| cross_sections | (1,3,1201) | float64 | SO2 at 295 K and O3 at 243 K in cm²/molecule, followed by dimensionless Ring pseudo-cross-section |
| elapsed_seconds | (1,360) | float64 | Time since first measured spectrum, seconds |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Spectrometer | Ocean Optics Flame-S FLMS02101 | instrument ID | Field UV spectrometer |
| Acquisition date | 2018-01-14 | calendar date | Masaya, Nicaragua |
| Integration time | 100 | ms | Per individual spectrum |
| Coadds | 10 | count | Instrument-reported averaging |
| View | Zenith | direction | Telescope mounted to car |
| Traverse geometry | Road approximately 3 km west of crater | km | Two plume crossings; output remains ordered by acquisition |
| Fit wavelengths | 310–320 inclusive | nm | Gas retrieval interval |
| Stray-light wavelengths | 280–290 inclusive | nm | Mean dark-corrected intensity estimates additive stray signal |
| Calibration model grid | 309–321 inclusive, spacing 0.01 | nm | Padded radiative-transfer grid; supplied values authoritative |
| ILS support | −2 inclusive to 2 exclusive, spacing 0.01 | nm | Normalized asymmetric super-Gaussian kernel |
| Grid convention | External molecular/solar calibrations in vacuum wavelengths; detector registration fitted | nm | Shift/stretch accounts for registration mismatch |
| Flat correction | Unavailable; no correction applied | dimensionless | Published no-flat-field analysis condition |
| Electronics flags | Dark correction false; nonlinearity correction false in instrument headers | boolean | Stored readings require supplied dark subtraction |
| Boundary | Zero extension for finite convolution; padded interior sampled | counts | No periodic spectral wrapping |
| Noise | No per-pixel covariance supplied | counts | Residual intensity/correction/model errors remain |

The external spectra have been cubic-interpolated onto the supplied model grid without target observations. They are legitimate spectroscopic calibration, not a reconstruction or a target-derived response.

**Evaluation certification:** The current numerical evaluator is not yet certified as a method-independent public benchmark; standard-method agreement must not be interpreted as independent physical accuracy.

## Method Hints

Use intensity-space spectral fitting with Beer–Lambert attenuation. Multiply the high-resolution solar spectrum, a smooth broadband polynomial and all molecular transmission factors before applying one instrument convolution. This avoids weak-absorption approximations that fail for stronger columns. A normalized asymmetric super-Gaussian is a useful line-shape family; include wavelength shift/stretch as nuisance variables. Nonlinear least squares can jointly retrieve the column and nuisance response for each acquisition. Preserve the full count scale and wavelength conventions, and examine residual structure rather than interpreting covariance estimates uncritically.

## References

B. Esse, M. Burton, M. Varnam, R. Kazahaya, G. Salerno, *iFit: A simple method for measuring volcanic SO2 without a measured Fraunhofer reference spectrum*, Journal of Volcanology and Geothermal Research 402, 107000 (2020). [Article](https://doi.org/10.1016/j.jvolgeores.2020.107000). [Measured spectra archive](https://doi.org/10.6084/m9.figshare.12174675.v1).

S. Beirle et al., *Parameterizing the instrumental spectral response function and its changes by a super-Gaussian and its derivatives*, Atmospheric Measurement Techniques 10, 581–598 (2017). [Article](https://doi.org/10.5194/amt-10-581-2017).
