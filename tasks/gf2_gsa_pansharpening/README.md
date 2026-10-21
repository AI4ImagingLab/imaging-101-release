# Native GF2 Adaptive Gram-Schmidt Pansharpening

> Recover a spatially detailed four-band image from registered real multispectral and panchromatic satellite observations with mismatched spatial and spectral resolutions.
> Domain: earth science | Keywords: satellite imaging, multispectral fusion, spatial resolution | Difficulty: Medium

## Background

Earth-observing satellites commonly separate spectral and spatial information across sensors. A broad panchromatic channel collects enough light for finer spatial sampling, while narrower multispectral channels provide material and vegetation contrast at coarser sampling. Combining these observations supports detailed interpretation of the Earth's surface.

Pansharpened multispectral imagery is useful for urban mapping, land-use monitoring and environmental assessment. The broad spectral response of the panchromatic detector differs from any individual multispectral band, so blindly transferring intensity can alter spectral relationships. Fine spectral structure remains ambiguous without a physically known spectral scene at the panchromatic sampling scale.

## Problem Description

Reconstruct a four-band high-resolution radiometric image from an author-supplied GF2 scene identified as Beijing. The native registered input arrays are reused; their resolution difference was not synthetically created. A generic acquisition model is

```math
M_b=D(H_b*X_b)+\epsilon_b,\qquad
P=H_P*\left(\sum_{b=1}^4 a_bX_b\right)+\eta.
```

Here $X_b$ is unknown band $b$ at high spatial sampling, $M_b$ the observed low-resolution band, $P$ the panchromatic observation, $H_b$ and $H_P$ their spatial point-spread functions, $*$ convolution, $D$ decimation by four in both directions, $a_b$ nonnegative spectral-response weights, and $\epsilon_b,\eta$ measurement/model discrepancy. The data supplier gives registered pixel geometry but does not supply native PSFs, spectral-response weights, physical pixel size or acquisition date. These quantities must not be inferred from an unavailable high-resolution spectral target. Relative normalized radiometry and pixel coordinates define this task; absolute reflectance or geographic inference is not requested.

Save `output/reconstruction.npz` relative to the task workspace. Its required `reconstruction` is a real `uint8` array of shape `(1, 256, 256, 4)`, axes `(batch, row, column, band)`, preserving original band order1–4. The physical scale is the normalized per-band radiometry multiplied by255 and truncated toward zero after clipping to[0,1]; units are quantized normalized intensity levels0–255, not absolute reflectance. The PAN-grid output is registered to the supplied PAN image. Do not permute bands for display in the saved output. Evaluation uses cosine normalized cross-correlation (NCC, without subtracting means) and dynamic-range normalized RMS error (NRMSE). For flattened real estimate $a$ and comparison image $b$, NCC is $a^Tb/(\|a\|_2\|b\|_2)$ and NRMSE is $\sqrt{\mathrm{mean}[(a-b)^2]}/[\max(b)-\min(b)]$. Every required array must pass both criteria. Evaluation compares the exact declared grid and physical scale, with no crop, fitted rescaling, translation or other alignment. The comparison establishes agreement with a validated standard reconstruction; these real observations have no independently known target truth.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| multispectral | (1, 64, 64, 4) | float32 | Native observed bands in normalized dimensionless radiometry, retaining source band order. |
| panchromatic | (1, 256, 256, 1) | float32 | Native observed PAN in independently normalized dimensionless radiometry. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Source scene | GF2_BJ | dimensionless | Native author-supplied GF2 scene labeled Beijing. |
| Resolution ratio | 4 | PAN pixel/MS pixel | Same registered field of view, fourfold ratio along each spatial axis. |
| Native ROI | MS rows200:264/columns200:264;PAN rows800:1056/columns800:1056 | source pixel | Zero-based upper-exclusive crop preserves acquisition registration. |
| Geometry | row,column increase in supplied image coordinates | pixel | No supplied celestial/geographic north orientation; no assumed meter scale. |
| Radiometry | separate min-max scaling for each MS band and for PAN | dimensionless | Observed ROI normalization; the four original multispectral channels retain relative shape but not absolute cross-band gain. |
| Acquisition date, wavelength bandpass, ground sampling | not provided by data supplier | unknown | No unsupported sensor metadata is assumed. |
| Native PSF and PAN spectral coefficients | not provided by data supplier | unknown | A surrogate low-pass model may be used diagnostically; it is not a measured sensor PSF. |
| Statistical noise calibration | not provided | unknown | Do not interpret pixel residuals as calibrated chi-square. |
| Boundary convention | periodic wrap for local interpolation and filtering | dimensionless | Declared reconstruction discretization on this finite registered crop. |

## Method Hints

Adaptive component substitution uses the observed multispectral bands to predict an intensity component matched to the broad panchromatic response. Multivariate regression on spatially matched observations estimates that component without a hidden spectral target. Gram-Schmidt covariance gains transfer PAN detail into each spectral band. Compare against interpolation alone and a non-adaptive intensity to assess spatial detail and spectral distortion; no single surrogate diagnostic establishes true spectral accuracy.

## References

- Aiazzi, Baronti and Selva (2007), *Improving Component Substitution Pansharpening Through Multivariate Regression of MS+Pan Data*. [DOI](https://doi.org/10.1109/TGRS.2007.901007).
- Liu (2000), *Smoothing filter-based intensity modulation: a spectral preserve image fusion technique for improving spatial details*. [DOI](https://doi.org/10.1080/014311600750037499).
- Vivone et al. (2015), *A Critical Comparison Among Pansharpening Algorithms*. [DOI](https://doi.org/10.1109/TGRS.2014.2361734).
- Anjing Guo, Python pansharpening toolbox and supplied GF2 observations (2020). Copyright and software notices are preserved in the maintainer package.
