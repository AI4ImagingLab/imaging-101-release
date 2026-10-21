# RAW Burst Latent Bayer Reconstruction

> Reconstruct a spatial low-noise Bayer sensor image from ten real handheld RAW exposures despite shot/read noise, local scene motion and imperfect frame alignment.
> Domain: mathematics | Keywords: multiframe inverse imaging, statistical image reconstruction, computational photography | Difficulty: Hard

## Background

A digital camera records noisy photoelectron counts after spatial filtering and color sampling. Short exposures reduce blur and protect highlights but collect fewer photons. A burst supplies repeated indirect observations of the same scene, allowing shared spatial information to constrain the unknown image.

Hand motion and scene motion mean that pixels across exposures do not describe identical scene points. Combining them without accounting for those changes produces ghosts and lost structure. Reliable reconstruction balances temporal evidence against disagreement, while respecting the color filter lattice and signal-dependent sensor variance. Such reconstruction supports low-light mobile photography and scientific imaging with limited exposure budgets.

## Problem Description

Ten actual consecutive short-exposure Bayer frames observe one scene. The unknown is the two-dimensional latent Bayer signal in the first frame’s sensor coordinates. A local acquisition model is

```math
y_t(p)=b+g_t[\mathcal M\{x(W_t(p))\}-b]+\eta_t(p),\qquad
\operatorname{Var}(\eta_t(p)\mid y_t)=\lambda_s y_t(p)+\lambda_r.
```

Here t indexes exposure, p=(row,column) a detector coordinate, x the latent reference-exposure signal, W the local motion pullback, M the sensor color sampling and digitization operator, y the recorded digital signal in ADU, b the sensor black offset, g the relative exposure/gain response, eta the shot/read/model error in ADU, lambda_s the variance slope in ADU and lambda_r its intercept in ADU². The first nine frames share exposure/ISO; the tenth has a longer exposure and higher ISO, so its response differs. The requested spatial image is M{x} on the reference sensor lattice. Motion and interframe disagreement are nuisances, rather than the final requested output. Only even-pixel native displacements are applied to retain Bayer parity. Repeated observations do not provide an independent noise-free truth.

Save `output/reconstruction.npz` with:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 1024, 1024) | uint16 | Quantized reconstructed native Bayer digital signal in ADU; axes batch,row,column; includes the original black offset |

Retain the BGGR sensor lattice and row/column orientation, one native sensor pixel per output element. Do not demosaic, gamma-correct, subtract black, white-balance, rotate, crop, fit a gain or replace it with displacements. Quantization truncates toward zero after the spatial merge, preserving the selected standard method’s digital image convention. Evaluation uses full-array relative Euclidean error, norm(estimate-reference)/norm(reference), computed in floating point without alignment or scale fitting. This is a reproducible signal reconstruction in digital units, not absolute physical radiance.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| frames | (1, 10, 1024, 1024) | uint16 | Original real RAW measurements in ADU; axes batch,time,row,column; complete ten-frame sequence on a fixed native ROI |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Burst identifier | 33TJ_20150606_224837_294 | identifier | Measured mobile-camera sequence |
| Exposure count | 10 | frame | All deposited exposures retained in native chronological order |
| Reference frame | 0 | zero-based index | Native first-frame coordinate system |
| Native sensor array | 3120 rows by 4208 columns | pixel | Original deposited DNG dimensions |
| Fixed ROI | rows1048:2072,columns1592:2616 | zero-based half-open pixel | Central1024² region fixed before reconstruction |
| Color filter lattice | BGGR | dimensionless | Upper-left blue, upper-right green, lower-left green, lower-right red |
| Sensor black | 64 on all four sites | ADU | Offset retained during merging |
| Sensor white | 1023 | ADU | Nominal native10-bit white point stored in uint16 |
| ISO, frames0–8 | 1143 | ISO speed | Reference and first nine exposures |
| ISO, frame9 | 1551 | ISO speed | Last exposure has a different gain setting |
| Camera | Motorola Nexus 6 | identifier | Deposited sensor metadata |
| Exposure time, frames0–8 | 297140608/4294967295 (approximately0.0691834390) | s | First nine deposited exposures |
| Exposure time, frame9 | 429466976/4294967295 (approximately0.0999930725) | s | Last deposited exposure; all ten raw frames retained |
| Variance slope lambda_s | 3.55148388 | ADU | Independent author ISO-derived noise model |
| Variance intercept lambda_r | 516.6520187906699 | ADU² | Independent author ISO-derived read-noise model |
| Noise calibration provenance | no NoiseProfile tag; author ISO model | dimensionless | Does not use reconstructed scene truth |
| Coordinates | row down,column right | pixel | Native unrotated sensor orientation |
| Boundaries | symmetric16-pixel extension; finite clipped searches | pixel | Fixed finite detector region, no scene wraparound |
| Absolute pixel pitch/radiometric calibration | unavailable | unavailable | Output uses native pixel coordinates and ADU |

These observations are a fixed native crop of real measurements, without resizing, prealignment or synthetic noise. The noise curve is a documented approximation derived from acquisition ISO rather than a claim of independently measured exact variance.

The selected standard method does not photometrically normalize the different final exposure. It treats disagreement through its reference-guided Fourier weights, so that exposure mismatch remains a documented modeling limitation. Retain all ten observations and the declared digital output convention; do not silently discard or normalize the last exposure.

## Method Hints

Use coarse-to-fine local block matching on Bayer-averaged Gaussian pyramids, then reconstruct the spatial signal from the original unaveraged Bayer observations. Pairwise Fourier shrinkage suppresses inconsistent temporal coefficients while retaining the reference structure; a spatial Wiener filter can reduce remaining high-frequency noise. Half-overlapped tiles with a partition-of-unity cosine window prevent seams. A quadratic fit to an actual two-dimensional3×3 distance neighborhood can refine coarse displacements; form RMS noise levels in floating point to avoid integer-square overflow.

The selected author2021 comparison uses temporal strength75 and spatial strength0.1, all four pyramid scales and all ten frames. The temporal Fourier correction has weight |R-A|²/(|R-A|²+c V), where R/A are reference/alternate coefficients, V is the signal-dependent variance, and c includes temporal strength and transform scaling. This differs from simple registration followed by averaging.

## References

Monod, Delon and Veit, *An Analysis and Implementation of the HDR+ Burst Denoising Method*, Image Processing On Line11,142–169(2021). [Paper](https://doi.org/10.5201/ipol.2021.336). This task uses the author2021 method with explicitly corrected neighborhood indexing and integer variance arithmetic.

Hasinoff et al., *Burst photography for high dynamic range and low-light imaging on mobile cameras*, ACM TOG35(6)(2016). [Paper](https://doi.org/10.1145/2980179.2980254). [Measured dataset and attribution](https://hdrplusdata.org/dataset.html), CC BY-SA 4.0; the fixed crop and NPZ repackaging are adaptations under the same data license. Algorithm source is AGPL3-or-later, copyright2021Antoine Monod. The method selected here is the2021 author implementation, not proprietary original camera software.
