# Joint Camera Response and Bracketed Irradiance Reconstruction

> Recover an RGB irradiance image and nonlinear camera response from seven real exposure-bracketed photographs despite saturation and unknown photometric encoding.
> Domain: mathematics | Keywords: inverse imaging, photometric reconstruction, computational photography | Difficulty: Hard

## Background

A camera integrates light over its shutter interval and converts the collected exposure into digital values. Consumer photographs additionally apply nonlinear encoding, so doubling a pixel value does not generally mean twice the illumination. A single exposure loses detail where its digitizer saturates or where the recorded signal approaches its dark limit.

Exposure bracketing records a stationary scene at several shutter times. Short exposures retain bright features and longer ones reveal dark surfaces. Recovering the common photometric field supports illumination measurement, high-dynamic-range photography and image-based rendering. Absolute radiometric calibration requires additional measurements; photographic brackets alone determine relative irradiance.

## Problem Description

Let p identify an image pixel, c an RGB channel, j an exposure and t its recorded shutter time. The observed digital code z is approximately

```math
z_{jpc}=Q\!\left(f_c(E_{pc}t_j)\right)+\eta_{jpc},\qquad
g_c(z_{jpc})\simeq\log E_{pc}+\log t_j.
```

Here E is the unknown relative image-plane irradiance, f the unknown monotone camera encoding, Q quantization and clipping to integers0–255, and eta digital noise and model mismatch. The log inverse response is g=log(f inverse); its positive lookup table is h=exp(g). Recover both E and h. JPEG processing, small scene changes and saturation prevent an exact fit. Multiplying E and h by the same channel-dependent constant preserves the model; fix h(128)=1 independently in every channel. The optical exposure corresponding to code128 then defines one relative exposure unit. E has units of relative exposure per second, not watts per square metre.

The seven images show the same indoor scene and its bright window. Use the complete427×640 field. Save `output/reconstruction.npz` relative to the task workspace with exactly these arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 427, 640, 3) | float64 | Strictly positive real relative irradiance; axes batch,row,column,RGB; units relative exposure/s; original pixel coordinates. |
| response | (1, 256, 3) | float64 | Strictly positive, strictly increasing inverse response h; axes batch,digital code0–255,RGB; units relative exposure; code128 equals1 in each channel. |

Every value must be finite. No output alignment, crop, exposure gain fit, channel permutation or tone mapping is applied during evaluation. Tone mapping is only a display operation. Evaluation version2 checks the measured-code fit, smooth monotone response and declared photometric gauge without requiring identity with a finite sample-based method. The full weighted forward RMS described below must be <=7.105642315376502 ADU. Require max over channels |h(128)-1|<=1e-8. For g=log(h), require the sum over all3channels and interior code indices1,...,254 of (g(z-1)-2g(z)+g(z+1))^2 <=0.0017575872742814459. The smoothness budget is the independently executed feasible standard value plus relative1e-8 numerical allowance; it expresses the smooth response prior. All irradiances and responses are strictly positive and the response strictly increasing. These conditions define a noise-compatible joint photometric class; no noise-free radiance truth is available.

Forward evaluation predicts digital codes by linearly interpolating code index against log h at log(E t), clipping outside the response endpoints. It uses all observed codes from8 through247, with weight min(z,255−z); other codes have weight0. The residual is sqrt(sum(weight×squared code error)/sum(weight)), in ADU. This fixed support excludes unreliable saturation endpoints without selecting pixels using reconstruction error.

## Data Description

These are author-deposited real camera photographs, already processed into JPEGs. Their decoded integer pixels are preserved exactly; no synthetic noise, exposure generation or extra resizing has been applied. They are not native sensor RAW files. Recorded exposure times accompany the complete seven-frame sequence.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| frames | (1, 7, 427, 640, 3) | uint8 | Measured JPEG digital codes in ADU0–255; axes batch,exposure,row,column,RGB. |
| exposure_times | (1, 7) | float64 | Actual shutter times in seconds, matching the exposure axis. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Camera | Canon EOS REBEL T2i | model | Recorded in each image's EXIF. |
| Acquisition date | 2015-08-21 | date | First recorded time09:11:58; last09:13:39. |
| Focal length | 36 | mm | Shared recorded lens focal length. |
| F-number | 4.5 | dimensionless | Shared aperture setting. |
| ISO setting | 200 | ISO speed | Shared camera sensitivity setting. |
| Exposure order | 2,0.5,0.125,0.03333333333333333,0.008,0.001,0.00025 | s | Stored values are authoritative, in decreasing exposure order. |
| Public image dimensions | 427×640 | pixel | Complete deposited image field; sensor-native dimensions are not supplied. |
| Coordinate origin | upper-left | pixel | Row increases downwards, column rightwards; no rotation. |
| Channel order | RGB | channel | Processed camera channels, not a Bayer mosaic. |
| Encoding | 8-bit JPEG, EXIF sRGB | digital code | Common response is unknown; do not assume a known gamma curve. |
| Quantization and clipping | 0–255 | ADU | Codes at both endpoints may lack usable exposure information. |
| Noise calibration | unavailable | ADU | No measured per-pixel noise variance is provided. |
| Geometric assumption | stationary registered scene | convention | No geometric resampling is part of the target contract. |
| Spatial boundary | finite image support | convention | Each pixel is reconstructed from its own seven exposure observations. |
| Relative exposure gauge | h(128)=1 per channel | relative exposure | Resolves otherwise unidentifiable photometric gain. |

## Method Hints

Use weighted photometric least squares to estimate a smooth log inverse response jointly with sampled log irradiances. Saturated digital codes should carry little weight. A second-difference penalty regularizes response curvature; the midpoint gauge fixes scale. Once the response is estimated, combine exposure-corrected log measurements across shutter times and invert the logarithm. Check response monotonicity and reproduce the measured codes; a visually pleasing tone map alone cannot validate the recovered photometric values.

## References

1. P. E. Debevec and J. Malik. *Recovering High Dynamic Range Radiance Maps from Photographs*. SIGGRAPH1997. Debevec and Malik, SIGGRAPH 1997 paper.
2. F. Banterle, A. Artusi, K. Debattista and A. Chalmers. *Advanced High Dynamic Range Imaging*,2nd edition,CRC Press,2017. ISBN9781498706940. The published HDR Toolbox implementation supplies the measured demonstration exposures and reference numerical conventions.
