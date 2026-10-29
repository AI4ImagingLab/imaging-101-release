# Bayer Raw Colour Reconstruction

> Recover a full-resolution three-channel camera-colour image from a real single-channel Bayer sensor exposure with missing colour samples, sensor noise and fine spatial structure.
> Domain: mathematics | Keywords: inverse imaging, spatial sampling, colour reconstruction | Difficulty: Medium

## Background

A camera with a colour filter array measures one spectral channel at each detector pixel. Adjacent pixels carry different filters, so the detector never observes all three colour channels at the same spatial location. Recovering the missing channels is an underdetermined spatial inverse problem. Fine textures, object edges and spectral transitions make simple independent interpolation prone to false colours.

Joint spatial and spectral information helps reconstruct a complete colour image at the detector's native resolution. This operation is fundamental to scientific cameras, microscopy and digital photography. The resulting camera colour coordinates retain the sensor's spectral response; they are distinct from calibrated material reflectance or absolute scene radiance. Optical blur, defective pixels and measurement noise can remain after colour reconstruction.

## Problem Description

The observations are one real Canon PowerShot A540 exposure of a potted plant. The active sensor array is used in full, without resizing, rotating or adding noise. Define the black-corrected normalized measurement at row i and column j by:

```math
y(i,j)=\frac{d(i,j)-b(i,j)}{w-b(i,j)}
=\sum_{c\in\{R,G,B\}}M_c(i,j)x_c(i,j)+\eta(i,j).
```

Here d is the recorded sensor value in ADU, b the known black level in ADU, w the white level in ADU, x the unknown three-channel linear camera-colour image, M the known binary colour-sampling masks and eta the sensor/model error in normalized detector units. Each location has exactly one active colour mask. The coordinates increase downwards for rows and rightwards for columns. Values in x are relative to each channel's black-to-white sensor range; no absolute radiometric calibration is available. Two of three colour components are unobserved at every location. Fine edges, scene chromaticity and defective sensor pixels constrain the achievable recovery.

Save **`output/reconstruction.npz`**, relative to the task workspace, with the following required output:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 2128, 2840, 3) | float64 | Linear camera RGB in dimensionless black-to-white detector units; axes batch, row, column, colour; channel order R, G, B. |

The output is real, signed and unclipped. Negative values or values above one produced by reconstruction must be retained. Do not white-balance, apply gamma, rotate, crop or rescale the submitted array. White balance is available for display only.

Evaluation uses three jointly required quantities: uncentered cosine NCC, dynamic-range NRMSE and maximum absolute error at the observed colour samples. For flattened submitted image u and an independently computed method reference v, NCC is their dot product divided by the product of their Euclidean norms; a zero submitted norm gives NCC zero. NRMSE is the square root of the mean squared element error divided by max(v) minus min(v). The sampling error is the maximum absolute difference between the corresponding submitted colour channel and y over all detector pixels. Higher NCC and lower errors are better. All comparisons use the declared native grid and scale, with no alignment or hidden rescaling. Agreement with a method reference is not proof of physical accuracy in the unobserved channels.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| sensor_mosaic | (1, 2128, 2840) | uint16 | Native active sensor samples in ADU; 10-bit measurements stored in a 16-bit container. |
| black_levels | (1, 4) | float64 | Per-channel known sensor black levels in ADU, ordered R, first G, B, second G. |
| white_balance | (1, 4) | float64 | Dimensionless camera white-balance metadata, ordered R, G, B, unused fourth entry; for display only. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Camera | Canon PowerShot A540, CHDK DNG | — | Real camera exposure. |
| Active image height | 2128 | pixel | Number of retained detector rows. |
| Active image width | 2840 | pixel | Number of retained detector columns. |
| Original raw container | 2136 by 2888 | pixel | Contains optically inactive sensor margins. |
| Active origin in raw container | row 8, column 44 | pixel | Zero-based top-left active sensor position. |
| Bayer pattern | RGGB | — | Top-left2by2 grid: R G / G B; the active origin preserves this phase. |
| ADC signal depth | 10 | bit | White level1023 ADU. |
| White level w | 1023 | ADU | Known sensor saturation scale. |
| Black levels | 31, 31, 31, 31 | ADU | Matching the visible four-entry calibration array. |
| Sensor orientation | Unrotated; row down, column right | — | Submission follows detector coordinates. |
| Boundary convention | Half-sample symmetric reflection | — | Extend sensor-edge values by reflection for finite-support reconstruction. |
| Noise | Present; variance not independently calibrated | — | No fabricated variance or noise-free observation is supplied. |

The observed image was released as raw.pixls.us sample1272 under CC0-1.0. Raw decoding only exposes the native active pixel array and recorded camera metadata. No rendered camera JPEG is used as an observation or evaluation target. Pixel spacing in metres and absolute radiometric response are unavailable and are unnecessary for this pixel-coordinate reconstruction.

## Method Hints

A direct linear reconstruction can combine spatial interpolation with cross-channel gradient correction. Estimate a missing colour from neighbouring samples of that colour and correct it using spatial curvature in a channel observed more densely nearby. Select the appropriate correction according to the colour-sampling phase. This family handles local luminance edges more effectively than independent bilinear interpolation under its smooth-colour assumptions. A finite-support direct inverse does not require artificial iteration. Sensor noise and defective pixels are not removed by this model, so inspect edge effects and avoid confusing sharper rendering with validated physical accuracy.

## References

1. H. S. Malvar, L.-W. He, R. Cutler. *High-Quality Linear Interpolation for Demosaicing of Bayer-Patterned Color Images*. ICASSP2004, III485–488. DOI: [10.1109/ICASSP.2004.1326587](https://doi.org/10.1109/ICASSP.2004.1326587).
2. [raw.pixls.us](https://raw.pixls.us/), camera sample1272, CC0-1.0. [Licence](https://creativecommons.org/publicdomain/zero/1.0/).
