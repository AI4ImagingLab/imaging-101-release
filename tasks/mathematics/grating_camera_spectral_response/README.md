# Grating Camera Spectral Response

> Recover four cameras' RGB spectral sensitivities from measured direct and diffracted light images despite unknown grating efficiency and wavelength mapping.
> Domain: mathematics | Keywords: spectral calibration, inverse optics, constrained estimation | Difficulty: Hard

## Background

A camera integrates wavelength-dependent radiance through its color filters. Its RGB spectral response governs color reproduction, radiometric interpretation and spectral reconstruction. Cameras with similar visible colors can have different responses to narrow-band illumination.

A transmission grating spreads light across sensor columns. Images of a narrow illuminated slit therefore contain wavelength-resolved information, while an image without diffraction measures integrated channel brightness. Recovering the camera response from these observations requires separating detector sensitivity from optical throughput and sensor geometry.

## Problem Description

For channel c and sensor column p, use the discretized acquisition model

```math
d_c=\sum_j l_j s_{jc},\qquad y_{pc}=\sum_j K_{pj}(w) l_j a_j s_{jc}.
```

Here d is the direct RGB measurement, y is the diffracted RGB profile, l is the known relative LED spectrum, s is camera sensitivity on the illumination wavelength grid, a is unknown relative grating throughput, and K distributes wavelengths to pixels using an unknown mapping w. Index j runs over wavelength bins; c runs over red, green and blue. The discrete sensitivity includes the fixed wavelength-bin scale. Fluorescent observations supply narrow peaks that constrain w. Absolute gain is unidentifiable; relative throughput and response can trade off. A separate population of calibrated cameras supplies a low-dimensional response prior.

The inputs contain four real camera acquisitions, in order EOS RP, Sony Alpha 1, iPhone 15 Pro Max, DJI Pocket 3. Recover each response on the 31-point 400--700 nm grid. Save `output/reconstruction.npz` with exactly the key `reconstruction`, shape `(1, 4, 31, 3)`, float32 or float64, real finite values. Axes are batch, camera in the stated order, increasing wavelength, RGB. Values are dimensionless relative sensitivities; normalize each camera by its single maximum across all wavelengths and RGB channels, so that maximum is one. Preserve the joint RGB scale, including any small negative numerical estimates; no per-channel rescaling is used in evaluation.

Evaluation measures cosine NCC and dynamic-range NRMSE for every camera against independent measurements, with no phase, translation or gain alignment. NCC is the flattened dot product divided by both Euclidean norms. NRMSE is root mean squared error divided by the reference maximum minus minimum. Every camera and the unit-peak normalization requirement must pass.

## Data Description

`data/raw_data.npz` contains author-decoded and cropped real linear images, with no additional cropping or resampling. Direct images have no grating dispersion. Diffracted images use LED illumination; fluorescent images constrain pixel-to-wavelength geometry. Camera prefixes match the output order above. Calibration spectra and the population response set were supplied independently of these four targets.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `eos_direct` | (1, 88, 10, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `eos_diffracted` | (1, 88, 1628, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `eos_fluorescent` | (1, 88, 1628, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `sony_direct` | (1, 151, 47, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `sony_diffracted` | (1, 482, 1904, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `sony_fluorescent` | (1, 482, 1904, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `iphone15promax_direct` | (1, 127, 16, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `iphone15promax_diffracted` | (1, 129, 441, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `iphone15promax_fluorescent` | (1, 129, 441, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `dji_pocket3_direct` | (1, 26, 16, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `dji_pocket3_diffracted` | (1, 129, 600, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `dji_pocket3_fluorescent` | (1, 129, 600, 3) | float32 | relative linear RGB digital signal; axes batch, row, column, RGB |
| `population_sensitivity` | (1, 44, 31, 3) | float64 | dimensionless relative sensitivity |
| `population_wavelength` | (1, 31) | float64 | metres |
| `illumination_wavelength` | (1, 321) | float64 | metres |
| `led_spectrum` | (1, 321) | float64 | dimensionless peak-normalized illumination |
| `fluorescent_spectrum` | (1, 321) | float64 | dimensionless peak-normalized illumination |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Cameras | EOS RP; Sony Alpha 1; iPhone 15 Pro Max; DJI Pocket 3 | names | Fixed output order |
| Illumination grid | 400 through 720, inclusive, step 1 | nm | Known light spectra |
| Response grid | 400 through 700, inclusive, step 10 | nm | Recovered response and population samples |
| LED | Godox LED6R, setting 6500 | K | Real broad-band source |
| Fluorescent source | Panasonic EFD15EL | model | Real narrow-peak calibration |
| Slit width | 2.5 | mm | Illuminated object width |
| Sensor setting | ISO 100, daylight white balance, manual RAW acquisition | setting | Author acquisition protocol |
| Lighting | Dark room | condition | Suppressed ambient light |
| Grating | Transmission, nominal 500 slits/mm | lines/mm | Paper apparatus; exact pixel geometry is inferred |
| Coordinates | Zero-based increasing image column; wavelength in metres in arrays | convention | Quadratic pixel-to-bin map |
| Spectral interpolation | Linear, including endpoint extrapolation | convention | 31 response samples evaluated on 321 light samples |
| Mixing boundary | Mapping outside [0, 320.5] contributes zero; terminal valid bins clamp to 320 | bin | Finite spectral support |
| Spectral sum | Unit numerical bin weight | convention | Bin width and unknown gain absorbed into relative response |
| Noise and exposure | No calibrated noise model or per-file exposure supplied | unknown | Do not infer absolute radiometry |

## Method Hints

Use population-derived response subspaces and a smooth reciprocal-throughput basis. A narrow-line source can constrain wavelength mapping before joint response/throughput estimation. After mapping, the bilinear problem can be transformed to a constrained linear least-squares problem using reciprocal throughput. Direct-light measurements constrain each color channel's scale. This decomposition is useful because the calibration population limits otherwise ambiguous spectral shapes. The unconstrained basis may yield signed nuisance throughput; agreement with observations alone does not establish its physical accuracy.

## References

Makabe et al., *Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating*, ICCV 2025. [Paper, version 1](https://arxiv.org/abs/2508.00330v1).
