# Diffractive Lens Chromatic Radiance

> Recover red and blue scene channels from a real wavelength-dependent blurred photograph using the simultaneously measured, less-blurred green channel as structural guidance.
> Domain: mathematics | Keywords: blind deconvolution, inverse optics, multispectral restoration | Difficulty: Hard

## Background

Diffractive optical elements bend different wavelengths by different amounts. Compact lenses can therefore produce sharp structure in one color band while spreading other wavelengths into broad, position-dependent point-spread functions. The resulting photograph loses detail differently across its color channels.

Recovering color detail is useful in compact cameras and lightweight optical systems where a fully corrected lens would be too complex or bulky. Simultaneously acquired channels share scene structure, although their intensities and fine detail need not be identical. This structural relationship supplies information lost in the blurred channels, while uncertainty in the blur prevents ordinary fixed-kernel inversion.

## Problem Description

The full measured photograph has 1585 rows and 886 columns. For local image patch p and color c, the forward model is

```math
y_{pc}=h_{pc}*x_{pc}+n_{pc}.
```

y denotes observed relative digital intensity, x the desired sharper scene channel, h an unknown wavelength- and position-dependent blur kernel, n measurement/model error, and the star local convolution. The measured green channel supplies structural information but is not a sharp ground truth. Unknown kernels and nonidentical spectral reflectance make this blind inverse problem ambiguous. Local periodic convolution is the numerical approximation; overlapping regions connect the patch solutions.

Save `output/reconstruction.npz` with `reconstruction`, real float32 or float64, shape `(1,1585,886,3)`, axes batch,row,column,RGB. Units are relative digital intensity on the input's 0--1 scale. Restore red and blue and retain the measured green channel exactly up to 1e-7 absolute numerical tolerance. The reference method can overshoot below zero or above one: preserve these signed values without display clipping, rescaling, cropping or channel permutation in the saved output. All values must be finite.

Evaluation measures full-array cosine NCC (flattened dot product divided by the two norms) and dynamic-range NRMSE (RMS difference divided by the comparison image's maximum minus minimum), with no fitted gain, registration or hidden crop. It assesses agreement with a validated standard-method result, not independently measured sharp-scene accuracy. The green-channel condition must also pass.

## Data Description

The image is the complete real diffractive-lens example distributed by the authors, converted losslessly from its decoded numeric intensities. No artificial blur, noise or new scene was generated.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `measurements` | (1,1585,886,3) | float64 | Relative digital intensity, RGB order, decoded 8-bit samples divided by255; batch,row,column,color |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Modality | Color photograph through a diffractive lens | type | Real wavelength-dependent optical blur |
| Image height and width | 1585,886 | pixel | Complete supplied field of view |
| Color order | Red, green, blue | channels | Green is measured structural guidance |
| Numeric scale | Original decoded sample divided by255 | relative digital intensity | No gamma inversion or radiometric calibration supplied |
| Pixel coordinates | Zero-based row downward, column rightward | pixel | No rotation or geometric alignment |
| Local convolution | Periodic within each estimated blur patch | boundary | Overlap combines local results |
| Finite difference boundary | Replicate last row/column | convention | Forward difference is zero at exterior edge |
| Sensor pitch, spectral passbands, exposure | Not distributed | unknown | Do not infer absolute radiance or physical blur length |
| Noise | No independently calibrated distribution | unknown | Model residual includes optical approximation and sensor processing |

**Evaluation certification:** The current numerical evaluator is not yet certified as a method-independent public benchmark; standard-method agreement must not be interpreted as independent physical accuracy.

## Method Hints

Use alternating local blur estimation and cross-channel information transfer. A Fourier-domain regularized kernel estimate connects an evolving sharp-color estimate to its observed blurred channel. Applying that kernel to green produces a matched-blur guide; a local affine model in guide intensity and its spatial derivatives transfers fitted color information back to the less-blurred green structure. Overlap weighting controls discontinuities. This exploits shared edges without assuming equal RGB intensity or a known point-spread function.

## References

Tiancheng Sun, Yifan Peng and Wolfgang Heidrich. *Revisiting Cross-channel Information Transfer for Chromatic Aberration Correction*. ICCV2017. Paper and project.
