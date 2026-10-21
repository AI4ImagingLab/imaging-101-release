# C. Elegans Fluorescence Deconvolution

> Reconstruct an effective sharp fluorescence image from a processed Airyscan image of a C. elegans intestine under an assumed blur model, despite noise and lost spatial frequencies.
> Domain: biology | Keywords: fluorescence microscopy, image deconvolution, variational regularization | Difficulty: Medium

## Background

Fluorescent molecules emit light after optical excitation, allowing selected structures inside cells and organisms to be imaged. Diffraction spreads the light from each emitter over several pixels, mixing adjacent structures. Confocal detection reduces out-of-focus contributions; detector arrays and image processing can further improve resolution.

Recovering spatial detail helps examine organelles and tissue organization. However, weak signals, photodamage constraints, optical aberrations and processing of instrument exports complicate quantitative interpretation. An effective convolution model can support reproducible restoration, while its limitations must remain explicit.

## Problem Description

The observation is a repository-provided experimental, processed 2D Airyscan export of a C. elegans intestine. Its sampling scale, wavelength, detector gain and original instrument processing are unavailable. Work in pixel coordinates and arbitrary intensity units; the result is a restoration conditional on the stated effective blur model and is not a claim of calibrated molecular concentration or resolution recovery.

Let y be the observed image, x the latent effective sharp fluorescence image, h the supplied normalized effective point-spread function, and epsilon the residual noise/model mismatch. On the reflection-extended image grid, the acquisition model is

```math
y_{\mathrm{ext}} = Hx_{\mathrm{ext}} + \epsilon,\qquad
(Hx)(r)=\sum_s h(s)x(r-s).
```

Here r and s are integer two-dimensional pixel indices; H denotes circular convolution on the extended grid. Reflection extension excludes the endpoint pixel, then the circular convolution convention is applied. The central unpadded field is the requested output. Blur suppresses high spatial frequencies, so inversion amplifies noise and requires regularization. The PSF is an upstream Gaussian modeling assumption, not an experimentally measured calibration or a quantity inferred from hidden targets. No Poisson photon-count model is asserted for these processed intensities.

Save `output/reconstruction.npz` relative to the task workspace with exactly the required array `reconstruction`: shape `(1, 310, 310)`, real `float32` or `float64`, axis order `(batch, row, column)`, units arbitrary fluorescence intensity (a.u.), the same physical intensity scale as the input, no rescaling or alignment. Signed values are permitted because the standard linear estimator can overshoot. Pixel centers are indexed from zero; row increases downward and column rightward.

Evaluation compares all output pixels using cosine similarity (NCC, with no mean subtraction) and root-mean-square error divided by the reference dynamic range (NRMSE). Both numerical agreement criteria must pass. No phase, translation, magnitude, clipping, normalization or crop alignment is performed. Agreement measures reproducibility of the specified restoration model, not independent biological accuracy.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| observation | (1, 310, 310) | float32 | Experimental processed Airyscan fluorescence export, intensity in a.u.; upstream loader removes three border pixels on each side. |
| psf | (1, 13, 13) | float32 | Dimensionless unit-sum Gaussian effective PSF model, not measured calibration. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Modality | Confocal Airyscan processed export | none | Data provider's description; original instrument frames are unavailable. |
| Field size | 310 by 310 | pixel | Image after the provider's loader crop; no additional resampling. |
| Pixel pitch | 1 in both axes | pixel | Pixel-coordinate convention; physical pitch is unknown. |
| PSF standard deviation | 1.5 in both axes | pixel | Assumed effective Gaussian blur from the provider's example, not a measured optical width. |
| PSF support and center | 13 by 13; index (6, 6) | pixel | Unit-sum tabulated kernel. |
| Boundary extension | 13 pixels on each side, reflect excluding endpoint | pixel | Extended grid is 336 by 336; circular model on that grid, central crop for output. |
| Kernel origin | Move embedded center (168,168) to (0,0) | pixel | Embed kernel starting at (162,162) before circular convolution. |
| Frequency convention | Unnormalized forward DFT; inverse divided by grid size | none | Frequencies follow standard FFT ordering. |
| Noise parameters | Unknown | a.u. | Residual additive model; no known variance, photon gain or detector offset. |
| Physical sampling and optics | Not supplied | none | Do not infer micrometers, wavelength or numerical aperture. |

## Method Hints

A Laplacian-regularized Wiener/Tikhonov inverse balances fidelity to the blurred observations with suppression of unstable high-frequency components. With L the five-point discrete Laplacian, beta a positive regularization strength, and the norm the Euclidean norm over the extended grid, minimize

```math
\frac12\|Hx-y_{\mathrm{ext}}\|_2^2+\frac{\beta}{2}\|Lx\|_2^2.
```

The five-point Laplacian is four times the center minus its four immediate neighbors, using periodic indexing. Since both operators are diagonal in the discrete Fourier basis, the unconstrained minimizer has Fourier transform

```math
\widehat{x}(k)=\frac{\overline{\widehat{h}(k)}\widehat{y}_{\mathrm{ext}}(k)}{|\widehat{h}(k)|^2+\beta|\widehat{L}(k)|^2}.
```

Here k indexes discrete frequencies, a hat denotes a DFT, and the overline denotes complex conjugation. The inverse DFT yields the real extended image. The standard comparison uses a fixed regularization strength of 0.005, preserving the supplied demonstration's estimator. This scalar is a reconstruction choice, not an acquisition parameter.

## References

- Prigent et al. (2023), *SPITFIR(e): a supermaneuverable algorithm for fast denoising and deconvolution of 3D fluorescence microscopy images and videos*, Scientific Reports 13, 1489. https://doi.org/10.1038/s41598-022-26178-y . Provides the broader microscopy restoration context; this task uses the associated framework's standard 2D Wiener comparison, not SPITFIR(e).
- Sibarita (2005), *Deconvolution Microscopy*, Advances in Biochemical Engineering/Biotechnology 95, 201-243. https://doi.org/10.1007/b102215 .
- SDeconv 1.0.4, 2D Wiener example and C. elegans sample data; copyright 2021 STracking, BSD 3-Clause. Complete redistribution terms and attribution are consolidated in the maintainer validation record.
