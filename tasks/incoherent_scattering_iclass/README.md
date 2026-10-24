# Blind Fluorescence Imaging Through a Scattering Layer

> Recover a relative fluorescence image and the unknown detection transfer function from 150 experimentally measured speckle-illuminated frames whose spatial structure is hidden by scattering.
> Domain: Physics | Keywords: multiple scattering, blind imaging, covariance imaging | Difficulty: Hard

## Background

Fluorescent emitters radiate mutually incoherent light. A disordered optical layer spreads the light from each emitter into an extended intensity speckle pattern, so conventional widefield detection loses the object's spatial structure. Within the optical memory-effect range, neighboring emitters share shifted versions of approximately the same intensity point-spread function. This creates a convolutional imaging model even when that point-spread function is unknown.

Changing the illumination speckle produces independent fluctuations of the fluorescence. Correlations across many camera frames can retain object information that is absent from their average. This is useful when neither a guide star nor a directly measured scattering calibration is available, for example when inspecting fluorescent structures behind diffusers or thin biological tissue. Recovering a detection transfer function together with the image makes the problem a blind inverse problem.

## Problem Description

Let the nonnegative effective fluorescence distribution be $O(\mathbf r)$, the intensity point-spread function be $P(\mathbf r)$, and the unknown random illumination of frame $m$ be $S_m(\mathbf r)$. The camera observations satisfy

```math
I_m(\mathbf r)=P*[O S_m](\mathbf r)+\epsilon_m(\mathbf r),\qquad m=1,\ldots,150.
```

Here $*$ denotes discrete spatial convolution, $\epsilon_m$ represents detector noise and model discrepancy, and $\mathbf r$ indexes the two image coordinates. The optical transfer function is $H(\mathbf k)=\mathcal F_2 P(\mathbf k)$, where $\mathcal F_2$ is the two-dimensional spatial discrete Fourier transform and $\mathbf k$ is spatial frequency. Both $P$ and $S_m$ are unknown. No measured point-spread function or illumination maps are supplied.

The acquisition contains fluorescent microspheres imaged through a holographic diffuser. Each supplied frame represents the entire recorded field after the documented Fourier low-pass resampling. This retains all 150 illumination realizations. It does not extract a smaller central scene. A finite camera field, finite speckle correlations, photon noise, imperfect isoplanatism and strongly attenuated optical frequencies make joint recovery ill-conditioned. Absolute fluorescence and absolute transfer amplitude are unidentifiable from these observations; the reconstructed image describes effective fluorescence under the unknown illumination envelope.

Save **`output/reconstruction.npz`**, relative to the task workspace, containing all three arrays:

| Required output key | Shape | Numeric dtype | Units and convention |
|---|---|---|---|
| `reconstruction` | `(1, 400, 400)` | `float32` | Nonnegative relative effective fluorescence, peak normalized to one; axes batch, camera row, camera column; same full field as the input |
| `phase` | `(1, 400, 400)` | `complex64` | Unit-magnitude Fourier correction phasor, dimensionless; $\exp(i\phi)$, with $\phi$ in radians; centered spatial-frequency order, axes batch, vertical frequency, horizontal frequency |
| `mtf` | `(1, 400, 400)` | `float32` | Nonnegative relative modulation-transfer amplitude, peak normalized to one; centered spatial-frequency order, same axes as `phase` |

The correction phasor compensates the phase of the intensity OTF, not the coherent optical pupil phase. Its global piston is irrelevant. Object position has a translation ambiguity because a translation may be assigned to the unknown PSF. Image agreement is evaluated after an integer circular translation maximizing cross-correlation over all 400 by 400 possible shifts. Ties select the first maximum in row-major order (smallest nonnegative row shift, then column shift), and the selected shift is applied by periodic wrapping. There is no fitted intensity rescaling, crop, reflection or rotation. Agreement uses cosine NCC without mean subtraction and dynamic-range NRMSE.

Every output quantity is evaluated. In addition to image agreement, the evaluator checks the MTF against the public observation covariance, checks that the submitted image agrees with per-frame spatial inversion of the submitted correction and MTF, and checks coherent accumulation of the corrected covariance. For this consistency convention, each temporally centered, Hann-windowed frame spectrum is multiplied by `phase`, divided by `mtf + 0.01`, spatially inverse-transformed, and combined by the square root of the mean squared magnitude; the resulting image is peak normalized. The regularizer is dimensionless relative to the peak-normalized MTF. This fixes the output scale and noise convention rather than requiring a particular code layout.

## Data Description

The observations are real experimental camera measurements associated with the fluorescence-scattering study in References, public experimental Figure 3(D–F) data, `Run_1.mat`. The recorded intensity stack contains 150 frames on a 1200-by-1200 grid. Each complete frame is transformed with the unnormalized spatial FFT, the centered 400-by-400 frequency square is retained, and an inverse spatial FFT on that retained grid produces the released observations. This operation preserves the full field and changes the sampling interval by a factor of three. Because no FFT amplitude compensation is applied, the resulting DN scale is nine times the corresponding conventional amplitude-preserving interpolation scale. This known factor cancels in the relative output quantities. The stored `float32` values reproduce the upstream input convention.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `camera_frames` | `(1, 150, 400, 400)` | `float32` | Fourier-resampled measured intensities in source FFT-resampled camera digital numbers (DN); axes batch, illumination realization, camera row, camera column; no target or transfer calibration is included |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Illumination realizations | 150 | frames | All recorded random speckle illuminations are retained |
| Recorded array size | 1200 by 1200 | camera pixels | Actual measurement array dimensions before resampling |
| Released array size | 400 by 400 | samples | Whole-field Fourier resampling; no spatial subfield crop |
| Sampling interval | 3 original detector samples | detector-pixel spacing | Both spatial axes have the same factor-three sampling change; absolute object-plane pitch is not supplied |
| Fourier convention | Forward unnormalized, inverse divided by sample count | dimensionless | Negative exponential in forward FFT; centered frequencies |
| Coordinates | Row down, column right | samples | Image origin at top left; frequency origin at the center |
| Exposure | 0.2 | s | Measured exposure for every scattering frame |
| Excitation wavelength | 0.488 | micrometers | Archived acquisition value |
| Nominal detection wavelength | 0.52 | micrometers | Archived acquisition parameter; not a replacement for the filter center |
| Emission filter | Center 525; bandwidth 39 | nm | Optical filter specification reported for the experiment |
| Objective | 20 times; numerical aperture 0.6 | dimensionless | High-NA acquisition configuration |
| Tube lens focal length | 300 | mm | Archived optical configuration |
| Laser power setting | 40.2 | mW | Archived setting; not an absolute sample irradiance calibration |
| Diffuser–object separation | 7 | mm | Experiment category reported for Figure 3(D–F) |
| Illumination rotation increment | 2.4 | degrees | Recorded change between illumination realizations |
| Spatial apodization | Separable symmetric Hann, endpoints zero | dimensionless | Apply on the released 400-by-400 grid before temporal centering |
| Boundary convention | Periodic discrete transforms after Hann apodization | dimensionless | Finite-field approximation; not a claim that physical space is periodic |
| Detector noise | No calibrated variance supplied | DN squared | Infer robustly from measured data; do not assume noiseless observations |

The dataset's camera-size scalar metadata says 1400 whereas its stored array is 1200 by 1200; the stored array defines the bound observations. The public nominal wavelength and filter specification are preserved separately. Absolute object-plane pixel size cannot be established from the available calibration and is not invented.

## Method Hints

Treat the frame stack as a measurement matrix, with pixels as rows and illuminations as columns. Subtracting the temporal mean removes the illumination mean. For sufficiently decorrelated speckles, the resulting covariance approximates a virtual reflection matrix with the same unknown detection distortion on its two sides. Its ideal object term is weighted by the squared fluorescence and the illumination statistics.

A memory-efficient CLASS family method can estimate the OTF phase by coherent accumulation of equal spatial-frequency differences. The products needed for a phase update can be evaluated using correlations of the rectangular pixel-by-frame matrix, avoiding an infeasible square pixel covariance. The covariance diagonal in Fourier space estimates the relative MTF. Regularized amplitude compensation belongs on each phase-corrected frame spectrum before forming the final RMS image. Distinguish the spatial axes from the illumination axis when applying inverse transforms.

## References

Weinberg, G., Sunray, E., and Katz, O. “Noninvasive megapixel fluorescence microscopy through scattering layers by a virtual incoherent reflection matrix.” *Science Advances* 10(47), eadl5218 (2024). https://doi.org/10.1126/sciadv.adl5218. Main article and Supplementary Materials, especially the covariance derivation and regularized Fourier reweighting.

Author-published experimental data collection: https://drive.google.com/drive/folders/18A_W_JemctYMtloonCAh3RTWsZfOJyJN. Figure 3(D–F) measurements; the task binds all scattering observations through the explicitly documented whole-field resampling.
