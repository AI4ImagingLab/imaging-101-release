# Polarimetric Interreflection Harmonic Fields

> Reconstruct five spatial light-transport fields from a real set of illumination-and-analyzer polarization images, separating coupled angular harmonics despite noise and imperfect reflection-model assumptions.
> Domain: physics | Keywords: polarization imaging, light transport, inverse decomposition | Difficulty: Medium

## Background

Linear polarization carries information about the paths light follows through a scene. Specular reflection can preserve or rotate polarization, while scattering can reduce its directional preference. Rotating a polarizer in front of the illumination and another in front of the camera probes this behavior from both sides. The resulting intensity changes reveal information that a single ordinary photograph cannot provide.

Spatial polarization analysis is useful for inspecting reflective surfaces, understanding interreflections and improving optical measurements around shiny objects. Several transport contributions can reach the same camera pixel, so their measured intensities add. Their angular responses overlap, and real materials can produce effects beyond a simplified polarization model. Recovering the contributions therefore requires calibrated angular observations and a clear distinction between a fitted response and a uniquely identified physical path.

## Problem Description

The observations are sixteen real monochrome HDR images of a reflective scene containing a small object and surrounding surfaces. Each image records a known illumination-polarizer angle $g$ and camera-analyzer angle $c$, in radians. Both are the stored instrument-angle coordinates. Reconstruct the following pixelwise model at every image location $r$:

```math
I(r;g,c)=\frac12\bigl[U(r)+|C_s(r)|+|C_d(r)|\bigr]
+\frac12\operatorname{Re}\left[C_s(r)e^{-2i(c+g)}
+C_d(r)e^{-2i(c-g)}\right].
```

Here $I$ is measured intensity in relative instrumental radiance units, $U$ is the polarization-independent component, and $C_s,C_d$ are complex coefficients of the sum-angle and difference-angle harmonics. The subscripts denote these precisely defined angular families. They do not certify a unique number of reflection bounces. The five unknown real fields are $U$, the real and imaginary parts of $C_s$, and the real and imaginary parts of $C_d$. The amplitudes are $|C_s|,|C_d|$, and each component's polarization angle is one half of its complex argument, modulo pi. Returning the Cartesian fields avoids an undefined angle when an amplitude vanishes.

The model excludes terms that depend on only one polarizer angle, and it is an approximation for these real measurements. Noise and unmodeled transport can produce a signed fitted $U$; do not silently enforce positivity or upper-clip HDR observations. The recorded angles define the output phase frame. No spatial mask, fitted radiometric gain or rotation is part of the output convention.

Save `output/reconstruction.npz`, relative to the task workspace, containing exactly:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `reconstruction` | `(1, 5, 2048, 2448)` | float32 or float64 | Real fields in relative instrumental radiance units; axes batch, channel, image row, image column. Channel order is U, real C_s, imaginary C_s, real C_d, imaginary C_d. Rows increase downward and columns to the right. Signed values must be retained. |

Every field is evaluated over the entire declared image. Each channel uses the Euclidean norm of its error divided by the norm of the corresponding validated method reference. A full-observation forward-consistency metric uses the square root of summed squared intensity residual divided by summed squared observed intensity. All five fields and forward consistency must meet their respective limits. There is no crop, phase alignment, gain fit, channel permutation or magnitude substitution. This real-data task evaluates agreement with a validated decomposition; it does not assume an independently measured reflection-path truth.

## Data Description

`data/raw_data.npz` contains the existing full-resolution real observations and recorded polarizer angles. Arrays have a leading batch axis.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `intensity` | `(1, 16, 2048, 2448)` | float32 | Linear monochrome HDR intensity, in relative instrumental radiance units; axes batch, measured state, row, column. Values above one are valid. |
| `polarizer_psg` | `(1, 16)` | float64 | Illumination-side polarizer angles g, in radians, matched to the intensity state axis. |
| `polarizer_psa` | `(1, 16)` | float64 | Camera-side analyzer angles c, in radians, matched to the intensity state axis. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Measurement grid | All combinations of four illumination and four analyzer orientations | states | Sixteen existing measurements; no spatial crop or resolution reduction. |
| Orientations on each side | 0, 45, 90, 135 | degrees | Exact corresponding radians and state ordering are supplied in the angle arrays. |
| Image dimensions | 2048 rows by 2448 columns | pixels | Camera-image coordinates; physical pixel pitch is not supplied or needed for this angular inverse. |
| Radiometric representation | Linear monochrome HDR | relative instrumental radiance unit | No absolute radiometric calibration or wavelength-specific refractive index is supplied. |
| Published real-scene camera | FLIR BFS-U3-51S5P-C polarization camera | — | Supplies simultaneous analyzer states; the bundled observations are the author's prepared monochrome HDR images. |
| Published illumination | Epson EB-W05 projector and polarizer/quarter-wave combination | — | Intended to keep incident intensity constant while rotating linear polarization. |
| HDR preparation | Multiple camera exposures merged before this angular inverse | — | Individual exposure frames/times are not supplied; fit the existing linear HDR states. |
| Ambient illumination | Darkroom in the published real-scene setup | — | Reduces unmodeled environment light; does not guarantee the restricted model is exact. |
| Lower input bound | Zero | relative instrumental radiance unit | Apply the acquisition pipeline's lower clipping before fitting; this dataset already has positive values. |
| Upper intensity bound | None | relative instrumental radiance unit | Do not clip values above one or apply display gamma before inversion. |
| Polarizer convention | Stored numeric angles on each instrument | rad | Sum and difference in the stated forward equation are authoritative; mechanical viewing handedness is not independently documented. |
| Spatial domain | Entire camera image | pixels | Independent pixelwise angular model; no privileged foreground support or boundary calibration. |
| Noise variance | Not independently supplied | — | Do not assume a known noise level from a hidden target. |

The authors explicitly offer the measured data for research. A separate raw-data redistribution license was not identified; this is a local research package and makes no broader redistribution claim. Code licensing does not silently replace the data terms.

## Method Hints

Expand the angular equation into a constant and four products of sine and cosine functions of twice the recorded angles. This gives a small linear least-squares problem shared by all pixels. A full-rank pseudoinverse recovers the linear coefficients; sums and differences of the coefficients then give both complex harmonic fields and the remaining polarization-independent component. This closed-form decomposition retains all five degrees of freedom without iterative optimization.

Check the design rank and preserve the HDR scale. A richer linear Mueller response can provide a useful diagnostic of the terms omitted by the selected model, but it is a different comparison model and does not change the requested five-field output. Reflection-path interpretations require more geometric and polarimetric assumptions than the harmonic coefficients alone.

## References

1. R. Maeda and S. Hiura. *Polarimetric Light Transport Analysis for Specular Inter-reflection*. IEEE Transactions on Computational Imaging 10, 876–887 (2024). [Published article](https://doi.org/10.1109/TCI.2024.3404612).
