# Calibrated Brillouin Spectral Shifts in Water

> Reconstruct Brillouin shifts and spectral parameters at ten water acquisition positions from curved VIPA camera spectra and independent calibration frames despite nonlinear dispersion and instrument drift.
> Domain: physics | Keywords: Brillouin microscopy, spectral calibration, inverse spectroscopy | Difficulty: Hard

## Background

Brillouin scattering exchanges energy between light and thermally excited acoustic waves. The resulting optical sidebands encode the speed of longitudinal density fluctuations. Resolving their frequency displacement provides a contact-free probe of a material's high-frequency mechanical response. Applications include liquids, polymers and biological specimens.

A virtually imaged phased array (VIPA) spectrometer converts optical frequency into position on a camera. Multiple interference orders and two dispersive stages produce a curved spectral trace. The displacement-to-frequency relation is nonlinear and changes with alignment and temperature. Independent reference measurements are therefore essential for comparing spectral shifts over time. The measured linewidth also contains instrumental broadening; it is not by itself an intrinsic viscosity measurement.

## Problem Description

The observations contain ten sequential positions in a water measurement, two camera frames per position and two independent ten-frame calibration measurements. Position indices identify acquisitions; they do not define a two-dimensional tissue image.

Let a camera frame be $I(u,v)$ in digital counts, with row $u$ and column $v$. A circular trace with center $c$ and radius $R$ is sampled at angle $\phi$ and radial offset $d$:

```math
 a(\phi,d)=c+(R+d)(\cos\phi,\sin\phi),\qquad
 S(p)=\frac{1}{|D|}\sum_{d\in D} I(a(\phi_p,d)).
```

Here $p$ indexes 500 equally spaced angles, $D$ is the radial sampling strip and camera values at fractional coordinates use bilinear interpolation. The frequency assigned to arc sample $p$ is

```math
 f(p)=\frac{1}{a_0+a_1p+a_2p^2}-f_0,
 \qquad f_0=\frac{c_0}{\lambda_0}.
```

The speed of light is $c_0$, laser wavelength is $\lambda_0$, and calibration coefficients are $a_0,a_1,a_2$. Six peaks from the independent reference arm have relative frequencies $q_j+o_j F$, where $q_j$ is a known reference displacement, $o_j$ is the interference-order index and $F$ is the fitted free spectral range. Calibration axes and extraction traces vary linearly between the two calibration acquisition times and are held at their endpoints outside that interval.

Within a selected spectral region, the camera intensity follows

```math
 L(f)=B+A\frac{(\Gamma/2)^2}{(f-\mu)^2+(\Gamma/2)^2}+\epsilon(f).
```

$\mu$ is the peak center, $\Gamma$ is its full width at half maximum, $A$ is its height, $B$ is background and $\epsilon$ includes detector noise and model mismatch. Each frame has two Brillouin regions and two elastic Rayleigh regions. A Brillouin shift is the smallest absolute frequency difference between that Brillouin center and either Rayleigh center. The nonlinear geometry, overlapping calibration doublets, unknown backgrounds and calibration drift make independent peak localization and frequency calibration necessary.

Save **`output/reconstruction.npz`** with the following real `float64` arrays. Every axis begins with a batch dimension of size one. No phase, scale, translation or intensity alignment is applied during evaluation.

| Key | Shape | Units and axis order |
| --- | --- | --- |
| `reconstruction` | `(1, 10, 2, 2)` | Hz; batch, acquisition position, camera frame, Brillouin region |
| `peak_parameters` | `(1, 10, 2, 4, 4)` | Batch, position, frame, region, parameter. Regions: Brillouin 1, Brillouin 2, Rayleigh 1, Rayleigh 2. Parameters: center (Hz), FWHM (Hz), height (ADU), background (ADU). |
| `frequency_axis` | `(1, 10, 2, 500)` | Hz relative to laser; batch, position, frame, arc sample |
| `extracted_spectra` | `(1, 10, 2, 500)` | ADU; batch, position, frame, arc sample |
| `calibration_frequency` | `(1, 2, 10, 500)` | Hz relative to laser; batch, calibration, frame, arc sample |
| `calibration_arcs` | `(1, 2, 500, 5, 2)` | Camera pixels; batch, calibration, arc sample, radial offset, coordinate (row, column). Radial offsets are -2,-1,0,1,2 pixels. |
| `calibration_parameters` | `(1, 2, 10, 4)` | Batch, calibration, frame, coefficient: $a_0$ (Hz$^{-1}$), $a_1$ (Hz$^{-1}$ pixel$^{-1}$), $a_2$ (Hz$^{-1}$ pixel$^{-2}$), $F$ (Hz). Here pixel denotes the dimensionless arc-sample index. |

Each quantity is evaluated by relative Euclidean error against an independently executed standard-method result. Mixed-unit parameter components are evaluated separately. All required quantities must pass; reconstruction agreement is a method-reproduction measure, not proof of exact physical truth.

## Data Description

`data/raw_data.npz` contains the complete measured camera records in their documented analysis orientation, without spatial cropping or reduced precision.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `images` | `(1, 10, 2, 400, 400)` | `uint16` | Water camera frames in ADU; batch, position, frame, row, column. |
| `calibration_images` | `(1, 2, 10, 400, 400)` | `uint16` | Independent mixed-standard reference-arm frames in ADU; batch, calibration, frame, row, column. |
| `calibration_time` | `(1, 2)` | `float64` | Start time of each calibration relative to acquisition origin, in seconds. |
| `measurement_time` | `(1, 10)` | `float64` | First-frame start time at each water position, in seconds. |
| `exposure` | `(1, 10)` | `float64` | Camera exposure and intra-position frame interval, in seconds. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Specimen | Water, ten sequential acquisition positions | — | Homogeneous liquid; index order is not a tissue geometry. |
| Acquisition date | 2020-11-03 | — | Native camera record date. |
| Analysis orientation | Clockwise 90 degrees from stored camera array | — | Already applied to the supplied arrays; apply no further rotation. |
| Detector axes | Row increases downwards, column rightwards; centers 0 through 399 | pixel | Coordinates refer to analysis-oriented images. |
| Laser wavelength | 780.24 | nm | $\lambda_0$. |
| Vacuum light speed | 299792458 | m/s | $c_0$. |
| Camera pixel pitch | 6.5 | µm | Physical detector sampling pitch. |
| Imaging lens focal length | 0.2 | m | VIPA frequency-model initialization. |
| VIPA cavity thickness | 0.006743 | m | Instrument geometry. |
| VIPA refractive index | 1.45367 | 1 | Instrument geometry. |
| VIPA incidence angle | 0.8 | degree | Instrument geometry; use radians in trigonometry. |
| Relative interference order | 0 | 1 | Initialization order relative to nearest laser resonance. |
| Methanol reference displacement | 3.78e9 | Hz | Known property of the independent calibration-arm standard. |
| Water reference displacement | 5.066e9 | Hz | Known property of the independent calibration-arm standard; not a prescribed value for the measured water positions. |
| Ordered calibration displacements | 0, 3.78e9, 5.066e9, -5.066e9, -3.78e9, 0 | Hz | $q_j$, in ascending detector order. |
| Calibration order indices | 0, 0, 0, 1, 1, 1 | 1 | $o_j$, paired with the displacements above. |
| Calibration nominal temperature | 295.15 | K | Metadata associated with the fixed supplied standard displacements. |
| Brillouin frequency windows | [4e9,6e9], [9e9,11e9] | Hz | Two inelastic regions relative to laser. |
| Rayleigh frequency windows | [-2e9,2e9], [13e9,17e9] | Hz | Two elastic interference orders. |
| Trace angular sampling | 500 equally spaced angles | 1 | Endpoint angles from circle/image-boundary intersections. |
| Trace radial sampling | -2,-1,0,1,2 | pixel | Five samples averaged at each angular position. |
| Extraction boundary convention | Rectangle [0,400] × [0,400]; interpolation valid at pixel centers 0…399 | pixel | Outside-image values omitted from radial mean. |
| Camera binning | 1 | 1 | No binning. |
| Noise model | Unspecified readout/photon noise | ADU | No known variance or synthetic noise addition. |

**Evaluation certification:** The current numerical evaluator is not yet certified as a method-independent public benchmark; standard-method agreement must not be interpreted as independent physical accuracy.

## Method Hints

Use bright calibration spots to estimate a circular spectral trace, then interpolate the trace and nonlinear frequency calibration in acquisition time. Fitting independent reference doublets and Rayleigh peaks constrains the VIPA dispersion and free spectral range without setting the unknown specimen shift to a nominal value.

The selected standard method fits Lorentzian shapes with bounded single-peak parameters and unconstrained calibration doublets. Its spectral optimizer receives squared residuals, making its effective objective quartic rather than ordinary least squares. Preserve this convention when reproducing the standard method; distinguish it from a separately labeled ordinary least-squares comparison. Resolve both Brillouin regions and both Rayleigh regions, including their backgrounds and widths.

## References

1. R. Schlüßler et al., *Mechanical Mapping of Spinal Cord Growth and Repair in Living Zebrafish Larvae by Brillouin Imaging*, Biophysical Journal 115, 911–923 (2018). [Published method](https://doi.org/10.1016/j.bpj.2018.07.027). The selected water observations are an independent software example, not the paper's zebrafish data.
2. S. Xiao, A. M. Weiner and C. Lin, *A Dispersion Law for Virtually Imaged Phased-Array Spectral Dispersers Based on Paraxial Wave Theory*, IEEE Journal of Quantum Electronics 40, 420–426 (2004). [VIPA dispersion](https://doi.org/10.1109/JQE.2004.825210).
3. bmlab developers, *Python library for the post-measurement analysis of Brillouin microscopy data sets* (software, 2022 onward). The selected implementation uses two independent calibration substances and its bundled measured-water example.
