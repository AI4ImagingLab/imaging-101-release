# Porcine Optical Attenuation Field

> Reconstruct a depth-resolved optical attenuation field from a complete porcine bladder OCT cross-section while accounting for confocal response, spectral sensitivity and finite-depth bias.
> Domain: medicine | Keywords: optical tomography, quantitative imaging, inverse problems | Difficulty: Hard

## Background

Optical coherence tomography measures backscattered near-infrared light as a function of depth. Interference localizes the scattering signal to narrow depth intervals, producing cross-sectional images of subsurface tissue. Light returning from deeper tissue is attenuated along both inward and outward paths, so the signal contains information about local optical attenuation as well as the structures that scatter it.

The signal also varies with focal-plane position and the instrument's sensitivity at different optical path lengths. Quantitative attenuation estimates must separate these instrumental effects from tissue-dependent loss. Such measurements support studies of layered tissue, including bladder, skin and retina. Speckle, low signal at depth and finite observation windows complicate interpretation; an algorithmic attenuation estimate is not automatically a calibrated measurement of tissue composition or disease.

## Problem Description

Infer the complete depth-by-lateral attenuation field of the supplied ex-vivo porcine bladder scan. Use the known author optical calibration and the depth-resolved confocal model; no automatic estimation of the focal plane or Rayleigh range is required.

For axial coordinate $z$ in mm, lateral scan index $x$, attenuation $\mu(z,x)$ in mm^-1 and depth-independent instrument/backscatter scale $C$, the single-scattering signal model is

```math
I(z,x)=C f(z)h(z)\mu(z,x)\exp\left[-2\int_0^z\mu(s,x)\,ds\right]+e(z,x).
```

Here $e$ represents additive measurement contamination, $h$ is the confocal response and $f$ is spectral sensitivity fall-off. The actual observed array is already the author's nonnegative linear signal; do not convert it from decibels again. Depth samples represent narrow signal integration intervals, with the selected native inverse using a Riemann-sum approximation.

The confocal function and sensitivity are

```math
h(z)=\frac{1}{1+((z-z_0)/z_R)^2},\qquad
f(z)=\left(\frac{\sin\zeta}{\zeta}\right)^2
\exp\left[-\frac{(\delta\lambda/\Delta\lambda)^2\zeta^2}{2\log2}\right],
\qquad\zeta=\frac{\pi z}{2z_{\mathrm{RD}}}.
```

The value at $\zeta=0$ is its continuous limit1. The focal coordinate is $z_0$, $z_R$ is the apparent Rayleigh range, $\delta\lambda$ is spectral resolution, $\Delta\lambda$ is wavelength spacing per detector pixel, and $z_{\mathrm{RD}}=\lambda^2/(4\Delta\lambda)$ is ranging depth, converted from meters to mm when using the wavelength values below. The central wavelength is $\lambda$.

The inversion is sensitive to weak deep signal and the finite observation boundary. The benchmark targets the declared native discrete estimator, including its endpoint convention: the last recorded depth sample is omitted from reverse cumulative denominators and the last output row is zero. The resulting penultimate-depth saturation is an estimator limitation that must remain in the full output; it does not establish a large true tissue attenuation.

Save `output/reconstruction.npz` with this required real array:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,512,512) | float64 | Nonnegative attenuation in mm^-1 on the author's axial coordinate; axes batch,depth,lateral; preserve every native pixel, zero final row and full physical scale |

Evaluation uses cosine NCC without mean subtraction, dynamic-range-normalized RMSE and the normalized residual of the declared discrete integral identity. All512×512 values are evaluated with no crop, scale fit, translation or support-dependent output omission. The integral-identity residual is evaluated on every nonfinal row, as defined by the estimator; the final row is still included in reference comparisons. These scores establish reproducibility of this quantitative method, not independent biological attenuation truth.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| oct_signal | (1,512,512) | float64 | Complete nonnegative author-linear OCT intensity in normalized instrument units; axes batch,depth,lateral; original observed speckle retained |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | Ex-vivo porcine bladder | — | Complete original author real-tissue scan |
| Signal representation | Linear intensity | normalized instrument units | Supplied array is already converted from author log signal and divided by55743 |
| Axial coordinates | linspace(0,2.358,512) | mm | Author tissue-loader coordinate convention |
| Axial spacing | 2.358/511 | mm | Consecutive sampled coordinate spacing |
| Lateral spacing | 0.00502 | mm | Consecutive A-scan coordinate spacing; lateral origin0 |
| Focal position | 1.0 | mm | Known nominal author calibration |
| Apparent Rayleigh range | 0.29017970000000004 | mm | Author105.905µm range multiplied by2×1.37 |
| Nominal medium index | 1.37 | dimensionless | Used in the author's apparent Rayleigh calibration; no additional axial rescaling |
| Central wavelength | 1.325e-6 | m | Source wavelength |
| Detector wavelength spacing | 7.1875e-11 | m/pixel | Spectral sampling parameter |
| Spectral resolution | 1.1055e-10 | m | Spectrometer resolution parameter |
| Noise power | 2.5e-5 | squared normalized instrument units | Author restoration parameter; not an independently verified per-pixel variance |
| Propagation model | Single backscatter, double-pass attenuation | — | Unknown depth-independent scale cancels in the inverse |
| Depth boundary | Native finite window | — | Reverse denominators exclude the last observed row; output last row0 |
| Extent provenance | 2.358 axial,2.56522 lateral | mm | Specific tissue loader governs this task; generic data note's2.57×2.57mm statement differs |

## Method Hints

Use an observation-derived tissue support and a confocal/sensitivity-corrected depth-resolved ratio. The selected surface convention is the first upward crossing of a per-column half-range threshold: half the sum of the mean of sorted samples0–50 and the mean of sorted samples460–511, using zero-based indices. A column without such a crossing starts at row0. The binary support includes that crossing and deeper rows, followed by a square10 dilation and square35 erosion; dilation exterior is0, erosion exterior is1, and the even-square anchor follows the centered SciPy maximum-filter convention with origin0. These specify the native support rather than requiring an external segmentation or hidden label.

Let $A$ be the public intensity multiplied by this support, $P$ the supplied noise-power scalar, and $J=A(hf)^{-1}A^2/(A^2+P)$ the restoration-corrected signal. For depth row $m$, width $\Delta z$ and total depth count $M=512$, the selected closed-form inverse is

```math
\widehat\mu_{m,x}=\frac{J_{m,x}}{2\Delta z\,S_{m,x}},\qquad
S_{m,x}=\sum_{r=m}^{M-2}J_{r,x},\qquad 0\le m\le M-2.
```

Use zero where the denominator vanishes and on row $M-1$. This known endpoint convention differs from summing through the final sample and explains the tail artifact. Cumulative sums make the inverse efficient; explicit optics and boundary conventions are essential. Optional lateral regularization can study the influence of speckle and layered structure, but the required output is the unsmoothed declared depth-resolved estimator.

## References

Smith et al., “Automated, Depth-Resolved Estimation of the Attenuation Coefficient From Optical Coherence Tomography Data,” *IEEE Transactions on Medical Imaging*34(12),2592–2602(2015), [doi:10.1109/TMI.2015.2450197](https://doi.org/10.1109/TMI.2015.2450197).
