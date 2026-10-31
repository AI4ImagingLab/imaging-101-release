# EPR Radical Density Separation

> Reconstruct two overlapping two-dimensional radical density fields from magnetic-gradient electron paramagnetic resonance spectra, despite spectral mixing, limited angular sampling and receiver noise.
> Domain: Chemistry & Material Science | Keywords: magnetic resonance imaging, spectral imaging, variational reconstruction | Difficulty: Hard

## Background

Electron paramagnetic resonance detects unpaired electron spins through their response to an applied magnetic field. Different chemical radicals have different resonance spectra. Their signals add when several species are present, so a measured spectrum alone describes chemical content without locating it.

A magnetic-field gradient makes the local resonance field depend on position. Recording spectra for several gradient directions encodes spatial information into the resonance shifts. This permits imaging of paramagnetic tracers, material defects and chemical reaction distributions. Broad or overlapping spectral signatures and receiver noise make spatial separation substantially harder than identifying an isolated spectral peak.

## Problem Description

Recover the two spatial radical distributions, ordered **TAM, TEMPO**, in an author-defined synthetic two-dimensional experiment using real spectral response templates. The underlying density fields are nonnegative. A signed reconstruction is permitted as an unconstrained statistical estimate; negative estimated values must not be interpreted as actual negative chemical concentration.

Let $u_j(x,y)$ denote the relative spin amount per square centimetre of species $j$, and let $h_j(B)$ be its calibrated receiver response per unit relative spin amount. The projection indexed by $p$ is

```math
s_p(B)=\sum_{j=1}^{2}\int_{\Omega}u_j(x,y)\,
h_j\!\left(B+g_{xp}x+g_{yp}y\right)\,dx\,dy+\eta_p(B).
```

Here $B$ is the swept homogeneous field in gauss, $(g_{xp},g_{yp})$ is the applied gradient in G/cm, $\Omega$ is the square image domain, and $\eta$ is independent Gaussian receiver noise. The template amplitudes define the relative density scale; absolute molecular concentration is not calibrated.

The supplied acquisition uses a periodic field-axis Fourier discretization. With $N$ field samples, spacing $\Delta B$, image spacing $\delta$, centered image coordinates $x_b=\delta(b-50)$, $y_a=\delta(a-50)$, and unnormalized template DFT $H_j[k]$, its retained Fourier coefficients satisfy

```math
Y_p[k]=\delta^2\sum_{j=1}^{2}H_j[k]
\sum_{a,b}u_j[a,b]\exp\!\left(
\mathrm{i}\frac{2\pi k}{N\Delta B}
\left[g_{xp}x_b+g_{yp}y_a\right]\right).
```

The DFT convention is $H_j[k]=\sum_n h_j[n]\exp(-2\pi\mathrm{i}kn/N)$. Retain only nonnegative frequencies satisfying

```math
\|g_p\|\,k<\frac{N\Delta B}{2\delta},\qquad k<N/2.
```

All remaining coefficients are zero. Real inverse DFT uses the $1/N$ normalization and the conjugate negative frequencies. The finite angular coverage and different template bandwidths make the joint spatial inverse ill-conditioned; the two species must be evaluated separately.

Save **`output/reconstruction.npz`**, relative to the task workspace, containing exactly:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `reconstruction` | `(1, 2, 100, 100)` | `float64` | Real signed density estimates in relative spin amount/cm²; axes are batch, species `(TAM,TEMPO)`, increasing $y$, increasing $x$. |

The first pixel center on either spatial axis is −1.50 cm and the last is +1.47 cm. Preserve this coordinate origin, amplitude scale and species order. Evaluation uses each complete field's cosine NCC, dynamic-range normalized RMS error and relative integrated-amount error. It also checks that the full predicted spectra have residual RMS consistent with the known noise and limited adjacent-channel correlation. No crop, translation, rescaling or species permutation is fitted during evaluation.

## Data Description

`data/raw_data.npz` contains only observations and available calibration:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `projections` | `(1, 40, 1200)` | `float64` | Noisy signed receiver spectra in arbitrary receiver units; axes batch, gradient direction, homogeneous field. |
| `magnetic_field_G` | `(1, 1200)` | `float64` | Homogeneous field sampling coordinates in G, increasing along the spectral axis. |
| `gradients_G_cm` | `(1, 2, 40)` | `float64` | Applied gradient components in G/cm; component order $x,y$, followed by projection order. |
| `reference_spectra` | `(1, 2, 1200)` | `float64` | TAM and TEMPO response templates in arbitrary receiver units per relative spin amount, sampled on the same field axis. |

The observations are newly generated using an original author-defined acquisition experiment, not experimental spatial measurements. The spectral templates come from publicly offered author experimental demonstrations. Stored float64 arrays are lossless promotions of the author's float32 observations and calibration; their coordinate rounding is retained.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Species order | TAM, TEMPO | — | Two additive chemical responses. |
| Spatial image size | 100 × 100 per species | pixel | Full output field. |
| Pixel spacing $\delta$ | 0.03 | cm | Equal horizontal and vertical spacing. |
| Pixel coordinates | $(a-50)\delta,(b-50)\delta$ | cm | Array row is $y$; column is $x$. |
| Projection count | 40 | — | All supplied gradient directions are required. |
| Nominal gradient magnitude | 10 | G/cm | Actual rounded components are supplied in the NPZ. |
| Gradient angle range | 0 to $\pi$, inclusive | rad | Actual components, including endpoint rounding, are authoritative. |
| Field sample count $N$ | 1200 | — | Samples per spectrum and template. |
| Field range | 334.0 to 463.89166259765625 | G | Supplied field coordinates are authoritative. |
| Fourier field step $\Delta B$ | 0.10833740234375 | G | First supplied difference, used by the acquisition discretization. |
| Fourier field period $N\Delta B$ | 130.0048828125 | G | Periodic spectral convention. |
| Noise mean | 0 | arbitrary receiver unit | Additive independent Gaussian noise, up to stored float32 rounding. |
| Noise standard deviation | 5000 | arbitrary receiver unit | Known measurement noise scale. |
| Spatial boundary | Finite centered square | cm | No periodic spatial wrap; the field-axis DFT convention is distinct. |
| Density scale | Relative spin amount/cm² | — | Fixed by supplied template amplitude; no absolute molar calibration. |

## Method Hints

A joint least-squares model with spatial total variation can separate the two spectral responses while suppressing noise. Keep the source coupling in the data term: fitting the mixture independently with each species can assign one radical's signal to the other.

A suitable objective is

```math
\frac12\|Au-s\|_2^2+\lambda\sum_{j=1}^{2}\sum_{a,b}
\sqrt{(D_yu_j[a,b])^2+(D_xu_j[a,b])^2},
```

where $A$ is the complete acquisition operator, $s$ denotes measured spectra, $\lambda>0$ controls spatial regularity, and $D_x,D_y$ are forward differences with zero outward boundary differences. Primal-dual splitting handles the nonsmooth regularizer. The normal operator has a cross-species convolution structure that can accelerate iteration. Positivity is physically natural, but with a large zero-density region it can bias integrated amounts by truncating signed noise fluctuations; evaluate spatial accuracy and amount recovery together.

## References

- R. Abergel, S. Durand and Y.-M. Frapart, “PyEPRI: A CPU & GPU compatible python package for electron paramagnetic resonance imaging,” *Journal of Magnetic Resonance* 376, 107891 (2025). [Paper](https://doi.org/10.1016/j.jmr.2025.107891).
- M. Boussâa, R. Abergel, S. Durand and Y.-M. Frapart, “Ultrafast multiple paramagnetic species EPR imaging using a total variation based model,” *Journal of Magnetic Resonance* 357, 107583 (2023). [Paper](https://doi.org/10.1016/j.jmr.2023.107583).
- A. Chambolle and T. Pock, “On the ergodic convergence rates of a first-order primal-dual algorithm,” *Mathematical Programming* 159, 253–287 (2016). [Paper](https://doi.org/10.1007/s10107-015-0957-3).
