# Brain Phosphorus Resonance Quantification

> Recover all overlapping phosphorus resonance amplitudes, frequencies, linewidths and their common phase from a complete author-simulated brain-like complex free-induction decay with noise and coupled multiplets.
> Domain: medicine | Keywords: magnetic resonance spectroscopy, metabolic imaging, constrained inversion | Difficulty: Hard

## Background

Magnetic resonance spectroscopy distinguishes molecular environments through their nuclear resonance frequencies. Phosphorus spectroscopy can resolve signals associated with cellular energy metabolism, membrane synthesis and phosphate pools, providing information that complements anatomical imaging. Phosphocreatine and adenosine-triphosphate resonances are particularly useful for studying energy storage and turnover in tissue.

A receiver measures a complex free-induction decay after excitation. Frequency offsets encode chemical shifts, exponential damping broadens resonances, and receiver timing and phase affect the observed interference. Neighboring metabolites and spin-coupled multiplets overlap in the spectrum. Quantifying their separate contributions therefore requires an inverse model rather than assigning each spectral maximum to a single molecule.

## Problem Description

The data are the original author's complete, previously released synthetic phosphorus FID benchmark representing a nominal 7 T human-brain-like resonance mixture. They are not newly generated observations for this task and are not a measured clinical patient record. Let $y_n$ be complex receiver signal at time $t_n$, with sample index $n=0,\ldots,1023$. Model it as

```math
y_n=\sum_{k=1}^{16}a_k\exp(i\phi)\exp(-\pi w_k t_n)
          \exp(i2\pi f_k t_n)+\epsilon_n.
```

Here $i^2=-1$, $a_k\ge0$ is the extrapolated time-zero amplitude in the original receiver signal units, $f_k$ is frequency offset in Hz, $w_k>0$ is the Lorentzian full width at half maximum in Hz, $\phi$ is the common receiver phase in radians and $\epsilon_n$ is complex measurement noise. Each resonance has damping coefficient $d_k=\pi w_k$ in s⁻¹. The lineshapes are Lorentzian; no Gaussian mixing parameter is unknown in this binding. The positive exponential sign defines frequency convention. Amplitudes retain the original signal scale; absent an absolute concentration calibration, they must not be labeled mM or rescaled to sum to one.

The acquisition samples the signal after a nonzero deadtime, rather than at time zero. Frequencies, linewidths and amplitudes must therefore be inferred jointly. Chemical resonance priors and multiplet physics constrain an otherwise correlated inverse problem. The following component order and prior ranges specify the bound physical model; ppm is parts per million relative to the carrier, with frequency in Hz equal to ppm times the stated nuclear frequency in MHz.

| Component | Chemical-shift range (ppm) | Linewidth range (Hz) | Multiplet constraint |
|---|---|---|---|
| BATP | [−16.30, −16.00] | [54.335, 61.911] | Independent parent. |
| BATP2 | [−16.30, −16.00] | Same as BATP | Half parent amplitude, parent frequency −15 Hz, parent linewidth. |
| BATP3 | [−16.30, −16.00] | Same as BATP | Half parent amplitude, parent frequency +15 Hz, parent linewidth. |
| AATP | [−7.72, −7.42] | [31.226, 33.327] | Independent parent. |
| AATP2 | [−7.72, −7.42] | Same as AATP | Parent amplitude, parent frequency −16 Hz, parent linewidth. |
| GATP | [−2.65, −2.39] | [36.892, 41.157] | Independent parent. |
| GATP2 | [−2.65, −2.39] | Same as GATP | Parent amplitude, parent frequency −16 Hz, parent linewidth. |
| UDPG | [−9.76, −9.68] | [29.125, 35.619] | Independent resonance. |
| NAD | [−8.25, −8.17] | [34.823, 46.155] | Independent resonance. |
| PCr | [−0.5, 0.5] | [13.21, 17.603] | Independent resonance. |
| GPC | [2.94, 2.96] | [18.557, 21.359] | Independent resonance. |
| GPE | [3.49, 3.51] | [17.348, 20.849] | Independent resonance. |
| Pin | [4.81, 4.83] | [19.353, 22.727] | Independent resonance. |
| Pex | [5.19, 5.29] | [22.091, 39.725] | Independent resonance. |
| PC | [6.22, 6.26] | [15.756, 24.16] | Independent resonance. |
| PE | [6.71, 6.81] | [20.658, 24.605] | Independent resonance. |

The same receiver phase applies to all 16 components and lies in [−π, π]. Amplitudes have no finite upper prior bound. These ties leave 37 identifiable free parameters: 12 independent amplitudes, 12 parent/independent frequencies, 12 parent/independent linewidths and one phase. The component amplitudes are resonance contributions, not multiplet-integrated metabolite concentrations. A parent label does not authorize omitting its coupled components.

Save **`output/reconstruction.npz`**, relative to the task workspace, containing all four required arrays:

| Output key | Shape | Dtype | Meaning and conventions |
|---|---|---|---|
| `reconstruction` | `(1, 16)` | `float64` | Nonnegative time-zero amplitudes, in original receiver signal units; batch then the exact component order above. |
| `frequency_hz` | `(1, 16)` | `float64` | Real frequency offsets in Hz, with the positive complex-exponential convention above and the same component order. |
| `linewidth_hz` | `(1, 16)` | `float64` | Positive real Lorentzian FWHM values in Hz, in the same order; not angular damping coefficients. |
| `phase_rad` | `(1, 1)` | `float64` | The single shared receiver phase, in radians in [−π, π]. |

Amplitudes are evaluated with flattened cosine NCC and dynamic-range-normalized RMS error. All frequencies and linewidths are evaluated by maximum absolute physical error; phase is compared by the absolute principal angle of its phase difference, reflecting 2π periodicity without fitting any phase alignment. A complex FID observation-consistency metric also checks the declared physical outputs jointly on all 1024 samples. There is no normalization, spectral cropping, magnitude conversion or fitted scale in evaluation. Every required metric must meet its boundary.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `measurements` | `(1, 1024)` | `complex128` | Full original complex receiver FID, in arbitrary receiver signal units; batch then increasing acquisition time, with real and imaginary channels retained. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation provenance | Author-released synthetic brain-like benchmark | — | Existing fixed observations; no actual-patient claim. |
| Nucleus | Phosphorus-31 | — | Resonance species. |
| Nominal field label | 7 | T | Protocol label; use the exact stated nuclear frequency in all formulas. |
| Nuclear frequency | 120 | MHz | Conversion from chemical shift in ppm to frequency offset in Hz. |
| Spectral width | 10000 | Hz | Uniform sampling bandwidth. |
| Dwell interval | 0.0001 | s | Reciprocal of spectral width. |
| Receiver deadtime | 300 | µs | Time of the first observed sample after excitation. |
| Sample times | $t_n=0.0003+n/10000$ | s | Increasing time, with $n$ from 0 through 1023. |
| Number of samples | 1024 | dimensionless | All complex observations are used. |
| Receiver convention | $\exp(i2\pi f t)$ | — | Positive frequency corresponds to the positive complex exponential. |
| Amplitude scale | Original receiver units; unnormalized | arbitrary signal units | No absolute molar calibration is available. |
| Lineshape | Lorentzian, $d=\pi w$ | s⁻¹ and Hz | Exponential damping coefficient versus FWHM linewidth convention. |
| Common phase | One shared receiver phase | rad | No independent per-component phase or fitted global alignment in evaluation. |
| Noise | Complex noise, variance not supplied | signal units² per component | Infer fit quality from real and imaginary residuals; no hidden noise seed is input. |
| Boundaries | Finite sampled FID; no extrapolated observations | — | Time-zero amplitude is a model parameter, not an observed sample. |

## Method Hints

A constrained time-domain resonance fit can exploit known multiplet amplitude ratios, frequency splittings and common linewidth/phase relationships. A Levenberg–Marquardt initializer followed by a bounded trust-region reflective least-squares fit is well suited to this smooth nonlinear model. Representing real and imaginary residuals explicitly preserves receiver phase and avoids the information loss of magnitude-only fitting. Keep all components and distinguish physical linewidth from exponential damping.


Fit all1024 complex observations jointly with the unnormalized constrained least-squares objective

```math
\min_{\theta}\sum_{n=0}^{1023}|y_n-s(t_n;\theta)|^2,
```

where y is the observed FID, s is the complete16-resonance model above, and theta contains its37 free physical quantities with the stated resonance bounds and ties. Equivalently, minimize the sum of squares of interleaved real and imaginary residuals, preserving their original units and equal weight. No fitted amplitude normalization or phase alignment is part of this objective. A bound-preserving internal parameter transform can support Levenberg–Marquardt initialization, followed by physical-coordinate trust-region refinement. Check convergence of all physical parameters as well as the residual energy: weak overlapping peaks can stop changing in total loss before their individual linewidths stabilize. The receiver quadratures need not have equal perturbation variance; no circular-noise calibration is supplied.

## References

Xu, J., Vaeggemose, M., Schulte, R. F., Yang, B., Lee, C.-Y., Laustsen, C., and Magnotta, V. A. (2024). PyAMARES, an Open-Source Python Library for Fitting Magnetic Resonance Spectroscopy Data. *Diagnostics*, 14, 2668. [Published method and benchmark study](https://doi.org/10.3390/diagnostics14232668).

Ren et al. (2015). Prior resonance model used for the brain-like phosphorus mixture. *NMR in Biomedicine*, 28, 1455–1462. [Resonance-prior source](https://doi.org/10.1002/nbm.3384).
