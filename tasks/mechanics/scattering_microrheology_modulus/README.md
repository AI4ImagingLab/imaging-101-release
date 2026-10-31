# Scattering Microrheology of a Measured Soft Material

> Recover three-dimensional tracer mean-squared displacement and complex shear modulus from three measured light-scattering correlation curves whose finite lag range and uncertain intercept amplify inverse errors.
> Domain: mechanics | Keywords: microrheology, light scattering, viscoelastic inversion | Difficulty: Hard

## Background

Thermal fluctuations move microscopic tracer particles inside a soft material. The surrounding medium resists their motion through viscous drag and elastic restoring forces. A laser scattered by many tracers produces fluctuating intensity; its temporal correlation encodes their displacement statistics without resolving individual particles.

The fluctuation-dissipation relation connects those passive displacements to mechanical response. This permits rheological characterization of small, optically accessible samples and fragile gels that can be difficult to load into a conventional rheometer. Applications include polymer networks, mucus, cell-containing materials, and formulations whose mechanics change over time.

The inference depends on tracer size, optical geometry, and the assumption that tracer motion samples a locally homogeneous continuum. Particle adhesion, multiple scattering, aging and incomplete equilibration can invalidate a bulk interpretation. A reference reconstruction quantifies repeatability of the selected estimator rather than independently certifying material properties.

## Problem Description

Analyze all three complete author-supplied repeat acquisitions of one author-labeled patient-derived soft-material sample. The public data are instrument-exported, author-processed intensity correlations, not camera images or unprocessed photon timestamps. The original sample's disease status is not established by these files and is not assumed.

Let t be lag in microseconds, c(t)=g2(t)-1 the dimensionless intensity correlation, beta its zero-lag intercept, q the scattering-vector magnitude in inverse nanometres, and M(t) the three-dimensional mean-squared displacement in square nanometres. For an ergodic sample with Gaussian isotropic displacement and single scattering,

```math
q=\frac{4\pi n}{\lambda}\sin(\theta/2),\qquad
g_1(t)=\exp[-q^2M(t)/6],\qquad
c(t)=\beta g_1(t)^2+\epsilon(t).
```

Here n is the medium refractive index, lambda the vacuum wavelength in nm, theta the scattering angle, g1 the field correlation, and epsilon measurement/model residual. The unknowns are M, its local log slope alpha=d log M/d log t, beta, and the storage/loss moduli G' and G''. For a local power-law displacement approximation, the generalized Stokes-Einstein relation is

```math
|G^*(\omega)|=\frac{k_B T\,10^{27}}{\pi a M(t)\Gamma[1+\alpha(t)]},\qquad
G'=|G^*|\cos(\pi\alpha/2),\quad
G''=|G^*|\sin(\pi\alpha/2),\quad \omega=10^6/t.
```

The tracer radius a is in nm, absolute temperature T in K, Boltzmann constant kB in J/K, Gamma is the gamma function, and both moduli are in Pa. The factor 10^27 converts cubic nanometres to cubic metres. G* means G'+iG'', using positive loss for a viscous material. A finite lag interval cannot uniquely determine an arbitrary relaxation kernel; local power-law continuation is an explicit approximation.

Use the declared reliable lag domain: discard native indices 0,1,2; stop before the first correlation below 0.05, or before the last native index if no such point exists. The resulting lengths are 188,145,155. Concatenate repeat 1 then 2 then 3, preserving increasing lag within each repeat; corresponding frequency decreases. Save `output/reconstruction.npz` relative to the task workspace with all required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,488,2) | float64 | Real modulus, Pa; axes batch, concatenated lag, component (storage G', loss G'') |
| msd | (1,488) | float64 | Real three-dimensional MSD, nm²; same concatenated-lag order |
| exponent | (1,488) | float64 | Real local displacement log slope alpha, dimensionless |
| intercept | (1,3) | float64 | Real zero-lag correlation beta, dimensionless; repeat order |

All values must be finite. No magnitude conversion, clipping, gain fitting, coordinate alignment, cropping or permutation is performed. Each modulus component, MSD and exponent is scored separately with uncentered cosine similarity NCC and reference-dynamic-range-normalized RMS error NRMSE. Intercepts use maximum relative error. Forward correlation consistency is the largest repeat RMS residual between beta exp(-q²M/3) and observed c over the stated lag domain. Every criterion must pass. The selected analysis may produce small negative storage values where the fitted local exponent exceeds one; preserve these signed estimates rather than clipping them.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| time_lag | (1,3,192) | float64 | Native lag samples, microseconds; axes batch, repeat, lag |
| correlation | (1,3,192) | float64 | Author-processed c=g2-1, dimensionless, same axes |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument | Malvern Zetasizer Nano ZS | - | Benchtop dynamic light scattering |
| Sample | Author supplied soft-material sample, three repeats | - | Disease status not inferred |
| Tracer diameter | 500 | nm | Author example measurement configuration |
| Temperature | 310.15 | K | Author example configuration, 37°C |
| Medium refractive index | 1.333 | dimensionless | Water convention |
| Vacuum wavelength | 633 | nm | Laser wavelength |
| Scattering angle | 173 | degree | Backscattering geometry |
| Boltzmann constant | 1.38e-23 | J/K | Original numerical convention |
| Model | Ergodic, Gaussian isotropic tracer displacement, continuum, single scattering | - | Conditional rheological interpretation |
| Optical boundary | Bulk scattering volume; no spatial boundary reconstruction | - | Finite observed lag domain |
| Observation processing | Native exported correlation, with vendor distribution-fit field-correlation values squared plus exported baseline at matching native lags | dimensionless | Original author import convention; no hidden-target calibration |
| Noise | Unknown correlation covariance | - | No independent Gaussian noise standard deviation supplied |
| Reliable lag rule | Indices 3 through before first c<0.05; otherwise before last index | zero-based index | Fixed public observation-derived domain |

The complete 192-point correlation curves are retained for all repeats. The vendor-processed correlation has been preserved exactly; no output modulus or MSD is included in the observations.

**Evaluation certification:** The current numerical evaluator is not yet certified as a method-independent public benchmark; standard-method agreement must not be interpreted as independent physical accuracy.

## Method Hints

A Siegert inversion converts field correlation into mean-squared displacement. Estimating the short-lag intercept with a stretched exponential avoids a systematic displacement offset. Cross-validation over short fitting windows can stabilize this extrapolation. Gaussian-weighted log-linear regression then estimates the displacement and its local power-law slope, which enter the generalized Stokes-Einstein relation. Direct Laplace continuation is a useful comparison but can have strong finite-window bias at both frequency extremes.

## References

- Cai et al. (2021), *Dynamic light scattering microrheology for soft and living materials*, Soft Matter 17, 1929–1939. [Article](https://doi.org/10.1039/D0SM01597K).
- Mason (2000), *Estimating the viscoelastic moduli of complex fluids using the generalized Stokes–Einstein equation*, Rheologica Acta 39, 371–378. [Article](https://doi.org/10.1007/s003970000094).
