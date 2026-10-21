# Pulsed EPR Molecular Distance Distribution

> Recover a molecular spin-distance probability distribution and nuisance physical parameters from measured complex four-pulse EPR echoes with intermolecular background decay and missing distance resolution.
> Domain: chemistry & material science | Keywords: spin spectroscopy, molecular structure, nonlinear inversion | Difficulty: Hard

## Background

Pulsed electron paramagnetic resonance probes magnetic dipole interactions between unpaired electron spins. The interaction frequency varies as the inverse cube of their separation, allowing nanometre distances to be inferred from a time-domain echo without directly resolving the atoms in an image.

Spin labeling makes this approach useful for molecular conformation, soft materials and chemical structure studies. A powder average mixes molecular orientations, while interactions between different molecules create a smooth decaying background. Finite observation time and noise make broad or overlapping distance populations difficult to distinguish.

## Problem Description

For evolution time t in microseconds and spin separation r in nm, the orientation-averaged intramolecular kernel is
```math
K(t,r)=\int_0^1\cos\!\left[\frac{D\,10^{21}}{r^3}|t-t_0|(1-3u^2)\right]du,
\qquad V(t)=B(t)\int[(1-\lambda)+\lambda K(t,r)]P(r)dr.
```
The orientation variable u is the cosine of the angle between spin separation and the magnetic field, D is the electron dipolar constant in rad·m³/s, t0 is refocusing time, lambda is modulation depth, P is the normalized distance density (1/nm), and B is homogeneous intermolecular decay. Instrument quadrature data have an unknown overall phase and amplitude. B(t)=exp[-κ lambda c D |t-t0| 10^-6], where κ=8π²/(9√3) and c is spin number density in m^-3. A concentration C in micromol/L gives c=C·10^-3·NA, with NA the Avogadro constant. The model assumes randomly oriented spin pairs, identical relaxation rates and no orientation selection or hyperfine modulation.

The complete bound observation contains 263 complex echoes. The requested model has two Gaussian distance populations and estimates both background concentration and refocusing time; limited evolution time cannot uniquely establish arbitrary substructure.

Save `output/reconstruction.npz` relative to your task workspace. Required arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `reconstruction` | `(1, 450)` | `float64` | Normalized real nonnegative P(r) in 1/nm, axis (batch, distance node), trapezoidal integral one. |
| `fitted_echo` | `(1, 263)` | `float64` | Real predicted phase-corrected peak-normalized echo, dimensionless, axis (batch, time). |
| `parameters` | `(1, 9)` | `float64` | Axis (batch, parameter), ordered [a1,a2,C,mu1,mu2,lambda,t0,sigma1,sigma2]. a1,a2 dimensionless Gaussian amplitudes in the upstream common-basis convention; C micromol/L; mu1,mu2 and sigma1,sigma2 nm; lambda dimensionless; t0 microsecond. Label populations by increasing mean mu1 <= mu2. |

Evaluation compares every required physical quantity with a validated standard-method result using relative Euclidean error, norm(estimate-reference)/norm(reference). Complex arrays retain both quadratures under the Hermitian norm. Mixed-unit parameter components are scored separately; every declared error must be at most 1e-5. No cropping, scale fitting, phase rotation or translation is performed. This is reproducibility against a standard method, not a claim of directly measured object truth.

## Data Description

The input is `data/raw_data.npz`; all arrays include batch size one.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `time` | `(1, 263)` | `float64` | Native recorded time coordinates in microsecond before origin correction. |
| `echo` | `(1, 263)` | `complex128` | Measured in-phase plus i quadrature echo in instrument arbitrary units. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| First interpulse delay | 0.3 | microsecond | Nominal four-pulse timing |
| Second interpulse delay | 4.0 | microsecond | Nominal four-pulse timing |
| Pulse length | 0.016 | microsecond | Refocusing-time admissible uncertainty is three pulse lengths |
| Evolution time | time - time[0] + 0.1 | microsecond | Declared origin convention |
| Distance output grid | 1.5 + 0.01 j, j=0..449 | nm | Trapezoidal probability density convention |
| mu0 | 1.25663706212e-6 | N/A² | CODATA 2018 magnetic constant |
| Bohr magneton | 9.2740100783e-24 | J/T | CODATA 2018 value |
| Electron g | 2.00231930436256 | 1 | Identical free-electron g factors |
| Planck constant | 6.62607015e-34 | J s | hbar = h/(2π) |
| Avogadro constant | 6.02214076e23 | 1/mol | Concentration conversion |
| Dipolar constant | mu0*(g*muB)^2/(2 h) | rad m³/s | D in the forward equation |
| Noise | Uncalibrated quadrature noise | instrument unit | No known variance supplied |

## Method Hints

Use separable nonlinear least squares: at each proposed distance means, widths, modulation depth, concentration and refocusing time, solve a two-column nonnegative linear amplitude problem. A two-Gaussian density imposes a compact molecular model; the powder kernel can be evaluated with Fresnel integrals.

## References

Fábregas Ibáñez, Jeschke and Stoll, DeerLab: a comprehensive software package for analyzing dipolar electron paramagnetic resonance spectroscopy data, Magnetic Resonance 1 (2020), 209–224. https://doi.org/10.5194/mr-1-209-2020


