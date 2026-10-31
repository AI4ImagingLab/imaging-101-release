# BBO Photon Polarization Density

> Recover the complex two-photon polarization density matrix from measured analyzer counts while enforcing a positive quantum state despite inconsistent experimental projections.
> Domain: physics | Keywords: quantum tomography, polarization imaging, statistical inversion | Difficulty: Hard

## Background

Spontaneous parametric downconversion converts one pump photon into a correlated pair of lower-energy photons. Polarization measurements on many independently generated pairs reveal quantum coherences that cannot be determined from a single horizontal/vertical intensity measurement. A density matrix represents both coherent superpositions and statistical mixtures of these states.

Quantum tomography characterizes entangled-light sources and tests components for quantum communication and information processing. It combines measurements in incompatible analyzer bases. Finite measurements and detector imperfections can make a direct linear reconstruction nonphysical; positivity, normalization and complex phase conventions are consequently essential.

## Problem Description

Reconstruct a two-photon polarization state from the complete set of 36 experimentally measured, author-corrected analyzer outcomes. The acquisition uses adjacent nonlinear crystals and four detectors. The recorded observations were corrected for accidental counts and detector efficiencies before publication; they are not unprocessed independent Poisson draws.

```math
\mu_j=s\operatorname{Re}\operatorname{tr}(P_j\rho),\qquad
P_j=|a_j b_j\rangle\langle a_j b_j|,
\qquad\rho=\rho^\dagger,\quad\rho\succeq0,\quad\operatorname{tr}\rho=1.
```

Here j indexes an analyzer pair, P is its known rank-one projector, rho is the unknown dimensionless complex 4-by-4 density matrix, s is an unknown common corrected-count scale, and mu is the predicted count. The ordered state basis is HH, HV, VH, VV; the first symbol denotes photon 1. The dagger denotes conjugate transpose. Noise, efficiency corrections and drift make the observed counts mutually inconsistent with an exact single state. The common-scale model does not assert that all remaining errors follow a calibrated Poisson law.

Save `output/reconstruction.npz` relative to the task workspace with both required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 4, 4) | complex64 or complex128 | Dimensionless unit-trace density matrix; axes batch, row state, column state; basis HH/HV/VH/VV |
| intensity | (1, 1) | float32 or float64 | Positive common corrected-count normalization s, in counts per analyzer exposure |

No phase conjugation, global phase, permutation, rescaling or alignment is applied during evaluation. The density is compared by complex Frobenius distance and intensity by relative absolute error to a standard-method solution. Evaluation also checks Hermitian symmetry, positive semidefiniteness, unit trace, and RMS count residual divided by s. The latter is an observation-consistency diagnostic, not proof of physical truth or shot-noise-limited accuracy.

## Data Description

`data/raw_data.npz` contains:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| counts | (1, 36) | float64 | Published corrected experimental counts, in counts per analyzer exposure; analyzer pairs flattened row-major |
| projectors | (1, 36, 4, 4) | complex128 | Dimensionless known polarization projectors in the declared state basis |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Pump wavelength | 351 | nm | Pump illuminating adjacent BBO crystals |
| Downconverted wavelength | 702 | nm | Polarization-entangled photon pairs |
| Filter bandwidth | 10 | nm | Detection optical filter shown in experimental setup |
| Detectors | 4 | count | Two outcomes per photon arm |
| Joint analyzer settings | 9 | count | Cartesian products of three orthogonal polarization bases |
| Outcomes | 36 | count | Four outcomes per joint setting |
| Analyzer order | H, V, D, A, R, L | dimensionless | Each arm; first arm outer index, second arm inner index |
| Corrections | Accidental subtraction; detector-efficiency normalization | dimensionless | Already applied to the published counts; do not apply again |
| Exposure | Common relative exposure 1; absolute duration unavailable | relative | Overall intensity is inferred, not externally fixed |
| Noise | Residual measurement/correction uncertainty, no supplied covariance | counts | Pure Poisson error bars are only an approximation |
| Boundary conditions | Finite two-qubit Hilbert space; no spatial boundary | dimensionless | Four-dimensional state representation |

H=(1,0), V=(0,1), D=(H+V)/sqrt(2), A=(H−V)/sqrt(2), R=(H+iV)/sqrt(2), and L=(H−iV)/sqrt(2), where i is the imaginary unit. These vectors define the supplied projectors and fix the circular-polarization sign convention. No target-derived calibration is supplied.

## Method Hints

To specify the conditional statistical estimate, minimize sum over all 36 outcomes of (m_j-count_j)²/m_j over a positive common scale s and a positive-semidefinite, Hermitian, unit-trace rho, where m_j=max(s Re trace(P_j rho),0.01 count). The 0.01-count floor applies to both numerator and denominator. There is no sample-count normalization, background parameter or additional state regularizer. This Pearson-type objective is a selected fit criterion for corrected counts, rather than a claim that those counts have an exact Poisson likelihood.

Use constrained statistical quantum tomography. A lower-triangular complex matrix T gives a positive unnormalized state TT-dagger; dividing by its trace produces a physical density matrix while the trace sets the count scale. Fit the Born projections jointly. A Pearson-type residual weights the difference between observed and expected counts by the square root of expected counts. Linear inversion is useful for diagnosis but may produce negative eigenvalues. Incomplete analyzer sets can leave imaginary coherences unidentifiable.

## References

J. B. Altepeter, D. F. V. James and P. G. Kwiat, *Quantum State Tomography*, 2004 chapter manuscript, particularly the measured two-photon example and maximum-likelihood formulation. [Author-hosted chapter](https://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf).

D. F. V. James, P. G. Kwiat, W. J. Munro and A. G. White, *Measurement of qubits*, Physical Review A 64, 052312 (2001). [DOI](https://doi.org/10.1103/PhysRevA.64.052312).
