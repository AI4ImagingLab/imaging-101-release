# Optical Quantum State and Wigner Field from Homodyne Measurements

> Reconstruct a complex optical density matrix and its signed two-dimensional Wigner field from calibrated single-shot homodyne measurements, where noise, phase distortion and finite photon-number support complicate inversion.
> Domain: physics | Keywords: quantum imaging, homodyne tomography, statistical inversion | Difficulty: Hard

## Background

Balanced homodyne detection combines a weak optical signal with a stronger phase-controlled local oscillator. Their interference measures a field quadrature: a phase-dependent projection of the optical mode's canonical position and momentum. Repeating this measurement at different phases provides complementary views of the optical quantum state. Individual shots fluctuate even for a perfectly stable source, so a single field amplitude cannot describe the observations.

A density matrix describes both photon populations and coherence between photon-number states. Its Wigner representation maps the same information into a two-dimensional canonical phase space. This field can have negative values, which carry information about nonclassical behavior. Such characterization supports photonic quantum information, non-Gaussian optical sources and the calibration of quantum sensors. Finite samples, detection noise, phase drift and restrictions on the represented photon numbers make recovery statistically and numerically challenging.

## Problem Description

The observations come from a real integrated-photonic optical experiment conditioned on three photon-number detectors each reporting three photons. The requested state is the effective state inferred at the homodyne detector, with the stated confidence selection; do not introduce an additional unprovided loss correction. Represent its density matrix in the photon-number basis from zero through thirty. The matrix element is the bra-ket expectation between the corresponding number states. The density must be Hermitian, positive semidefinite and have unit trace.

Let a measured canonical quadrature be the dimensionless real number $q$, and let the local-oscillator phase be $\theta$ in radians. For photon index $n$, define

```math
\psi_n(q)=\frac{H_n(q)e^{-q^2/2}}{\pi^{1/4}\sqrt{2^n n!}},\qquad
A_n(q,\theta)=e^{in\theta}\psi_n(q),\qquad
p(q\mid\theta,\rho)=\mathbf A^\dagger\rho\mathbf A.
```

Here $H_n$ is the physicists' Hermite polynomial, defined by $H_0(q)=1$, $H_1(q)=2q$ and $H_{n+1}(q)=2qH_n(q)-2nH_{n-1}(q)$. The dagger denotes complex conjugate transpose; $\rho$ is the unknown density matrix. The function $p$ is a continuous probability density in $q$. Samples belong to different phases and are not noiseless projections. The canonical commutator is $[Q,P]=i$, so vacuum quadrature variance is one half.

Each shot supplies a digitizer value $s$ and an acquisition-file index $f$. With the supplied file mean $\mu_f$ and vacuum variances $V_{f,j}$ over the sixty-four phase settings, the quadrature calibration is

```math
q=\frac{s-\mu_f}{\sqrt{2\overline V_f}},\qquad
\overline V_f=\frac{1}{64}\sum_{j=0}^{63}V_{f,j}.
```

The raw phase is its stored phase index times the corresponding file's phase step. Its calibrated phase follows $\theta=\theta_0-b_f+a_f\sin(\theta_0-c_f)$, where $\theta_0$ is the raw phase and $b_f,a_f,c_f$ are nuisance phase-calibration parameters determined from the public variance scans. Determine these nuisance parameters by unweighted least squares of0.5times signal_variance/vacuum_variance to the variance model in Method Hints. Use the canonical calibration branch with the lower variance extremum u<=v, offset b in[-pi/2,pi/2), distortion |a|<1(monotonic phase scan), and c in[-pi/2,pi/2) with signed a. These fix equivalent phase-parameter branches, without prescribing a fitting algorithm. Apply the strict confidence selection in the acquisition table to define the reconstructed conditional ensemble. All accepted quadratures enter the likelihood, including measurements outside the displayed Wigner window.

The signed phase-space field is the canonical Wigner transform of this same density:

```math
W(Q,P)=\frac{1}{\pi}\int_{-\infty}^{\infty}
e^{2iPy}\langle Q-y|\rho|Q+y\rangle\,dy.
```

Here $Q,P$ denote dimensionless canonical coordinates, $y$ is the integration variable and the kets denote position eigenstates. The transform integrates to one on the full plane. Its finite output window must not be independently renormalized, clipped to positive values or rotated to fit a reference.

Save one file, `output/reconstruction.npz`, relative to the task workspace. It must contain exactly these arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `density_matrix` | `(1, 31, 31)` | complex64 or complex128 | Dimensionless matrix; axes batch, bra photon index, ket photon index; indices run from 0 through 30. |
| `wigner` | `(1, 201, 201)` | float32 or float64 | Real signed canonical phase-space density; axes batch, P, Q. Both axes contain 201 uniformly spaced coordinates from -7 to 7, inclusive. |

Evaluation requires a Hermitian positive-semidefinite density with trace one and a Wigner field consistent with that density. The mean log Born probability density over every accepted calibrated shot must be at least the independently validated standard likelihood minus1e-8 nats/shot; better likelihood passes. No density-matrix or Wigner-array agreement to a particular finite optimizer iterate is required. The trace and Hermiticity tolerances are1e-6, minimum eigenvalue at least?1e-6, and maximum Wigner-density discrepancy at most1e-6. No phase-space rotation, gain fitting or cropping is performed.

The conditional maximum-likelihood target maximizes the arithmetic mean of log p(q|theta,rho) over the confidence-selected shots described below, subject to rho being Hermitian positive semidefinite with trace one and photon-number cutoff30. This finite-dimensional convex state inference fixes the statistical target without fixing a positivity-preserving iteration history.

## Data Description

`data/raw_data.npz` contains existing real measurements and their acquisition calibration, stored without Python objects. The leading axis is a single experiment.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `adc` | `(1, 3756232)` | int16 | Single-shot balanced-homodyne readings, in digitizer units. |
| `pnr_distance` | `(1, 3756232, 3)` | uint16 | Confidence distances for the three heralding detectors, in stored classifier-distance units; lower is more confident. |
| `phase_index` | `(1, 3756232)` | int16 | Phase-setting indices from 0 through 63, dimensionless. |
| `file_index` | `(1, 3756232)` | int16 | Index of the corresponding calibration file, from 0 through 99, dimensionless. |
| `quadrature_mean` | `(1, 100)` | float64 | File-specific homodyne offset, in digitizer units. |
| `phase_step` | `(1, 100)` | float64 | File-specific phase increment, in radians per index. |
| `vacuum_variance` | `(1, 100, 64)` | float64 | Independent vacuum calibration at each phase, in squared digitizer units. |
| `signal_variance` | `(1, 100, 64)` | float64 | Unconditional signal variance scan used for phase calibration, in squared digitizer units. |
| `variance_phase` | `(1, 100, 64)` | float64 | Raw angles at which the variance calibrations were measured, in radians. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Measurement type | Balanced homodyne, single optical mode | — | Each shot measures one canonical quadrature. |
| Heralding condition | 3, 3, 3 | photons | Reported outcomes of the three heralding detectors. |
| Acquisition blocks | 100, chronological | — | Each block has its own public offset and variance calibration. |
| Raw phase settings per block | 64 | — | Nominal scan covers one full two-pi phase period. |
| Raw phase step | pi/32 | rad | The exact stored float64 values in `phase_step` are authoritative. |
| Canonical scale | hbar = 1; vacuum variance = 0.5 | — | Fixes both quadrature and Wigner normalization. |
| Detector-1 confidence cutoff | 5554.373134328358 | classifier-distance unit | Keep only shots with strictly smaller first distance. |
| Detector-2 confidence cutoff | 5811.4427860696524 | classifier-distance unit | Keep only shots with strictly smaller second distance. |
| Detector-3 confidence cutoff | 7281.3880597014922 | classifier-distance unit | Keep only shots with strictly smaller third distance. |
| Joint confidence selection | All three inequalities | — | Defines 1,881,028 accepted samples in this experiment. |
| Additional detector-loss inversion | None | — | Estimate the effective measured state in the specified phase frame. |
| Wigner axis convention | Columns Q, rows P; increasing coordinates | — | Positive phase rotates the measured quadrature toward positive canonical momentum. |

The classifier cutoffs are fixed acquisition-selection values obtained from the public distance observations. Vacuum and unconditional variance scans are independent calibration channels, not a reconstructed conditional state. The supplied arrays suffice to repeat calibration and confidence selection. There is no synthetic observation generator.

The authors publicly offer these research data. A separate raw-data redistribution license was not identified; this benchmark is a local research package and makes no broader redistribution claim. The implementation's code license does not silently replace the data terms.

## Method Hints

Estimate each block's phase distortion using the normalized variance scan. With raw angle $x$, its model is $V(x)=u\cos^2\theta(x)+v\sin^2\theta(x)$, where $u,v$ are the two fitted variance extrema and $\theta(x)=x-b+a\sin(x-c)$. The public vacuum scan supplies the quadrature scale. The conditional state can then be estimated directly from individual calibrated shots without histogramming them for inversion.

A positive maximum-likelihood method starts from a full-rank normalized density and uses the measurement projectors to build an operator $R$. A useful update family is

```math
R(\rho)=\sum_k\frac{\mathbf A_k\mathbf A_k^\dagger}
{\mathbf A_k^\dagger\rho\mathbf A_k},\qquad
\rho^{\mathrm{new}}=\frac{R\rho R}{\mathrm{Tr}(R\rho R)}.
```

The index $k$ runs over accepted individual shots, and $\mathbf A_k$ is the Hermite-amplitude vector at that shot's quadrature and phase. This update preserves positivity and trace normalization. Photon-number truncation limits the representation; convergence and sensitivity should be examined rather than inferred from the existence of a finite iterate. Compute the Wigner field from the recovered full complex matrix, including off-diagonal coherence, using canonical oscillator basis functions or an equivalent explicit recurrence.

## References

1. *Integrated photonic source of Gottesman–Kitaev–Preskill qubits*. Nature 642, 587–591 (2025). [Published article](https://doi.org/10.1038/s41586-025-09044-5).
2. A. I. Lvovsky. *Iterative maximum-likelihood reconstruction in quantum homodyne tomography*. Journal of Optics B 6, S556–S559 (2004). [Published method](https://doi.org/10.1088/1464-4266/6/6/014).
