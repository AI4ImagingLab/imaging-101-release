# Experimental Structural Modal Reconstruction

> Recover seven damped structural modes, complex spatial residues and residual terms from six measured accelerance responses with overlapping resonances and finite frequency coverage.
> Domain: mechanics | Keywords: structural dynamics, system identification, vibration imaging | Difficulty: Hard

## Background

A mechanical structure responds to applied forces through resonant modes with characteristic frequencies, damping and spatial motion patterns. Measuring acceleration divided by input force across frequencies produces complex accelerance responses that retain both amplitude and phase.

Modal identification supports vibration control, fatigue analysis and structural condition assessment. A finite experimental frequency band contains contributions from resonances outside that band as well as measurement noise. Separating these contributions and estimating small damping ratios requires a shared dynamic model across measurement locations.

## Problem Description

For angular frequency omega=2πf, the complex accelerance at roving excitation index l is modeled as
```math
H_l(\omega)=-\omega^2\sum_{k=1}^{7}\left[\frac{A_{lk}}{\mathrm{i}\omega-p_k}+\frac{\overline{A_{lk}}}{\mathrm{i}\omega-\overline{p_k}}\right]+L_l-\omega^2U_l.
```
Here f is frequency in Hz, p is a stable modal pole in s^-1, A is a complex residue in m/(N·s), the overbar is complex conjugation, L is the lower residual in m/(N·s²), and U is the upper residual in m/N. Natural frequency is abs(p)/(2π) Hz, and damping ratio is -Re(p)/abs(p). L and U represent contributions outside the fitted frequency band. Real physical impulse responses give conjugate pole pairs.

The bound observation consists of all 9999 frequency bins for each of six roving excitations at the fixed response index selected in the author showcase. Identify seven modes within the 10–5000 Hz analysis band and their complete six-location complex residues. Pole-residue factorization is sensitive to noise, polynomial order and out-of-band contributions; no measured spatial distances are supplied.

Save `output/reconstruction.npz` relative to your task workspace. Required arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `natural_frequency` | `(1, 7)` | `float64` | Seven positive natural frequencies in Hz, ascending mode order; axes (batch, mode). |
| `damping_ratio` | `(1, 7)` | `float64` | Seven positive dimensionless damping ratios; axes (batch, mode). |
| `reconstruction` | `(1, 6, 7)` | `complex128` | Complete complex modal residues A in m/(N s), axes (batch, roving excitation, mode). |
| `fitted_accelerance` | `(1, 6, 4999)` | `complex128` | Complete complex fitted accelerance in m/(N s²), axes (batch, roving excitation, declared 4999-bin output frequency). |
| `lower_residual` | `(1, 6)` | `complex128` | Complex L in m/(N s²), axes (batch, roving excitation). |
| `upper_residual` | `(1, 6)` | `complex128` | Complex U in m/N, axes (batch, roving excitation). |

Evaluation compares every required physical quantity with a validated standard-method result using relative Euclidean error, norm(estimate-reference)/norm(reference). Complex arrays retain both quadratures under the Hermitian norm. Mixed-unit parameter components are scored separately; every declared error must be at most 1e-5. No cropping, scale fitting, phase rotation or translation is performed. This is reproducibility against a standard method, not a claim of directly measured object truth.

## Data Description

The input is `data/raw_data.npz`; all arrays include batch size one.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `frequency` | `(1, 9999)` | `float64` | Native measured frequency samples in Hz, ascending order. |
| `accelerance` | `(1, 6, 9999)` | `complex128` | Complex measured acceleration/force in m/(N s²); axes (batch, roving excitation, frequency). |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Input frequency convention | Cycles per second | Hz | Use omega=2πf |
| Analysis band | 10 to 5000 | Hz | Closest stored bins; upper cutoff is exclusive before fitting |
| Transfer quantity | Accelerance | m/(N s²) | Measured acceleration divided by force |
| Time convention | Laplace s=i omega | 1 | Stable poles have negative real part |
| Response reference | Index 1 in original seven-response author dataset | 1 | Entire six-excitation acquisition for that fixed response |
| Spatial axis | Roving excitation index 0..5 | 1 | No physical coordinate scale is supplied |
| Model size | Seven conjugate-paired flexible modes | 1 | All seven mode quantities are required |
| Fitted output grid | Prepend zero frequency, retain native bins before closest 5000 Hz bin | Hz | 4999 entries; zero-padding convention is explicit |
| Noise covariance | Not supplied | 1 | Unweighted frequency-domain model |

## Method Hints

Use frequency-domain least-squares pole identification with a stabilization diagram across polynomial orders, followed by least-squares fitting of complex conjugate residues and both residual terms. Jointly using the six responses stabilizes shared poles while preserving spatial mode information.

## References

Peeters et al., The PolyMAX frequency-domain method: a new standard for modal parameter estimation?, Shock and Vibration 11 (2004), 395–409. https://doi.org/10.1155/2004/523692

Data and executable method source are the author-maintained experimental/operational modal-analysis package and its measured accelerance showcase; this is a modern numerical port, not the original commercial software implementation.
