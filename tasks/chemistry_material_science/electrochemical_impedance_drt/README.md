# Lithium-Ion Cell Impedance Relaxation Spectrum

> Recover a nonnegative electrochemical relaxation spectrum, series resistance and inductance from measured lithium-ion-cell impedance whose overlapping processes are poorly resolved in frequency.
> Domain: chemistry & material science | Keywords: electrochemical spectroscopy, spectral inversion, regularization | Difficulty: Hard

## Background

Electrochemical impedance spectroscopy measures the small-signal voltage response to an alternating current. Charge transfer, double-layer charging and transport contribute on different time scales, so a frequency sweep probes processes that a single resistance measurement cannot separate.

Relaxation spectra are useful for studying batteries, fuel cells and electrode degradation. The measured response combines several physical processes and inductive wiring effects; interpreting their overlap requires a stable inverse model rather than assigning each visible arc to one mechanism.

## Problem Description

Let Z(f) be complex impedance in ohms at positive frequency f in Hz. The passive relaxation model is
```math
Z(f)=R+\mathrm{i}2\pi f L+\int_{\ln\tau_{\min}}^{\ln\tau_{\max}}\frac{\gamma(\tau)}{1+\mathrm{i}2\pi f\tau}\,d\ln\tau+\epsilon(f).
```
Here R is series resistance (ohm), L is series inductance (H), tau is relaxation time (s), gamma is a nonnegative resistance density per logarithmic time interval (ohm), i is the imaginary unit, and epsilon is measurement/model error. Finite bandwidth and similar responses from nearby relaxation times make the inversion ill-conditioned.

The acquisition is the complete 107-point author-deposited lithium-ion-cell spectrum. Recover the sampled relaxation spectrum and both series quantities on the supplied physical scale. No object truth or electrochemical reaction assignment is assumed.

The finite-bandwidth spectral estimate is defined by the nonnegative regularized inverse below, so the requested spectrum has a specified meaning despite ill conditioning. Set $u_j=\ln(1/f_j)$, $\tau_j=1/f_j$, with trapezoidal weights $w_0=(u_1-u_0)/2$, $w_{N-1}=(u_{N-1}-u_{N-2})/2$, and $w_j=(u_{j+1}-u_{j-1})/2$ at interior nodes. Define $A_{ij}=w_j/(1+\mathrm{i}2\pi f_i\tau_j)$, $B=[\mathrm{i}2\pi f,\mathbf{1},A]$, and $x=[L,R,\gamma_0,\ldots,\gamma_{N-1}]$. The output is the minimizer

```math
\underset{x\geq0}{\operatorname{argmin}}\;
\sum_i |(Bx)_i-Z_i|^2+
10^{-3}\sum_{j=0}^{N-2}\left(\frac{\gamma_{j+1}-\gamma_j}{u_{j+1}-u_j}\right)^2.
```

Here N=107, both quadratures contribute to the complex squared modulus, neither sum is divided by its sample count, and the series parameters are unpenalized. Report fitted impedance as Bx. The coefficient is fixed for these physical units; normalizing the impedance or inductance column requires the corresponding change of variables in this objective.

Save `output/reconstruction.npz` relative to your task workspace. Required arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `reconstruction` | `(1, 107)` | `float64` | Nonnegative gamma on tau=1/frequency, axis (batch, relaxation node), units ohm. |
| `series_parameters` | `(1, 2)` | `float64` | Axis (batch, parameter), ordered [inductance L in H, resistance R in ohm], both nonnegative. |
| `fitted_impedance` | `(1, 107)` | `complex128` | Complex predicted impedance in ohm, axis (batch, frequency), same order and +i time convention as input. |

Evaluation compares every required physical quantity with a validated standard-method result using relative Euclidean error, norm(estimate-reference)/norm(reference). Complex arrays retain both quadratures under the Hermitian norm. Mixed-unit parameter components are scored separately; every declared error must be at most 1e-5. No cropping, scale fitting, phase rotation or translation is performed. This is reproducibility against a standard method, not a claim of directly measured object truth.

## Data Description

The input is `data/raw_data.npz`; all arrays include batch size one.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `frequency` | `(1, 107)` | `float64` | Measured frequency in Hz, native descending acquisition order. |
| `impedance` | `(1, 107)` | `complex128` | Complex voltage/current ratio in ohm, real and imaginary components retained. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Time convention | exp(+i 2π f t) | 1 | Passive capacitive response has negative imaginary impedance |
| Relaxation grid | tau[j] = 1 / frequency[j] | s | Ascending logarithmic nodes; preserve all 107 stored frequencies |
| Quadrature | Trapezoidal in ln(tau) | 1 | Half-width endpoint weights |
| Acquisition extent | 0.00500083202 to 999.040405 | Hz | Native measured frequencies; not rounded for computation |
| Input noise | Not calibrated | ohm | No Gaussian noise variance supplied |
| Boundary | Zero spectrum outside declared time interval | 1 | Finite-bandwidth inverse model |

## Method Hints

Use nonnegative Tikhonov inversion with piecewise-linear spectral discretization. Penalizing the first difference with respect to logarithmic relaxation time suppresses unstable oscillations while retaining separate real and imaginary observations; include the unpenalized series resistance and inductance.

## References

Saccoccio et al., Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods, Electrochimica Acta 147 (2014), 470–482. https://doi.org/10.1016/j.electacta.2014.09.058

Wan et al., Influence of the discretization methods on the distribution of relaxation times deconvolution, Electrochimica Acta 184 (2015), 483–499. https://doi.org/10.1016/j.electacta.2015.09.097
