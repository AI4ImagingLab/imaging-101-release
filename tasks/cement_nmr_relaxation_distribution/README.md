# Cement Proton-NMR Relaxation Spectral Evolution

> Recover all 120 nonnegative proton relaxation spectra of cement paste from measured complex echo trains whose exponential kernels strongly blur neighboring water environments.
> Domain: chemistry & material science | Keywords: magnetic resonance, porous materials, spectral inversion | Difficulty: Hard

## Background

Proton nuclear magnetic resonance measures the decay of water magnetization in porous materials. Water in different environments relaxes at different rates because of confinement, surface interactions and molecular mobility. Echo measurements can therefore reveal changes in cement pore water while avoiding destructive drying or intrusion.

A relaxation spectrum describes how much detected signal belongs to each transverse relaxation time. Recovering this spectrum is an inverse Laplace problem: nearby decay rates produce similar signals, weak populations overlap and noise can create unstable spectral oscillations. Relaxation times alone do not identify a unique pore size without independent surface-relaxivity calibration.

## Problem Description

For acquisition index s and echo time t in seconds, the phase-corrected signal obeys
```math
y_s(t)=\sum_{j=0}^{254} a_{sj}\exp(-t/T_j)+\epsilon_s(t),\qquad a_{sj}\geq0.
```
Here a is discrete echo amplitude in instrument arbitrary units, T is transverse relaxation time (s), y is the real coherent signal after constant quadrature-phase correction, and epsilon is measurement/model error. T[j]=10^(-5+5j/254) s. The amplitudes are discrete bin coefficients, not a density per second or per logarithmic interval. No constant offset is included in the complex-acquisition model.

Recover the full 120-acquisition series. Each stored trace has 256 measured complex echoes; discard only the first echo after estimating each acquisition's phase from echoes 1 through 4. The inverse problem is ill-conditioned even when the exponential model is exact. Acquisition order is known; hydration ages, pore radii and specimen recipe are not inferred from missing metadata.

Save `output/reconstruction.npz` relative to your task workspace. Required arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `reconstruction` | `(1, 120, 255)` | `float64` | Nonnegative discrete amplitudes in instrument arbitrary units on the declared T grid; axes (batch, acquisition, relaxation node). No per-scan normalization. |
| `fitted_echo` | `(1, 120, 255)` | `float64` | Real predicted coherent echoes in instrument arbitrary units after first-echo removal; axes (batch, acquisition, retained echo). |

Evaluation compares every required physical quantity with a validated standard-method result using relative Euclidean error, norm(estimate-reference)/norm(reference). Complex arrays retain both quadratures under the Hermitian norm. Mixed-unit parameter components are scored separately; every declared error must be at most 1e-5. No cropping, scale fitting, phase rotation or translation is performed. This is reproducibility against a standard method, not a claim of directly measured object truth.

## Data Description

The input is `data/raw_data.npz`; all arrays include batch size one.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `echo_time` | `(1, 256)` | `float64` | Native echo-center times in seconds, shared by all acquisitions. |
| `echo` | `(1, 120, 256)` | `complex128` | Measured real+i imaginary magnetization amplitude in instrument arbitrary units; axes (batch, acquisition, echo). |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Echo times | Stored individually, 256 values | s | Native irregular logarithmic echo schedule |
| Quadrature convention | real + i imaginary | a.u. | Coherent transverse magnetization signal |
| Phase estimate | Mean of four angles at zero-based echoes 1,2,3,4 | rad | Use atan(Im/Re), add π when Re<0 |
| First echo treatment | Discard zero-based echo 0 after phasing | 1 | Preserves author treatment of initial echo artifact |
| Relaxation grid | 10^(-5+5j/254), j=0..254 | s | Discrete exponential model required for output interpretation |
| Scans | 120 | 1 | Complete OPC example, filename-sorted acquisition order |
| Offset model | 0 | a.u. | Complex-acquisition branch has no magnitude offset |
| Noise variance | Not supplied | a.u.² | No known Gaussian uncertainty is asserted |
| Boundary | No contribution outside relaxation grid | 1 | Finite-grid inversion assumption |

## Method Hints

Use nonnegative Tikhonov-regularized inverse Laplace fitting on a logarithmic relaxation grid. The identity regularizer stabilizes amplitudes that fit nearly the same echo decay; compare individual-scan inversion with inversion after coherent averaging to assess noise and nonlinearity.

## References

Georget et al., Reconciliation of pore structure characterization methods: The simple case of PC-limestone cement pastes, Cement and Concrete Research 184 (2024), 107624. https://doi.org/10.1016/j.cemconres.2024.107624

The bound observation is the independent author-deposited OPC example series, not claimed to be a particular specimen or result figure from this paper.
