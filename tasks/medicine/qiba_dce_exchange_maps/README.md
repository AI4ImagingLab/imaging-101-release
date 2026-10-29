# Dynamic Contrast MRI Exchange Maps

> Reconstruct contrast-transfer and extracellular-volume maps from a noisy dynamic MR signal series, where signal nonlinearity and weak washout make kinetic parameters difficult to separate.
> Domain: medicine | Keywords: tracer kinetics, quantitative MRI, compartment imaging | Difficulty: Hard

## Background

Dynamic contrast MRI follows the passage of a paramagnetic contrast agent through tissue. The agent changes the longitudinal relaxation rate of water, which changes the signal of a repeated gradient-echo acquisition. The resulting time series contains information about exchange between blood plasma and the extracellular space.

Quantitative exchange maps help characterize vascular permeability and the tissue space accessible to contrast agent. Their interpretation depends on the supplied plasma input, calibration, sampling interval and compartment assumptions. Noise can create very different parameter estimates even when the fitted concentration curves look similar.

## Problem Description

The observations are an existing noisy digital reference phantom, with a complete 60-row by 50-column raster. Reconstruct the spatial transfer constant and extracellular volume fraction. Every raster position belongs to the evaluation support.

Let $S(t)$ denote the measured MR signal at one raster position, $S_b$ the mean of its first two signal samples, $R_0$ its supplied precontrast relaxation rate, $T_R$ the repetition time, and $\theta$ the excitation flip angle. The noise-free spoiled-gradient-echo model, after eliminating the unknown common signal scale, is given below; the recorded ratio also contains magnitude noise and noise in the measured baseline.

```math
\frac{S(t)}{S_b}=
\frac{1-e^{-T_R R(t)}}{1-e^{-T_R R(t)}\cos\theta}
\frac{1-e^{-T_R R_0}\cos\theta}{1-e^{-T_R R_0}},
\qquad R(t)=R_0+r_1 C(t).
```

Here $R(t)$ is the effective longitudinal relaxation rate, $r_1$ is the known relaxivity, and $C(t)$ is the tissue contrast concentration. For the standard exchange model,

```math
C(t)=K\int_0^t C_p(\tau)
\exp\!\left[-\frac{K}{v}(t-\tau)\right]d\tau .
```

$K$ is the transfer constant, $v$ is the extracellular volume fraction, $C_p(t)$ is the supplied plasma concentration, and $\tau$ is integration time. Use a consistent time unit inside this equation. The vascular-volume contribution is fixed to zero in this standard model. The unknown maps are $K$ and $v$; their ratio is the exchange efflux rate. Finite temporal sampling, magnitude noise and uncertain baseline normalization make this inverse problem nonlinear and poorly conditioned in some regions.

Save exactly these two real arrays in `output/reconstruction.npz`, relative to your task workspace:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `ktrans` | `(1, 60, 50)` | float32 or float64 | Transfer constant $K$, in min⁻¹; batch, native row, native column; finite values in[0,5]. |
| `extracellular_fraction` | `(1, 60, 50)` | float32 or float64 | Extracellular fraction $v$, dimensionless; batch, native row, native column; finite values in[0,1]. |

Preserve the entire raster and its orientation. Both maps are evaluated against independently specified digital-phantom parameters using cosine similarity without mean subtraction and RMS error divided by the parameter map's dynamic range. Every required metric must pass. No alignment, fitted scale, crop or output-dependent mask is applied. A zero estimate has cosine similarity zero. Reconstruction agreement does not establish performance on measured patient data.

## Data Description

`data/raw_data.npz` contains only the following visible observations and calibration arrays. All arrays include the leading batch dimension.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `signal` | `(1, 1321, 60, 50)` | float64 | Existing noisy magnitude MR signal in arbitrary signal units; batch, time, row, column. |
| `relaxation_rate` | `(1, 60, 50)` | float64 | Known precontrast longitudinal relaxation calibration $R_0$, in s⁻¹; batch, row, column. |
| `time_seconds` | `(1, 1321)` | float64 | Native acquisition time samples, in seconds. |
| `plasma_concentration` | `(1, 1321)` | float64 | Supplied plasma contrast concentration $C_p$, in mM; already corrected to plasma concentration. Preserve its stored precision, including roundoff near zero. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Dataset type | Existing noisy QIBA version6 digital reference phantom | — | Author-provided simulated observations, not patient measurements. |
| Repetition time $T_R$ | 0.005 | s | Gradient-echo signal timing. |
| Excitation angle $\theta$ | 30 | degree | Constant flip angle; convert to radians for trigonometric functions. |
| Relaxivity $r_1$ | 4.5 | s⁻¹mM⁻¹ | Gd-DTPA relaxation/concentration conversion assumed at3T. |
| Dynamic sample spacing | 0.5 | s | Native time sampling; use the stored time vector. |
| Observation duration | 660 | s | First through final native time sample. |
| Baseline signal support | First2 dynamic samples | sample | Measured baseline normalization in the selected signal relation. |
| Input concentration convention | Plasma | — | The stored input already includes the source's hematocrit correction; do not apply it again. |
| Nominal source noise |0.2 times the first noise-free tissue signal | relative complex RMS | Existing source magnitude observations were made from independent real/imaginary Gaussian components, each with this scale divided by√2. This specifies the dataset's noise convention, not a hidden clean input. |
| Raster coordinates | Native row/column indices | pixel index | Independent kinetic curves on a60×50 reference raster; no physical in-plane spacing is needed by this voxelwise model. |
| Spatial support | Entire60×50 raster | — | All3000 positions are required; no spatial boundary coupling is assumed. |
| Initial tissue concentration | Zero | mM | Causal model initial condition. |
| Vascular volume contribution | Zero | dimensionless | Fixed standard-model assumption, not an additional estimated field. |

## Method Hints

A nonlinear least-squares compartment fit can separate transfer amplitude from washout. It is often convenient to optimize a transfer constant and an efflux rate, then recover the extracellular fraction from their ratio. Evaluate the causal exponential convolution using piecewise-linear interpolation of the sampled plasma input, retaining the exponential factor analytically over each interval.

A damped Gauss–Newton or Levenberg–Marquardt method is suitable for this small nonlinear inverse at each raster position. The published standard estimator also projects its final physical maps onto finite parameter ranges. Examine the effects of weak washout and noise rather than removing difficult positions. The signal-normalization and concentration units must remain consistent with the supplied calibration.

## References

1. Smith DS, Li X, Arlinghaus LR, Yankeelov TE, Welch EB. *DCEMRI.jl: a fast, validated, open source toolkit for dynamic contrast enhanced MRI analysis.* PeerJ3:e909(2015). [Article](https://doi.org/10.7717/peerj.909).
2. Flouri D, Lesnic D, Sourbron SP. *Fitting the two-compartment model in DCE-MRI by linear inversion.* Magnetic Resonance in Medicine76:998–1006(2016). Appendix derives the exponential-convolution recurrence. [Article](https://doi.org/10.1002/mrm.25991).
3. Duke University/Quantitative Imaging Biomarkers Alliance. Version6 Tofts digital reference objects. Educational and research data-use terms are preserved in the accompanying notice.
