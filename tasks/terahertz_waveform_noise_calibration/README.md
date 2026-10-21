# Terahertz Waveform Noise and Drift Calibration

> Reconstruct a common terahertz waveform, acquisition drift and three signal-dependent noise amplitudes from fifty repeated photocurrent traces whose variability mixes detector noise with amplitude and timing fluctuations.
> Domain: physics | Keywords: terahertz spectroscopy, noise calibration, maximum likelihood | Difficulty: Hard

## Background

Terahertz time-domain spectroscopy measures the electric-field response of a short pulse by scanning an optical sampling delay. A photoconductive receiver converts this response into a photocurrent waveform. The time trace preserves amplitude and phase information, making the modality useful for characterizing dielectric response, conductivity and molecular resonances.

Reliable spectroscopy requires an instrument-noise description as well as a material-response model. Detector electronics produce additive noise, optical power fluctuations produce amplitude-dependent noise, and sampling-time jitter produces noise that grows with waveform slope. Slower amplitude and timing drift across repeated scans can resemble these stochastic processes. Separating drift from noise enables meaningful uncertainty estimates for later physical measurements.

## Problem Description

The observations are fifty repeated air-reference acquisitions, each containing all 267 native time samples. Let $x_{jk}$ denote photocurrent in scan $j$ at relative time $t_k=k\Delta t$, where $j=0,\ldots,M-1$, $k=0,\ldots,N-1$, $M=50$ and $N=267$. The measurement model is

```math
 x_{jk}=\zeta_{jk}+\epsilon_{jk},\qquad
 \zeta_j=a_j S(\eta_j)\mu,\qquad
 \epsilon_{jk}\sim\mathcal N(0,v_{jk}),
```

where $\mu$ is the common latent photocurrent waveform, $a_j$ is relative amplitude drift and $\eta_j$ is delay drift. The band-limited shift $S(\eta)$ evaluates $\mu(t-\eta)$ using the periodic discrete Fourier convention. If $\widehat\mu_l$ is a Fourier coefficient at angular frequency $\omega_l$, the shifted coefficient is $e^{-i\omega_l\eta}\widehat\mu_l$. The derivative operator $D$ multiplies Fourier coefficients by $i\omega_l$ and returns a time derivative in pA/ps.

The independent sample variances are

```math
 v_{jk}=\alpha^2+\beta^2\zeta_{jk}^2+\tau^2(D\zeta_j)_k^2.
```

Here $\alpha$ is additive-noise amplitude in pA, $\beta$ is dimensionless multiplicative-noise amplitude and $\tau$ is timing-jitter amplitude in ps. The waveform and variance must be inferred jointly: strong signal variability can come from noise, slow drift or an altered latent waveform. Fix $a_0=1$ and $\eta_0=0$ to remove the otherwise arbitrary amplitude and time-origin gauges. The latent waveform is consequently in the first scan's reference gauge, rather than an independently known noiseless signal.

Save **`output/reconstruction.npz`** containing these real `float64` arrays:

| Key | Shape | Units and axis order |
| --- | --- | --- |
| `reconstruction` | `(1, 267)` | pA; batch, time sample; latent waveform in first-scan gauge. |
| `amplitude` | `(1, 50)` | Dimensionless; batch, scan; first entry exactly 1. |
| `delay` | `(1, 50)` | ps; batch, scan; positive values delay the waveform; first entry exactly 0. |
| `noise` | `(1, 3)` | Batch, component: additive $\alpha$ (pA), multiplicative $\beta$ (1), jitter $\tau$ (ps). Report the finite-sample corrected amplitudes defined below. |
| `expected_waveforms` | `(1, 50, 267)` | pA; batch, scan, time; $a_j S(\eta_j)\mu$. |
| `noise_std` | `(1, 50, 267)` | pA; batch, scan, time; square root of the stated variance law evaluated at `expected_waveforms` using the reported corrected `noise` values. |

The standard method estimates positive noise amplitudes by likelihood and reports each amplitude multiplied by $\sqrt{M/(M-1)}$. This finite-sample correction is part of the output convention; it does not guarantee removal of every nuisance-parameter bias. Preserve it for all three noise components. Evaluate agreement separately for the differently dimensioned noise parameters and for every other required array, using relative Euclidean error. No waveform rescaling, delay alignment, cropping or gauge fitting is applied by evaluation.

## Data Description

`data/raw_data.npz` contains the full author-deposited measured traces. Voltage was converted to calibrated photocurrent with the acquisition gain. No samples were cropped, denoised or synthetically generated.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `waveforms` | `(1, 50, 267)` | `float64` | Photocurrent in pA; batch, repeated scan, native time sample. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Measurement condition | Air-reference repeated THz time traces | — | Instrument-noise/drift experiment, without a material specimen inversion. |
| Acquisition date | 2017-03-20 | — | Author data record. |
| Repeated acquisitions | 50 | 1 | $M$; original filename order. |
| Native samples per trace | 267 | 1 | $N$; entire recorded window. |
| Sampling interval | 0.05003462406015038 | ps | $\Delta t$; mean native time-coordinate spacing. |
| Relative time origin | 0 | ps | First supplied sample; absolute time offset is absorbed by the waveform gauge. |
| Transimpedance gain | 5e7 | V/A | Original voltage divided by this gain yields current in amperes. |
| Public amplitude conversion | Original voltage × 1e12 / 5e7 | pA | Exact author-specified conversion. |
| Shift convention | $\mu(t-\eta)$ | — | Positive delay shifts a feature toward larger sample time. |
| Fourier boundary | Periodic, real transform on all 267 samples | — | Odd sample count has no separate Nyquist coefficient. |
| Drift gauge | First amplitude 1; first delay 0 | — | Fixes amplitude and delay nonidentifiability. |
| Sample-noise assumptions | Gaussian; independent conditional on drift and waveform | — | Variance depends on each scan's expected waveform and its derivative. |
| Noise parameters | Unknown and shared across all scans | — | Additive, multiplicative and timing-jitter amplitudes must all be inferred. |

## Method Hints

Use a heteroscedastic maximum-likelihood formulation that includes both the logarithm of the variance and the variance-weighted residual. Estimate the latent waveform jointly with amplitude drift, delay drift and the three positive noise amplitudes. Fourier shift and derivative operators provide explicit gradients through the waveform-dependent variance; treating variance as fixed changes the inverse problem.

Log-variance variables and parameter-dependent scaling improve conditioning. A variance regression on the mean waveform and its derivative supplies initial noise estimates. General-purpose BFGS with the full analytic gradient is suitable for the selected method. A tiny likelihood change can coincide with a line-search precision-loss stop, so retain actual optimizer diagnostics and assess gradient/curvature quality independently instead of fabricating a successful termination flag.

## References

1. L. Mohtashemi et al., *Maximum-likelihood parameter estimation in terahertz time-domain spectroscopy*, Optics Express 29, 4912–4926 (2021). [Published noise and drift model](https://doi.org/10.1364/OE.417724).
2. J. Posada Loaiza et al., *THzTools: data analysis software for terahertz time-domain spectroscopy*, Journal of Open Source Software 9(104), 7542 (2024). [Selected modern software method](https://doi.org/10.21105/joss.07542).
3. J. S. Dodge et al., *MATLAB functions for maximum-likelihood parameter estimation in terahertz time-domain spectroscopy*, version 1.0.2 (2021). [MIT-licensed measured-data deposit](https://doi.org/10.5281/zenodo.4876388). The present benchmark applies the modern joint method to all 267 samples; it does not claim reproduction of the historical MATLAB experiment's 256-sample numerical table.
