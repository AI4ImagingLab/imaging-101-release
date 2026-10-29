# Holmium SHG-FROG Pulse Retrieval

> Recover the temporal complex electric-field envelope of a femtosecond laser pulse from a measured delay–frequency intensity trace whose optical phase is missing.
> Domain: physics | Keywords: phase retrieval, nonlinear optics, ultrafast imaging | Difficulty: Hard

## Background

An optical pulse can vary much faster than an electronic detector can resolve. A nonlinear crystal provides an optical gate: two delayed copies of the pulse interact to generate light at twice their carrier frequency. Measuring the spectrum of this light for many delays produces a frequency-resolved optical gating trace. Its two coordinates encode information about the pulse's temporal amplitude and phase.

Such measurements help characterize short laser pulses used in spectroscopy, material processing, nonlinear frequency conversion and plasma generation. A spectrum alone cannot distinguish pulses with different spectral phases. A nonlinear delay scan adds the constraints needed for pulse retrieval, although noise, finite recording windows and weak pulse structure limit the recovered phase.

## Problem Description

Recover the **time-domain complex envelope**, sampled at 512 times, of the Ho:CALGO regenerative-amplifier output after its linear compressor. The observation is a complete author-processed experimental SHG-FROG trace. This is a temporal field reconstruction, rather than a two-dimensional spatial image.

Let the complex envelope be $E(t)$ and let $\tau$ denote the delay between identical pulse copies. Under instantaneous thin-crystal second-harmonic generation, the gated signal is $S(\tau,t)=E(t)E(t-\tau)$. With signal detuning $\omega=\Omega-2\omega_0$, the measurement model is

```math
\mathcal F[E](\omega)=\frac{1}{2\pi}\int E(t)e^{i\omega t}\,dt,
\qquad
D(\tau,\Omega)\simeq \mu\left|\mathcal F[S(\tau,\cdot)](\Omega-2\omega_0)\right|^2.
```

Here $D$ is the dimensionless processed detector intensity, $\Omega$ is the measured signal angular frequency in rad/s, $\omega_0$ is the fundamental carrier angular frequency, and $\mu>0$ absorbs the unknown detector gain and pulse scale. The model uses a flat effective spectral response on the supplied processed frequency grid. Phase matching, detector response and experimental preprocessing can contribute to the residual; no independent physical field truth is available. Absolute optical carrier phase and absolute pulse energy cannot be inferred.

Use periodic Fourier interpolation to evaluate fractional delays. The scan delay spacing and reciprocal Fourier time spacing differ slightly; **preserve the provided delays**. The inverse is quartic and nonconvex, with weak satellite structure and phase uncertainty.

Save `output/reconstruction.npz` relative to the task workspace, containing `reconstruction`, a **complex128** array of shape **(1, 512)**. Axis order is batch, time; use the supplied increasing `time` coordinate and the complex-envelope convention above. Values have units s$^{-1/2}$ and must satisfy the discrete unit-energy convention

```math
\Delta t\sum_{k=0}^{511}|E_k|^2=1.
```

The full complex field is evaluated: its relative complex L2 error and coherent overlap are measured after global phase, continuous periodic time translation and optional conjugate time reversal $E(t)\mapsto E^*(-t)$. No weak-field mask, magnitude-only comparison, frequency shift or additional amplitude fit is used for field comparison. An independent trace-consistency metric is the RMS intensity residual divided by the maximum observed intensity, after fitting the single nonnegative detector gain. These checks measure agreement with a validated method and the observations; they do not establish high-precision physical phase truth.

## Data Description

`data/raw_data.npz` contains only the observations and coordinate calibration:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| trace | (1, 512, 512) | float64 | Author-processed SHG intensity, dimensionless, axes batch/delay/signal frequency; maximum normalized to one. |
| delay | (1, 512) | float64 | Actual scan delay in s, increasing. |
| time | (1, 512) | float64 | Periodic envelope time samples in s, increasing. |
| angular_frequency | (1, 512) | float64 | Signal angular frequency in rad/s, increasing. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Measurement | Ho:CALGO regenerative-amplifier output after Martinez compressor | — | Complete processed `FROG_Regen/a.dat` record. |
| Nonlinear gate | Identical pulse copies, SHG, thin-crystal instantaneous envelope model | — | Signal field is the product of delayed and undelayed envelopes. |
| BBO thickness / slit width | 10 / 50 | µm | Acquisition identifiers in the author experiment log. |
| Laser repetition rate | 100 | kHz | Acquisition identifier. |
| Fundamental carrier angular frequency | 903760270468365.0 | rad/s | Signal detuning is signal frequency minus twice this value. |
| Speed of light | 299792458 | m/s | Used to interpret the supplied wavelength coordinates. |
| Frequency increment | 407474458795.6243 | rad/s | Equidistant fitted signal-frequency and envelope-detuning increment. |
| Fourier time increment | 3.0116847910804345e-14 | s | Reciprocal grid: 512 times this increment times the frequency increment equals $2\pi$. |
| Actual scan delay increment | 3.0096e-14 | s | Use `delay`; it is not the Fourier time increment. |
| Time / detuning origins | -256 times their respective increments | s / rad/s | Centered 512-point periodic Fourier grid. |
| Boundary convention | Periodic envelope, fractional delay by spectral phase | — | The supplied window is about 15.42 ps. |
| Intensity scale | Observed maximum divided out | — | A single unknown positive gain remains in the forward model. |
| Noise / response | Experimental, not independently calibrated | — | No known Gaussian noise variance or response curve is supplied. |
| Spectral-coordinate precision | Maximum rounding residual 0.002282 frequency pixels | — | Linear fit to the printed nonlinear wavelength coordinates; no intensity interpolation. |

The author had already subtracted background, extracted a portion of the original detector record and centered/resampled the trace. The bundled values retain all 512×512 processed samples, including zeros. Only spectral order reversal, transpose and division by the observed maximum were applied here. The input is **processed experimental data**, not the original CCD acquisition. Historical author pulse reconstructions are not observations or independent truth.

## Method Hints

Projection-based nonlinear phase retrieval can alternate between fitting the measured signal magnitude and enforcing its factorization into two delayed copies of one field. COPRA combines stochastic local updates with global residual descent; the global stage is useful when noisy measurements cannot be satisfied exactly. The optimal scalar gain for a predicted trace $T$ is

```math
\mu=\frac{\sum_{m,n}D_{mn}T_{mn}}{\sum_{m,n}T_{mn}^{2}},
\qquad
S'=\mathcal F^{-1}\!\left[\sqrt{D/\mu}\,
\frac{\mathcal F[S]}{|\mathcal F[S]|}\right].
```

Indices $m,n$ enumerate delay and frequency; $S'$ denotes the magnitude-projected gated field, and zero predicted Fourier components may be assigned zero phase. A complex gradient of the signal-matching loss propagates both factors of the delayed product. Fractional delays require the full Fourier phase factor, rather than integer array rolls. Different initial guesses can reveal uncertainty in weak features; a small trace residual alone does not guarantee a unique complete field.

## References

- N. C. Geib et al., *Common pulse retrieval algorithm: a fast and universal method to retrieve ultrashort pulses*, Optica 6, 495–505 (2019), [doi:10.1364/OPTICA.6.000495](https://doi.org/10.1364/OPTICA.6.000495); method version [arXiv:1810.04780v1](https://arxiv.org/abs/1810.04780v1).
- A. Suzuki et al., *High peak-power 2.1-µm femtosecond Holmium amplifier at 100 kHz*, [arXiv:2412.09096v1](https://arxiv.org/abs/2412.09096v1).
- Suzuki and collaborators, experimental Holmium-amplifier FROG measurement dataset (2024), CC BY 4.0. The observations and legitimate calibration in this task are documented above.
