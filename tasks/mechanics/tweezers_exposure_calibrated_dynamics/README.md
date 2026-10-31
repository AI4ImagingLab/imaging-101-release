# Exposure-Calibrated Magnetic Tweezer Dynamics

> Recover a tethered bead's effective spring constant, drag coefficient and tracking-noise amplitude from a measured video trajectory distorted by exposure averaging and temporal aliasing.
> Domain: mechanics | Keywords: force spectroscopy, stochastic dynamics, parameter inference | Difficulty: Hard

## Background

Magnetic tweezers exert forces on microscopic magnetic beads attached to molecules. Thermal collisions move the bead around its equilibrium position, while the stretched tether provides a restoring force. The fluctuations therefore contain information about the effective mechanical stiffness and viscous drag. Such measurements support single-molecule mechanics and instrument force calibration.

A camera records a time average of the bead position during each exposure. Finite sampling aliases fast motion into the observable spectrum, and imperfect image localization introduces additional frame-to-frame noise. These effects can resemble changes in the mechanics, so displacement variance alone cannot reliably separate the unknown quantities.

## Problem Description

The complete supplied trajectory is an author-processed experimental lateral displacement of a bead tethered to double-stranded DNA of approximate contour length2.8 micrometres. The underlying overdamped displacement x obeys

```math
\gamma\dot{x}(t)+\kappa x(t)=\sqrt{2\gamma k_BT}\,\xi(t),\qquad
y_n=\frac{1}{\Delta}\int_{n\Delta}^{(n+1)\Delta}x(t)\,dt+\eta_n.
```

Here gamma is viscous drag in pN s/nm, kappa is the effective lateral spring constant in pN/nm, xi is unit white noise with covariance delta(t−s), kBT is thermal energy, Delta is the sampling/exposure interval, and independent localization noise eta has variance epsilon squared in square nanometres. An arbitrary constant position offset is irrelevant. Equal exposure and frame intervals are the declared zero-dead-time modeling assumption, not a separately measured exposure metadata item.

With relaxation time c=gamma/kappa and sampling frequency f_s=1/Delta, the two-sided power spectral density at positive frequency f is

```math
P(f)=\frac{2k_BT c}{\kappa}
\left[1+2cf_s\sin^2(\pi f/f_s)
\frac{\sinh[1/(cf_s)]}{\cos(2\pi f/f_s)-\cosh[1/(cf_s)]}\right]
+\frac{\epsilon^2}{f_s}.
```

P has units nm²/Hz. The finite trajectory gives a fluctuating estimate of this expectation. Drag, stiffness and tracking noise have overlapping spectral effects. All three must be inferred; an assumed bead radius or fluid viscosity must not fix the drag. No independent physical parameter truth is supplied. This is a nonspatial mechanical inverse problem, not a molecular-image reconstruction or a determination of axial force: the tether height needed to convert lateral stiffness into axial force is unavailable.

Save `output/reconstruction.npz` relative to the task workspace with key `reconstruction`, shape `(1,3)`, dtype float32 or float64, finite real values. Axis order is batch,parameter. The parameter order is `(gamma,kappa,epsilon)` with units `(pN s/nm,pN/nm,nm)`; all must be strictly positive and epsilon denotes a standard deviation, not its variance. There is no rescaling, sign alignment or offset fitting of these parameters.

Evaluation requires the maximum componentwise relative parameter difference and an observation-consistency residual to meet their bounds. For the latter, compute a two-sided periodic-Hann Welch spectrum: segment length floor(2N/36), half-segment integer overlap, subtract each segment's mean, retain strictly positive frequencies, and average the periodograms. N is the supplied trajectory length. The residual is RMS of `(observed P−modeled P)/(observed P/sqrt(35))`. The weight35 preserves the stated spectral estimator convention; it is not a claim of35 statistically independent segments. Reference agreement evaluates reproducibility of the declared estimator, not independent physical accuracy.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| position | (1,13631) | float64 | Complete author-processed measured lateral bead positions in nm; axes batch,time; native order and absolute offset retained |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sampling frequency | 400 | Hz | Native experimental acquisition |
| Sample interval | 0.0025 | s | Uniform frame spacing |
| Frame count | 13631 | count | All supplied real trajectory samples |
| Sampled time span | 34.075 | s | Time from first to final sample, (N−1)/400 |
| Thermal energy kBT | 4.1 | pN nm | Published analysis convention; physical estimates are conditional on this value |
| Exposure model | Equal to0.0025s frame interval | s | Zero-dead-time approximation used for the original real-data analysis; no separate exposure measurement supplied |
| Tether | Double-stranded DNA, approximate contour length2.8 | micrometre | Experimental sample identity; contour length is not measured tether extension |
| Instrument | Custom magnetic tweezer with video tracking | none | Author's experimental apparatus; no raw camera movie included |
| Position origin | Arbitrary constant | nm | Removed through per-segment centering; no absolute coordinate inference |
| Noise | Independent zero-mean Gaussian localization error | nm | Standard deviation epsilon unknown; thermal fluctuations are distinct |
| Mechanical model | Stationary harmonic lateral confinement, overdamped drag | none | No hydrodynamic-memory correction or externally driven motion |
| Temporal boundary | Finite native record | none | No periodic concatenation of the trajectory; FFT is applied to windowed segments |

The observed trajectory has already undergone the authors' video localization and conversion to nm. No target-derived mechanical calibration or fitted parameter is part of the visible input. The declared thermal energy and exposure assumptions bound the interpretation of the reconstructed physical quantities.

## Method Hints

Use maximum-likelihood fitting of an exposure- and aliasing-aware Langevin spectrum with an additive localization-noise term. Averaged spectral ordinates can be approximated by gamma-distributed observations with mean P and shape35; positive finite parameters are physically meaningful. Equipartition can initialize stiffness but does not replace the exposure-aware inference. An Allan-variance analysis provides a complementary time-domain check, and fits without localization noise expose a systematic bias. Inspect residuals and stability across window lengths and portions of the record.

## References

Morgan,I.L. and Saleh,O.A. (2021), *Tweezepy: A Python package for calibrating forces in single-molecule video-tracking experiments*, PLOS ONE16(12),e0262028. [Published paper](https://doi.org/10.1371/journal.pone.0262028).

Morgan,I.L., author data and figure-analysis archive, Zenodo data release, CC-BY4.0. The bundled trajectory is the real measurement used in the paper's experimental comparison.
