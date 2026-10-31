# Alexa488 Confocal Population Dynamics

> Recover effective fluorescent molecule number, dark-state population and diffusion residence time from a measured fluorescence correlation curve whose short-lag processes overlap.
> Domain: biology | Keywords: fluorescence fluctuation spectroscopy, molecular transport | Difficulty: Medium

## Background

Fluorescent molecules moving through a focused laser detection volume generate intensity fluctuations. Their time correlations reveal how long molecules remain in the observed region and how many independently emitting molecules contribute. Confocal correlation spectroscopy probes molecular mobility at timescales much shorter than conventional image acquisition.

Temporary nonemissive states also modulate fluorescence. Separating these fast photophysical fluctuations from molecular passage is necessary for interpreting fluorescent labels in solution and cells. Such measurements support studies of molecular transport, molecular association and intracellular environments.

## Problem Description

The observation is an author-distributed Alexa488 confocal fluorescence cross-correlation measurement, expressed as excess correlation above its long-lag baseline. Recover the three parameters of the conditional Gaussian-volume, single-diffusing-species model with one fixed short-lived dark-state timescale:

```math
G(t)=\frac{1+\frac{T}{1-T}\exp(-t/\tau_T)}{N(1+t/\tau_D)\sqrt{1+t/(S^2\tau_D)}}.
```

Here t is lag time in milliseconds, N is effective molecule number (dimensionless), T is the nonemissive-state fraction, tauD is diffusion residence time in ms, tauT is the fixed dark-state residence time, and S is the axial/lateral confocal aspect ratio. The measured correlation is G plus empirical correlation-estimation error. Independent detector-background data and beam-radius calibration are unavailable; report effective population and residence time, without claiming absolute concentration or a diffusion coefficient.

The conditional target is the global minimum of the sum of squared residuals over all 200 measured lag channels, with every channel assigned unit weight. The admissible physical model has 0<N<=100000, 0<=T<=0.95, and 0.002<=tauD<=1000 ms. Fixed model conditions follow the distributed analysis session: tauT=0.002 ms, S=5 and additive correlation offset zero. These are explicit modeling assumptions rather than independently measured truth. Noise, small triplet amplitude and coupled short-lag decay make inference sensitive to model choice.

Save `output/reconstruction.npz`, relative to the workspace, with both required real finite arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 3) | float64 | Parameters in order N (dimensionless effective count), T (dimensionless fraction), tauD (ms) |
| correlation | (1, 200) | float64 | Dimensionless G(t) at every supplied lag, preserving native order |

Evaluation checks all parameters through the maximum componentwise relative difference against an independently reproduced standard analysis (limit 0.0001), full-curve relative L2 difference (limit 0.000001), and agreement between the reported parameters and reported curve (relative L2 limit 1e-10). Observation SSE must be at most 0.003. Every criterion must pass; no fitted normalization, permutation or alignment is performed. This evaluates reproducibility of a conditional analysis, not independently validated physical ground truth. Undefined zero-reference metric configurations are rejected by the evaluator.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| lag_ms | (1, 200) | float64 | Positive increasing temporal lag, ms |
| correlation | (1, 200) | float64 | Measured excess fluorescence correlation, dimensionless |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Fluorescent label | Alexa488 | 1 | Identity given in author session |
| Observation type | Confocal cross-correlation | 1 | Correlation derived from photon fluctuations |
| Observation axes | batch, lag | 1 | Native lag order, no resampling |
| Channel uncertainties | unavailable | 1 | Empirical error, no independent noise standard deviation supplied |
| Background correction | none | 1 | No independent detector background measurement available |
| Absolute beam radius | unavailable | 1 | Absolute concentration and diffusion coefficient are outside this target |

Fixed model assumptions are defined in Problem Description; they are not acquisition calibration. The original numeric observations are preserved as float64 without cropping, smoothing or synthetic noise.

## Method Hints

Fit the diffusion-volume correlation multiplied by a short-lived dark-state factor using bounded nonlinear least squares. Molecule number primarily sets amplitude, residence time controls the broader decay and the dark-state fraction affects short lags. A no-dark-state fit helps assess whether that additional physical process materially improves the measured-curve residuals. Parameter sensitivity and residual structure are useful diagnostics.

## References

Müller P., Schwille P., Weidemann T. PyCorrFit—generic data evaluation for fluorescence correlation spectroscopy. Bioinformatics 30,2532–2533 (2014). https://doi.org/10.1093/bioinformatics/btu328
