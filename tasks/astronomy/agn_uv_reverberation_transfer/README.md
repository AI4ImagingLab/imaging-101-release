# AGN UV Reverberation Transfer

> Recover latent ultraviolet emission histories and broad-line response delays from irregular, noisy continuum and emission-line observations of an active galaxy.
> Domain: astronomy | Keywords: reverberation mapping, stochastic inference, astronomical spectroscopy | Difficulty: Hard

## Background

Variations in radiation from an accreting galactic nucleus illuminate surrounding gas. The gas reprocesses that radiation into emission lines after a distribution of light-travel delays. Comparing continuum and line variability therefore probes a region too small to resolve directly in an ordinary telescope image.

Irregular observing cadence, measurement noise and finite monitoring duration obscure the driving signal and smear its echoes. A time delay characterizes a response scale, but does not determine a unique three-dimensional gas geometry. Recovering a continuous emission history and its transfer response is an inverse problem with potentially multiple plausible solutions.

## Problem Description

The measurements are the complete three processed IUE light curves of NGC5548 from the1989campaign:1337 Angstrom continuum, Lyman alpha and C IV. Model the zero-mean driving continuum as a stationary damped random process $c(t)$ with covariance

```math
\operatorname{Cov}[c(t),c(t')]=\sigma^2\exp(-|t-t'|/\tau).
```

Here $t,t'$ are observing times in days, $\sigma$ is continuum variability amplitude in the supplied continuum flux units and $\tau$ is its damping time in days. For emission line $j$, let $d_j,w_j,a_j$ denote response-centroid delay, width and integrated response scale. The line signal $s_j$ and measured flux $y$ satisfy

```math
s_j(t)=\frac{a_j}{w_j}\int_{d_j-w_j/2}^{d_j+w_j/2}c(t-u)\,du,
\qquad y_i(t)=\mu_i+s_i(t)+\epsilon_i(t),\qquad s_0=c.
```

The channel means $\mu_i$ are unknown. Independent zero-mean Gaussian errors $\epsilon_i$ have the supplied standard deviations. Flux units differ by channel; the numerical scale $a_j$ converts the supplied continuum unit into the supplied line unit. Delay is in the observer frame; no redshift correction is applied. The top-hat response is a model approximation, not a claim of a uniform physical shell. Finite sampling produces coupled lag, smoothing and variability parameters and multiple local solutions.

Save `output/reconstruction.npz` relative to the task workspace:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1,3,600) | float32 or float64 | Real total latent flux including inferred means; axes batch, channel, time; continuum/Lyman alpha/C IV channel order, continuum in1e-15 erg s^-1 cm^-2 Angstrom^-1, lines in1e-13 erg s^-1 cm^-2 |
| transfer | (1,2,3) | float32 or float64 | Batch, Lyman alpha/C IV line, parameter; columns delay(day), width(day), scale(line flux unit per continuum flux unit); all widths/scales positive and delays nonnegative |
| continuum | (1,2) | float32 or float64 | Batch then variability amplitude $\sigma$ in continuum flux units and damping time $\tau$ in days; both positive, not logarithms |

The output grid has600equally spaced times from7486.597to7769.101, in JD-2440000days. It includes10%extrapolation at each end of the acquired235.42day baseline. No alignment, time shifting, normalization or cropping is performed during scoring. Evaluation checks each channel's RMS difference, scaled absolute transfer-parameter errors and logarithmic continuum-parameter errors, together with measured-flux consistency. For the latter, interpolate the submitted grid linearly to the exact observed epochs and compute RMS of residual divided by supplied uncertainty. There is no independently observed continuous latent truth; method agreement and observation consistency do not establish unique gas geometry or black-hole mass.

## Data Description

All180measurements are retained. The source table combines observations within a single IUE observing shift, at most about4hours; these are real processed satellite measurements, not simulated curves.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| observations | (1,3,60,3) | float64 | Batch, channel, epoch, column; columns JD-2440000(day), measured flux,1-sigma flux error. Channel0 continuum uses1e-15 erg s^-1 cm^-2 Angstrom^-1; channels1/2 integrated lines use1e-13 erg s^-1 cm^-2 |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Target/instrument | NGC5548, International Ultraviolet Explorer | dimensionless | Actual1989monitoring campaign |
| Channels | Continuum1337; Lyman alpha1216; C IV1549 | Angstrom | Nominal wavelength identifiers; integrated line fluxes |
| Epoch convention | JD-2440000, observer frame | day | Retain supplied irregular times; no rest-frame conversion |
| Acquired range | 7510.139 to 7745.559 | day | Full supplied campaign |
| Observations per channel | 60 | count | No fit-based filtering or downsampling |
| Error model | Supplied flux standard deviation, independent diagonal noise | channel flux unit | Cross-channel calibration covariance unavailable |
| Baseline/noise preprocessing | Source-processed integrated flux; no new flux calibration | dimensionless | Unknown channel means must be inferred |
| Boundary convention | Stationary continuum on the real time axis | day | No periodic extension or zero padding; extrapolation is model dependent |
| Instrument response/raw spectra | Not included in these processed tables | dimensionless | Do not infer unavailable raw spectral or camera calibration |

## Method Hints

The evaluated conditional estimate maximizes the following mean-marginalized Gaussian objective. Stack all measured fluxes into y, subtract each channel's arithmetic sample mean, and let L be the three-column channel-indicator matrix. C is the covariance obtained by convolving the stated continuum covariance with the two top hats and adding the supplied diagonal measurement-error variances. Define q=(L-transpose C-inverse L)-inverse L-transpose C-inverse y. Maximize

```math
-\tfrac12(y-Lq)^\mathsf{T}C^{-1}(y-Lq)
-\tfrac12\log\det C-\tfrac12\log\det(L^\mathsf{T}C^{-1}L)
-\sum_{j=1}^2\max\left(0,\log\frac{d_j}{0.3T}\right).
```

Here T=235.42 days is the acquired time baseline, sigma,tau and both response scales are positive, each delay lies in [0,80] days and each width in [0.1,80] days. The penalty is defined as zero when a delay is zero. These are analysis priors rather than acquisition calibrations. No division by observation count occurs. At the selected maximum, the reported history is the Gaussian conditional mean with channel means restored as sample means plus q; parameter-uncertainty averaging is not part of this target.

Use a Gaussian-process reverberation model with an exponential continuum covariance and analytic top-hat convolution covariances. Marginalizing the constant channel means avoids treating sample averages as exact. Jointly fit variability and transfer parameters with multiple initializations, then condition on all observations to recover the dense emission histories. Positive widths and scales are essential; local maxima and lag-width coupling should be checked. Conditional predictive error bars at a selected fit do not include parameter uncertainty.

## References

1. Zu, Kochanek and Peterson(2011), An Alternative Approach to Measuring Reverberation Lags in Active Galactic Nuclei. https://doi.org/10.1088/0004-637X/735/2/80 ; https://arxiv.org/abs/1008.0641
2. Clavel et al.(1991), Steps toward determination of the size and structure of the broad-line region in active galactic nuclei.I. An8month campaign of monitoring NGC5548 with IUE. https://doi.org/10.1086/169540
3. AGNWatch project measurement archive: https://www.asc.ohio-state.edu/astronomy/agnwatch/n5548/lcv/
