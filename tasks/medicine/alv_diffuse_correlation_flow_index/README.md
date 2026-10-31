# Diffuse correlation flow-index reconstruction

> Reconstruct a model-conditioned scatterer flow index and optical coherence from measured intensity autocorrelation curves despite noise and their coupled effects on the decay.
> Domain: medicine | Keywords: diffuse optics, correlation spectroscopy, parameter inference | Difficulty: Medium

## Background

Coherent near-infrared light develops speckle after multiple scattering. Motion of scatterers changes the speckle over time, so temporal intensity correlations contain information about microscopic motion. In biological tissue this effect enables noninvasive monitoring of microvascular perfusion.

Diffuse correlation measurements are useful for bedside optical monitoring, where penetration beyond the superficial layers matters. The inferred motion parameter depends on the assumed scattering, absorption and geometry; it is a flow index rather than a direct volumetric perfusion measurement. Unknown optical properties and coherence can make parameter recovery ambiguous.

## Problem Description

The observations are390 complete four-channel ALV hardware recordings supplied as real measurements by the data author. Sample identity is unknown; these must not be interpreted as human measurements or assigned physiological event labels. Infer one conditional flow index and coherence factor per recording under a homogeneous semi-infinite diffusive medium, using the supplied nominal analysis assumptions. Absolute optical calibration is unavailable.

For correlation lag tau, normalized field correlation g1 is obtained by a source and image-source Green function:

```math
g_1(\tau)=\left|\frac{e^{-K(\tau)r_1}/r_1-e^{-K(\tau)r_2}/r_2}
{e^{-K(0)r_1}/r_1-e^{-K(0)r_2}/r_2}\right|,\quad
g_2(\tau)=1+\beta g_1(\tau)^2,
\qquad K(\tau)^2=3\mu_a\mu_s'+6(\mu_s')^2 k^2 F\tau.
```

Here F=alpha D is the flow index (moving-scatterer fraction alpha times effective diffusivity D), beta is dimensionless coherence, absorption mu_a and reduced scattering mu_s' are inverse length, and k=2pi n/lambda0 is the **medium** wavevector for vacuum wavelength lambda0 and refractive index n. Only the product F is identifiable; alpha is fixed to1 by convention. With source-detector distance rho, define z0=1/(mu_a+mu_s'), reflectance R=-1.440/n²+.710/n+.668+.0636n, extrapolation distance zb=[2(1+R)/(3(1-R))]/mu_s', and r1=sqrt(rho²+z0²), r2=sqrt(rho²+(z0+2zb)²). The medium occupies a half-space with extrapolated zero-field boundary. Consistent length units are essential.

Save `output/reconstruction.npz` with all three required float64 real arrays:

| Key | Shape | Unit and axes |
|---|---|---|
| beta | (1,390) | dimensionless coherence, batch/record |
| flow_index | (1,390) | alpha D in cm²/s, batch/record; absolute conditional scale |
| predicted_g2 | (1,390,199) | dimensionless full normalized correlation, batch/record/lag |

No offset removal or amplitude rescaling is permitted. Parameter agreement is evaluated separately in physical units (absolute beta error, relative flow-index error), together with full predicted-curve error and global/worst-record observed g2 RMS residual. For observation consistency only, use the fixed author support: count-weight the four channels linearly by countrate, start strictly above100ns, end at the earlier of the last lag not exceeding1ms and the last weighted value above1.13; include all bins between those endpoints. This avoids early hardware dead-time bins and noise-dominated tails; all raw bins remain available. All390records are evaluated, including coherence-boundary fits. No independent physical flow truth is available.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| g2 | (1,390,4,199) | float64 | Normalized dimensionless intensity correlation; native stored g2-1 converted by adding1 |
| lag_times | (1,199) | float64 | Lag in seconds, native milliseconds multiplied by.001 |
| countrate | (1,390,4) | float64 | Native mean photon count rate in kHz, used for linear channel weights |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Hardware | ALV-7004/USB-FAST | none | Native correlator header |
| Acquisition date | 11/11/2025 | native date text | Source header; sample/memo fields blank |
| Complete recordings | 390 | records | All complete source recordings; author excludes one terminal incomplete file |
| Integration duration | 1 | s | Native duration; first header FloatDur944ms |
| Source-detector separation rho | 2.5 | cm | Author nominal analysis geometry, not independently measured calibration |
| Vacuum wavelength lambda0 | 785 | nm | Author nominal analysis setting; native generic header wavelength0 is invalid/unusable |
| Analysis refractive index n | 1.4 | dimensionless | Author nominal analysis assumption |
| Generic header refractive index | 1.332 | dimensionless | Preserved discrepancy; not substituted for the selected analysis assumption |
| Analysis absorption mu_a | .1 | cm^-1 | Author nominal assumption, not measured optical calibration |
| Analysis reduced scattering mu_s' | 10 | cm^-1 | Author nominal assumption, not measured optical calibration |
| Generic header temperature | 298.16000 | K | Native instrument metadata; does not independently identify sample |
| Generic header viscosity | .89000 | cP | Native metadata; not used to derive hidden motion parameters |
| Boundary | Homogeneous semi-infinite | none | Extrapolated image-source solution defined above |
| Coordinate conversion | 10mm per cm; 1e-6mm per nm | length | Forward model uses mm; output flow index uses cm²/s |
| Noise | Unknown | dimensionless g2 | No calibrated pointwise noise variance supplied |

## Method Hints

Use constrained nonlinear least squares on the explicit correlation-diffusion model. The coherence controls the short-lag amplitude, while the motion index controls the decay, so fitting them jointly and retaining a physical nonnegative motion parameter helps separate their effects. Channel count rates provide legitimate observation weights. Inspect residual structure and sensitivity to nominal optical assumptions instead of interpreting a conditional flow index as independently calibrated perfusion.


The standard recovery is the following explicitly constrained estimator, not an arbitrary nonnegative diffusion fit. For record r, let J(r) be its fixed fitting-lag support and let w(r,c) be channel c's countrate divided by the sum over the four channels. Define the measured channel average gbar(r,j)=sum over c of w(r,c)g2(r,c,j). The declared recovery minimizes the unweighted lag residual:

```math
(\widehat\beta_r,\widehat F_r)
=\underset{0.46\le\beta\le0.54,\ F\ge0}{\operatorname{argmin}}
\sum_{j\in J(r)}\left[g_2(\tau_j;\beta,F)-\overline g(r,j)\right]^2.
```

The coherence interval is a selected analysis prior, not measured hardware calibration; boundary solutions are retained and evaluated. F has the physical cm²/s convention defined above. Use the fitted parameters to predict all199 original lags, including those outside J(r); no further amplitude normalization is part of this estimator.

## References

- Nabacino et al., Fast Multi-Distance Time-Domain NIRS and DCS System for Clinical Applications, Sensors2024. [DOI](https://doi.org/10.3390/s24227375). Instrument/modality context; the bound author-distributed ALV data are not asserted to be that paper's experiment.
- Wu et al., Improved accuracy of cerebral blood flow quantification in the presence of systemic physiology cross-talk using multi-layer Monte Carlo modeling, Neurophotonics2021. [DOI](https://doi.org/10.1117/1.NPh.8.1.015001). Selected code's semi-infinite analytical baseline only, not the multilayer Monte Carlo inverse.
- Sakadzic, Boas and Carp, Theoretical model of blood flow measurement by diffuse correlation spectroscopy, JBO2017. [DOI](https://doi.org/10.1117/1.JBO.22.2.027006).
