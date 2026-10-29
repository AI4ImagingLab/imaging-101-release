# OGLE Stellar Microlensing Geometry

> Reconstruct a stellar microlensing trajectory and source/blend flux from measured I-band photometry despite nonlinear geometric degeneracy and unresolved background light.
> Domain: astronomy | Keywords: gravitational lensing, time-domain astronomy, nonlinear inversion | Difficulty: Medium

## Background

Gravity deflects light from a distant star when a foreground object passes near its line of sight. The resulting images usually remain unresolved, but their combined brightness changes as the source and lens move. This temporal magnification allows otherwise faint stellar and planetary lenses to be studied without detecting their own light.

Ground-based surveys measure crowded stellar fields repeatedly. Light from neighboring objects mixes with the magnified source, while irregular sampling and photometric scatter limit the geometric information available. Recovering the event trajectory and separating source from constant background flux is therefore an inverse problem. A single ordinary light curve constrains dimensionless geometry and a crossing time, but does not by itself determine lens mass or distance.

## Problem Description

Recover the point-source point-lens geometry of the 2010 stellar subevent at OGLE-2008-BLG-092 using all 383 published OGLE-IV photometric epochs. This bound acquisition covers the final stellar subevent, for which a single-lens description is appropriate; it does not contain the earlier planetary anomaly.

```math
u(t)=\sqrt{u_0^2+\left(\frac{t-t_0}{t_E}\right)^2},\qquad
A(u)=\frac{u^2+2}{u\sqrt{u^2+4}},\qquad
F(t)=F_s A[u(t)]+F_b.
```

Time t and closest approach time t0 are in heliocentric Julian date minus 2450000, in days. The impact parameter u0 and instantaneous separation u are in units of the Einstein angular radius. The Einstein crossing time tE is in days. A is dimensionless total magnification; Fs is unlensed source flux and Fb is constant blend flux. Flux units are defined by magnitude 22 having unit flux. For measured magnitude m and its supplied standard deviation sigma-m, use

```math
F=10^{0.4(22-m)},\qquad \sigma_F=0.4\ln(10)F\sigma_m.
```

The unknown geometry and blending are correlated. The sign of u0 cannot be determined by this model, so use the positive convention. The source flux and tE must also be positive. Blend flux is a fitted additive background and may be negative within photometric noise; no positivity prior on it is assumed.

Save `output/reconstruction.npz` relative to the task workspace with both required float32 or float64 arrays:

| Output key | Shape | Meaning |
|---|---|---|
| reconstruction | (1,3) | Batch then [t0,u0,tE], respectively days, dimensionless Einstein-radius units, days |
| source_blend_flux | (1,2) | Batch then [Fs,Fb], in magnitude-22 flux units |

All values are real and finite. Use the stated physical scale, without normalization, logarithms or implicit transformations. Evaluation checks absolute errors in each geometric parameter, the L2 source/blend error divided by reference source flux, and reduced chi-square in observation flux space with 383-5 degrees of freedom. No alignment or scale fitting is performed. The numerical reference is a standard-method estimate, not independently measured physical truth.

## Data Description

`data/raw_data.npz` supplies the complete measured light curve:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| time | (1,383) | float64 | Observation epochs, HJD minus 2450000 in days, acquisition order |
| magnitude | (1,383) | float64 | Author-processed I-band apparent magnitude, mag |
| magnitude_error | (1,383) | float64 | Supplied magnitude standard deviations, mag |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Telescope | Warsaw telescope, Las Campanas, Chile | site | Ground-based photometry |
| Aperture | 1.3 | m | Primary diameter |
| Camera | OGLE-IV, 32 CCD chips, 4096 by 2048 pixels each | pixels | Survey instrument |
| Detector pixel size | 15 | micrometre | Physical CCD sampling |
| Angular pixel scale | 0.26 | arcsec/pixel | Angular sampling before photometric extraction |
| Camera field | 1.4 | square degrees | Total field of view |
| Band | I | photometric band | Measured brightness |
| Acquisition | First two OGLE-IV seasons, 2010 and 2011 | year | Complete published O4 sequence |
| Photometry | Difference image analysis; transformed photometric zero point | method | Author processing of measured exposures |
| Epoch convention | HJD minus 2450000 | day | Supplied values are authoritative |
| Error convention | 0.0068, baseline dispersion assigned to each epoch | mag | No additional uncertainty rescaling |
| Flux convention | Magnitude 22 corresponds to flux 1 | relative flux | Fixes the numerical source/blend scale |
| Motion model | Rectilinear relative motion | dimensionless | No annual/satellite parallax term |
| Source model | Point source, unresolved images | dimensionless | No finite-source or limb-darkening correction |
| Boundaries | Independent irregular epochs; no periodic extension | day | No interpolation of observed measurements |

No instrument response or fitted parameter is supplied as hidden-target-derived calibration. The source magnitudes and error bars are existing processed telescope observations.

## Method Hints

Use nonlinear least squares with variable projection: for any trial geometry, source and blend flux enter linearly and can be solved by a two-column weighted least-squares system. This leaves only three nonlinear geometric parameters. Preserve magnitude-to-flux uncertainty propagation, account for the reflection ambiguity, and inspect both residuals and the impact-parameter/timescale correlation. A derivative-free simplex search is suitable for this small nonlinear problem.

## References

R. Poleski and J. C. Yee, “Modeling microlensing events with MulensModel”, Astronomy and Computing 26, 35-49 (2019), [doi:10.1016/j.ascom.2018.11.001](https://doi.org/10.1016/j.ascom.2018.11.001).

R. Poleski et al., “Triple Microlens OGLE-2008-BLG-092L: Binary Stellar System with a Circumprimary Uranus-type Planet”, Astrophysical Journal 795, 42 (2014), [doi:10.1088/0004-637X/795/1/42](https://doi.org/10.1088/0004-637X/795/1/42). The selected stellar subevent is described by the independent point-lens model in Section 3.1.
