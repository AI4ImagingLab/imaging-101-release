# Cosmic Chronometer Acceleration

> Reconstruct cosmic acceleration and the present expansion rate from noisy differential-age expansion measurements, where differentiation amplifies sparse-data uncertainty.
> Domain: astronomy | Keywords: cosmological inference, Gaussian processes, derivative reconstruction | Difficulty: Hard

## Background

The expansion of the Universe stretches wavelengths and changes the relation between cosmic time and redshift. Differential ages of passively evolving galaxies provide a way to measure the expansion rate without assuming a particular late-time expansion history. Comparing nearby redshift populations cancels the unknown absolute formation time under the common-population approximation.

The acceleration parameter describes whether expansion speeds up or slows down. Its recovery requires a derivative of the expansion rate, which is much less directly constrained than the rate itself. Sparse high-redshift observations and stellar-population uncertainties make this an informative example of uncertainty-sensitive inverse inference.

## Problem Description

Use the complete 30-measurement cosmic-chronometer cohort provided below. These observations are published, derived expansion rates rather than raw galaxy spectra or individual stellar ages. The measurement relation is

```math
H(z)=-\frac{1}{1+z}\frac{dz}{dt},\qquad
H_i=H_0\exp\left[\int_0^{z_i}\frac{1+q(u)}{1+u}\,du\right]+\epsilon_i,
\qquad q(z)=(1+z)\frac{H'(z)}{H(z)}-1.
```

Here redshift $z$ and integration variable $u$ are dimensionless, $t$ is cosmic time, $H$ is the expansion rate, $H_0=H(0)$, a prime denotes differentiation with respect to redshift, $q$ is the deceleration parameter (negative for acceleration), and independent measurement errors $\epsilon_i$ have zero mean and supplied standard deviations. Reconstruct the four regularized solutions in axis order squared exponential, Matérn 5/2, Matérn 7/2, Matérn 9/2, on 200 equally spaced redshifts from 0 to 1.965 inclusive. Kernel choice is an explicit scientific comparison, not four independent observations.

Save all arrays in `output/reconstruction.npz`:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 4, 200) | float64 | Deceleration parameter q, dimensionless; batch, kernel, redshift |
| hubble_rate | (1, 4, 200) | float64 | Positive H, km s^-1 Mpc^-1; same axis order |
| hubble_derivative | (1, 4, 200) | float64 | H prime, km s^-1 Mpc^-1 per unit redshift; same axis order |
| hubble_constant | (1, 4) | float64 | Positive H at redshift zero, km s^-1 Mpc^-1 |
| point_covariance | (1, 4, 200, 2, 2) | float64 | Symmetric positive-definite marginal covariance of [H,H prime] at each redshift; entries carry the corresponding product units |

The covariance is a separate 2 by 2 marginal at each location; it does not encode covariance between distinct redshifts. All quantities must be finite. Evaluation measures q absolute discrepancy, H relative discrepancy, derivative discrepancy normalized by its marginal standard deviation, covariance discrepancy normalized by marginal standard-deviation products, agreement with the measured rates in error units, and the defining q/H0 identities. There is no amplitude or redshift alignment. Conditional Gaussian-process covariance and delta-method q errors do not constitute physical confidence intervals or account for stellar-population systematics.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| redshift | (1, 30) | float64 | Dimensionless measured redshift, in published order |
| hubble_observation | (1, 30) | float64 | Derived expansion rate H, km s^-1 Mpc^-1 |
| hubble_error | (1, 30) | float64 | Positive quoted one-standard-deviation error on H, km s^-1 Mpc^-1 |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Acquisition | Differential ages of passively evolving galaxies | none | Cosmic chronometer cohort; no BAO observations |
| Number of measurements | 30 | none | Complete chronometer cohort in the bound source compilation |
| Observed redshift support | 0.07 to 1.965 | none | Source decimal coordinates retained |
| Reconstruction grid | 200 uniform points, 0 to 1.965 inclusive | none | Redshift zero is an extrapolation |
| Measurement noise | Independent Gaussian with supplied hubble_error | km s^-1 Mpc^-1 | Selected statistical measurement model; no supplied off-diagonal covariance |
| Derivative convention | Derivative with respect to increasing redshift | none | Negative q denotes accelerating cosmic expansion |

## Method Hints

Use zero-mean Gaussian-process conditioning for H with measurement-error covariance and optimize the covariance amplitude and correlation length by the marginal likelihood. Analytically differentiating the covariance kernel gives joint marginal predictions for the rate and its redshift derivative. Compare the squared exponential and three differentiable Matérn kernels, since derivative inference can depend strongly on regularity assumptions. Propagate the H/H-prime covariance when diagnosing q uncertainty and assess weakly constrained high-redshift behavior rather than interpreting every smooth curve as a direct measurement.

## References

- Seikel, Clarkson and Smith, “Reconstruction of dark energy and expansion dynamics using Gaussian processes,” JCAP 06 (2012) 036, [doi:10.1088/1475-7516/2012/06/036](https://doi.org/10.1088/1475-7516/2012/06/036).
- Moresco et al., “A 6% measurement of the Hubble parameter at z approximately 0.45,” [arXiv:1601.01701](https://arxiv.org/abs/1601.01701), Table 4, chronometer compilation.
