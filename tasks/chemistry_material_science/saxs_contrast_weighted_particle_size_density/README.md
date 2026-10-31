# SAXS Contrast-Weighted Particle Size Density

> Recover a continuous particle-radius density and flat background from a complete small-angle scattering curve whose oscillatory averaging makes size inversion ill-conditioned.
> Domain: chemistry & material science | Keywords: small-angle scattering, particle characterization, nonnegative inversion | Difficulty: Hard

## Background

Small-angle X-ray scattering measures fluctuations of electron density over length scales larger than atomic spacings. A detector records the angular dependence of radiation scattered by interfaces between particles and their surroundings. For dilute suspensions or inclusions, particle form factors connect these observations to particle dimensions and morphology.

The method characterizes ensembles without resolving individual particles. It is used for colloids, nanoporous materials, precipitates and dispersions. Large particles contribute strongly at small scattering angles, while smaller particles and sharp interfaces affect higher angles. A broad distribution mixes these contributions, and instrumental background can dominate the weakest scattering.

## Problem Description

Infer a nonnegative, contrast-weighted volume density of spherical particle radii from the complete author-distributed `OneDist1pctLarge` demonstration curve. Its exact experimental versus simulated origin is unverified; it is an existing author dataset, not newly generated benchmark data. Assume dilute, noninteracting, uniform spheres with pinhole geometry and a constant additive background. The physical observation equation is

```math
I(q)=B+10^{-4}\int V(r)\,F(qr)^2\,Z(r)\,\mathrm{d}r+\epsilon(q),
\qquad V(r)=\frac{4\pi r^3}{3},
\qquad F(x)=\frac{3(\sin x-x\cos x)}{x^3},\quad F(0)=1.
```

Here q is the scattering-vector magnitude in inverse angstroms, r is particle radius in angstroms, I and the unknown constant B are in inverse centimeters, and epsilon denotes measurement error. V is the particle volume expressed numerically in cubic angstroms. Define Z(r)=C p(r), where p is the particle volume fraction per angstrom and C is the squared scattering-length-density contrast expressed numerically in units of 10^20 cm^-4. Thus Z has units 10^20 cm^-4 per angstrom. The factor10^-4 combines cubic-angstrom to cubic-centimeter conversion10^-24 with the contrast unit10^20. Contrast and volume fraction cannot be separately identified from this curve. Recover Z directly; no absolute volume fraction or known physical contrast is implied.

Use the declared logarithmic radius coordinates and bin convention:

```math
r_j=10^{\log_{10}(10)+j[\log_{10}(30000)-\log_{10}(10)]/499},\quad 0\leq j<500,
\qquad I_i\simeq B+\sum_j K_{ij}Z_jw_j,
\qquad K_{ij}=10^{-4}V(r_j)F(q_i r_j)^2.
```

The endpoint bin widths are w0=r1-r0 and w499=r499-r498; interior widths are half the distance between adjacent outer neighbors. Each Zj is the density at rj. These full endpoint widths are part of the declared discretization. Distinct radius populations can produce nearly indistinguishable scattering curves, especially near the limits of the acquired angular range. Recovering a stable distribution requires a nonnegativity constraint and controlled inverse fitting. There is no spatial image axis: this is an ensemble parameter-density inverse problem.

Save `output/reconstruction.npz` relative to the task workspace with both arrays:

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| reconstruction | (1, 500) | float64, real | Nonnegative Z(r), in10^20 cm^-4 per angstrom; axes batch,radius in increasing order defined above |
| background | (1, 1) | float64, real | Nonnegative constant B in cm^-1; axes batch,scalar |

Evaluation version 2 requires nonnegative density, observed-background agreement and the declared weighted observation discrepancy over all 500 samples. No crop, amplitude rescaling, size-axis warping or alignment is performed. Measurements need not identify a unique microscopic size distribution; the accepted class is conditional on the sphere model and supplied nominal uncertainties.

## Data Description

All acquired q values, intensities and supplied nominal uncertainties are retained in `data/raw_data.npz` in their original order. Every supplied uncertainty equals3% of the same visible intensity to floating-point precision; these are observation-derived fit weights, not independently verified instrument noise or an additional clean signal. No fitted density, contrast estimate, cached model or hidden object is input.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| scattering_vector | (1, 500) | float64 | Positive increasing scattering-vector magnitude q in angstrom^-1; batch,observation |
| intensity | (1, 500) | float64 | Complete observed scattering intensity I in cm^-1; batch,observation |
| intensity_uncertainty | (1, 500) | float64 | Supplied positive standard uncertainty of intensity in cm^-1; batch,observation |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation count | 500 | samples | Complete author curve, no angular crop |
| q range | 0.00009999999747378752 to1.0 | angstrom^-1 | Actual endpoints; full coordinates are in the array |
| Radius coordinates | 500 logarithmic centers,10 to30000 | angstrom | Declared unknown grid; radius, not diameter |
| Endpoint widths | Full neighboring spacing | angstrom | Discrete bin-mass convention in the forward equation |
| Morphology | Uniform spheres | dimensionless | Common particle shape, isotropic orientations |
| Interparticle coupling | Neglected | dimensionless | Dilute-limit model, no structure factor |
| Geometry | Pinhole | dimensionless | No slit convolution supplied or assumed |
| Contrast normalization | C absorbed into Z |10^20 cm^-4| Unit contrast is a change of unknown, not an instrument calibration |
| Additive background | Unknown constant | cm^-1 | Infer from visible observations; no blank sample supplied |
| Uncertainty convention | 0.03 times the visible observed intensity | cm^-1 | Supplied nominal standard-deviation weights; no independent stochastic calibration is established; preserve without inflation |
| Additional instrument parameters | Not supplied | not applicable | The bound inverse starts from reduced q/I data; no wavelength, detector distance or physical sample identity is invented |

## Method Hints

The v2 target is the nonnegative measurement-compatible histogram class: all 500 observations must satisfy mean(((B+K(Z*w)-I)/sigma)^2)<=1.01. Background B is the arithmetic mean of the observed intensities with 0.5<=q<=1 inverse angstrom and must agree with this statistic to1e-6 inverse centimetres. Neighboring radius distributions can be physically indistinguishable at the stated measurement precision; no actual generating distribution is identified.


A nonnegative histogram inverse can use an interior-point gradient method. Build the sphere form-factor operator explicitly, weight residuals by the supplied uncertainties, and limit each descent step to keep bin masses positive. A discrepancy criterion compares weighted residual energy with the number of observations. Near zero argument, evaluate the sphere amplitude with a Taylor expansion to avoid cancellation. Treat the constant background as an observable nuisance parameter; high-q data constrain its level. Monitor how radius discretization and early stopping affect inferred populations, rather than assuming every narrow feature is resolved.

Nonnegative weighted least squares or another discrepancy-controlled inverse can find eligible bin masses x=Z*w. The supplied weights do not establish a calibrated stochastic goodness-of-fit test. Retain all observations, including negative values after subtracting B, and preserve density units without additional normalization.

## References

Ilavsky, J. and Jemian, P.R. (2009). *Irena: tool suite for modeling and analysis of small-angle scattering*. Journal of Applied Crystallography42,347–353. [Published paper](https://doi.org/10.1107/S0021889809002222).

Ilavsky, J. *Irena Size Distribution manual*, including the scattering integral, interior-point inverse, background inference and uncertainty analysis. Author size-distribution documentation. The selected implementation follows the author's later Python port; it is not represented as Python code released in2009.
