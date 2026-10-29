# LO Peg Zeeman Vector Mapping

> Reconstruct the three-component stellar surface magnetic field from rotationally modulated circular-polarization line profiles with incomplete surface visibility.
> Domain: astronomy | Keywords: stellar imaging, spectropolarimetry, maximum entropy | Difficulty: Hard

## Background

Circular polarization of stellar absorption lines measures the component of the magnetic field along the line of sight through the weak Zeeman effect. Rotation shifts contributions from different surface locations to different wavelengths. A sequence of polarized spectra therefore encodes spatial information about the surface vector field.

Such maps help characterize stellar dynamos, activity and wind coupling. Cancellation of opposite polarities and the changing projected view limit sensitivity to small scales and create cross-talk between vector components. A surface map is an inference under a specified atmosphere and geometry, rather than a direct photograph.

## Problem Description

The observations are sixteen normalized least-squares-deconvolved (LSD) mean line profiles of LO Peg from August 2014. Treat these already extracted profiles as measured data; no unprovided raw echelle spectra or polarization calibration files are required. The supplied standard deviations are marginal errors. Use their diagonal weighting as a modeling assumption; the LSD extraction covariance is unavailable.

Let surface cell j have colatitude theta, longitude phi, area A, outward normal n and relative brightness b. At observing epoch o the unit viewing direction is u, projected rotational velocity is v, and mu = n dot u. Only mu > 0 contributes. Define the continuum weight W = A mu (1-eta+eta mu) g^beta on visible cells and zero otherwise, where eta is the linear limb coefficient and g is effective gravity normalized to polar gravity. Differential rotation has angular speed Omega(theta)=2 pi/P - dOmega cos(theta)^2; phase is Omega(theta)(JD-JDref). Each cell uses a Voigt absorption profile L at its Doppler-shifted rest wavelength lambda/(1+v/c). Its Gaussian width includes instrumental broadening in quadrature, while its Lorentz half-width is preserved.

```math
I_o(\lambda)=\frac{\sum_j W_{oj}b_j L_{oj}(\lambda)}{\sum_j W_{oj}b_j},\qquad
V_o(\lambda)=\frac{\sum_j W_{oj}b_j B_{oj}^{\rm los}D_{oj}(\lambda)}{\sum_j W_{oj}b_j},\quad
D=-4.6686\times10^{-12}\lambda_0^2 g_{\rm eff}\frac{dL}{d\lambda}.
```

Here I and V are continuum-normalized Stokes profiles, c is the speed of light in km/s, wavelengths are in nm, g_eff is the effective Lande factor, and B_los = u dot B is positive toward the observer in G. The coefficient is consequently in the nm/G convention. The measured velocity is converted to rest wavelength by lambda = lambda0[1+(velocity-Vr)/c]. The adopted scalar local absorption strength is calibrated from the measured mean intensity equivalent width using the uniform unit-brightness, zero-field model; it is not an independently supplied target map. Evaluate the discrete equivalent-width integral with right rectangular velocity-bin widths and repeat the last width at the final point.

The unknown is the spatial vector field B, with brightness fixed to b=1; the measured Stokes I channel is used only for line-strength calibration. The I-derived brightness task is not an input to this inversion. Opposite polarities cancel in V, and a single line-of-sight component at each phase cannot uniquely determine every surface vector.

Save `output/reconstruction.npz` relative to your task workspace with exactly the required physical array: `reconstruction`: float64, shape `(1, 3, 4584)`, axes batch, vector component, surface cell; components are outward radial, increasing-colatitude (southward), increasing-longitude (right-handed eastward), all in gauss (G). No phase, sign, amplitude, rotation or translation alignment is applied in evaluation. Evaluation version2 requires the field to lie in the stated degree1..15 vector-harmonic model, meet the original full-profile mean standardized squared residual <=1.002, and have negative signed entropy <=12657.736914129822 in the coefficient convention below. Harmonic reconstruction relative L2 error must be <=1e-10; a zero field has zero harmonic error but fails the measured profiles. No hidden map identity is required. The entropy limit is the independently reproduced feasible standard cost plus relative1e-8 numerical allowance, retaining its selected spatial prior. No actual surface-field truth exists.

Cell order is ring-major north to south, then increasing longitude. There are 60 rings, theta_i=pi(i+1/2)/60 for i=0,...,59. Ring i has N_i=round[120 sin(theta_i)] cells, and phi_j=2pi(j+1/2)/N_i. Use nearest-integer rounding; no midpoint tie occurs. This produces 4584 cells. The reference epoch defines longitude zero.

The surface is the Roche equipotential with dimensionless radius r(theta)=3 cos[(pi+acos(w sin(theta)))/3]/[w sin(theta)], normalized by the polar radius. Here w=(2pi/(86400P))/sqrt[(8/27)GM/Rp^3], G is the gravitational constant, M is stellar mass and Rp is the adopted radius scale. Use the gradient of the effective potential for the surface normal and effective gravity; integrate the actual curved surface area of each ring cell. The adopted velocity scale vsini/sin(inclination) is normalized to the equatorial cell radius and is independent of the adopted mass/radius shape parameters.

## Data Description

All arrays in `data/raw_data.npz` use a leading batch axis of one. Epoch order follows increasing Julian date; velocity samples remain in their measured order.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| velocity | (1, 16, 89) | float64 | Measured Doppler velocity before systemic-velocity subtraction, km/s. |
| stokes_i | (1, 16, 89) | float64 | Continuum-normalized intensity I, dimensionless. |
| sigma_i | (1, 16, 89) | float64 | Reported one-sigma uncertainty of normalized I, dimensionless. |
| julian_date | (1, 16) | float64 | Observation Julian date, days. |
| stokes_v | (1, 16, 89) | float64 | Circular polarization V divided by continuum intensity, dimensionless; source sign retained. |
| sigma_v | (1, 16, 89) | float64 | Reported one-sigma uncertainty of normalized V, dimensionless. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Inclination | 45 | deg | Rotation axis to line of sight. |
| Projected equatorial speed | 67.7 | km/s | Adopted vsini. |
| Equatorial period | 0.4232 | day | P in the phase and shape models. |
| Equator-pole shear | 0.02 | rad/day | dOmega; equator rotates faster. |
| Reference epoch | 2456892.015 | Julian day | Longitude-zero reference. |
| Systemic velocity | -19.8 | km/s | Vr for every observation. |
| Included velocity range | -80 to 80 | km/s | Relative to Vr; inclusive measured sample selection. |
| Stellar mass | 0.66 | solar mass | Roche shape input. |
| Adopted polar radius scale | 0.72 | solar radius | Shape input; independent of velocity scale. |
| Gravitational constant | 6.67408e-11 | m^3 kg^-1 s^-2 | Roche convention. |
| Solar mass | 1.98892e30 | kg | Mass conversion. |
| Solar radius | 6.955e8 | m | Radius conversion. |
| Speed of light | 299792.458 | km/s | Wavelength and Doppler conversion. |
| Mean wavelength | 650 | nm | lambda0 of the LSD line. |
| Intrinsic Gaussian width | 2.41 | km/s | sqrt(2) sigma, not FWHM. |
| Lorentz/Gaussian ratio | 0.89 | 1 | Intrinsic Lorentz HWHM divided by Gaussian width. |
| Effective Lande factor | 1.195 | 1 | Weak-field polarization coefficient. |
| Resolving power | 65000 | 1 | Gaussian instrumental FWHM c/R. |
| Limb coefficient | 0.66 | 1 | Linear continuum limb darkening. |
| Gravity exponent | 0.5 | 1 | Continuum scales as normalized effective gravity to beta. |
| Surface coordinate convention | right-handed longitude, colatitude from north | rad | Viewing longitude decreases as rotation phase increases. |
| Time/polarization convention | supplied Stokes V; B_los toward observer positive | 1 | Preserve the weak-field sign above. |

## Method Hints

Use the full vector spherical-harmonic space with degree l=1,...,15 and nonnegative order m=0,...,l, including independent complex alpha,beta,gamma coefficients. Let P_l^m(cos theta) be the Condon--Shortley associated Legendre function, c=sqrt[(2l+1)(l-m)!/(4 pi (l+m)!)], and

```math
Y=cP_l^m(\cos\theta)e^{im\phi},\qquad
X=\frac{imY}{(l+1)\sin\theta},\qquad
Z=\frac{c}{l+1}\frac{dP_l^m(\cos\theta)}{d\theta}e^{im\phi}.
```

Sum over all135(l,m) pairs. The field is Br=Re(sum alpha Y), Btheta=-Re(sum(beta Z+gamma X)), Bphi=-Re(sum(beta X-gamma Z)). Coefficients have unitsG. Imaginary m=0 coefficients are unobservable and fixed to zero. This convention excludes a magnetic monopole and defines the model space independently of a solver.

For each real or imaginary coefficient a, let prior width mB=100G and psi=sqrt(a^2+4mB^2). The signed entropy is S=sum l[psi-2mB-a asinh(a/(2mB))], where each of the six coefficient blocks carries the same degree weight. The numerical cost -S is inG in this coefficient convention. The criterion above bounds -S; it is not an area-weighted entropy of individual surface pixels. This strictly convex negative-entropy prior suppresses unconstrained fine structure while the full measured profiles set the fit quality. Constrained maximum entropy or general convex optimization can find admissible coefficients without reproducing a particular iteration trajectory.

## References

- Folsom et al. (2018), *The evolution of surface magnetic fields in young solar-type stars II*, MNRAS 474, 4956–4987. [Method paper](https://doi.org/10.1093/mnras/stx3021), [author manuscript v1](https://arxiv.org/abs/1711.08636v1).
- Cang et al. (2020), *Magnetic field and prominences of the young, solar-like, ultra-rapid rotator V530 Persei*, A&A 643, A39. [Oblate geometry](https://doi.org/10.1051/0004-6361/202037693).
- Skilling and Bryan (1984), *Maximum entropy image reconstruction: general algorithm*, MNRAS 211, 111–124. [Algorithm](https://doi.org/10.1093/mnras/211.1.111).
- C. P. Folsom, ZDIpy author-distributed LO Peg August 2014 normalized LSD example and associated code, GPL-2.0; see included LICENSE. These observations are reused without synthesizing an unknown target.
