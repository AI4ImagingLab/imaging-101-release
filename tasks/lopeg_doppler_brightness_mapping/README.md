# LO Peg Doppler Brightness Mapping

> Reconstruct a stellar surface brightness image from rotationally broadened intensity line profiles with incomplete surface visibility.
> Domain: astronomy | Keywords: stellar imaging, Doppler imaging, maximum entropy | Difficulty: Hard

## Background

A rotating star broadens absorption lines because approaching and receding surface elements have different Doppler shifts. A bright or dark surface region changes its contribution across the line as it rotates into view. Repeated spectra encode a spatial image even when the stellar disk cannot be resolved by a telescope.

Surface brightness maps help investigate starspots, stellar activity and rotational modulation. Limb darkening, inclination and partial phase coverage cause unequal spatial sensitivity. Fine mesh pixels therefore do not imply equally fine angular resolution, particularly on the far hemisphere.

## Problem Description

The observations are sixteen normalized least-squares-deconvolved (LSD) mean line profiles of LO Peg from August 2014. Treat these already extracted profiles as measured data; no unprovided raw echelle spectra or polarization calibration files are required. The supplied standard deviations are marginal errors. Use their diagonal weighting as a modeling assumption; the LSD extraction covariance is unavailable.

Let surface cell j have colatitude theta, longitude phi, area A, outward normal n and relative brightness b. At observing epoch o the unit viewing direction is u, projected rotational velocity is v, and mu = n dot u. Only mu > 0 contributes. Define the continuum weight W = A mu (1-eta+eta mu) g^beta on visible cells and zero otherwise, where eta is the linear limb coefficient and g is effective gravity normalized to polar gravity. Differential rotation has angular speed Omega(theta)=2 pi/P - dOmega cos(theta)^2; phase is Omega(theta)(JD-JDref). Each cell uses a Voigt absorption profile L at its Doppler-shifted rest wavelength lambda/(1+v/c). Its Gaussian width includes instrumental broadening in quadrature, while its Lorentz half-width is preserved.

```math
I_o(\lambda)=\frac{\sum_j W_{oj}b_j L_{oj}(\lambda)}{\sum_j W_{oj}b_j},\qquad
V_o(\lambda)=\frac{\sum_j W_{oj}b_j B_{oj}^{\rm los}D_{oj}(\lambda)}{\sum_j W_{oj}b_j},\quad
D=-4.6686\times10^{-12}\lambda_0^2 g_{\rm eff}\frac{dL}{d\lambda}.
```

Here I and V are continuum-normalized Stokes profiles, c is the speed of light in km/s, wavelengths are in nm, g_eff is the effective Lande factor, and B_los = u dot B is positive toward the observer in G. The coefficient is consequently in the nm/G convention. The measured velocity is converted to rest wavelength by lambda = lambda0[1+(velocity-Vr)/c]. The adopted scalar local absorption strength is calibrated from the measured mean intensity equivalent width using the uniform unit-brightness, zero-field model; it is not an independently supplied target map. Evaluate the discrete equivalent-width integral with right rectangular velocity-bin widths and repeat the last width at the final point.

The unknown is the spatial brightness b, with magnetic field fixed to zero; Stokes V is not supplied or needed. Continuum normalization makes a uniform brightness rescaling unidentifiable, so the unit photospheric scale and regularization convention fix this ambiguity. The far southern hemisphere has low or zero sensitivity, and longitude/latitude resolution is limited by the local line width and rotation.

Save `output/reconstruction.npz` relative to your task workspace with exactly the required physical array: `reconstruction`: float64, shape `(1, 4584)`, axes batch, surface cell; strictly positive continuum brightness relative to the unit unspotted photosphere, dimensionless. Do not multiply by limb or gravity darkening. No phase, sign, amplitude, rotation or translation alignment is applied in evaluation. Every surface cell is scored using cosine NCC and RMS error divided by the corresponding reference dynamic range. A forward standardized residual over all measured samples is also required. The benchmark compares a validated standard-method reconstruction, not actual surface truth.

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

Use positive-image maximum entropy with a unit default photosphere and cell-area weighting. Differentiate the continuum-normalized disk integral, including the brightness dependence of the denominator. This discourages artificial structure where observations provide little information. A small search subspace built from entropy and likelihood gradients can control curvature and enforce a target chi-square. Maximum entropy selects one regularized spatial solution; it does not remove the acquisition nullspace or establish the physical uniqueness of a recovered feature.

## References

- Folsom et al. (2018), *The evolution of surface magnetic fields in young solar-type stars II*, MNRAS 474, 4956–4987. [Method paper](https://doi.org/10.1093/mnras/stx3021), [author manuscript v1](https://arxiv.org/abs/1711.08636v1).
- Cang et al. (2020), *Magnetic field and prominences of the young, solar-like, ultra-rapid rotator V530 Persei*, A&A 643, A39. [Oblate geometry](https://doi.org/10.1051/0004-6361/202037693).
- Skilling and Bryan (1984), *Maximum entropy image reconstruction: general algorithm*, MNRAS 211, 111–124. [Algorithm](https://doi.org/10.1093/mnras/211.1.111).
- C. P. Folsom, ZDIpy author-distributed LO Peg August 2014 normalized LSD example and associated code, GPL-2.0; see included LICENSE. These observations are reused without synthesizing an unknown target.
