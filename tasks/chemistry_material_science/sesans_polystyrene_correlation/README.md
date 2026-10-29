# Polystyrene Microstructure Correlation from SESANS

> Reconstruct the radial two-phase microstructure autocovariance of a polystyrene suspension from a measured neutron spin echo depolarisation curve that encodes a projected spatial correlation.
> Domain: chemistry & material science | Keywords: neutron imaging, microstructure reconstruction, inverse scattering | Difficulty: Medium

## Background

Spin echo small-angle neutron scattering encodes scattering angle in the precession of a polarized neutron spin. Scattering from microscopic density variations reduces the measured polarization. Changing the spin echo length probes different real-space correlation distances, allowing structures much larger than those accessible with conventional small-angle scattering resolution to be investigated.

The measurement probes bulk microstructure, including materials that are optically opaque. Applications include colloidal suspensions, emulsions, porous media, powders and food materials. For an isotropic sample, the signal is related to a projection of its three-dimensional density correlation rather than a direct photograph of individual particles. Structural interpretation requires inverting that projection with an explicit microstructure model.

## Problem Description

The data are a real measurement of nominally two micrometer polystyrene spheres in a water/heavy water mixture. Infer the sphere radius R and particle volume fraction phi in an isotropic, monodisperse two-phase sphere model, then reconstruct its radial indicator autocovariance. This is a model-based bulk correlation reconstruction, not an image of particular particle locations or a general interacting suspension.

For momentum magnitude q in inverse Angstrom, the sphere amplitude and macroscopic scattering intensity are

```math
f(qR)=3\frac{\sin(qR)-qR\cos(qR)}{(qR)^3},\qquad f(0)=1,
\qquad I(q)=10^8(\Delta\rho)^2\phi(1-\phi)V f(qR)^2,
\qquad V=\frac{4\pi R^3}{3}.
```

Delta rho is the particle-minus-solvent scattering length density in Angstrom^-2, V is particle volume in Angstrom^3, and I has units cm^-1. The factor 10^8 converts inverse Angstrom to inverse centimeter. The model uses no additional interparticle structure factor. The complementary phase ambiguity of phi(1-phi) is resolved by requiring the particle fraction below one half.

Let z be spin echo length in Angstrom, lambda neutron wavelength in Angstrom, t sample thickness in cm, and P/P0 the sample-to-empty-beam polarization ratio. The supplied reduced depolarisation D follows

```math
D(z)=\frac{\ln(P(z)/P_0(z))}{t\lambda^2}=G(z)-G(0),
\qquad G(z)=\frac{1}{2\pi}\int_0^\infty q I(q)J_0(qz)\,dq.
```

J0 is the ordinary Bessel function of order zero. G and D have units Angstrom^-2 cm^-1. Measurement errors can make individual observed D values slightly positive. Keep their signed values.

The reduced task's finite forward-operator convention uses the released sampling-dependent logarithmic quadrature: qmin = 0.1(2 pi)/(N zmax), qmax = (2 pi)/(z[1]-z[0]), and samples `exp(log(qmin) + j log(1.0003))` below qmax, with N=40 measured lengths. The weight at sample j is q[j] times its backward q difference divided by 2 pi; the first difference equals the next difference. Apply these same weights to G(z) and G(0). The nominal angular acceptance is supplied below; every quadrature sample is inside it for these observations. This finite integration convention defines the numerical forward approximation, not an additional hidden calibration measurement.

The inferred dimensionless radial autocovariance C is

```math
C(r)=\phi(1-\phi)\gamma(r),\qquad
\gamma(r)=\begin{cases}
1-3r/(4R)+r^3/(16R^3),&0\le r\le2R,\\
0,&r>2R.
\end{cases}
```

Here r is positive three-dimensional separation in Angstrom; gamma is the normalized overlap correlation of a homogeneous sphere. The finite measured length range, projection and radius/concentration coupling constrain information about more general microstructures. The uniform sphere approximation fixes this otherwise underdetermined inversion.

Save `output/reconstruction.npz` relative to the task workspace with both arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 501) | float64 | Real dimensionless C(r); axes (batch, increasing radial separation), r=100 j Angstrom for j=0,...,500. |
| parameters | (1, 2) | float64 | Real `(radius, particle volume fraction)`, units `(Angstrom, dimensionless)`; radius is positive and particle fraction is between zero and one half. |

Evaluation checks both quantities and forward consistency against every measured depolarisation value. Profile NRMSE is RMS autocovariance error normalized by the reference autocovariance dynamic range. Parameter error is maximum absolute error normalized by `(10000 Angstrom, 0.1 volume fraction)`. Measurement consistency is the sum of squared `(predicted D - observed D)/standard_error`. No shift, rescaling, normalization or magnitude conversion is applied. Every metric must pass its declared boundary. The comparison is with a validated standard inversion of these real observations, without an independently measured exact microscopic truth.

## Data Description

`data/raw_data.npz` preserves all 40 reduced observations and their original precision.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| spin_echo_length | (1, 40) | float64 | Nominal spin echo correlation length z, Angstrom; axes (batch, observation). |
| depolarisation | (1, 40) | float64 | Signed reduced log depolarisation D, Angstrom^-2 cm^-1; axes (batch, observation). |
| standard_error | (1, 40) | float64 | One standard deviation uncertainty of D, Angstrom^-2 cm^-1; axes (batch, observation). |
| wavelength | (1, 40) | float64 | Nominal neutron wavelength lambda, Angstrom; axes (batch, observation). |
| length_standard_error | (1, 40) | float64 | Reported uncertainty in nominal spin echo length, Angstrom; axes (batch, observation). |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | polystyrene spheres in 53% H2O, 47% D2O | — | Real sample identified in the published data header; nominal diameter is approximate, not truth. |
| Measurement date | 2014-07-10 | calendar date | Date in original acquisition file. |
| Sample thickness t | 0.2 | cm | Physical path length used by the published reduced depolarisation. |
| Particle SLD | 1.41 | 10^-6 Angstrom^-2 | Known polystyrene scattering length density. |
| Solvent SLD | 2.70 | 10^-6 Angstrom^-2 | Known mixture scattering length density. |
| Encoding direction | Z | — | Laboratory gradient axis from original instrument file, not the radial reconstruction coordinate r. |
| Maximum vertical acceptance | 0.0168 | radian | Nominal acceptance angle in the encoding direction. |
| Maximum horizontal acceptance | 0.0168 | radian | Nominal orthogonal acceptance angle. |
| Scan type | sine, one element scan | — | Instrument acquisition mode. |
| D1 and D2 apertures | 20 by 8 | mm | Slit dimensions in the acquisition header. |
| Sample aperture | 16 by 10 | mm | Sample illumination dimensions. |
| Wavelength treatment | nominal stored values | Angstrom | No extra spectral convolution in the selected reduced sphere model. |
| Length uncertainty treatment | nominal coordinate fit | Angstrom | Reported errors retained; the selected standard model does not convolve over them. |
| Radial symmetry | isotropic, homogeneous spheres | — | Sphere overlap correlation model, no spatial particle localization. |
| Correlation boundary | zero beyond one particle diameter | — | Model support of the single-sphere overlap term. |

The reduced D and its standard errors are authoritative measurements. Do not recompute them from a rounded polarization table or infer unreported sample truth. No raw flipper counts or hidden calibration image is needed.

## Method Hints

Fit a physical sphere scattering model through the explicit Bessel/Hankel projection, using the measured depolarisation uncertainties. A bounded nonlinear least squares fit can estimate radius and particle fraction together. The sphere overlap correlation then maps the inferred physical quantities back to three-dimensional separation space.

Use a stable small-argument form of the sphere amplitude and retain the intensity unit conversion and G(0) subtraction. A Gaussian substitute for the correlation or a direct identification of projected depolarisation with C(r) changes the inverse problem.

## References

Kadletz, E., Bouwman, W. G. and Pappas, C. (2022). *Radial spin echo small-angle neutron scattering method: concept and performance*. Journal of Applied Crystallography 55, 1072–1084. [Published formalism, including conventional linear SESANS](https://doi.org/10.1107/S1600576722007245).

Andersson, R., van Heijkamp, L. F., de Schepper, I. M. and Bouwman, W. G. (2008). *Analysis of spin-echo small-angle neutron scattering measurements*. Journal of Applied Crystallography 41, 868–885. [Analysis background](https://doi.org/10.1107/S0021889808026770).
