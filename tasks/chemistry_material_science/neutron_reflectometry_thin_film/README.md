# Neutron Reflectometry of a Buried Thin Film

> Reconstruct the depth dependent scattering length density of a buried thin film from a measured neutron specular reflectivity curve with finite momentum resolution and intensity uncertainty.
> Domain: chemistry & material science | Keywords: neutron imaging, interface reconstruction, inverse scattering | Difficulty: Hard

## Background

Neutrons reflect from changes in nuclear scattering length density at material interfaces. Interference between waves reflected from different depths encodes the thickness and composition of films far thinner than the spatial resolution of a conventional neutron camera. Hydrogen and deuterium have strongly different coherent scattering lengths, making neutron contrast particularly useful for soft matter in aqueous environments.

Specular reflectometry measures structure normal to an interface, averaged over the illuminated lateral area. Applications include polymer coatings, buried solid–liquid interfaces, membranes, corrosion layers, and adsorption. The detector measures intensity and loses phase, so many depth profiles can produce similar curves; finite angular and wavelength resolution further limits the available depth information.

## Problem Description

The observations are a real PLATYPUS neutron reflectivity measurement of a film at a silicon/oxide/heavy water interface. Reconstruct a one dimensional nuclear scattering length density (SLD) profile using a silicon fronting medium, a uniform oxide layer, a uniform film, and a heavy water backing medium. Use this layer ordering in the direction of increasing depth. The two finite layers have unknown thicknesses; the film SLD and the three interface roughnesses are unknown. An unknown intensity scale and a nonnegative, momentum independent background are nuisance parameters. Absorption is zero in this selected model.

Let depth be z, momentum transfer be Q, wavelength be lambda, and the grazing angle be theta, with Q = 4 pi sin(theta)/lambda. For layer j, its wavevector normal to the interfaces is


```math
k_j(Q)=\sqrt{Q^2/4-4\pi(\rho_j-\rho_0)},\qquad
r_j(Q)=\frac{k_j-k_{j+1}}{k_j+k_{j+1}}
\exp[-2k_jk_{j+1}\sigma_{j+1}^2].
```


Here rho is SLD in inverse square angstroms, rho0 is the silicon fronting SLD, sigma is root mean square interface roughness in angstroms, and the square root follows the outgoing wave convention (an infinitesimal negative imaginary argument). Propagation through each finite thickness d contributes phase factors exp(plus or minus i k d). Multiplying the interface and propagation matrices gives a complex reflected amplitude a(Q). The ideal reflectivity is its squared magnitude.

The measured intensity at point i follows


```math
R_i=s\int G_i(Q-Q_i)|a(Q)|^2\,dQ+b+\epsilon_i,
\qquad
\rho(z)=\rho_0+\sum_{j=1}^{3}(\rho_j-\rho_{j-1})
\frac{1+\operatorname{erf}[(z-z_j)/(\sqrt{2}\sigma_j)]}{2}.
```


G is the normalized Gaussian instrument resolution function with the supplied FWHM; s is intensity scale, b is background, epsilon is measurement error with supplied standard deviation, and erf is the Gaussian error function. The interface positions are 0, oxide thickness, and oxide plus film thickness. A uniform layer reflectivity model with Nevot–Croce interface damping defines the acquisition approximation; the displayed error function profile is its conventional depth representation. These approximations need not describe every microscopic interfacial feature exactly.

Save `output/reconstruction.npz` relative to the task workspace with both required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 821) | float64 | Real SLD in units of 10^-6 Angstrom^-2; axes (batch, increasing depth), z = -30 + 0.5 j Angstrom for j = 0,...,820. No magnitude, normalization, offset or alignment is applied. |
| parameters | (1, 8) | float64 | Real parameters in order: intensity scale (dimensionless), background reflectivity (dimensionless), oxide thickness (Angstrom), Si/oxide roughness (Angstrom), film thickness (Angstrom), film SLD (10^-6 Angstrom^-2), oxide/film roughness (Angstrom), film/water roughness (Angstrom). |

Evaluation checks both arrays. Profile NRMSE is RMS error divided by the reference profile dynamic range. Parameter error is the maximum absolute difference divided by the corresponding normalization scale `(0.6, 8.999e-6, 35, 14, 100, 2.9, 14, 14)` in the parameter units above. A forward consistency check evaluates the sum of squared log10 intensity residuals using propagated standard deviations `standard_error/(reflectivity ln(10))`. All metrics must meet their declared numerical agreement and consistency limits. No depth translation, intensity rescaling, cropping, or phase alignment occurs. Agreement with a standard inversion of these observations is the objective; it does not establish independently measured microscopic truth.

## Data Description

`data/raw_data.npz` contains all 408 measured points, in their original precision and ordering.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| q | (1, 408) | float64 | Momentum transfer Q, Angstrom^-1; axes (batch, observation). |
| reflectivity | (1, 408) | float64 | Measured specular intensity ratio, dimensionless; axes (batch, observation). |
| standard_error | (1, 408) | float64 | One standard deviation reflectivity uncertainty, dimensionless; axes (batch, observation). |
| q_fwhm | (1, 408) | float64 | Point dependent Gaussian Q resolution FWHM, Angstrom^-1; axes (batch, observation). |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument | PLATYPUS | — | Neutron reflectometer; supplied curve is already reduced to specular reflectivity. |
| Fronting SLD | 2.07 | 10^-6 Angstrom^-2 | Semi-infinite silicon incident medium. |
| Oxide SLD | 3.47 | 10^-6 Angstrom^-2 | Fixed uniform native oxide layer SLD. |
| Backing SLD | 6.36 | 10^-6 Angstrom^-2 | Semi-infinite heavy water exit medium. |
| Imaginary SLD | 0 | Angstrom^-2 | Absorption neglected for all four media. |
| Depth origin | 0 | Angstrom | Silicon/native oxide interface. |
| Positive depth | silicon toward water | — | Layer order: silicon, oxide, film, water. |
| Lateral convention | uniform layers | — | Specular approximation, laterally averaged structure. |
| Interface convention | Nevot–Croce | — | Gaussian roughness damping of interface amplitudes. |
| Resolution model | Gaussian, per observation FWHM | Angstrom^-1 | Widths supplied by q_fwhm; sigma = FWHM/(2 sqrt(2 ln(2))). |
| Boundary convention | semi-infinite fronting and backing | — | No second reflection beyond bounding media. |
| Noise information | supplied standard errors | dimensionless | Independent point uncertainty approximation; residual systematics may remain. |

The supplied Q values and widths are sufficient to formulate the reduced measurement model; unreleased wavelength or detector geometry is not required. No extra calibration observations are assumed.

## Method Hints

Use a stratified medium transfer matrix or equivalent reflection recursion with roughness and instrument resolution. The two uniform finite layers give a low dimensional physical parameterization of the depth profile. An error function representation makes the reconstructed interfaces readable in depth.

Bounded global maximum likelihood fitting can handle the oscillatory, nonconvex thickness dependence. Weighting log reflectivity by propagated uncertainties preserves sensitivity to high momentum fringes across the large intensity range. Treat intensity scale and background as nuisance parameters. This task selects the maximum likelihood reconstruction stage; posterior sampling is an optional uncertainty analysis, not a required output.

## References

Nelson, A. R. J. and Prescott, S. W. (2019). *refnx: neutron and X-ray reflectometry analysis in Python*. Journal of Applied Crystallography 52, 193–200. [Published paper](https://doi.org/10.1107/S1600576718017296).

Parratt, L. G. (1954). *Surface Studies of Solids by Total Reflection of X-Rays*. Physical Review 95, 359–369. [Physical model](https://doi.org/10.1103/PhysRev.95.359).
