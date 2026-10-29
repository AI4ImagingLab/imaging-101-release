# Spinel Film Magnetization Depth from Polarized Neutrons

> Recover the magnetic and nuclear depth profiles of a spinel film from two measured polarized-neutron reflectivity channels while separating a suppressed interface layer from structural roughness and beam broadening.
> Domain: physics | Keywords: neutron imaging, magnetic depth profiling, wave scattering | Difficulty: Hard

## Background

Neutrons interact with both atomic nuclei and magnetic moments. Reflectivity measured with spin polarization parallel or antiparallel to a film's magnetization probes different effective scattering potentials. Their contrast provides sensitivity to magnetization buried beneath a surface, while the shared reflectivity structure constrains thickness and composition.

Thin magnetic insulators can contain interfacial regions whose magnetization is much smaller than that of the film interior. Such regions influence spin transport and device performance. Depth-sensitive measurements are valuable because a bulk magnetization measurement alone cannot locate the suppressed region. Reflectometry remains an intensity-only inverse problem, so finite momentum coverage and instrumental resolution create ambiguity between layer thickness, interface mixing and magnetic structure.

## Problem Description

Use all 84 measured points in each of the non-spin-flip ++ and -- channels from a saturated Mg-Al-Fe spinel film on crystalline MgAl2O4. The known substrate nuclear scattering length density (SLD) is fixed; film nuclear SLD, magnetic SLD, thickness, an interfacial magnetically suppressed layer, two roughnesses, intensity normalization, angular offset and additional sample broadening are unknown. Assume collinear magnetization and no spin-flip scattering. The depth origin is the mean substrate/film interface; positive depth points toward the film surface and air.

For spin sign s=+1 or -1, the layer wave number is

```math
k_{j,s}(Q)=\sqrt{(Q/2)^2-4\pi[\rho_j+s\rho^M_j]+4\pi i\eta},\qquad
Q=\frac{4\pi}{\lambda}\sin(\theta+\delta).
```

Here j indexes layers from incident air toward the substrate, Q is momentum transfer in inverse Angstrom, lambda is neutron wavelength in Angstrom, theta is measured grazing angle and delta is its unknown offset (convert supplied degrees to radians). Nuclear and magnetic SLDs rho and rhoM are in inverse Angstrom squared. The infinitesimal eta=1e-16 inverse Angstrom squared fixes the same absorbing branch convention in both spin channels. A Gaussian interface of roughness sigma has amplitude factor

```math
f_{j,s}=\frac{k_{j,s}-k_{j+1,s}}{k_{j,s}+k_{j+1,s}}
\exp[-2k_{j,s}k_{j+1,s}\sigma_j^2].
```

Multiple reflection through finite layer thicknesses determines the total amplitude r; measured intensity is I times the resolution convolution of |r|², where I is an unknown dimensionless normalization. The film has one nuclear SLD throughout, but zero magnetic SLD in a thickness d adjacent to the substrate and uniform magnetic SLD m above it. Both buried boundaries use the fitted substrate-interface roughness; the surface uses its own roughness. The returned smooth profiles use Gaussian error-function interfaces. Therefore a nonzero d denotes a suppressed interface region, not an exactly zero smooth magnetic profile over a sharp interval.

Save `output/reconstruction.npz` with both required real float32 or float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 521, 2) | float32 or float64 | Batch, supplied depth coordinate, SLD channel; channel0 nuclear and channel1 magnetic, both in 10^-6 Angstrom^-2 |
| parameters | (1, 9) | float32 or float64 | Batch, physical/nuisance parameter in the exact order below; no rescaling or phase convention |

The parameter order, physical units and admissible ranges are:

| Index | Quantity | Unit | Admissible range |
|---|---|---|---|
| 0 | Incident intensity normalization I | dimensionless | [0.98,1.02] |
| 1 | Additional sample angular broadening omega, FWHM | degree | [0,0.08] |
| 2 | Grazing-angle offset delta | degree | [-0.02,0.02] |
| 3 | Buried nuclear/magnetic interface roughness | Angstrom | [1,15] |
| 4 | Film/air interface roughness | Angstrom | [1,15] |
| 5 | Suppressed magnetic-layer thickness d | Angstrom | [0,30] |
| 6 | Interior magnetic SLD m | 10^-6 Angstrom^-2 | [0,2] |
| 7 | Film nuclear SLD | 10^-6 Angstrom^-2 | [2,7] |
| 8 | Total film thickness h | Angstrom | [60,180] |

The suppressed layer must be thinner than the film. All parameters and both profile channels are evaluated. Metrics include maximum parameter error normalized by the declared parameter ranges, separate dynamic-range-normalized RMS errors for nuclear and magnetic profiles, profile/parameter physical consistency, and chi-square against all measured uncertainties with 168-9 degrees of freedom. No profile shift, amplitude rescaling, channel swap or cropping is permitted. Agreement with an independently reproduced standard-method result is not a measurement of error to a known magnetic ground truth.

## Data Description

The two channels are actual NCNR polarized-beam reflectometer measurements. They were corrected by the instrument reduction chain for measured background, polarization and footprint before distribution. Supplied uncertainty is treated diagonally; correction-induced covariance is unavailable. The sample broadening is deliberately unknown because it was not independently calibrated. Public arrays contain no inferred film properties.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reflectivity | (1, 2, 84) | float64 | Batch, spin channel (++ then --), measured point; reflected/incident intensity ratio, dimensionless |
| uncertainty | (1, 2, 84) | float64 | One-standard-deviation reflectivity uncertainty, dimensionless |
| q | (1, 2, 84) | float64 | Nominal momentum transfer in Angstrom^-1 before fitted angular offset |
| q_resolution | (1, 2, 84) | float64 | Instrument-only Gaussian Q standard deviation in Angstrom^-1 |
| angle | (1, 2, 84) | float64 | Grazing angle theta in degrees |
| angle_resolution | (1, 2, 84) | float64 | Instrument-only angular divergence FWHM in degrees |
| wavelength | (1, 2, 84) | float64 | Neutron wavelength in Angstrom |
| wavelength_resolution | (1, 2, 84) | float64 | Wavelength FWHM in Angstrom |
| depth | (1, 521) | float64 | Required profile coordinate from -40 to220 Angstrom in0.5 Angstrom increments |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument | NCNR Polarized Beam Reflectometer | dimensionless | Actual polarized neutron acquisition |
| Acquisition record | 171115B_30A45607 through45620, January2018 | instrument run ID | Distributed saturated-film measurements |
| Incident medium | air, nuclear and magnetic SLD0 | 10^-6 Angstrom^-2 | Front-side reflection |
| Substrate | crystalline MgAl2O4, nuclear SLD5.377, magnetic SLD0 | 10^-6 Angstrom^-2 | Known composition/density, semi-infinite |
| Neutron wavelength | 4.75 | Angstrom | Monochromatic beam |
| Relative wavelength FWHM | 0.029999999329447746 | dimensionless | Original file metadata; converted absolute values retained in array |
| Spin convention | ++ parallel, -- antiparallel to collinear film magnetization | dimensionless | No spin-flip observation is inferred |
| Background | 0 after measured-background subtraction | dimensionless | No fitted arbitrary background |
| Depth boundary | air/film/possibly suppressed film/substrate | dimensionless | Semi-infinite fronting and backing media |
| Interface model | Nevot-Croce Gaussian roughness | Angstrom | Same buried roughness at nuclear interface and magnetic onset |
| Resolution kernel | Gaussian; piecewise-linear intensity on measured-angle Q grid | dimensionless | Integral truncates at exp(-6.90775527898213703123) cutoff, retaining whole crossing intervals and renormalizing |
| Profile convention | Sum of error-function interface steps | dimensionless | Smooth output; fixed interface origin, no alignment |

Angular broadening adds to instrument angular FWHM before quadrature with wavelength dispersion. If sigmaQ is supplied Q standard deviation, theta is the original angle, sigmaTheta is supplied angular FWHM converted to radians and divided by sqrt(8 log2), and w is omega converted the same way, then

```math
\sigma_{Q,\mathrm{eff}}^2=\sigma_Q^2+
\left[\frac{4\pi}{\lambda}\cos\theta\right]^2(2w\sigma_\theta+w^2).
```

The offset shifts both evaluation Q and convolution centres; the width formula uses original angles. The resolution integral ends at the available Q range and renormalizes its truncated area. Use the original 84-point grid, without inserting an unreported oversampling grid.

## Method Hints

Collinear spin channels decouple into scalar layered-wave equations. Compute each interface amplitude and accumulate multiple reflections from the substrate toward air, preserving the complex propagation phase through each finite layer. Apply intensity resolution convolution before evaluating noise-weighted residuals. Bounded derivative-free optimization can jointly estimate structure, magnetic suppression and beam nuisance parameters. Compare the suppressed-interface model with a uniform magnetic film to assess the phase of spin-asymmetry oscillations; finite Q coverage and roughness make uncertainty and parameter correlation important.

## References

- Wisser et al. (2019), “Ultrathin interfacial layer with suppressed room temperature magnetization in magnesium aluminum ferrite thin films”, Applied Physics Letters115,132404. https://doi.org/10.1063/1.5111326
- Maranville et al. (2016), “Measurement and modeling of polarized specular neutron reflectivity in large magnetic fields”, arXiv1510.07912v2. https://arxiv.org/abs/1510.07912v2

The bound data and selected finite-layer model are the instrument author's educational spinel case. The scientific comparison is uniform versus suppressed interfacial magnetization; this task does not claim to reproduce the paper's complete MCMC posterior or supplementary measurements.
