# HD 209458b atmospheric transmission retrieval

> Infer atmospheric temperature, composition, clouds and scattering together with planetary and stellar nuisance parameters from measured multi-instrument exoplanet transit depths despite degenerate spectral signatures and heterogeneous uncertainties.
> Domain: astronomy | Keywords: atmospheric retrieval, transmission spectroscopy, Bayesian inverse problems | Difficulty: Hard

## Background

During a planetary transit, starlight passes through the atmosphere near the planetary limb. At wavelengths with stronger molecular absorption or scattering, a larger effective planetary area blocks the star. Measuring this wavelength-dependent transit depth allows remote inference of atmospheric conditions for planets that cannot be spatially resolved.

Gas abundances, atmospheric scale height, aerosol scattering and opaque clouds can produce similar spectral changes. Stellar and planetary geometry also affect the signal. Observations from multiple instruments broaden wavelength coverage but introduce heterogeneous uncertainties and calibration histories. Atmospheric retrieval therefore requires a physically constrained forward model and explicit uncertainty treatment.

## Problem Description

Use all 42 measured bands of the hot Jupiter HD209458b to recover eight parameters. A spherically symmetric, isothermal, hydrostatic atmosphere has local extinction coefficient alpha(lambda,r), chord optical depth tau(lambda,b), and transit depth

```math
\tau(\lambda,b)=\int_{\mathrm{chord}}\alpha(\lambda,r)\,\mathrm{d}s,\qquad
D(\lambda)=\frac{r_c^2+2\int_{r_c}^{r_t}[1-\exp(-\tau(\lambda,b))]b\,\mathrm{d}b}{R_s^2}.
```

Here lambda is wavelength in m, r is radius in m, s is chord distance in m, b is the ray impact radius in m, r-c is the opaque cloud/base radius, r-t is the top atmospheric radius, R-s is stellar radius, and D is blocked stellar flux fraction. Hydrostatic equilibrium satisfies dP/dr=-G M-p mu m-u P/(k-B T r squared), with pressure P in Pa, planet mass M-p in kg, mean molecular mass mu in atomic mass units, atomic mass constant m-u, Boltzmann constant k-B and gravitational constant G. The radius R-p is defined at 1 bar. Fixed equilibrium chemistry determines species mixing ratios from temperature T, pressure, metallicity and C/O ratio. Gas extinction, enhanced Rayleigh scattering and collision-induced absorption all contribute; clouds are opaque below their top pressure.

Each observed band depth is the stellar-flux-weighted average of D on its wavelength interval. Treat measurements as independent Gaussian values with their supplied nominal uncertainties multiplied by a common unknown positive error factor. Jointly infer this factor rather than silently inflating reported errors. Stellar radius and planetary mass have independent Gaussian constraints listed below. The remaining parameters are degenerate, particularly metallicity, clouds, scattering and reference radius; no unique atmospheric truth is available from these bands alone.

Save `output/reconstruction.npz` relative to the task workspace with all three real float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,8) | float64 | Component-wise posterior medians, in parameter order below |
| best_fit | (1,8) | float64 | Highest-posterior sampled parameter vector, same order/units |
| transit_depth | (1,42) | float64 | Dimensionless band depths computed physically from best_fit, in exact input band order |

Parameter order is stellar radius R-s [m], planetary mass M-p [kg], reference planetary radius R-p [m], atmospheric temperature T [K], log10 scattering enhancement [dimensionless], log10 metallicity relative to solar [dimensionless], log10 cloud-top pressure [Pa], and Gaussian uncertainty multiplier [dimensionless]. All values must be finite. Radii, mass and error factor are positive; T lies strictly between 300 and 3000 K, log scattering between 0 and 5, log metallicity between -1 and 3, and log cloud pressure between -0.99 and 5. The selected finite sampling protocol summarizes all 50walkers and 1000transitions without burn-in removal; this is a reproducibility convention, not a proof of asymptotic convergence. Initialization and sampler details may be guided by the cited method.

Evaluation checks every parameter in both summaries using fixed parameter-wise scales, every band depth, physical consistency between the best vector and spectrum, unscaled observation chi-square, and normalized log posterior including the inferred error multiplier. There is no fitted rescaling, band dropping, alignment or crop. Agreement with a standard method is distinct from knowledge of the true planetary atmosphere.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| wavelength_bins | (1, 42, 2) | float64 | m; batch, observed band, lower/upper edge; upper edge exclusive |
| transit_depth | (1, 42) | float64 | dimensionless stellar flux fraction blocked; 42 author-processed observed bands |
| uncertainty | (1, 42) | float64 | dimensionless nominal one-sigma uncertainty per observed depth |
| wavelength_grid | (1, 2575) | float64 | m; retained common high-resolution spectral grid |
| pressure_grid | (1, 13) | float64 | Pa; ascending thermochemical/opacity pressure nodes |
| temperature_grid | (1, 29) | float64 | K; ascending calibration temperature nodes |
| metallicity_grid | (1, 41) | float64 | log10 solar metallicity units; equilibrium chemistry nodes |
| co_grid | (1, 2) | float64 | dimensionless C/O number ratio nodes bracketing fixed 0.53 |
| log_abundance_grid | (1, 41, 2, 34, 29, 13) | float64 | log10 dimensionless number mixing ratios; axes batch,metallicity,C/O,species,temperature,pressure |
| opacity_grid | (1, 28, 29, 13, 2575) | float64 | m^-1 per unit number mixing fraction at tabulated T/P; axes batch,absorber,temperature,pressure,wavelength |
| absorber_species | (1, 28) | int64 | dimensionless zero-based indices into species list; absorber order |
| species_mass | (1, 34) | float64 | atomic mass units (u); species mean particle masses |
| species_polarizability | (1, 34) | float64 | m^3; molecular polarizability volumes; zeros for omitted Rayleigh species |
| collision_coefficients | (1, 13, 29, 2575) | float64 | m^5; pair collision coefficient; axes batch,pair,temperature,wavelength |
| collision_species | (1, 13, 2) | int64 | dimensionless zero-based species indices; axes batch,pair,pairmember |
| stellar_weights | (1, 2575) | float64 | relative stellar photon-flux times wavelength-spacing weights; common arbitrary normalization cancels; wavelength order |

The first 10bands are HST/STIS, the next 28HST/WFC3, and final 4Spitzer. They are real published observations as assembled by the author example, with its 60 ppm STIS offset and Spitzer radius-ratio combination weights1/sigma retained. These processed values are not raw telescope exposures; do not undo those operations. All bands and all active physics calibration entries are retained. Opacity and equilibrium-chemistry tables are independent scientific calibration inputs, not atmosphere reconstructions.

Species index order: 0:H, 1:He, 2:C, 3:N, 4:O, 5:Na, 6:K, 7:H2, 8:N2, 9:O2, 10:OH, 11:HF, 12:MgH, 13:SiH, 14:HCl, 15:CO, 16:NO, 17:SiO, 18:TiO, 19:VO, 20:HCN, 21:H2CO, 22:CH4, 23:OCS, 24:CO2, 25:C2H2, 26:C2H4, 27:H2O, 28:H2S, 29:NH3, 30:PH3, 31:NO2, 32:SO2, 33:O3.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Target | HD209458b | identifier | Transiting hot Jupiter |
| Reference pressure | 100000 | Pa | Planetary radius definition |
| Stellar effective temperature | 6091 | K | Known stellar spectrum selection |
| Stellar radius constraint | 1.19 +/-0.02 | solar radii | Independent Gaussian constraint |
| Planet mass constraint | 0.73 +/-0.04 | Jupiter masses | Independent Gaussian constraint |
| Solar radius | 695700000 | m | Unit conversion |
| Solar mass | 1.98848e30 | kg | Stellar-mass convention in atmospheric binding guard |
| Gravitational constant | 6.67408e-11 | m^3 kg^-1 s^-2 | Fixed forward-model convention |
| Boltzmann constant | 1.38064852e-23 | J K^-1 | Fixed opacity/hydrostatic convention |
| Atomic mass constant | 1.660539040e-27 | kg | Conversion from mean particle mass in u |
| Radial quadrature | 250 | pressure nodes | Uniform log-pressure forward discretization |
| Jupiter radius | 71492000 | m | Unit conversion |
| Jupiter mass | 1.89819e27 | kg | Unit conversion |
| Carbon/oxygen ratio | 0.53 | dimensionless | Fixed equilibrium chemistry input |
| Scattering spectral index | 4 | dimensionless | Rayleigh wavelength dependence |
| Starspots | absent | convention | No additional spot correction |
| Atmospheric geometry | spherical hydrostatic isothermal | convention | Top/bottom pressures supplied in pressure_grid; opaque lower cloud boundary |
| Bin edge rule | lower inclusive, upper exclusive | convention | Stellar-weighted discrete wavelength selection |
| Noise | supplied nominal independent Gaussian errors | depth fraction | A common positive scale is inferred jointly |

## Method Hints

The Bayesian target has independent Gaussian stellar-radius and planet-mass constraints from the metadata. In the declared parameter coordinates, the remaining log-prior is zero inside the strict support Rp>0, 300<T<3000 K, 0<logscatter<5, -1<logZ<3, -.99<logcloud<5 and errorfactor>0, and minus infinity outside. These logarithms are base10, with cloud pressure in Pa; radius and mass use the stated SI conversions. The log likelihood is -.5 times the sum of squared residuals divided by (errorfactor times supplied error)^2 plus log(2 pi (errorfactor times supplied error)^2). The normalization must remain because the noise scale is unknown.

An equilibrium-chemistry radiative-transfer model can be combined with affine-invariant ensemble sampling. Use the fixed250node log-pressure radial grid. Interpolate log number mixing ratios linearly in temperature, log pressure, log metallicity and log C/O; interpolate extinction linearly in temperature and pressure. Integrate spherical hydrostatic structure and midpoint layer extinction along chord paths, then use lower-boundary radius times layer width in the area quadrature and average predicted depths over supplied bands using stellar weights. The visible profile includes only pressures strictly below the cloud-top pressure; mean particle mass is the sum of species mixing ratio times species mass without additional renormalization. An opaque cloud deck removes deeper atmospheric shells. Include Gaussian ancillary constraints and the full Gaussian likelihood normalization when the noise scale is unknown. Degeneracies make posterior distributions and chain diagnostics essential; a best-fit spectrum alone does not determine credible atmospheric parameters.

## References

- Zhang et al., *Forward modelling and retrievals with PLATON, a fast open source tool*,2019. [Published-method preprint](https://arxiv.org/abs/1811.11761v3). This task selects the author v4 implementation of that physics and retrieval family; its discretization is explicitly versioned.
- [STIS transit data](https://doi.org/10.1086/510111), [WFC3 transit data](https://arxiv.org/abs/1302.1141), and [Spitzer transit data](https://arxiv.org/abs/1504.05942), as processed in the author example.
- Goodman and Weare, affine-invariant ensemble samplers; Foreman-Mackey etal, [emcee](https://doi.org/10.1086/670067),2013.
