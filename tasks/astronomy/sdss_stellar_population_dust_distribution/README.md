# SDSS Stellar Population and Dust Distribution

> Reconstruct a galaxy's stellar population light distribution and dust attenuation from a measured optical spectrum whose age, metallicity and continuum effects are degenerate.
> Domain: astronomy | Keywords: spectral inversion, stellar populations, dust attenuation | Difficulty: Hard

## Background

An unresolved galaxy combines light from stars with different ages and chemical abundances. Absorption features encode their atmospheric conditions, while stellar evolution changes the relative numbers and luminosities of stars. Recovering a population distribution helps study how galaxies assembled and enriched their stars.

Interstellar dust changes the spectral continuum, and instrumental broadening blends narrow features. Old, metal-rich and dust-reddened populations can resemble one another. Full-spectrum inversion uses many absorption features together, but its population weights remain conditional on the adopted stellar library and calibration.

## Problem Description

The sample is the measured SDSS galaxy spectrum identified by plate0266, MJD51602, fiber0001. Let $y_i$ be its calibrated spectral flux density and $s_i$ its measured standard error at rest-frame vacuum wavelength $\lambda_i$. Let $T_{ji}$ be the independent template luminosity density for population node $j$, already matched to catalog velocity dispersion, instrumental resolution and foreground Galactic reddening. Unknown positive relative attenuation $a_i$ and nonnegative light weights $w_j$ obey

```math
\mu_i=\sum_{j=1}^{198}w_j B_{ji}(a),\qquad
B_{ji}(a)=\frac{T_{ji}a_i}{\mathrm{median}_{k}(T_{jk}a_k)}\mathrm{median}_{k}(y_k),\qquad
y_i=\mu_i+\epsilon_i,\quad \sum_jw_j=1.
```

Here $B$ is the attenuated template on the observation flux scale, $\mu$ the predicted spectrum, and $\epsilon$ the measurement/model residual. The supplied standard errors describe a diagonal noise approximation. Relative attenuation is normalized through the inferred continuum ratio; its absolute grey scale cannot be identified from normalized mixtures. The color excess $E=E(B-V)$ summarizes a compatible normalized Calzetti curve, rather than forcing the empirical curve to equal that analytic law. Age and metallicity coordinates are given for every template; weights refer to this exact index order, including distinct library nodes with equal coordinates.

Save `output/reconstruction.npz` with all three real numeric float32 or float64 arrays: `reconstruction` shape(1,198), dimensionless nonnegative light fractions in public template order summing to one within1e-5; `attenuation` shape(1,2308), strictly positive dimensionless relative transmission on the fitting wavelength axis; and `ebv` shape(1,1), color excess in magnitudes between0 and1.5. Positive common grey transmission is unidentifiable because each attenuated template is median-normalized. Version2 therefore treats curves differing by a positive constant as equivalent. Curve regularity and the color summary use a/median(a); this explicitly declared scale equivalence never uses a hidden target. No wavelength alignment or population rescaling is fitted.

Evaluation version2 defines a conditional nonnegative-population/smooth-dust class, retaining all original unknowns and every supported observed point. Require mean_i[((mu_i-y_i)/s_i)^2]<=1.0374388649929949 over all2308 points, without clipping or estimating new noise. The log-attenuation total variation is the sum of |log(a[i+1])-log(a[i])| over2307 consecutive supplied wavelength pairs, without periodic wrap or wavelength-spacing divisor. It must be <=0.2835785786811014. For normalized curve a0=a/median(a) and normalized Calzetti law c0 at the submitted E, require sqrt(mean_i[(log(a0[i])-log(c0[i]))^2])<=0.04561968339812343. These two new regularity/summary budgets are the independently reproduced feasible standard values at both declared output precisions, plus relative1e-8 and absolute1e-10 numerical allowance. They are not independently calibrated confidence limits.

For l=lambda/10000 in micrometres, c(E,lambda)=10^(-0.4 E k(l)); normalize by its median on the supplied2308-point grid. For l<0.63, k=2.659(-2.156+1.509/l-0.198/l^2+0.011/l^3)+4.05; for0.63<=l<=2.2, k=2.659(-1.857+1.040/l)+4.05. All supplied wavelengths lie in this definition's range. The E output remains in magnitudes and must lie in[0,1.5]. Its curve-distance criterion evaluates its physical summary role rather than comparing it with a private value.

Every weight enters the measured-spectrum prediction, attenuation enters both prediction and regularity, and E enters the summary law. No hidden population, attenuation or color-excess identity is required. This conditional class does not claim equivalence to finite FIREFLY candidate/BIC selection, uniqueness of the population, or independently known true dust.

## Data Description

All arrays reside in `data/raw_data.npz`; the first axis is a single acquisition batch. Float32 observations preserve original FITS precision (some are stored big-endian); templates and population coordinates use float64.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| wavelength | (1,2308) | float32 | Supported rest-frame vacuum wavelength, Angstrom |
| flux | (1,2308) | float32 | Measured supported flux density, 1e-17 erg s^-1 cm^-2 Angstrom^-1 |
| uncertainty | (1,2308) | float32 | One-standard-error flux uncertainty, same flux units |
| templates | (1,198,2308) | float64 | Independent matched SSP luminosity density, erg s^-1 Angstrom^-1 per solar mass; template then wavelength axes |
| age | (1,198) | float64 | Age of each template, Gyr |
| metallicity | (1,198) | float64 | Metal abundance relative to solar, dimensionless |
| observed_wavelength | (1,3835) | float32 | Complete original detector-spectrum observed vacuum wavelength, Angstrom |
| observed_flux | (1,3835) | float32 | Complete original measured flux density, same flux units |
| observed_inverse_variance | (1,3835) | float32 | Original inverse variance, inverse squared flux units; zero marks unavailable uncertainty |
| context_wavelength | (1,2327) | float32 | Matched pre-support-cut rest vacuum wavelength, Angstrom |
| context_flux | (1,2327) | float32 | Measured pre-support-cut context spectrum, same flux units |
| context_uncertainty | (1,2327) | float32 | Context one-standard-error uncertainty, same flux units |
| support_mask | (1,2327) | bool | Dimensionless fixed template-support/matching mask selecting the2308 fitting wavelengths |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Survey instrument | SDSS optical fiber spectrograph | identifier | Real galaxy acquisition, plate0266/MJD51602/fiber0001 |
| Right ascension / declination | 145.89219 / 0.059372 | degree | Catalog sky coordinates |
| Catalog redshift | 0.02127545326948166 | dimensionless | Rest wavelength = observed wavelength/(1+redshift) |
| Catalog stellar velocity dispersion | 135.89956665039062 | km/s | Independently cataloged spectral-broadening input |
| Instrument resolving power | 2000 | dimensionless | Original example convention used to match templates |
| Template library | M11 MILES, Kroupa initial mass function | identifier | Independent SSP luminosity templates; nominal native FWHM2.55 Angstrom |
| Foreground Galactic color excess | 0.0938291996717453 | mag | Independent SFD map value included in matched templates |
| Declared fitting support | 3750 to7400 | rest vacuum Angstrom | Interior to independent MILES7429-Angstrom boundary; original matching discards final incomplete bin |
| Actual fitting grid | 3751.087890625 to7397.02685546875 | Angstrom | 2308 valid matched samples; preserve supplied irregular gaps |
| Emission masking | 20 | Angstrom | Original recommended mask around HeII,NeV,OII,NeIII,H-zeta,H-epsilon,H-delta,H-gamma,OIII,ArIV,H-beta,NI,HeI,OI,NII,H-alpha,SII,ArIII lines |
| Noise | supplied standard errors | flux units | Diagonal approximation; no synthetic noise added |
| Boundary convention | supplied finite matched band | identifier | No extrapolation into template padding; context mask is fixed without target fitting |

Full raw observations and pre-cut context are preserved. The upper support removes the independently known template-convolution boundary, including zero padding; it does not select points by fitted residuals. The fitting arrays are exactly the context rows identified by `support_mask`. Catalog broadening and templates are legitimate prior calibration; no population or intrinsic attenuation estimated from this galaxy is supplied.

## Method Hints

Nonnegative mixtures of the calibrated stellar populations can explain absorption features while a smooth relative transmission captures continuum attenuation. For a fixed curve the population weights solve a simplex-constrained weighted spectral fit; a logarithmic curve representation preserves positivity. Constrained optimization or variable projection can combine measured-spectrum consistency with the stated total-variation and color-summary conditions. Age, metallicity and dust degeneracies imply an admissible family, not an exact reconstruction of an unobserved population catalog.

## References

Wilkinson et al., *FIREFLY: a full spectral fitting code*, MNRAS472,4297–4326 (2017), https://doi.org/10.1093/mnras/stx2215 ; arXiv1711.00865v1.

Maraston and Strömbäck, *Stellar population models at high spectral resolution*, MNRAS418,2785 (2011), https://doi.org/10.1111/j.1365-2966.2011.19738.x. SDSS observations are described above.
