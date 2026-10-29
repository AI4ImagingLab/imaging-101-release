# Iron Sites from Mössbauer Transmission

> Recover iron-site hyperfine shifts, quadrupole splittings, line widths and effective resonant optical thicknesses from four real calibrated Mössbauer spectra with overlapping absorption doublets.
> Domain: chemistry & material science | Keywords: nuclear spectroscopy, iron speciation, nonlinear spectral inversion | Difficulty: Hard

## Background

Recoil-free nuclear absorption can resolve energy changes far smaller than ordinary optical spectroscopy. In an iron-bearing solid, electron density shifts the nuclear transition energy and the surrounding electric field gradient splits the excited nuclear state. Sweeping a radioactive source's velocity Doppler-shifts its gamma rays through these resonances, allowing local iron environments to be distinguished even when their absorption lines overlap.

Mössbauer spectroscopy is used to study mineral chemistry, catalysts, corrosion and functional materials. Isomer shifts and quadrupole splittings constrain oxidation states and coordination environments. Line broadening and finite absorber thickness complicate these inferences: overlapping site contributions and instrumental convolution require a physical transmission model, and fitted site intensity alone is insufficient to establish absolute chemical abundance.

## Problem Description

Infer four effective doublet components for each of four measured vermiculite ores: ALB1VA00 (Louisa,Virginia), ALB5SA00 (Palabora,SouthAfrica), GDS469 (Enoree,SouthCarolina) and zonolite4 (Libby,Montana), in that order. The observations are calibrated, corrected real transmission spectra. The source velocity axis was independently calibrated using an iron foil; no unknown sample fit supplies that calibration.

For a site k, let delta(k) be its isomer shift, q(k)>0 its full quadrupole splitting, gamma(k)>0 its absorber Lorentzian full width at half maximum, and I(k)>=0 its effective resonant optical thickness. All shifts/widths are in mm/s. Its two line centers are delta(k)±q(k)/2. Define a unit-area Lorentzian and the absorber optical depth by

```math
L(v;\Gamma)=\frac{\Gamma/2}{\pi[v^2+(\Gamma/2)^2]},\qquad
\tau(v)=\frac{\pi\Gamma_0}{4}\sum_{k=1}^{4} I(k)
\left[L(v-\delta(k)+q(k)/2;\gamma(k))+
L(v-\delta(k)-q(k)/2;\gamma(k))\right].
```

Here Gamma0 is the natural nuclear line width. With normalized source line density S(u)=L(u;Gamma0), velocity v and multiplicative baseline B, the ideal normalized intensity is T(v)=B integral S(u)exp[-tau(v+u)]du. The sample is modeled as an isotropic powder with no magnetic splitting and no extra Gaussian broadening. Effective absorber widths include unresolved broadening and are not independently measured intrinsic nuclear lifetimes. I(k) is effective optical thickness; unknown site-dependent recoil fractions prevent interpreting it as a measured atom fraction.

Use the following fixed64-node quadrature convention for the reported spectrum. Set e(j) uniformly from−0.989 to0.989, deltaE=e(2)−e(1), c=(0.027181,22.7718,−33.6937), and u(j)=sum over l=1..3 of c(l)log[(l+e(j))/(l−e(j))]/0.28 in mm/s. Define w(j)=S(u(j)) deltaE sum over l of [2l c(l)/(l²−e(j)²)]/0.28. Then T(v)=B{sum w(j)exp[−tau(v+u(j))]+1−sum w(j)}. The final term treats unsampled distant source-wing photons as unabsorbed. The known energy-node map is acquisition-model discretization, not sample-derived calibration.

Save `output/reconstruction.npz` with every real **float64** array below. Sites are ordered by increasing isomer shift, breaking ties by increasing splitting; arbitrary permutations are not accepted after this canonical ordering. No scale or spectral-axis alignment is applied.

| Output key | Shape | Units and axis order |
|---|---|---|
| reconstruction | (1,4,4,3) | Batch, ore specimen, site, parameter; parameter order is isomer shift, full quadrupole splitting, absorber FWHM, all mm/s |
| optical_thickness | (1,4,4) | Batch, specimen, site; dimensionless effective resonant optical thickness |
| baseline | (1,4) | Batch, specimen; dimensionless multiplicative level |
| transmission | (1,4,512) | Batch, specimen, source velocity point; normalized transmitted intensity, dimensionless |

Evaluation checks every site quantity and every spectrum, separately comparing shifts/widths, thicknesses, baselines and transmitted curves. It also checks observation residual RMS for each specimen and verifies that the submitted parameters reproduce the submitted spectrum under the declared model. These method-comparison criteria do not establish unique true site chemistry; weak components can be poorly identifiable.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| velocity | (1,4,512) | float64 | Independently calibrated source velocity, mm/s; batch, specimen, channel |
| transmission | (1,4,512) | float64 | Measured corrected intensity divided by100, dimensionless; batch, specimen, channel |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Absorber temperature | 295 | K | Real acquisition temperature |
| Spectrometer | WEB Research Co. WT302 | — | Instrument at Mount Holyoke College |
| Radiation source | 57Co in Rh,100–60mCi | — | Conventional radioactive Mössbauer source |
| Nuclear transition | approximately14.4 | keV | Recoil-free57Fe nuclear absorption |
| Natural FWHM Gamma0 | 0.098 | mm/s | Known natural source linewidth used by selected model |
| Sample mass | approximately10–30 | mg | Ground ore mixed with powdered sugar |
| Preferred orientation | minimized by sample preparation; isotropic model | — | Gently packed washer-shaped absorber |
| Acquisition duration | 6–24 | h | Author measurement runs |
| Native channels | 1024 | — | Original spectrum before supplied processing |
| Supplied samples | 512 per specimen | — | Author calibrated/processed spectral rows |
| Velocity calibration | independent iron foil (thickness unit ambiguous in source metadata) | — | Corrected to linear velocity scale |
| Baseline correction | Compton contribution subtracted | — | Source122keV scattering background removed |
| Data normalization | supplied percent-scale intensity /100 | — | Values near1 represent transmission baseline |
| Instrument Gaussian width | 0 | mm/s | Selected conventional-source model assumption |
| Unknown backgrounds | one multiplicative baseline per specimen | — | No additive background after correction |

## Method Hints

Fit the nonlinear source-convolved transmission model rather than adding isolated transmitted spectra. A damped least-squares method with explicit parameter bounds can optimize effective thicknesses and hyperfine parameters. Independent chemically reasonable initial mixtures help detect poor local minima. Compare the need for additional site components and inspect residuals and weak-component sensitivity; additional fitted components are not automatically separate chemical species.

## References

Yaroslavtsev, “SYNCmoss software package for fitting Mössbauer spectra measured with a synchrotron Mössbauer source,” Journal of Synchrotron Radiation30(2023),596–604, DOI [10.1107/S1600577523001686](https://doi.org/10.1107/S1600577523001686). The method also supports conventional sources as used here.

Swayze etal., “Characterizing the source of potentially asbestos-bearing commercial vermiculite insulation using in situ IR spectroscopy,” American Mineralogist103(2018),517–549, DOI [10.2138/am-2018-6022](https://doi.org/10.2138/am-2018-6022). Measured Mössbauer data release: [10.5066/F7JM27SR](https://doi.org/10.5066/F7JM27SR).

This benchmark applies the selected SYNCmoss conventional-source method to independently published USGS observations; it does not reproduce the original SYNCmoss paper's sample or figures.
