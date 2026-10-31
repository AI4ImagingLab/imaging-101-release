# SMC24 Nebular Temperature and Density

> Recover electron temperature and density diagnostics of an ionized nebula from optical and infrared emission-line ratios that respond differently to collisions and radiative decay.
> Domain: astronomy | Keywords: emission spectroscopy, plasma diagnostics, statistical equilibrium | Difficulty: Hard

## Background

Ionized nebulae emit strong forbidden atomic lines when free electrons excite metal ions and the excited levels decay radiatively. Competition between collisional excitation, collisional de-excitation and photon emission makes different line ratios sensitive to electron temperature and density. These diagnostics support studies of stellar ejecta, chemical enrichment and the thermal state of interstellar gas.

Several ions can sample different parts of a nebula. Their diagnostic curves need not meet at one point, so preserving the differences between diagnostic combinations matters physically. Atomic transition data and extinction corrections also affect inferred conditions.

## Problem Description

Use the author-published SMC24 line-flux compilation and independent atomic coefficients to infer four ordered temperature-density pairs. The low-density, optically thin model assumes Maxwellian electrons, collisional excitation/de-excitation and spontaneous radiative transitions, with no fluorescence pumping or ionization-equilibrium fitting. For level i, a fractional population pi satisfies

```math
\sum_{j\ne i}p_j[n_e q_{ji}(T_e)+A_{ji}]
=p_i\sum_{j\ne i}[n_e q_{ij}(T_e)+A_{ij}],\qquad \sum_i p_i=1.
```

Here Te is electron temperature in Kelvin, ne electron density in cm^-3, q is a collisional rate coefficient in cm^3/s, and A is a spontaneous transition probability in s^-1 (zero for upward radiative transitions). For upper level j and lower level i,

```math
q_{ji}=\frac{K\Upsilon_{ji}(T_e)}{g_j\sqrt{T_e}},\qquad
q_{ij}=\frac{g_j}{g_i}e^{(E_i-E_j)/(bT_e)}q_{ji}.
```

Statistical weights are gi, energies Ei are in inverse Angstrom, effective collision strengths are Upsilon, K=8.629132419846745e-6 in the matching cgs/K convention, and b=6.95034760642295e-9 inverse Angstrom/K. Line emissivity per ion/electron pair is pj Aji h c (Ej-Ei) times1e8/ne, in erg cm^3/s; h=6.62606957e-27 erg s and c=29979245800 cm/s. Ratios of lines from the same ion cancel abundance and geometric path length. This is a physical-condition inverse problem conditional on each diagnostic pair; do not force all ions to share one exact condition.

Save `output/reconstruction.npz` with real float64 key `reconstruction`, shape `(1, 4, 2)`. Axes are batch,diagnostic pair,quantity. Quantity order is Te in Kelvin then ne in cm^-3. Pair order is [NII]5755/6584 with [OII]3726/3729; [NII]5755/6584 with [SII]6731/6716; [OIII]4363/5007 with [OII]3726/3729; [OIII]4363/5007 with [SII]6731/6716. Wavelengths in these labels are Angstrom. Use absolute positive conditions without alignment or rescaling. Each returned pair must reproduce both of its extinction-corrected observed line ratios with absolute fractional residual at most 0.001. Every one of the eight residuals must pass. Temperature must be inside the collision-table temperature range of both selected ions, and density must be between 1 and 1e8 cm^-3. Any simultaneous solution satisfying these conditions is accepted; no finite iteration history or reference-coordinate agreement is required.

## Data Description

All arrays in `data/raw_data.npz` are batch first. Fluxes are author-provided processed astronomical observations, not simulated truth. Atomic arrays are independently published calibration/model coefficients, not estimated from the unknown nebular temperature/density.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| N2_energy | (1, 6) | float64 | Atomic excitation wavenumber, inverse Angstrom; axes batch,level. |
| N2_weight | (1, 6) | float64 | Atomic statistical weight, dimensionless; axes batch,level. |
| N2_A | (1, 6, 6) | float64 | Radiative transition probability, s^-1; axes batch,upper level,lower level. |
| N2_collision_temperature | (1, 11) | float64 | Published collision-table temperature coordinate, log10(K); axes batch,temperature node. |
| N2_collision_strength | (1, 6, 6, 11) | float64 | Published effective collision strength, dimensionless; axes batch,lower level,upper level,temperature node. Only levels supported by both atomic/radiative and collision data participate. |
| O2_energy | (1, 5) | float64 | Atomic excitation wavenumber, inverse Angstrom; axes batch,level. |
| O2_weight | (1, 5) | float64 | Atomic statistical weight, dimensionless; axes batch,level. |
| O2_A | (1, 5, 5) | float64 | Radiative transition probability, s^-1; axes batch,upper level,lower level. |
| O2_collision_temperature | (1, 19) | float64 | Published collision-table temperature coordinate, log10(K); axes batch,temperature node. |
| O2_collision_strength | (1, 5, 5, 19) | float64 | Published effective collision strength, dimensionless; axes batch,lower level,upper level,temperature node. Only levels supported by both atomic/radiative and collision data participate. |
| O3_energy | (1, 5) | float64 | Atomic excitation wavenumber, inverse Angstrom; axes batch,level. |
| O3_weight | (1, 5) | float64 | Atomic statistical weight, dimensionless; axes batch,level. |
| O3_A | (1, 5, 5) | float64 | Radiative transition probability, s^-1; axes batch,upper level,lower level. |
| O3_collision_temperature | (1, 25) | float64 | Published collision-table temperature coordinate, log10(K); axes batch,temperature node. |
| O3_collision_strength | (1, 5, 5, 25) | float64 | Published effective collision strength, dimensionless; axes batch,lower level,upper level,temperature node. Only levels supported by both atomic/radiative and collision data participate. |
| S2_energy | (1, 5) | float64 | Atomic excitation wavenumber, inverse Angstrom; axes batch,level. |
| S2_weight | (1, 5) | float64 | Atomic statistical weight, dimensionless; axes batch,level. |
| S2_A | (1, 5, 5) | float64 | Radiative transition probability, s^-1; axes batch,upper level,lower level. |
| S2_collision_temperature | (1, 11) | float64 | Published collision-table temperature coordinate, log10(K); axes batch,temperature node. |
| S2_collision_strength | (1, 8, 8, 11) | float64 | Published effective collision strength, dimensionless; axes batch,lower level,upper level,temperature node. Only levels supported by both atomic/radiative and collision data participate. |
| S3_energy | (1, 6) | float64 | Atomic excitation wavenumber, inverse Angstrom; axes batch,level. |
| S3_weight | (1, 6) | float64 | Atomic statistical weight, dimensionless; axes batch,level. |
| S3_A | (1, 6, 6) | float64 | Radiative transition probability, s^-1; axes batch,upper level,lower level. |
| S3_collision_temperature | (1, 8) | float64 | Published collision-table temperature coordinate, log10(K); axes batch,temperature node. |
| S3_collision_strength | (1, 6, 6, 8) | float64 | Published effective collision strength, dimensionless; axes batch,lower level,upper level,temperature node. Only levels supported by both atomic/radiative and collision data participate. |
| Ne3_energy | (1, 5) | float64 | Atomic excitation wavenumber, inverse Angstrom; axes batch,level. |
| Ne3_weight | (1, 5) | float64 | Atomic statistical weight, dimensionless; axes batch,level. |
| Ne3_A | (1, 5, 5) | float64 | Radiative transition probability, s^-1; axes batch,upper level,lower level. |
| Ne3_collision_temperature | (1, 16) | float64 | Published collision-table temperature coordinate, log10(K); axes batch,temperature node. |
| Ne3_collision_strength | (1, 5, 5, 16) | float64 | Published effective collision strength, dimensionless; axes batch,lower level,upper level,temperature node. Only levels supported by both atomic/radiative and collision data participate. |
| flux_S4_10.5m | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_Ne2_12.8m | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_Ne3_15.6m | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_S3_18.7m | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_O2_3726A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_O2_3729A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_Ne3_3869A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_Ne3_3968A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_S2_4069A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_S2_4076A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_O3_4363A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_O3_5007A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_N2_5755A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_S3_6312A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_O1_6300A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_O1_6364A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_N2_6548A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_N2_6584A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_S2_6716A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_S2_6731A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_Ar3_7136A | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_O2_7319A+ | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |
| flux_O2_7330A+ | (1, 1) | float64 | Observed relative line flux, Hbeta=100 convention; dimensionless, axes batch,object. Original published input parsing precision preserved. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Object | SMC24 | identifier | Author-published planetary-nebula optical/IR flux compilation |
| Observation scope | 23 valid line entries | - | Original supplemental33.6micron SIII label is invalid in the selected atomic data and is not one of the seven diagnostics; no substitute line is supplied |
| Flux normalization | Hbeta=100 at publication | relative flux | No Hbeta renormalization after extinction correction |
| Logarithmic extinction c(Hbeta) | 0.047 | dimensionless | Author-supplied reddening calibration; held fixed |
| Extinction law | CCM89, RV=3.1 | dimensionless | Optical0.303-0.909micron polynomial and nearIR0.909-3.333micron power law; original convention uses zero extinction beyond3.333micron |
| Converted E(B-V) | 0.0325583778321743011 | mag | Archived original float32 conversion from supplied c(Hbeta); preserve value rather than recompute with different rounding |
| Extinction correction | Fcorrected=Fobserved times10^(0.4 E(B-V) X(lambda)) | relative flux | X(lambda)=RV*a+b from CCM89; absolute correction, not a new normalization to Hbeta |
| Collision interpolation | linear in supplied log10(K) coordinate; endpoint clamp | - | Collision arrays store lower,upper,node; returned collision rates use physical upper,lower order |
| Model levels | NII6; OII5; OIII5; SII5; SIII6; NeIII5 | levels | Minimum levels supported by radiative and collision data; extra tabulated collision levels do not add states |
| Atomic data | NII FFT04/T11; OII Z82-WFD96/Kal09; OIII FFT04-SZ00/SSB14; SII RGJ19/TZ10; SIII FFTI06/TG99; NeIII GMZ97/McLB00 | identifiers | Supplied probabilities and collision coefficients fix the model; no external atomic library required |
| Geometry | optically thin unresolved line emission | - | Abundance/path length cancel in same-ion ratios |
| Noise | no measured uncertainty supplied | - | Reference review uses an explicitly assumed5percent component-flux error; it is not an instrumental variance measurement |

Flux ordering is identified by key. Atomic energy order increases from the ground state at index0, and all probability/strength matrices use that same ordering. Temperature and density diagnose a plasma model; the supplied source compilation does not provide spatially resolved imaging geometry or an independent exact physical truth.

## Method Hints

Solve the statistical-equilibrium balance to predict each pair of line ratios, interpolating collision strengths linearly in log10 temperature on the supplied nodes. A bounded nonlinear root or least-squares solve in log temperature and log density handles the different physical scales. Evaluate both diagnostic residuals at the same final pair; convergence of alternating updates alone does not establish consistency. The four independent pairs may describe different nebular conditions.

## References

V. Luridiana, C. Morisset and R.A.Shaw. PyNeb: a new tool for analyzing emission lines. I. Code description and validation of results. Astronomy & Astrophysics573,A42 (2015). [DOI10.1051/0004-6361/201323152](https://doi.org/10.1051/0004-6361/201323152), [arXiv1410.6662v2](https://arxiv.org/abs/1410.6662v2). The task uses the modern atomic-data set explicitly identified above, not an unspecified2015 default.

J.A.Cardelli, G.C.Clayton and J.S.Mathis. The relationship between infrared, optical, and ultraviolet extinction. Astrophysical Journal345,245 (1989). [DOI10.1086/167900](https://doi.org/10.1086/167900).
