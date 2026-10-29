# S100A14 Zinc Binding Thermodynamics

> Recover temperature-dependent zinc binding thermodynamics of human S100A14 from measured injection heats while separating changing occupancy from dilution heat.
> Domain: biology | Keywords: molecular thermodynamics, calorimetry, parameter inference | Difficulty: Hard

## Background

Binding between proteins and metal ions releases or absorbs heat. Isothermal titration calorimetry measures this thermal response as small ligand aliquots enter a protein-containing cell. Changes in molecular occupancy link the measured heat to affinity and binding enthalpy, supporting studies of metalloprotein regulation and molecular recognition.

An injection also dilutes the sample, displaces solution out of the cell and mixes buffer components. These effects can produce heat even without binding. Acquiring titrations at several temperatures helps distinguish thermal response and affinity, but instrument noise, concentration uncertainty and imperfect equilibrium assumptions remain relevant limitations.

## Problem Description

The data are six complete measured titrations of recombinant human S100A14 with zinc chloride. Infer a common reference association constant, a constant binding enthalpy and an effective site/concentration factor, with temperature-specific linear dilution contributions. There are14injections per experiment; injection0is supplied but excluded from parameter fitting because the author identifies it as a first-shot transient. All14predicted heats are required, including that diagnostic extrapolation.

Let experiment index be e and injection index j. Protein and ligand concentrations before injection0are S0 and0. For cell volume V, injected volume v and syringe concentration L0, overflow dilution gives

```math
P_{ej}=\prod_{r=0}^{j-1}(1-v_{er}/V),\quad
S_{ej}=f S_0P_{ej},\quad L_{ej}=L_0(1-P_{ej}).
```

P at j=0 is1. The effective factor f multiplies the nominal protein concentration; it can exceed1and is not a biological fraction constrained below1. At absolute temperature T,

```math
K_e=K_{\mathrm{ref}}\exp\left[-\frac{H}{R}(1/T_e-1/298.15)\right],\quad
C_{ej}=\frac{b_{ej}-\sqrt{b_{ej}^2-4S_{ej}L_{ej}}}{2},\quad
b_{ej}=S_{ej}+L_{ej}+K_e^{-1}.
```

Kref is association constant at298.15K, H is the constant molar enthalpy, R the gas constant and C the bound complex concentration. Occupancy is u=C/S. The discrete measured heat model is

```math
q_{ej}=VS_{e,j+1}H(u_{e,j+1}-u_{ej})+m_e L_{e,j+1}+c_e+\epsilon_{ej}.
```

The dilution slope and offset are m and c; epsilon denotes model/measurement discrepancy. With concentrations in mol/L, V in microlitres and H in cal/mol, q is in microcalories. The model assumes one effective independent binding site, constant enthalpy over this temperature range, complete mixing and the stated overflow convention. Strong coupling between affinity, concentration scale and dilution makes inversion difficult. No independent physical parameter truth is available.

Save`output/reconstruction.npz`relative to your task workspace, with exactly the following real float64 arrays. All are batch first; temperature order matches the six observation rows.

| Output key | Shape | Units and convention |
|---|---|---|
| enthalpy | (1,1) | H, cal/mol, signed |
| association_constant | (1,1) | Kref, M^-1, positive |
| effective_site_factor | (1,1) | f, dimensionless; effective site/concentration correction |
| dilution_slope | (1,6) | m, microcal/M, signed |
| dilution_offset | (1,6) | c, microcal, signed |
| predicted_heats | (1,6,14) | q, microcal; axes batch,experiment,injection; include first injection |

Evaluation measures maximum absolute error separately for each output quantity in its physical unit and measured-heat RMSE over the fixed78nonfirst injections. All criteria must pass; no registration, scale fitting or hidden sample selection is applied. Agreement measures reproduction of a validated standard estimator. The first-shot discrepancy and systematic fit residuals must not be interpreted as exact physical accuracy.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| heats | (1,6,14) | float64 | Author integrated measured injection heats, microcal; batch,experiment,injection |
| injection_volumes | (1,6,14) | float64 | Recorded injection volumes, microlitre; same axes |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Actual temperatures, in row order | 5.33304,10.00685,17.13053,24.99623,29.99722,34.99955 | °C | Native header values; convert by adding273.15 |
| Nominal temperatures | 5,10,17,25,30,35 | °C | Labels only |
| Protein concentration | 0.00011 | mol/L | Initial nominal S100A14 in every cell |
| Zinc syringe concentration | 0.002 | mol/L | Author data headers; accompanying historical notebook prose says200mM, conflicting with2mM headers; this task binds to the native headers |
| Initial ligand concentration | 0 | mol/L | No zinc initially in cell |
| Cell volume | 201.3 | microlitre | Constant-volume overflow cell |
| Reference temperature | 298.15 | K | Defines Kref |
| Gas constant | 1.9872036 | cal/(mol K) | Numerical convention |
| Instrument | GE ITC200 | none | Isothermal titration calorimeter |
| Buffer | 25mM Trizma,100mM NaCl,pH7.4 | specified | Chelex treated,0.22micrometre filtered |
| First shot | index0 | zero-based index | Acquired and predicted, excluded from the78-point fit |
| Calibration/noise | No independent per-shot variances supplied | none | Do not interpret a numerical fit weight as measured instrumental uncertainty |
| Boundary/mixing | Complete mixing, fractional volume replacement | none | No spatial or periodic boundary model |

## Method Hints

A global nonlinear least-squares estimator can couple all temperatures through the van’t Hoff relation while retaining separate dilution terms. Compute equilibrium occupancy from mass action and injection-to-injection changes explicitly. Preserve actual temperatures, microlitre-to-microcal scaling and the overflow concentration convention. A concentration/site factor captures effective stoichiometry and nominal concentration mismatch; it is not independently identifiable as a literal active fraction.

## References

Duvvuri H., Wheeler L.C., Harms M.J., “pytc: Open-Source Python Software for Global Analyses of Isothermal Titration Calorimetry Data,” Biochemistry57,2578–2583(2018), [doi:10.1021/acs.biochem.7b01264](https://doi.org/10.1021/acs.biochem.7b01264).
