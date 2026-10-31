# GP15 Water-Mass and Remineralization Section

> Reconstruct the spatial mixing fractions of nine Pacific water masses and two biological oxygen-consumption contributions from real hydrographic and nutrient bottle measurements despite overlapping source signatures and nonunique mixtures.
> Domain: earth science | Keywords: oceanographic imaging, tracer mixing, constrained inversion | Difficulty: Hard

## Background

Ocean circulation transports water with characteristic temperature, salinity and dissolved chemical properties. Mixing redistributes these signatures across latitude and depth, while biological regeneration changes nutrients and consumes dissolved oxygen. Separating transport from biological alteration helps interpret carbon and nutrient cycling in the ocean interior.

A hydrographic section samples water at sparse bottle locations rather than observing the contributing source volumes directly. Different sources can have similar properties, and oxygen and nutrients are not conservative under biological activity. Source-water interpretation therefore requires coupled mass conservation, chemical stoichiometry and explicit assumptions about plausible geographic distributions.

## Problem Description

The observations are all576 complete intermediate/deep samples in the author-distributed GP15 bottle-section example. Recover nonnegative fractions F(i,k), with sum over k equal to one, and two nonnegative oxygen-consumption contributions R(i,l). The measurement model is

```math
Y_{ij}=\sum_{k=1}^{9}F_{ik}M_{kj}+\sum_{l=1}^{2}R_{il}C_{lj}+e_{ij}.
```

Indices i,j,k,l denote sample, tracer, water mass and reaction endmember. Y is observed tracer property, M is the supplied source-water property dictionary, C is the supplied stoichiometric change per unit oxygen consumed, and e represents measurement and model discrepancy. Tracers are potential temperature, practical salinity, silicate, nitrate, phosphate and oxygen, in that order. The source order is AAIW,PSUW,ESSW,NPIW,PDW1,UCDW,PDW23,LCDW,AABW. F is dimensionless; R is micromol oxygen/kg. Reaction channels have identical nitrate-to-oxygen ratios but different phosphate-to-oxygen ratios, permitting intermediate effective regeneration ratios through nonnegative mixtures. Temperature, salinity and silicate have zero reaction coefficients in this selected model.

The target is the constrained optimum of the following explicit conditional analysis, evaluated separately at each observed sample:

```math
J_i(F_i,R_i)=\sum_j\left[w_j\left(\sum_kF_{ik}M_{kj}+\sum_lR_{il}C_{lj}-Y_{ij}\right)\right]^2+\sum_k(P_{ik}F_{ik})^2.
```

The supplied weights w convert heterogeneous tracer residuals into dimensionless weighted residuals; they define the analysis, not calibrated measurement error bars. The nonnegative geographic usage penalties P are fixed from the observed coordinates. Let

```math
E(t;\alpha,\beta)=\min(10000,\beta[\exp(\alpha\max(t,0))-1]).
```

For density anomaly s in kg/m^3, latitude h in degrees north and depth d in metres, P for LCDW and AABW is E(27.72-s;0.5,1000). P for PSUW is E(10-h;0.05,100)+E(d-3000;0.001,50). All other penalties are zero. The two PSUW terms are individually capped before addition. These are explicit source-distribution assumptions, not additional observations. F and R must be nonnegative and every fraction row must sum to one. No spatial interpolation or smoothness is required between samples.

Nine water sources plus reactions cannot be uniquely inferred from six properties. The target is the complete class of feasible solutions meeting this convex conditional objective; individual fractions need not match a particular numerical optimizer. This is the author's nine-source example configuration. It does not require a separate circulation-model disambiguation or reproduce the later final publication's subtype analysis.

Save `output/reconstruction.npz` with both required real float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,576,9) | float64 | Dimensionless fractions; batch,sample,water mass in the stated order |
| remineralization | (1,576,2) | float64 | micromol oxygen consumed/kg; batch,sample,reaction channel in supplied reaction_ratios order |

Every sample and all11latent quantities are assessed jointly through the complete forward objective. Every sample cost may exceed an independently attained standard solution by at most1e-5 times max(1,that standard cost). Fraction-row sums must differ from1 by at most1e-7; all components must be at least-1e-10. An equal or better feasible solution passes regardless of its nullspace coordinates. There is no sample sorting, implicit rescaling, fitted alignment, crop or parameter permutation during evaluation. Agreement establishes consistency with the stated water-source model and priors, not independent water-mass truth.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| observations | (1,576,6) | float64 | Measured/author-derived tracer properties in the stated order; units degrees C, PSS-78 practical salinity, then micromol/kg for the four dissolved tracers |
| endmembers | (1,9,6) | float64 | Author-supplied source-water properties in matching source/tracer order and matching tracer units |
| sample_coordinates | (1,576,4) | float64 | Latitude degrees north, longitude degrees east, positive-down depth m, potential-density anomaly sigma0 kg/m^3 |
| tracer_weights | (1,6) | float64 | Weights in inverse matching tracer units; resulting weighted residual is dimensionless |
| reaction_ratios | (1,2,6) | float64 | Tracer changes per micromol oxygen consumed/kg; zero temperature/salinity/silicate entries; nutrient and oxygen entries in mol/mol |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Section | GEOTRACES GP15 Pacific,2018 | 1 | Bottle acquisition across latitude and depth |
| Sample order | Native576complete rows | 1 | No sorting or resampling |
| Weight order | 56,80,3,5,5,1 | inverse tracer unit | Stated weighted objective |
| Oxygen reaction coefficients | -1,-1 | mol/mol | Oxygen consumed in both channels |
| Nitrate reaction coefficients | 0.10330578512,0.10330578512 | mol/mol | Nitrate regeneration per oxygen consumed |
| Phosphate reaction coefficients | 0.01036168132,0.00327210989 | mol/mol | Two allowed regeneration ratios |
| Mass normalization | Sum of source fractions1 | 1 | Hard samplewise conservation |
| Density prior boundary | 27.72 | kg/m^3 | Density anomaly for LCDW/AABW usage penalty |
| PSUW latitude boundary | 10 | degrees north | Source usage prior |
| PSUW depth boundary | 3000 | m | Source usage prior |
| Prior cap | 10000 per term | 1 | Exponential-penalty conditioning limit |

All arrays preserve author-provided double precision. The endmember dictionary is calibration/model information from the author configuration, not a reconstruction or withheld fraction-derived input. Potential temperature,density and depth are retained as supplied; no new equation-of-state calculation is required.

## Method Hints

The forward model is linear in source fractions and reaction contributions. The objective is a convex quadratic with a simplex constraint for source fractions and nonnegative reaction contributions. Constrained quadratic optimization can solve each bottle independently. Inspect fraction ambiguity and tracer residuals rather than treating one optimizer's fraction vector as unique physical truth. Removing reactions or source-distribution priors provides informative diagnostics of identifiability.

## References

A.Shrikumar,R.Lawrence,K.L.Casciotti, PYOMPA version0.3:Technical Note,2022, doi:10.1002/essoar.10507053.4.

R.M.Lawrence et al., Analysis of nutrient and hydrographic patterns on the GEOTRACES Pacific Meridional Section(GP15) using PYOMPA,a new framework for water mass analysis, Marine Chemistry274,104598(2026), doi:10.1016/j.marchem.2025.104598. The benchmark selects the documented nine-source example configuration rather than the final subtype/circulation-model analysis.
