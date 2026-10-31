# SecB HDX Residue Protection Field

> Recover a conditional residue protection field from overlapping peptide deuterium-uptake measurements whose spatial averaging leaves individual residues underdetermined.
> Domain: biology | Keywords: molecular imaging, inverse kinetics, regularization | Difficulty: Hard

## Background

Hydrogen-deuterium exchange probes how readily protein backbone amide hydrogens exchange with solvent. Local folding and interactions protect these sites, slowing exchange relative to an unstructured reference. Mass spectrometry measures the accumulated mass increase in digested peptides at several exposure times.

The resulting measurements support studies of protein folding, binding and conformational dynamics. Each peptide combines many sites, and overlapping peptides give incomplete spatial information. A residue map therefore requires an explicit kinetic convention and regularization; it is not a uniquely measured atomic energy landscape.

## Problem Description

The observations are real apo ecSecB peptide mass increments. Let $X$ be the supplied peptide-by-residue membership matrix, $k_j$ the supplied intrinsic exchange rate, $t_m$ the exposure, and $G_j$ the unknown field in J/mol. The observed uptake after independent fully deuterated control correction is $y_{pm}=u_{pm}C_p/F_p$, where $u$ is `measurements`, $F$ is `control` and $C$ is `capacity`.

```math
a_{pm}(G)=f\sum_j X_{pj}\left[1-\exp\left(-\frac{k_j t_m}{1+\exp(G_j/(RT))}\right)\right].
```

Here $f$ is the solvent deuterium fraction, $R$ the gas constant and $T$ the absolute temperature. The factor $f$ is essential: the infinite-time capacity is $C_p$, not the unlabelled number of sites. Intrinsic rates of prolines are zero; their membership does not create uptake. This task explicitly corrects a missing fraction in an older implementation's forward model.

Recover the regularized conditional target defined by

```math
J(G)=\frac{1}{63\cdot7}\sum_{p,m}(a_{pm}(G)-y_{pm})^2+
\frac{2\times10^{-4}}{145}\sum_{j=1}^{145}|G_{j+1}-G_j|.
```

The TV coefficient has units Da^2 mol/J. The full field has 146 adjacent entries for residues 10 through 155, including 23 auxiliary support sites marked false in `exchanges` (7 prolines and 16 uncovered sites). Only the 123 true entries are interpreted as conditional exchange-site protection energies. The auxiliary values preserve adjacency and the objective; they are not physical energies at unobserved or nonexchangeable sites. No unique residue truth is assumed, and no positivity constraint on $G$ is imposed.

Save `output/reconstruction.npz` with both real float64 arrays: `reconstruction`, shape `(1, 146)`, batch then increasing residue, in J/mol; and `uptake`, shape `(1, 63, 7)`, batch, peptide, native exposure index, in Da. All entries must be finite. Every field entry contributes to the full-domain objective, and every uptake entry is checked against the stated forward model. Evaluation accepts objective values J<=1.0361689959301181 and requires reported uptake to agree with the forward calculation to relative L2 error $10^{-9}$, normalized by the norm of the corrected observations. There is no alignment, rescaling or cropping and no strict matching of unidentifiable auxiliary values.

## Data Description

All arrays are in `data/raw_data.npz`; the first axis is the single acquisition batch. Native floating-point values are preserved.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements | (1, 63, 7) | float64 | Measured peptide mass increments, Da |
| control | (1, 63) | float64 | Independent fully deuterated peptide control increments, Da |
| capacity | (1, 63) | float64 | Finite-label exchangeable peptide capacities, Da |
| membership | (1, 63, 146) | int32 | Peptide incidence matrix, dimensionless; early nonexchanging peptide positions excluded |
| intrinsic_rates | (1, 146) | float64 | Unstructured-reference intrinsic rates at the stated conditions, 1/s |
| exchanges | (1, 146) | bool | Physical exchange-site support, dimensionless |
| time_s | (1, 7) | float64 | Native exposure times, s |
| residue_number | (1, 146) | int32 | Protein sequence position, dimensionless |
| peptide_start | (1, 63) | int32 | Original peptide start position, dimensionless |
| peptide_stop | (1, 63) | int32 | Original exclusive peptide stop position, dimensionless |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Protein state | Apo ecSecB | dimensionless | Native author measurement series |
| Temperature | 303.15 | K | 30 degC exchange experiment |
| pH | 8.0 | dimensionless | Source labeling-buffer convention used for intrinsic rates |
| Solvent deuterium fraction | 0.9 | dimensionless | Independent source preparation specification |
| Gas constant | 8.31446261815324 | J/(mol K) | Thermal energy conversion |
| Exposure ordering | 0, 10.02, 30, 60, 300, 600, 6000.00048 | s | Exact native coordinates also stored in time_s |
| Field boundary | Open chain | dimensionless | 145 consecutive residue differences, no periodic wrap |
| Control correction | measurements * capacity / control | Da | Independent peptide-specific recovery correction |

The intrinsic-rate calibration is calculated from the unstructured sequence reference at the stated solvent conditions, rather than inferred from the unknown protection field. Raw uptake can contain experimental departures from the ideal kinetic model. Control correction does not remove these model limitations.

## Method Hints

A full-field total-variation penalty connects overlapping peptide constraints while allowing protected regions to change sharply. A multi-exponential initialization can estimate local exchange time scales before joint nonlinear fitting. General differentiation and optimization tools are suitable provided the peptide mixing, finite-label fraction and full-domain penalty are implemented explicitly. Different valid optimization paths may qualify under the same objective; auxiliary values need not reproduce a particular optimizer's output.

## References

[Probing Universal Protein Dynamics Using Hydrogen-Deuterium Exchange Mass Spectrometry-Derived Residue-Level Gibbs Free Energy](https://doi.org/10.1021/acs.analchem.1c02155). The task uses a documented finite-label forward-model correction and does not claim unchanged reproduction of that implementation.
