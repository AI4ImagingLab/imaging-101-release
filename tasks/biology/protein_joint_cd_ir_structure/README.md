# Protein Joint CD–IR Structure

> Recover four protein secondary-structure compositions from measured circular dichroism and infrared spectra despite overlapping spectral signatures and disagreement between sample conditions.
> Domain: biology | Keywords: molecular spectroscopy, multimodal inference, structural biology | Difficulty: Hard

## Background

Peptide arrangements affect both electronic circular dichroism and vibrational infrared absorption. Helical, sheet, turn and disordered structures contribute differently across these spectra, allowing bulk secondary-structure composition to be inferred without resolving every atom. Such measurements are useful for studying protein folding, stability and structural changes.

Individual spectral bands overlap, and different proteins can produce similar signals. Complementary spectroscopic measurements and independent proteins with established structural annotations help constrain the inverse. Measurements in solution and dry films can nevertheless probe different conformational populations, so their combination requires care and does not guarantee agreement with a crystal structure.

## Problem Description

**Scientific contract status: not certified for benchmark release.** A validated public statistical target for the discrete self-consistent solution ensemble is unresolved. The physical model, observations and output layout below remain documented; historical method-comparison criteria are not a certified public challenge.

Four target proteins—myoglobin, lysozyme, ubiquitin and concanavalin A—have measured CD and infrared spectra. Infer their helix, sheet, turn and other fractions. The empirical mixture model is

```math
q\approx Bf,\qquad A\approx BF,
```

where $q$ is a target spectrum, $f$ its four-component secondary-structure fraction vector, $B$ unknown effective component spectra, $A$ the spectra of independent calibration proteins, and $F$ their known fractions. A variable-subspace formulation uses coefficients $c$ with $q\approx Ac$ and $f\approx Fc$. The coefficients need not be nonnegative. A single universal component spectrum per structure is an approximation; sequence, structural context and sample condition affect the observed response. The four targets and their structure annotations are absent from the calibration library.

Save `output/reconstruction.npz` relative to the task workspace with:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 4, 4) | float64 | Estimated fractions, dimensionless; axes batch, protein, structure; protein order myoglobin/lysozyme/ubiquitin/concanavalin A; structure order helix/sheet/turns/other |
| spectra | (1, 4, 127) | float64 | Mean refitted spectrum; axes batch, protein, spectral bin; first66CD values at240–175nm in descending1nm steps, followed by61IR values at1600–1720cm⁻¹ in ascending2cm⁻¹ steps; CD units Δε in M⁻¹cm⁻¹ per residue, IR normalized absorbance |

Return unscaled IR values in `spectra`. Preserve the estimates' numerical scale, including small negative fractions allowed by approximate subspace fitting; do not clip or renormalize. Evaluation measures maximum absolute fraction deviation, separate CD and IR refit RMS errors, near-unit fraction sums and a lower fractional bound. No alignment or output rescaling is applied. There is no independent same-condition physical truth; evaluation assesses agreement with a standard method rather than atomic accuracy.

## Data Description

The public NPZ includes the complete spectra for all target and calibration proteins; all values use the original loader's float32 convention.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| observations | (1, 4, 267) | float32 | Target measured spectra:66CD samples in M⁻¹cm⁻¹ per residue, then201IR samples in normalized absorbance |
| library_spectra | (1, 24, 267) | float32 | Independent calibration protein spectra in the same mixed units and spectral order |
| library_fractions | (1, 24, 4) | float32 | Established calibration protein secondary-structure fractions, dimensionless, helix/sheet/turns/other order |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| CD spectral grid | 240–175, decreasing step1 | nm | First66bins |
| IR full spectral grid | 1600–1800, increasing step1 | cm⁻¹ | Last201bins |
| Required refit IR grid | 1600–1720, increasing step2 | cm⁻¹ |61selectedbins; no interpolation needed |
| CD convention | Molar differential extinction per residue | M⁻¹cm⁻¹ | Author processed circular dichroism from the SP175/SMP180 collections |
| IR condition | Protein samples dried on an ATR crystal | — | RaSP50 dry-film measurements; not the earlier liquid-sample data |
| IR preprocessing | Baseline subtraction, zero at1718cm⁻¹, normalization to unit maximum | dimensionless | Applied by the source data providers; the supplied processed values are preserved |
| CD/IR pairing | Same named protein, different measurement conditions | — | Does not assert identical conformation in solution and dry film |
| Calibration fractions | Structural annotations from independent crystal structures | dimensionless | Other proteins only; no target annotation supplied |
| Spatial boundary condition | Not applicable | — | Spectral samples, not a spatial differential operator |
| Noise covariance | Not supplied | — | No calibrated noise weighting |
| File-specific instrument serial, concentration and pathlength | Not included in the processed archive | — | These are absorbed in the published normalized spectra; no raw absorbance calibration can be reconstructed |

Calibration row order is apoferritin, hemoglobin, citrate synthetase, phospholipase A2, cytochrome c, insulin, triose phosphate isomerase, peroxidase, phosphoglycerate kinase, subtilisin Carlsberg, alpha-lactalbumin, ovalbumin, glucose oxidase, papain, ribonuclease A, monellin, pepsinogen, thaumatin, carbonic anhydrase, chymotrypsinogen A, superoxide dismutase, lentil lectin, soy trypsin inhibitor and avidin. No target fraction is needed to use these calibration rows.

## Method Hints

A self-consistent variable-selection method can sort calibration proteins by spectral similarity, augment the spectral library with the target, and iterate its provisional structure estimate across truncated singular-vector subspaces. Filter candidate solutions by fraction sums, lower bounds, spectral refit quality and an independent low-rank helix estimate. Average retained solutions and their spectral refits. The declared joint analysis uses the published SELCON3 family with a fifteen-fold IR block weight; undo that weight for the output refit. Relative CD/IR scaling affects the subspaces and should be examined; the method is not an unconstrained single least-squares regression.

## References

1. Hoffmann, Jones and Rodger(2025). Protein secondary structure determined from independent and integrated infra-red absorbance and circular dichroism data using the algorithm SELCON. *QRB Discovery*6,e10. https://doi.org/10.1017/qrd.2025.4
2. Goormaghtigh, Ruysschaert and Raussens(2006). Evaluation of the information content in infrared spectra for protein secondary structure determination. *Biophysical Journal*. https://doi.org/10.1529/biophysj.105.072017
3. Pinto Corujo et al.(2022). Spectroscopic analysis of protein structure using self-organizing maps. https://doi.org/10.3389/fchem.2021.784625
