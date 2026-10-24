# Pancreatic Spatial Cell-Type Mixtures

> Reconstruct twenty spatial cell-type mixture fields from tissue transcript counts and an independently assayed single-cell reference, despite sparse counts and correlated expression profiles.
> Domain: biology | Keywords: spatial transcriptomics, mixture inversion, spatial regularization | Difficulty: Hard

## Background

A spatial transcriptomics assay captures RNA on an array of locations while preserving where each measurement originated in a tissue section. Each location can contain several cells. Its transcript counts therefore combine signals from multiple cell populations. Recovering their spatial mixture can help characterize tissue organization, interactions between cell populations, and changes associated with disease.

Single-cell RNA sequencing supplies expression profiles for individual cell types, but it measures a separate assay with different capture efficiency and sampling variation. Rare cell populations, shared marker genes, low transcript counts, and differences between assays make mixture inversion uncertain. Spatial proximity provides useful additional information because adjacent tissue regions often have related compositions.

## Problem Description

The measurements are the complete author-supplied pancreatic cancer spatial count matrix and its separate, matched-patient single-cell reference. Reconstruct both the nonnegative mixing weights and their row-normalized conditional cell-type fractions at **all 428 observed locations and all 20 declared types**. Locations are the native measured positions; no interpolated positions are requested. These estimates are conditional on the reference expression model and are not independently measured cell compositions.

Let G be the number of retained informative genes, N=428 the number of tissue locations, and K=20 the number of types. Let X denote the normalized and source-scaled spatial expression matrix, B the independently derived and source-scaled single-cell basis, V the spatial mixing weights, and E the residual expression matrix. The selected measurement approximation is

```math
X=BV^{\mathsf T}+E,\qquad V_{ik}\geq0,\qquad
P_{ik}=\frac{V_{ik}}{\sum_{j=1}^{K}V_{ij}}.
```

Here i indexes tissue locations, k and j index cell types, and P contains conditional fractions. X has shape (G,N), B has shape (G,K), and V and P have shape (N,K). All are dimensionless after the documented normalization. The Gaussian approximation for E models normalized expression; raw sequencing counts need not have constant Gaussian noise. The basis combines cell-type expression frequencies and mean reference-cell library sizes. The full reference counts and labels allow its derivation without any unknown target mixture.

Counts are sparse and some basis columns are similar. A noisy expression vector can admit several plausible mixtures; assay differences also produce model mismatch. Preserve all types and native locations, including weakly supported rare populations. Do not delete locations or types after fitting.

Save **`output/reconstruction.npz`**, relative to the task workspace, with exactly these arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `mixing_weights` | `(1, 428, 20)` | real `float32` or `float64` | Dimensionless nonnegative source-scaled RNA weights; axes batch, native location, declared type. These are not absolute cell counts. |
| `cell_type_proportions` | `(1, 428, 20)` | real `float32` or `float64` | Dimensionless nonnegative conditional fractions P; same axes and type order; each location sums to one. |

All values must be finite. Evaluation applies no crop, permutation, rescaling, alignment, or discarded type. Every type in each output is scored separately by cosine NCC, reference-dynamic-range NRMSE, and maximum absolute error, with respective boundaries 0.999999, 0.000001, and 0.000001. Zero prediction norm has NCC zero; the selected comparison fields have nonzero range and norm. Fractions must agree with row-normalized weights to within 0.000002. Relative excess squared forward residual over the selected standard estimator must not exceed 0.000001. Agreement measures reproduction of the conditioned estimator; it does not establish physical truth.

## Data Description

`data/raw_data.npz` contains real measured counts and independent reference observations. Read with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `spatial_counts` | `(1, 25753, 428)` | `int32` | Native transcript counts; units counts; axes batch, gene, spatial location. |
| `single_cell_counts` | `(1, 19736, 1926)` | `int32` | Independent reference transcript counts; units counts; axes batch, gene, reference cell. |
| `cell_type_index` | `(1, 1926)` | `int16` | Zero-based reference-cell type index; units dimensionless identifier. |
| `cell_sample_index` | `(1, 1926)` | `int16` | Reference sample-group index; units dimensionless identifier; one group in this binding. |
| `spatial_coordinates` | `(1, 428, 2)` | `float64` | Native x,y coordinates; units author grid units; no inferred micrometre conversion. |
| `spatial_gene_names` | `(1, 25753)` | Unicode `<U25` | Spatial gene names; units dimensionless string identifier. |
| `single_cell_gene_names` | `(1, 19736)` | Unicode `<U16` | Reference gene names; units dimensionless string identifier. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | PDAC-A, GSE111672 | identifier | Human pancreatic cancer section and matched-patient reference. |
| Spatial assay | Spatial Transcriptomics | assay | Spots capture mixtures of cellular transcript signals. |
| Independent reference assay | inDrop single-cell RNA sequencing | assay | Separate dissociated-cell observations; not inferred from target mixtures. |
| Native spatial locations | 428 | locations | Complete supplied spatial acquisition. |
| Native reference cells | 1926 | cells | Complete supplied reference; all 20 types retained. |
| Coordinate axes | x,y in supplied order | grid units | No rotation, interpolation, or physical-length conversion. |
| Library sizes | Sums of supplied counts | counts | Known per-location and per-cell sequencing totals. |
| Noise | Sparse count variation and assay mismatch; no supplied variance calibration | dimensionless after normalization | Residual variance is estimated. |
| Boundary | Finite observed spot set | convention | Pairwise native-coordinate coupling; no periodic image boundary. |
| Type axis | Zero-based order below | identifier | Identical for both outputs. |

| Type index | Cell type |
|---|---|
| 0 | Acinar cells |
| 1 | Ductal terminal ductal like |
| 2 | Ductal CRISP3 high-centroacinar like |
| 3 | Cancer clone A |
| 4 | Ductal MHC Class II |
| 5 | Cancer clone B |
| 6 | mDCs A |
| 7 | Ductal APOL1 high-hypoxic |
| 8 | Tuft cells |
| 9 | mDCs B |
| 10 | pDCs |
| 11 | Endocrine cells |
| 12 | Endothelial cells |
| 13 | Macrophages A |
| 14 | Mast cells |
| 15 | Macrophages B |
| 16 | T cells & NK cells |
| 17 | Monocytes |
| 18 | RBCs |
| 19 | Fibroblasts |

## Method Hints

Use reference-based conditional autoregressive nonnegative matrix factorization. Derive the basis from reference cells, retaining cell-size information: average gene frequencies within each type and sample, average across samples, and multiply by mean reference-cell library size. Select common informative genes using natural-log fold change greater than 1.25 relative to other types; remove the highest one percent of mean within-type variance-to-mean dispersion. The selected estimator uses input-only quality controls: genes expressed in more than five spatial spots, spots with 100 to 1000000 total retained counts, removal of zero-count reference genes/cells, and exclusion of case-sensitive `mt-` names. This binding retains all 428 spots and selects 7050 genes. Use case-insensitive ASCII gene collation. Normalize each selected spatial library to sum one; independently scale X and B to mean 0.1.

Subtract coordinate minima and divide both axes by the largest resulting extent. For normalized coordinate vectors r, define Gaussian weights W, their row-sum diagonal D, and CAR precision L:

```math
W_{ij}=\exp\!\left(-\frac{\|r_i-r_j\|^2}{2(0.1)^2}\right)\quad(i\ne j),\qquad
W_{ii}=0,\qquad L=D-\phi W.
```

Phi is spatial correlation strength. Compare the published candidate strengths using a uniform-proportion initialization. Use multiplicative nonnegative mixture updates with analytic intercept and variance updates. The inverse-gamma prior has shape alpha=1 and scale beta=N/2; these specify the prior in the score below.

The complete score for comparing different phi values is

```math
\ell=-\frac{GN}{2}\log s-\frac{\|X-BV^{\mathsf T}\|_F^2}{2s}
-\frac{N}{2}\sum_k\log\lambda_k
-\frac12\sum_k\frac{(V_k-b_k\mathbf1)^{\mathsf T}L(V_k-b_k\mathbf1)}{\lambda_k}
-\sum_k\left((\alpha+1)\log\lambda_k+\frac{\beta}{\lambda_k}\right)
+\frac K2\log\det L.
```

Here s is residual variance, lambda are positive type variances, b are type intercepts, Vk denotes column k of V, the ones vector has N entries, and the Frobenius norm sums squared entries. The determinant term is constant during fitting at fixed phi but **cannot be omitted when comparing different phi**. This task explicitly includes that published normalization correction; use the complete score to select its last maximizer. Row-normalize the selected weights to obtain P. Sparse rare-type markers and cross-assay mismatch remain limitations.

## References

- Ma, Y. and Zhou, X. Spatially informed cell-type deconvolution for spatial transcriptomics. *Nature Biotechnology* 40, 1349–1359 (2022). [Paper and supplement](https://doi.org/10.1038/s41587-022-01273-7).
- Moncada, R. et al. Integrating microarray-based spatial transcriptomics and single-cell RNA-seq reveals tissue architecture in pancreatic ductal adenocarcinomas. *Nature Biotechnology* 38, 333–342 (2020). [Acquisition paper](https://doi.org/10.1038/s41587-019-0392-8); [GEO record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111672).

Software is GPL-3.0; see LICENSE and NOTICE. The author offers these real sample data for scientific reproduction, but a separate derivative-data redistribution license was not identified. Local research provenance does not establish unrestricted public redistribution rights.
