# Yeast Chromosome Contact Geometry

> Reconstruct a three-dimensional consensus geometry for five yeast chromosomes from noisy, visibility-biased contact counts whose spatial scale and orientation are not directly measured.
> Domain: biology | Keywords: chromatin imaging, statistical inference, genome architecture | Difficulty: Hard

## Background

Chromosome conformation capture turns spatial proximity into DNA ligation products. Sequencing these products measures how frequently pairs of genomic regions occur near one another in a population of cells. These observations connect genome sequence coordinates to the three-dimensional organization of the nucleus, helping study chromosome territories, regulatory interactions and genome accessibility.

Contact frequency is an indirect spatial measurement. Chromatin changes between cells, while digestion, mapping and sequencing introduce locus-dependent visibility. A consensus geometry summarizes population measurements and does not describe the exact conformation of every cell. Without a separately measured contact-to-distance calibration, its coordinates have a model-defined scale rather than a scale in nanometres.

## Problem Description

The supplied observations cover all350 native10kb bins on yeast chromosomes1–5. There are343 bins with observed contacts; seven bins have no contacts and therefore have no identifiable position. Let the coordinate of bin index $i$ be the vector $\mathbf{x}_i$ in three-dimensional model-distance units, and let $c_{ij}$ be the observed number of contacts between two bins. A distance-dependent Poisson model has expected count

```math
\lambda_{ij}=b_i b_j\,\beta\,\lVert\mathbf{x}_i-\mathbf{x}_j\rVert_2^{\alpha}.
```

Here $b_i$ and $b_j$ are positive dimensionless visibility factors, $\lVert\cdot\rVert_2$ denotes Euclidean length, $\alpha=-3$ is the fixed contact-distance exponent, and $\beta=1$ fixes the arbitrary model-distance scale. The data specify unique contact pairs in the upper triangle. The selected standard estimator uses every positive-count pair and omits zero-count pairs from its Poisson objective; zero counts do not imply an infinite distance. Visibility factors depend only on the observed count matrix, not on an inferred geometry.

Recover one consensus coordinate for each supported bin, keeping the native bin order. Contact measurements determine distances but cannot determine an origin, rotation or handedness. The geometry remains statistically uncertain because finite counts, population averaging and the assumed contact-distance law do not uniquely identify a physical single-cell structure.

Save `output/reconstruction.npz`, relative to the task workspace, containing exactly this array:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | `(1, 350, 3)` | float32 or float64 | Real coordinates in model-distance units; axes are batch, native genomic-bin index, Cartesian coordinate `(x,y,z)`; all three coordinates are exactly zero at bins with `bead_support=False`. |

Coordinates and all resulting pair distances and model contact predictions must be finite. Translation, rotation and reflection on supported bins are equivalent. No scale alignment, locus permutation, independent chromosome alignment or omitted supported bin is allowed. Evaluation compares all58,653 unique distances among the343 supported bins using cosine NCC and dynamic-range-normalized RMS error, and checks the positive-pair Poisson objective. It measures agreement with the selected standard estimator; this is not evidence of agreement with independently measured chromosome coordinates. For predicted and comparison distance vectors $\widehat{\mathbf d}$ and $\mathbf d$, with $M$ components, the metrics are

```math
\mathrm{NCC}=\frac{\widehat{\mathbf d}\cdot\mathbf d}
{\lVert\widehat{\mathbf d}\rVert_2\lVert\mathbf d\rVert_2},
\qquad
\mathrm{NRMSE}=\frac{\sqrt{M^{-1}\sum_{k=1}^{M}(\widehat d_k-d_k)^2}}
{\max_k d_k-\min_k d_k}.
```

## Data Description

`data/raw_data.npz` contains existing author-distributed experimental contact counts and genomic bin metadata. Every stored count is preserved, without spatial subsetting of the supplied five-chromosome sample, count thinning or invented contact values.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| contact_counts | `(1, 350, 350)` | int64 | Unique observed contacts in counts; strictly upper triangular, with diagonal and lower triangle zero by storage convention. |
| bead_support | `(1, 350)` | bool | Dimensionless input support: true exactly when a bin has at least one observed contact; false bins require zero output padding. |
| chromosome_index | `(1, 350)` | int16 | Dimensionless zero-based chromosome labels0–4, corresponding to chromosomes1–5; bin counts24,82,32,154,58. |
| locus_bounds | `(1, 350, 2)` | int64 | Native inclusive genomic start/end coordinates in base pairs; the first bin starts at base pair1. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Organism | Saccharomyces cerevisiae | — | Budding yeast experimental contact sample. |
| Genomic coverage | Chromosomes1–5, all350 supplied bins | — | Full author-distributed sample, including its seven unobserved bins. |
| Native bin width | 10000 | base pairs | Genomic binning, rather than a physical distance between bins. |
| Positive unique pairs | 53883 | pairs | Every stored positive upper-triangle entry is used. |
| Total unique contact count | 1902039 | counts | Sum over the upper triangle; do not double it by treating the mirrored display as independent observations. |
| Locus indexing | Zero-based array index; native BED bounds are one-based inclusive | — | Keep array identities fixed when reconstructing coordinates. |
| Spatial calibration | No absolute distance or handedness calibration supplied | — | Model-distance units are fixed by the contact-law scale convention. |
| Observation model | Count fluctuations and locus visibility bias | — | No additive Gaussian noise level is specified for these sequencing counts. |
| Boundary conditions | No nuclear-envelope or chain-length constraint supplied | — | Do not invent nanometre distances, nuclear confinement or coordinates for unsupported bins. |
| Library identity | Not separately identified in the distributed sample metadata | — | Do not invent a restriction-enzyme replicate or a biological replicate. |

The sample is distributed by the method authors for scientific analysis. A separate per-file data redistribution license is not identified; the accompanying notice preserves this limitation.

## Method Hints

Use iterative equal-visibility correction to estimate the visibility factors, preserving the total unique count. The selected sparse ICE convention stops when the summed absolute change of its bias vector falls below $10^{-4}$. Unobserved loci have neutral bias and do not enter the objective. Weighted multidimensional scaling provides a useful starting geometry through desired distances $\delta_{ij}=(c_{ij}/(b_i b_j\beta))^{1/\alpha}$.

The PM1 family then minimizes the positive-pair Poisson negative log likelihood. With $D$ denoting the set of positive observed pairs, the relevant two objectives are

```math
S(\mathbf X)=\sum_{(i,j)\in D}
\frac{(\lVert\mathbf{x}_i-\mathbf{x}_j\rVert_2-\delta_{ij})^2}{\delta_{ij}^2},
\qquad
L(\mathbf X)=\sum_{(i,j)\in D}\left(\lambda_{ij}-c_{ij}\log\lambda_{ij}\right),
```

where $\mathbf X$ collects all bin coordinates and $\log$ is the natural logarithm. The omitted Poisson factorial term is independent of the coordinates. The benchmark's conditional standard estimator initializes each coordinate uniformly on $[-1,1]$ using the MT19937 seed0, uses L-BFGS-B for both stages with maximum5000 iterations, and retains the standard relative-function stopping factors $10^{12}$ for the initial stress and $10^7$ for the Poisson objective, multiplying machine epsilon. These conventions fix the selected nonconvex solution; centroid shifts do not affect its distances. Different optimizer versions or starting states can converge to other geometries with similar likelihood, which is a limitation of interpreting a method result as a physical structure.

## References

1. Varoquaux N., Ay F., Noble W.S. and Vert J.-P. *A statistical approach for inferring the3D structure of the genome.* Bioinformatics30, i26–i33 (2014). [Published article](https://doi.org/10.1093/bioinformatics/btu268).
2. Duan Z. et al. *A three-dimensional model of the yeast genome.* Nature465,363–367 (2010). [Experimental data publication](https://doi.org/10.1038/nature08973).
