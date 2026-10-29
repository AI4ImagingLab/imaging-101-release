# Ascidian Cell Interface Force Atlas

> Recover a three-dimensional atlas of relative membrane tensions and cell pressures from a real embryo's segmented membrane mesh, despite noisy contact geometry and an unknown absolute force scale.
> Domain: mechanics | Keywords: force inference, tissue mechanics, inverse geometry | Difficulty: Hard

## Background

Contractile cell cortices and intercellular contacts exert forces that shape an embryo. When shape changes are slow compared with mechanical relaxation, the interfaces approximately balance like the films of a heterogeneous foam. Three membrane tensions balance along a triple junction, while pressure differences balance the variation of membrane surface energy.

Three-dimensional microscopy can measure cell geometry much more readily than all of these forces. Inferring a mechanical atlas from that geometry helps study compaction, morphogenesis and the organization of living tissues. Segmentation errors, nonideal junctions and departures from mechanical equilibrium make the inference approximate. Geometry determines force ratios rather than an absolute tension scale.

## Problem Description

The observations are an author-provided triangular membrane mesh of a real 64-cell *Phallusia mammillata* embryo. The mesh is already a processed geometry measurement, rather than raw microscope images. Infer one effective isotropic tension for every observed interface and one pressure for every cell, including the exterior medium.

Let interface index $i$ identify an unordered pair of regions, with tension $\gamma_i$ and area $A_i$. Let $p_c$ and $V_c$ be the pressure and signed enclosed volume of cell $c$. The quasistatic surface-energy model is

```math
\mathcal L=\sum_i\gamma_i A_i-\sum_{c>0}p_c V_c,
\qquad
\nabla_{\boldsymbol x}\mathcal L
=\sum_i\gamma_i\nabla_{\boldsymbol x}A_i
-\sum_{c>0}p_c\nabla_{\boldsymbol x}V_c\approx0.
```

Here $\boldsymbol x$ is the vector of all three-dimensional vertex coordinates; gradients differentiate the measured piecewise planar geometry. At a triple junction the corresponding relation is $\sum_i\gamma_i\boldsymbol\nu_i\approx\boldsymbol0$, where the sum runs over its three incident membranes and $\boldsymbol\nu_i$ is a unit direction in the membrane, perpendicular to the junction. All tension and pressure values share a multiplicative ambiguity. Fix the arithmetic mean of the interface tensions to one and the exterior pressure to zero. Measurement defects make the equations inconsistent, so the target is a conditional mechanical estimate rather than independently measured biological forces.

Save **`output/reconstruction.npz`**, relative to the task workspace, containing both of these arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `tensions` | `(1, 357)` | float32 or float64 | Real relative tensions; dimensionless after fixing their arithmetic mean to one. Columns follow the lexicographically sorted unique unordered pairs in `region_labels`. |
| `pressures` | `(1, 65)` | float32 or float64 | Real pressures in mean-tension units per author mesh coordinate unit; columns follow the ascending unique region labels found in `region_labels`, with exterior region 0 first and fixed to zero. The measured labels are nonconsecutive integers ranging from 0 to 90. |

Both arrays must be complete and finite. Every interface and region is evaluated with cosine similarity and RMS error divided by the corresponding reference dynamic range. There is no fitted scale, permutation, crop or spatial alignment. Evaluation also checks the tension gauge and consistency of all cell pressure normal equations with the supplied geometry. Agreement with a standard mechanical estimate does not establish absolute force accuracy.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`. The leading axis is the single observed specimen.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `vertices` | `(1, 70433, 3)` | float64 | Vertex coordinates in author mesh coordinate units, inherited from voxel geometry; last axis is x, y, z. No micrometre conversion is supplied. |
| `triangles` | `(1, 144312, 3)` | int64 | Zero-based vertex indices of each membrane triangle, dimensionless. |
| `region_labels` | `(1, 144312, 2)` | int64 | The two region labels separated by each triangle, dimensionless; its right-hand normal points from the first listed region toward the second. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Specimen | 64-cell *Phallusia mammillata* embryo | — | Real microscopy-derived geometry distributed by the method authors. |
| Exterior | Region 0 | — | All other region labels identify the 64 cells. |
| Geometry | Full provided membrane mesh | — | No new spatial crop or geometric smoothing is applied. |
| Coordinate metric | Euclidean distance on supplied x, y, z | author mesh unit | The original force model uses these coordinates directly; physical voxel spacing is unrecorded in the provided mesh. |
| Interface convention | Unordered distinct region pairs | — | Label swaps must be accompanied by reversing triangle winding. |
| Surface discretization | Planar triangles | — | Area and signed-volume derivatives use the provided piecewise planar surfaces. |
| Boundary | Closed free embryo, exterior medium | — | No imposed shell or artificial crop face is supplied. |
| Observational uncertainty | No numerical noise variance supplied | — | Segmentation and meshing errors remain part of the measured geometry. |
| Triple-junction validity | Three incident triangles, three distinct interface pairs, three regions | — | Other edge valences carry no triple-junction angle observation; the geometric rule is determined entirely from the input. |

The author mesh retains 3,130 unused vertices and 128 malformed edges touching three regions. Unused vertices can be reindexed without changing any triangle. Retain all observed triangles for area and volume calculations; malformed junctions do not supply a guessed contact angle. The coordinate scale permits relative pressures in the stated units, but does not support pressures in pascals or a physical length calibration.

## Method Hints

A suitable family combines junction force balance with variational pressure inference. For each valid triple line, average its chord-based interior wedge angles over the measured edges and weight the line equations by its length divided by total junction length. A conditional estimate is specified by

```math
\min_{\boldsymbol\gamma}\sum_{\ell}w_\ell^2
\left\|C_\ell\boldsymbol\gamma_\ell\right\|_2^2,
\qquad \frac{1}{357}\sum_i\gamma_i=1,
\qquad
\boldsymbol p=\arg\min_{\boldsymbol q}
\left\|D^{\mathsf T}\boldsymbol\gamma-V^{\mathsf T}\boldsymbol q\right\|_2^2.
```

Here $\ell$ indexes valid triple lines, $w_\ell$ is their relative length, and $\boldsymbol\gamma_\ell=(\gamma_{ab},\gamma_{bc},\gamma_{ac})^{\mathsf T}$ contains the three incident tensions for ordered region labels $a<b<c$. The arithmetic mean wedge angles inside regions $a,b,c$ are $\theta_a,\theta_b,\theta_c$. The scalar Young–Dupré balance matrix is

```math
C_\ell=
\begin{pmatrix}
1&\cos\theta_b&\cos\theta_a\\
\cos\theta_b&1&\cos\theta_c\\
\cos\theta_a&\cos\theta_c&1
\end{pmatrix}.
```

Matrix $D$ contains the gradients of all interface areas in its rows; $V$ contains the gradients of the 64 cell volumes, excluding the exterior. Their columns follow vertex-major x, y, z order. Vector $\boldsymbol q$ contains cell pressures with exterior pressure fixed separately to zero, and $\boldsymbol p$ is its minimizer. Thus the pressure step satisfies $VV^{\mathsf T}\boldsymbol p=VD^{\mathsf T}\boldsymbol\gamma$. Angle conventions, label orientation, the tension normalization and the medium-pressure gauge matter as much as the linear algebra. Alternative force-balance formulations can differ on noisy measured geometry.

## References

Ichbiah et al., “Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy,” *Nature Methods* (2023), [published article](https://doi.org/10.1038/s41592-023-02084-7) and its supplementary mathematical derivations.

The real mesh `Phallusia_Guignard_et_al.rec` is provided in the [authors' example data collection](https://www.dropbox.com/scl/fo/fanim64m3dwy6in9dkq6v/AGnqONanjcZUMEtvFNFy9wo?rlkey=mx7ffzbea3v93rsqrkdr9ok1q&dl=0). The processed example is offered for use with the method; a separate redistribution license for that mesh is not stated. See the accompanying notice for the distinction between software and data rights.
