# Muon Statue Scattering Volume

> Reconstruct a three-dimensional angular-scattering strength field from six-plane cosmic-muon detector hits, where distributed scattering, sparse voxel occupancy and nearly parallel trajectories obscure internal structure.
> Domain: physics | Keywords: particle imaging, scattering tomography, geometric inversion | Difficulty: Hard

## Background

Cosmic-ray muons penetrate stone, metal and other dense materials. Their Coulomb interactions with nuclei produce angular deflections that depend on material composition, path length and particle momentum. Position-sensitive trackers placed above and below an object measure the incoming and outgoing trajectories without an artificial radiation source.

Scattering tomography supports inspection of dense structures, concealed materials and cultural-heritage objects that are difficult to transport. Finite particle flux and distributed interactions limit spatial resolution. An angular-scattering image is a relative material-contrast measurement; converting it into absolute mass density requires additional momentum and path-length information.

## Problem Description

Use every supplied event from an author-provided GEANT4/CRY simulation of a sculpture-scale specimen. The observations are simulated detector hit coordinates, not a laboratory acquisition. All six recorded three-dimensional hit positions are retained, including their small within-detector z variation. No additional detector noise or event losses are injected.

For event n, the upper and lower tracker observations obey

```math
\mathbf h^{\pm}_{nj}=\mathbf p^{\pm}_n+t^{\pm}_{nj}\mathbf u^{\pm}_n+\boldsymbol\epsilon^{\pm}_{nj},
\qquad \|\mathbf u^{\pm}_n\|=1.
```

Here h is a recorded detector position in mm, p is a point on the corresponding straight trajectory, t is distance along it in mm, u is a dimensionless direction vector oriented towards decreasing z, and epsilon describes deviation from the fitted line. The superscripts denote the incoming and outgoing trackers. The geometric deflection is

```math
\theta_n=\arccos\frac{\mathbf u^-_n\cdot\mathbf u^+_n}
 {\|\mathbf u^-_n\|\,\|\mathbf u^+_n\|}.
```

Theta is in radians. Multiple interactions throughout the specimen contribute to this deflection. A single-dominant-interaction model assigns an effective scattering location inside the reconstruction volume. Recover the voxel field q whose value represents the root-mean-square angular deflection associated with that location. Its population interpretation is a conditional angular second moment. The required finite-data statistic uses the robust truncation convention below. This conditional angular contrast has units rad and is not an absolute density in g/cm^3. Unknown momenta, the single-interaction approximation and finite sampling prevent a unique absolute material-density interpretation. Empty voxels have zero estimated strength.

Save `output/reconstruction.npz`, relative to the task workspace, containing the following required array. Load/save NPZ without object arrays.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 36, 80, 40) | float32 | Real angular-scattering RMS q, rad; axes batch,x,y,z; nonnegative native physical scale |

Voxel centers are (-175+10i, -395+10j, -855+10k) mm for indices i=0,...,35, j=0,...,79 and k=0,...,39. Evaluation uses every voxel, cosine similarity without mean subtraction and RMS error divided by the reference dynamic range. Both criteria must pass. There is no spatial alignment, cropping, gain fitting, renormalization or conversion to mass density.

## Data Description

`data/raw_data.npz` contains only the full bound detector observations.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| detector_hits | (1, 522963, 6, 3) | float32 | Detector positions, mm; axes batch,event,plane,Cartesian coordinate; Cartesian order x,y,z |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Data class | Author GEANT4/CRY simulation | dimensionless | Recorded observations reused without new synthesis |
| Recorded events | 522963 | muon | Every event traversing all six detector planes in the supplied acquisition |
| Generated exposure | 5000000 source particles; approximately 8 hours | muon; h | Author cosmic-ray source normalization; only complete recorded traversals appear in the array |
| Detector arrangement | Three upper planes, then three lower planes | dimensionless | Plane indices0–2 are incoming,3–5 outgoing |
| Coordinate frame | x,y transverse; z increases upward | mm | Incoming particles travel towards decreasing z |
| Nominal upper-face z coordinates | -9.5,-409.5,-459.5,-859.5,-909.5,-1309.5 | mm | Plane order0–5; use each actual recorded z coordinate rather than snapping to these nominal values |
| Recorded z extent | Within approximately1 mm below each listed upper face | mm | Preserved source hit-depth variation |
| Nominal detector transverse coverage | 1000 by1000 | mm | Planar acceptance; keep all recorded x,y positions |
| Added detector position noise | 0 | mm | No new noise or smearing is added to the stored source hits |
| Added event loss | 0 | fraction | Full recorded source event table is retained |
| Reconstruction box | x[-180,180],y[-400,400],z[-860,-460] | mm | Inclusive physical boundary; exterior effective interactions are excluded |
| Voxel dimensions | 10 by10 by10 | mm |36 by80 by40 voxels; C-order flattened z fastest, array axes explicitly x,y,z |
| Boundary assignment | Internal shared edge selects upper-index cell; global maximum stays in final cell | dimensionless | Fixed discretization convention |

Particle energies and source material labels are not supplied or required for the unweighted angular field. The full detector positions suffice to reconstruct the trajectories and relative scattering image.

## Method Hints

The point-of-closest-approach family first fits straight trajectories by orthogonal least squares. The principal right singular vector of centered detector hits gives the line direction. Reverse the entire direction vector when needed to orient it downward; changing only one component does not preserve the fitted line. A nonparallel test uses the norm of the cross product, rather than requiring every cross-product component to be nonzero.

For incoming/outgoing lines p+t u and r+s v, solve for the two points with the shortest perpendicular connector, and use its midpoint as the effective interaction location. If w=v×u, the small linear system is

```math
[\mathbf u,-\mathbf v,\mathbf w]
\begin{bmatrix}t\\s\\a\end{bmatrix}=\mathbf r-\mathbf p.
```

Here a is the auxiliary coefficient of the common perpendicular. The two fitted trajectory directions are unit vectors oriented toward decreasing z. Pairs with cross-product norm below1e-7 are geometrically unresolvable. Compute geometry in double precision before the final float32 field; coordinate quantization does not authorize single-precision intersection arithmetic. The stable angle atan2(norm(u cross v),u dot v) is equivalent to the displayed geometric angle and avoids near-parallel cancellation. For retained in-volume events let T be the empirical0.999 angle quantile, using
linear interpolation between adjacent sorted values at fractional index0.999(N-1).
Set t_n=min(theta_n,T). For each voxel with N_v events form

```math
r_v=\sqrt{\frac{1}{N_v}\sum_{n\in v}t_n^2},\qquad
q_v=\min(r_v,Q_{0.999}(r)).
```

Empty voxels have r_v=0. The second quantile includes every voxel, including
empty cells, with the same interpolated-quantile definition. These define a
robust finite-data angular-moment estimand, not an absolute material density.
Event retention uses only the input geometry and fixed physical box, without
hidden material labels. Preserve the complete vector geometry and voxel edges.

## References

- Giammanco et al. (2025), *Toward using cosmic rays to image cultural heritage objects*, iScience28,112094. [Published article](https://doi.org/10.1016/j.isci.2025.112094). [Author preprint, v1](https://arxiv.org/abs/2405.10417v1), including AppendixA on point-of-closest-approach geometry.
- Lagrange et al. (2025), *Muon tomography for cultural heritage objects*, Journal of Applied Physics138,104902. [Article](https://doi.org/10.1063/5.0273541).
- Author Muograph source and accompanying simulated observations are redistributed under AGPL-3.0 terms preserved in `LICENSE`. The selected angular-RMS implementation is a qualified geometric reproduction with disclosed source corrections, not an assertion of unchanged-source numerical identity.
