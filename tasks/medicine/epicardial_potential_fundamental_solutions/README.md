# Epicardial potential reconstruction from jacket measurements

> Reconstruct the cardiac-surface potential distribution from sparse jacket potentials and nested geometry by solving an ill-conditioned harmonic continuation problem.
> Domain: medicine | Keywords: electrocardiographic imaging, harmonic continuation, regularization | Difficulty: Hard

## Background

Electrical activity in the heart produces potentials throughout the surrounding conductive volume. Measurements on an outer surface contain a smoothed, attenuated signature of the cardiac distribution. Recovering the inner distribution can reveal localized electrical features that are difficult to see directly in sparse external measurements.

Potential-based electrocardiographic imaging treats the region between cardiac and outer surfaces as a source-free conductor. Volume conduction strongly suppresses spatial detail, so small measurement errors can become large changes at the cardiac surface. Geometry and boundary conditions are therefore essential components of the inverse problem.

## Problem Description

Infer247 inner sock-location potentials from one frame of191 author-supplied recorded jacket potentials, a closed torso/tank mesh and a closed heart mesh. The exact sample and acquisition identity are unspecified; this is an author-recorded demonstration of unknown sample type, not an asserted human dataset. Absolute voltage calibration is unknown, so retain the original native potential scale and reference offset without guessing millivolts. No independent cardiac potential truth is supplied.

Assume a homogeneous isotropic source-free conductor between the two closed surfaces. The potential phi obeys:

```math
\nabla^2\phi=0\quad\mathrm{in}\ \Omega,\qquad
\phi(x_i)=y_i,\qquad\partial_n\phi(x_i)=0.
```

Omega is the region exterior to the heart and interior to the outer mesh; xi are the jacket electrode positions on the outer boundary, yi their measured potentials, and n its outward unit normal. The derivative condition represents an insulating outer boundary. The target is phi at each supplied sock position. This continuation is ill-conditioned: many fine inner patterns give almost identical outer data. Measurements are noisy and satisfy the boundary conditions only approximately.

Save `output/reconstruction.npz` with the sole required key `reconstruction`, real float64 shape(1,247), axes(batch,sock location), in the same native potential unit and reference offset as the measured potentials. No arbitrary normalization, amplitude scaling, centering, sign reversal or alignment is allowed. Evaluation uses uncentered cosine similarity, dynamic-range-normalized RMS error, and maximum absolute potential error against a validated standard-method reference. These measure numerical agreement, not independent physiological accuracy.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| heart_vertices | (1,337,3) | float64 | Closed inner mesh x/y/z coordinates in mm |
| heart_faces | (1,670,3) | int64 | Zero-based oriented triangle vertex indices, dimensionless |
| torso_vertices | (1,284,3) | float64 | Closed outer mesh x/y/z coordinates in mm |
| torso_faces | (1,564,3) | int64 | Zero-based oriented triangle vertex indices, dimensionless |
| sock_vertices | (1,247,3) | float64 | Desired cardiac-surface evaluation positions x/y/z in mm |
| electrode_indices | (1,191) | int64 | Zero-based jacket-electrode indices into torso_vertices, dimensionless |
| potentials | (1,1,191) | float64 | Recorded outer potentials, native potential unit, axes(batch,frame,electrode) |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Coordinate system | Native Cartesian x/y/z | mm | Identical frame for both meshes, electrodes and sock locations |
| Face orientation | Outward from each enclosed mesh | dimensionless | Both meshes closed, consistently oriented |
| Measured frame count | 1 | frame | Full author demonstration frame, no inferred time sampling |
| Outer sensors | 191 | electrodes | All measured jacket potentials retained |
| Target locations | 247 | sock electrodes | Independent supplied evaluation geometry |
| Conductivity | Homogeneous isotropic constant | S/m unspecified | Constant cancels in the source-free Laplace equation and zero-current boundary |
| Boundary condition | Zero normal derivative on outer boundary | native potential unit/mm | Insulating outer surface approximation |
| Reference potential | Supplied native reference | native potential unit | Preserve its offset; absolute reference identity unspecified |
| Noise | Not calibrated | native potential unit | No independent uncertainty supplied |
| Sample/acquisition identity | Unknown | none | Do not assert human, animal species, stimulus or acquisition date |

## Method Hints

A meshless fundamental-solution representation places virtual sources outside the conducting region and expresses the potential as a constant plus a weighted sum of1/(4pi distance) kernels. Collocating potential and normal-derivative constraints gives a linear Cauchy system. Singular-value regularization suppresses poorly determined directions; an automatic regularization-choice criterion can balance residual and source-coefficient growth. Check source placement, normal orientation and boundary residuals because a small outer-data error alone does not establish a unique cardiac solution.


The evaluated standard recovery fixes the otherwise nonunique numerical representation and regularization as follows. Use dimensionless coordinates q=x/(1mm), with x the supplied physical coordinate. At each mesh vertex, the outward normal is the normalized sum of adjoining unit triangle normals weighted by that vertex's triangle angle. For each mesh let d be its bounding-box diagonal length in q coordinates. Virtual sources are q+0.1d n for the outer mesh and q-0.1d n for the heart mesh. Keep the outer sources before the inner sources.

For source s define k(q,s)=1/(4pi norm(q-s)). Represent phi(q)=a0+sum over sources of a(s)k(q,s). The collocation matrix A has a potential row [1,k] and an outward-normal row [0,partial-n k] at every measured jacket vertex; stack all potential rows before all normal rows. Its right hand side b is the measured potentials followed by zeros. The derivative uses q, so its physical unit is converted by1mm before combining the rows. With the thin singular-value decomposition A=U diag(sigma) V-transpose, set z=U-transpose b and define

```math
C(\lambda)=\sum_i
\frac{\sigma_i^2 z_i^2(\sigma_i^2-3\lambda^2)}
     {(\sigma_i^2+\lambda^2)^3},\qquad
\lambda_* = \operatorname{argmax}_{\lambda\in\Lambda} C(\lambda),
\qquad
\Lambda=\left\lbrace10^{-8}+j\frac{10^{-3}-10^{-8}}{9999}:j=0,\ldots,9999\right\rbrace.
```

Choose the first grid value if tied. The coefficient vector, including its constant term, is

```math
a=V\operatorname{diag}\!\left(\frac{\sigma_i}{\sigma_i^2+\lambda_*^2}\right)U^\mathsf{T}b.
```

Evaluate this same expansion at every sock position. Sigma denotes a singular value, z its projected datum, and lambda the dimensionless numerical regularization parameter under the stated coordinate convention. These definitions specify the canonical estimate used for this task's standard-method comparison. A different source placement or regularization rule still estimates the same physical potential, but does not define that canonical estimate merely by achieving a small outer-boundary fit.

## References

- Wang and Rudy, Application of the Method of Fundamental Solutions to Potential-based Inverse Electrocardiography, Annals of Biomedical Engineering2006. [DOI](https://doi.org/10.1007/s10439-006-9131-7).
- Johnston/Tate, SCI Forward/Inverse Toolkit normal-displacement MFS demonstration2016/2017. The selected implementation variant uses mesh normals rather than the paper's radial scaling. The task uses independently written mathematical code and the complete author demonstration inputs.

This is an internal research package: the original demonstration has no verified explicit source/data redistribution license. Permission must be checked before external distribution; no permissive license is implied.
