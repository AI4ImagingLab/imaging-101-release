# Stanford Range Similarity Registration

> Reconstruct a three-dimensional similarity transform and aligned surface samples from two real laser scans with unknown correspondence and partial overlap.
> Domain: mathematics | Keywords: geometric imaging, probabilistic registration, point clouds | Difficulty: Hard

## Background

Laser triangulation measures visible surface positions by combining an illuminated spot with calibrated camera geometry. Each view records a partial surface: self-occlusion prevents a single scan from capturing a complete object. Aligning these measurements supports digitization, shape measurement and combining complementary views.

The scanner records spatial coordinates rather than corresponding surface labels. Changes in viewpoint, nonuniform sampling and missing surface regions make nearest-neighbor associations ambiguous. A coherent transformation must explain many observations simultaneously while allowing unmatched points.

## Problem Description

Two measured point sets describe the Stanford bunny: the target scan has 40,256 samples and the source scan has 40,097 samples. Estimate an orientation-preserving isotropic similarity in their native coordinate system:

```math
z_j=s y_j R+t,\qquad s>0,\quad R^\mathsf{T}R=I,\quad\det R=1.
```

Here each source point $y_j$ and aligned point $z_j$ is a row vector in metres, $R$ is a dimensionless three-dimensional rotation, $s$ is dimensionless scale, and $t$ is a row-vector translation in metres. Target points $x_i$ sample an overlapping surface; their correspondence to source points is unknown. A useful measurement model combines isotropic Gaussian displacements about transformed source samples with a uniform outlier component. The noise variance is inferred, while the conditional uniform-outlier fraction is fixed to0.1 as a model assumption, not independently measured sensor calibration. Partial overlap means residual distance need not vanish even for a useful alignment.

Save `output/reconstruction.npz` relative to the task workspace with every array below. Use finite real `float64` values, preserve source observation order and use no additional coordinate normalization in the saved results.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 40097, 3) | float64 | Aligned source points, axes batch/source-point/xyz, metres in target coordinates |
| rotation | (1, 3, 3) | float64 | Row-vector rotation R, dimensionless |
| translation | (1, 3) | float64 | Row-vector translation t, xyz metres |
| scale | (1, 1) | float64 | Positive isotropic scale s, dimensionless |

Evaluation uses relative Euclidean error for each declared array, defined as estimate-reference difference norm divided by reference norm. It also checks transform consistency, rotation orthogonality and determinant, and RMS nearest-target distance using every aligned source point. There is no post-submission alignment, scale adjustment or cropping. These measures establish agreement with a validated method and measured geometry; no independent exact pose is supplied.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| target_points | (1, 40256, 3) | float64 | Native bun000 measured laser samples, batch/point/xyz, metres |
| source_points | (1, 40097, 3) | float64 | Native bun045 measured laser samples, batch/point/xyz, metres |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Measurement modality | Cyberware laser triangulation | — | Two raw range scans; no fused mesh |
| Target scan | bun000 | — | Native target coordinate table |
| Source scan | bun045 | — | Native source coordinate table |
| Coordinate axes | xyz, original scanner convention | — | Row-vector coordinates; no axis swap or reflection |
| Coordinate scale | 1 | metre per stored coordinate unit | Both tables retain their acquisition scale |
| Correspondences | Unknown | — | No matching labels or scanner pose supplied |
| Surface support | Partially overlapping, open | — | Missing surfaces are unobserved, not zero-valued samples |
| Measurement uncertainty | Not independently specified | metre | Do not assume a calibrated Gaussian variance |
| Boundary conditions | None | — | Unstructured spatial samples, not a periodic image |

The two arrays retain all native vertices and their ordering. The original scanner pose file is not an input. Research use and free redistribution require acknowledgment of the Stanford Computer Graphics Laboratory; commercial-product use requires the source provider's permission, as recorded in `NOTICE`.

## Method Hints

The conditional statistical target uses every40th native point, starting at index0, for fitting (1007target,1003source); all native source points must still be transformed in the output. Write o for the selected target mean and a=sqrt(mean over all its scalar coordinates of (x-o)^2) in metres. Normalize selected coordinates as X=(x-o)/a and Y=(y-o)/a. With Ntarget points, Msource points, fixed outlier fraction w=0.1 and inferred dimensionless variance v>0, the observed negative log likelihood is

```math
J(R,s,t,v)=-\sum_{i=1}^{N}\log\left[\frac{1-w}{M(2\pi v)^{3/2}}
\sum_{j=1}^{M}\exp\left(-\frac{\|X_i-sY_jR-t\|^2}{2v}\right)+\frac{w}{N}\right].
```

Here t is normalized-coordinate translation. Save the native translation a t+o-s o R. Select a stable nondegenerate stationary fit of J in proper rotation, positive scale, translation and positive variance, satisfying native full-source nearest-target RMS<=0.0024525798338538444m; among stable stationary solutions satisfying this geometry condition, use the smallest J. Stable means a local minimum in translation, log scale, log variance and the three rotation tangent directions. The unrestricted global mixture likelihood has a zero-variance boundary singularity and is not claimed to identify an absolute global pose optimum. This stationary-fit convention does not prescribe EM initialization, update count or stopping arithmetic.

Coherent point drift treats the transformed source points as centers of a Gaussian mixture and estimates soft correspondences. A uniform component reduces the influence of unmatched measurements. Expectation-maximization alternates posterior correspondence weights with a weighted similarity fit: centering, a determinant-corrected singular value decomposition, positive scale and translation, followed by a variance update. Jointly normalizing the measured coordinate sets can improve numerical conditioning, provided the final transformation is converted back to metres. A single scalar scale is allowed; an affine deformation is outside this problem.

## References

- A. Myronenko and X. Song, “Point Set Registration: Coherent Point Drift,” IEEE TPAMI 32(12), 2262–2275 (2010). https://doi.org/10.1109/TPAMI.2010.46 ; https://arxiv.org/abs/0905.2635
- Stanford Computer Graphics Laboratory, “The Stanford 3D Scanning Repository,” raw bunny range scans and research-use terms. https://graphics.stanford.edu/data/3Dscanrep/
