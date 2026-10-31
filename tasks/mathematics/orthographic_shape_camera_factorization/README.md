# Orthographic Shape and Camera Factorization

> Reconstruct three-dimensional landmark geometry and camera motion from complete two-dimensional tracks when neither scene depth nor viewing directions are known.
> Domain: mathematics | Keywords: inverse geometry, structure from motion, matrix factorization | Difficulty: Hard

## Background

A moving camera observes different projections of a rigid object's landmarks. When the viewing distance is large relative to object depth, an orthographic model approximates image formation. The changes across frames contain geometric information that a single image cannot provide.

Recovering geometry and motion together supports visual metrology, robotic mapping and three-dimensional scene interpretation. The two unknown factors interact bilinearly: many affine shapes and camera matrices explain the same tracks. Physical camera constraints reduce this ambiguity, while a global coordinate-frame ambiguity remains unavoidable.

## Problem Description

Use the complete supplied sequence of 51 frames and 491 corresponding landmarks. The acquisition origin and scene identity are not independently documented; these are author-provided numerical tracks, with no claim of a particular measured scene or known three-dimensional truth. All observations are reused without synthesis or cropping.

For frame $f$ and point $j$, the observation $w(f,j)$ is a two-component image-coordinate vector in pixels. The forward model is

```math
w(f,j)=R(f)X(j)+t(f)+e(f,j).
```

Here $X(j)$ is the unknown three-component landmark position in pixel-equivalent scene units, $R(f)$ is an unknown dimensionless 2-by-3 camera matrix, $t(f)$ is the two-component image translation in pixels and $e(f,j)$ denotes measurement/model error. An ideal orthographic camera has $R(f)R(f)^T=I_2$, where $I_2$ is the 2-by-2 identity matrix. The rigid shape is shared by all frames. The calibrated numerical standard fits approximate orthographic cameras to imperfect tracks; row-orthogonality residuals need not be exactly zero.

Fix the translation gauge by requiring the landmark centroid to be zero; then $t(f)$ is the frame's observed landmark centroid. Fix scale by the unit camera-row convention and the input pixel coordinate scale: there is no unknown focal length or millimetre conversion. The remaining global orthogonal ambiguity includes reflection. If an orthogonal 3-by-3 matrix $Q$ transforms every row landmark vector as $X^T\mapsto X^TQ$, the same $Q$ must transform every camera as $R\mapsto RQ$. This leaves all image predictions unchanged. Independent camera/shape alignment is invalid.

Save `output/reconstruction.npz`, relative to the task workspace, containing exactly these three real arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 491, 3) | float32 or float64 | Zero-centroid landmark positions in pixel-equivalent units; axes batch, landmark, Cartesian component |
| cameras | (1, 51, 2, 3) | float32 or float64 | Dimensionless camera rows mapping Cartesian shape to image x then image y; axes batch, frame, image component, scene component |
| translations | (1, 51, 2) | float32 or float64 | Image translations in pixels; axes batch, frame, image x/y component |

All values must be finite. Correspondence and frame order must remain unchanged. Evaluation fits one global orthogonal matrix, including possible reflection, using all landmark coordinates and applies it jointly to shape and cameras. No additional scale, translation, crop or point permutation is fitted. Each complete array is compared using cosine NCC without mean subtraction and RMS error divided by its reference dynamic range. An additional RMS image reprojection error in pixels evaluates the three submitted quantities jointly on every observation. Reference agreement measures reproduction of a standard geometric inference; no independent three-dimensional accuracy is asserted.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| tracks | (1, 51, 491, 2) | float64 | Author-provided point positions in image-coordinate pixels; axes batch, frame, landmark, image x then image y |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Projection | Orthographic approximation | — | Common rigid landmark geometry across frames |
| Frames / points | 51 / 491 | count / count | Complete supplied sequence and correspondence set |
| Track scale | Unchanged author image coordinates | pixel | No centring or rescaling has been applied to public observations |
| Coordinate order | Image x, image y | pixel | Supplied author axis orientation retained; no camera intrinsics are supplied |
| Shape centroid | (0, 0, 0) | pixel-equivalent | Fixes the translation gauge |
| Camera row scale | Approximately one | dimensionless | Orthographic metric-upgrade convention fixes scene scale |
| Global shape orientation | Unspecified, reflection permitted | — | Joint orthogonal gauge only |
| Missing data | None | — | Every landmark is present in every frame |
| Noise model | Not independently calibrated | pixel | Do not assume the residual is an instrument-noise variance |
| Acquisition origin / scene | Not independently documented | — | Author-provided tracks; no unsupported real-scene claim |

## Method Hints

A rank-three factorization of the centred measurement matrix separates affine motion from affine structure. Metric constraints on camera-row inner products determine a symmetric 3-by-3 correction, removing the affine ambiguity up to a global orthogonal gauge. Preserve the image coordinate scale when returning shape and translation.

To identify the selected numerical standard, form interleaved x/y rows, subtract each row's mean, and normalize by the sum of the maximum absolute centred x coordinate and maximum absolute centred y coordinate. If the rank-three singular-value decomposition gives affine factors $\widehat M$ and $\widehat S$ with symmetric square-root singular-value splitting, fit a symmetric matrix $C$ so each frame's two rows of $\widehat M$ have unit norms and zero mutual inner product under $C$. Use the least-squares solution of all three constraints per frame. For $C=U\Sigma V^T$, define $G=U\sqrt\Sigma$; on this case $C$ is positive definite.

The native rigid helper adds a joint row-orthogonality correction. Let $N=\widehat M G$ and let $D$ stack, frame by frame, the transpose-compatible pseudoinverses $(N(f)^T)^+$ of its 2-by-3 blocks $N(f)$. With $H$ the least-squares solution of $NH=D$, use $Q=G\sqrt H$, where $\sqrt H$ is the principal matrix square root. The upgraded factors are $\widehat M Q$ and $Q^{-1}\widehat S$; restore the normalization scale to the shape. These formulas define the selected closed-form rigid helper, rather than a later iterative non-rigid reconstruction algorithm. Check reprojection and camera Gram matrices independently; a Cholesky factor with the wrong transpose convention can preserve reprojection while corrupting metric geometry.

## References

Carlo Tomasi and Takeo Kanade, *Shape and motion from image streams under orthography: a factorization method*, International Journal of Computer Vision 9, 137–154 (1992). https://doi.org/10.1007/BF00129684

Marco Paladini, Alessio Del Bue, Marko Stošić, Marija Dodig, João Xavier and Lourdes Agapito, *Factorization for non-rigid and articulated structure using metric projections*, CVPR (2009). The selected implementation is its closed-form rigid factorization helper, not the complete iterative non-rigid algorithm.
