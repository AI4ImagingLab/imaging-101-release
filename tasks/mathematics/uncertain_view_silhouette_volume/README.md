# Uncertain View Silhouette Volume

> Reconstruct a three-dimensional occupancy-density volume and eight camera orientations from silhouettes with uncertain angular readings.
> Domain: mathematics | Keywords: geometric inverse problems, silhouette imaging, deep image prior | Difficulty: Hard

## Background

Silhouettes carry information about the outer support of an object while discarding texture and much of its internal structure. Combining silhouettes from several viewing directions can recover a three-dimensional shape and supports applications in object digitization, inspection and geometric modeling.

Sparse views leave concave or hidden regions ambiguous. Angular calibration errors also change which rays intersect the object, so treating uncertain camera readings as exact can distort the recovered shape. Jointly estimating a volume and orientations requires both geometric consistency and a prior that selects plausible solutions among many matching projections.

## Problem Description

The supplied author-generated synthetic case contains eight128-by128 silhouette-opacity images and eight noisy Euler-angle readings. The unknowns are a128-by128-by128 dimensionless density volume and one three-angle orientation for each image. The known world frame and physical angular readings anchor the coordinate gauge; outputs must use this frame without posthoc rigid alignment or reflection.

For a row-coordinate rotation $R(v)$ specified by an Euler triple $v$, the opacity model is

```math
I_v(x,z)=1-\exp\left[-\tau\sum_y\mathcal T_{R(v)}\rho(x,y,z)\right].
```

Here $\rho$ is density, $I$ is dimensionless opacity, $\tau=0.5$ is a dimensionless projection factor, and $\mathcal T$ rotates/resamples the volume. This is the selected differentiable silhouette approximation rather than a calibrated material absorption experiment. The three row-coordinate matrices are

```math
R(v)=R_z(\gamma)R_y(\beta)R_x(\alpha),\qquad
R_x=\begin{pmatrix}1&0&0\\0&\cos\alpha&\sin\alpha\\0&-\sin\alpha&\cos\alpha\end{pmatrix},
```

```math
R_y=\begin{pmatrix}\cos\beta&0&-\sin\beta\\0&1&0\\\sin\beta&0&\cos\beta\end{pmatrix},\qquad
R_z=\begin{pmatrix}\cos\gamma&\sin\gamma&0\\-\sin\gamma&\cos\gamma&0\\0&0&1\end{pmatrix}.
```

The Euler triple is $v=(\alpha,\beta,\gamma)$ in radians. Sample normalized row coordinates $q$ inclusively on $[-1,1]^3$, multiply by $R$, then map each component to the voxel index $(qR+1)(N-0.5)/2$ with $N=128$. Clamp indices to $[0,N-1]$ and use eight-corner interpolation. Preserve the selected interpolation convention: for each scalar coordinate $s$, the lower and upper corners are independently $\lfloor s\rfloor$ and $\lceil s\rceil$. The factors multiplying the lower and upper corner values are $|\lceil s\rceil-s|$ and $|s-\lfloor s\rfloor|$, respectively. At an exact integer both factors vanish, including coordinates clamped to a boundary. This source convention is imperfect geometrically but defines the bound model consistently.

The readings follow independent circular von Mises errors around the true Euler components, with concentration100. Silhouettes alone cannot determine invisible concavities uniquely, and jointly recovering viewpoints introduces additional ambiguities.

Save `output/reconstruction.npz` relative to the task workspace with exactly these finite real arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 128, 128, 128) | float32 or float64 | Dimensionless density in[0,1], axes batch,x,y,z on the inclusive normalized cube[-1,1]³ |
| viewpoints | (1, 8, 3) | float32 or float64 | Euler angles in radians, axes batch,image,angle; angle order alpha,beta,gamma and row rotations defined above |

Density allows absolute1e-6 numerical roundoff outside[0,1]. Evaluation uses full-volume uncentered cosine NCC, RMS error normalized by the reference density dynamic range, occupancy IoU at density>0.5, and the RMS and maximum SO(3) geodesic orientation errors over all eight views. Geodesic error is the angle of the relative rotation, so equivalent Euler representations are accepted. No free alignment, scaling, cropping, threshold fitting or target-derived gauge correction is applied. Every condition must pass.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| silhouettes | (1, 8, 128, 128) | float32 | Dimensionless opacity in[0,1]; axes batch,image,x,z |
| view_prior | (1, 8, 3) | float32 | Noisy measured Euler readings in rad; axes batch,image,alpha/beta/gamma |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Volume/image size |128 | voxel/pixel | Cubic volume and square projection sampling |
| Views |8 | image | Supplied order is authoritative |
| Coordinate frame |x,y,z inclusive[-1,1] | normalized | Voxel centers ordered as defined in the forward model |
| Ray integration axis |y | axis | Images use rowx,columnz after volume rotation |
| Projection factor tau |0.5 | dimensionless | Maps ray density sum to opacity |
| Angular-error concentration |100 | dimensionless | Independent circular errors for each Euler component |
| Added image-noise standard deviation |0 | opacity | Bound source acquisition has no added image noise |
| Exterior convention |Clamped index then floor/ceil weights | voxel | Exact integer/boundary samples vanish under the selected model |
| World orientation |Anchored by supplied angular readings | rad | No unannounced alignment freedom |

## Method Hints

A three-dimensional convolutional generator can serve as an untrained deep prior: fit its weights to these observations rather than loading a trained model. Jointly optimize its density output and camera angles. Mean absolute opacity error provides the data term; a negative cosine penalty on angle differences encodes the circular measurement prior. The generator architecture restricts the volume representation while viewpoint regularization controls the coupled geometry ambiguity. Preserve the explicit acquisition and coordinate conventions when forming derivatives.

The complete selected training path requires a CUDA GPU in the validated runtime. CPU forward/gradient calculations are supported for small mathematical checks; complete128³ training on CPU has not been validated.

## References

Matheus Gadelha, Rui Wang and Subhransu Maji, *Shape Reconstruction using Differentiable Projections and Deep Priors*, ICCV2019. https://openaccess.thecvf.com/content_ICCV_2019/html/Gadelha_Shape_Reconstruction_Using_Differentiable_Projections_and_Deep_Priors_ICCV_2019_paper.html
