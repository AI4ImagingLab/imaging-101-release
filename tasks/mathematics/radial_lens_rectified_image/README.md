# Radial Lens Rectified Image

> Recover a geometrically rectified photograph and radial lens parameters from a measured distorted image and observed straight-edge samples whose curvature obscures the scene geometry.
> Domain: mathematics | Keywords: geometric imaging, camera calibration, algebraic inversion | Difficulty: Hard

## Background

A pinhole camera maps straight scene lines to straight image lines. Wide-angle optical systems can bend those projections because radial magnification varies with distance from the optical center. This distortion compromises measurements of shape, position and orientation even when the photograph remains visually recognizable.

Geometric rectification is useful in architectural measurement, machine vision and scientific photography. Observations of known-straight physical edges constrain the optical mapping without requiring a complete three-dimensional calibration target. Recovering geometry requires separating perspective, which preserves lines, from nonlinear lens distortion, which bends them.

## Problem Description

The supplied 688 by506 RGB photograph shows an actual photographed printed calibration pattern. Two author-selected edge groups contain15 and14 observed points. Their latent undistorted points lie on two unknown straight lines. No pristine digital pattern is supplied as a target. Infer radial parameters and rectify the complete photograph.

Let x be a measured two-dimensional pixel coordinate, c the fixed optical center, r its distance from c, u the undistorted coordinate, and L the radial scale:

```math
r=\lVert x-c\rVert,\qquad u=c+L(r)(x-c),\qquad L(r)=1+k_2r^2+k_4r^4.
```

The unknown coefficients have units pixel⁻² and pixel⁻⁴. The center is fixed by convention, not estimated from an unavailable ground-truth image. Square pixels and radial symmetry are assumed. Each measured point group maps to an unknown straight line plus localization error. Two finite segments constrain only a low-order radial approximation; perspective, tangential aberration and arbitrary scene depth are not recovered.

Save `output/reconstruction.npz` with both arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 506, 688, 3) | uint8 | Rectified RGB intensity,0–255 digital codes; axes batch,row y,column x,color channel; no extra crop,rotation or contrast change. |
| distortion | (1, 5) | float64 | Real radial polynomial coefficients ordered k0,k1,k2,k3,k4, with k0=1 and k1=k3=0; units pixel to the negative power of coefficient index. |

The output frame preserves visibility of the transformed source corners. Set s=L(R), where R is the center-to-corner radius using corner coordinates(0,0),(688,0),(0,506),(688,506). For output coordinate v, evaluate the undistorted coordinate u=c+s(v−c), invert the radial mapping, then bilinearly sample the measured RGB image. Invalid samples are black. Truncate valid interpolated nonnegative intensity to uint8. A constant interpolation neighborhood must retain its exact integer intensity. There is no independent geometric alignment during evaluation.

Evaluation measures cosine NCC of flattened images, RMS image error divided by the reference intensity range, maximum relative error of the two nonzero radial coefficients, and deviations from the declared fixed/zero coefficients. If a is an estimate and b a reference vector, NCC is aᵀb divided by their Euclidean norms; an all-zero estimate receives zero. Both image and geometry must pass. This is reproducibility against a standard geometric inference, not independently known distortion truth.

## Data Description

Load `data/raw_data.npz` using `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| image | (1, 506, 688, 3) | uint8 | Complete measured RGB photograph,0–255 digital codes; axes batch,row,column,RGB. |
| line_points | (1, 29, 2) | float64 | Observed edge positions in pixels; axes batch,point,coordinate; each pair is(x,y). |
| line_ids | (1, 29) | int32 | Dimensionless group identity: first15 points belong to0, remaining14 to1. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Width,height | 688,506 | pixel | Complete author photograph; no downsampling |
| Coordinate origin | Upper-left pixel center(0,0) | pixel | x rightward,y downward |
| Optical center | (344,253) | pixel | Fixed midpoint convention |
| Pixel aspect ratio | 1 | dimensionless | Square-pixel radial model assumption |
| RGB scale | 0–255 | digital code | Author-distributed processed photograph; no radiometric linearity claimed |
| Point geometry | Two straight physical edges | constraint | Lines have unknown orientation and offset |
| Point localization variance | Unavailable | pixel² | No calibrated statistical noise model |
| Camera/lens/exposure | Not documented for this frame | — | No invented instrument identity or focal-length inference |
| Radial convention | Distorted pixel to undistorted pixel | mapping | Positive quadratic/quartic terms expand outer points |
| Image boundary | Black outside valid source support | digital code | Renormalize available bilinear weights at valid boundary cells |
| Output framing | Complete transformed corners visible | dimensionless | Scale s specified above; same output dimensions |

## Method Hints

An algebraic straight-line method minimizes the determinant of each transformed point group's scatter matrix. Because transformed coordinates are affine in two unknown radial coefficients, this objective is quartic. Fixing the central scale avoids the trivial zero-map solution. A Sylvester resultant of the two cubic derivative polynomials produces candidate stationary points; compare their objective values. Radius normalization improves conditioning. Inverse radial root solving then supplies the geometric warp for image resampling. This is a joint geometric calibration and rectification problem, not simple denoising.

## References

L. Alvarez,L. Gomez,J.R. Sendra, *Algebraic Lens Distortion Model Estimation*, Image Processing On Line1,1–10(2010). [Paper and original materials](https://doi.org/10.5201/ipol.2010.ags-alde).

L. Alvarez,L. Gomez,R. Sendra, *An Algebraic Approach to Lens Distortion by Line Rectification*, Journal of Mathematical Imaging and Vision35,36–50(2009). [Paper](https://doi.org/10.1007/s10851-009-0153-2).
