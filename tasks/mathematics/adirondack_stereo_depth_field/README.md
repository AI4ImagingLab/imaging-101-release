# Adirondack Stereo Depth Field

> Reconstruct a dense optical-axis depth field from two calibrated real photographs despite occlusions, weak texture and radiometric differences.
> Domain: mathematics | Keywords: inverse geometry, stereo imaging, discrete regularization | Difficulty: Hard

## Background

Two cameras observe a scene from different positions. A visible surface point projects to a different horizontal position in each image, and the displacement depends inversely on distance. Calibrated stereo uses this parallax to recover three-dimensional geometry without projecting a measurement pattern during the photographic acquisition.

Dense geometric reconstruction supports mapping, robotic navigation, inspection and scene modeling. Occluded surfaces appear in only one camera, while repeated textures and nearly uniform surfaces make correspondences ambiguous. Reflection and illumination differences also limit the validity of exact brightness agreement.

## Problem Description

The observations are the complete, author-distributed quarter-resolution Adirondack rectified stereo pair. No additional crop or resampling is applied. For a surface point with left-camera coordinates X,Y,Z in millimetres, the ideal rectified projection is

```math
u_L=fX/Z+c_L,\qquad u_R=f(X-B)/Z+c_R,\qquad
v_L=v_R=fY/Z+c_y.
```

Here u and v are horizontal and vertical pixel coordinates, L and R identify the left and right cameras, f is focal length in pixels, B is baseline in millimetres, and cL,cR,cy are principal-point coordinates. The positive optical axis points into the scene. Define disparity D=uL-uR and the known rectified coordinate-origin offset a=cR-cL. Then

```math
D=fB/Z-a,\qquad Z=\frac{fB}{D+a}.
```

For a mutually visible approximately Lambertian surface, the photometric observation relation is approximately IL(u,v)=IR(u-D(u,v),v), where IL and IR are the observed RGB images. Occlusions, non-Lambertian reflection, small residual rectification errors and sampling prevent this relation from holding everywhere. A solution must resolve ambiguous correspondences while preserving depth discontinuities.

Save `output/reconstruction.npz`, relative to the task workspace, with exactly the following two arrays. Neither output depends on a particular solver module layout.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,496,718) | float32 or float64 | Real optical-axis depth Z in mm, axes batch,row,column, on the complete left-image grid; positive and finite everywhere. |
| disparity | (1,496,718) | float32 or float64 | Real left-minus-right disparity D in pixels, axes batch,row,column; finite everywhere, D+a positive, and geometrically consistent with reconstruction. |

Evaluation uses cosine NCC without mean subtraction, dynamic-range-normalized RMS error for each quantity, and the fraction of disparities whose absolute error exceeds one pixel. All available independent measured reference pixels are evaluated, including occluded pixels; there is no fitted-value quality mask or alignment. RMS error is divided by the maximum-minus-minimum reference value. A separate maximum relative depth/disparity consistency error is evaluated over every output pixel, with tolerance 3.5762786865234375e-7. No implicit crop, rescaling, translation or depth convention conversion is performed. Independent structured-light reference measurements have finite accuracy and incomplete availability.

## Data Description

The public arrays are decoded RGB values from the real, already rectified author images. Their native order and pixel values are retained.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| left_image | (1,496,718,3) | uint8 | Left RGB photograph; axes batch,row,column,channel, digital intensity count in 0..255. |
| right_image | (1,496,718,3) | uint8 | Right RGB photograph; same axes and intensity-count units. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Acquisition | Middlebury 2014 Adirondack, MiddEval3 Q distribution | none | Complete author-provided Q pair, not an agent-created resize. |
| Image width,height | 718,496 | pixel | Native distributed grid. |
| Focal length f | 1038.018 | pixel | Equal horizontal/vertical focal lengths of rectified cameras. |
| Left principal point cL,cy | 322.037,243.393 | pixel | Supplied rectified left camera. |
| Right principal point cR,cy | 375.308,243.393 | pixel | Supplied rectified right camera. |
| Rectified origin offset a | 53.271 | pixel | Difference of rectified principal-point abscissae; must not be omitted in triangulation. |
| Baseline B | 176.252 | mm | Right optical centre lies along positive left-camera X. |
| Image coordinates | u increases right; v increases down; integer array indices | pixel | First array element has u=0,v=0; no additional half-pixel offset. |
| Camera coordinates | X right; Y down; Z into scene | mm | Left-camera frame used by the projection equation. |
| Rectification | Supplied initial-calibration rectification and crop | none | Small vertical residuals can remain; no target-based correction is provided. |
| Photometric calibration | Native RGB counts; no calibrated response curve supplied | digital count | Exact radiometric equality is an approximation. |
| Noise | No independent sensor-noise variance supplied | digital count | Correspondence ambiguity and occlusion dominate this task. |
| Image boundary | No periodic extension; points outside an image are unobserved | none | Correspondence support ends at each acquired image edge. |

## Method Hints

A discrete Markov-field formulation combines a local correspondence cost with a spatial discontinuity penalty. Census comparisons of neighboring intensity ranks reduce sensitivity to monotone radiometric changes. A truncated label-difference penalty permits sharp jumps while preferring coherent surfaces. Directional min-plus aggregation provides an approximate inverse; averaging messages from two causal predecessors reduces the streaking associated with independent one-dimensional paths. Correct for repeated counting of the unary cost before selecting the best label. This approximation is not an exact global minimum certificate. Calibrated triangulation then converts disparity into optical-axis depth.

## References

- Facciolo, de Franchis and Meinhardt. *MGM: A Significantly More Global Matching for Stereovision*. BMVC 2015. https://doi.org/10.5244/C.29.90 .
- Scharstein et al. *High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth*. GCPR 2014. https://www.cs.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf .
- Middlebury Stereo Evaluation v3, real Adirondack Q image acquisition and camera calibration. The provider permits use/publication with the GCPR2014 citation; original permission terms and required attribution are retained by the maintainers.
