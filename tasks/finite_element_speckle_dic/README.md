# Finite Element Speckle Displacement Imaging

> Reconstruct a continuous solid-surface displacement field from two speckle images despite subpixel motion and spatially varying deformation.
> Domain: mechanics | Keywords: deformation imaging, finite element inversion | Difficulty: Medium

## Background

Optical deformation imaging follows an irregular surface pattern as a solid changes shape. Under stable illumination, the pattern acts as a material label: its brightness approximately follows the same material point between exposures. A single camera measures the apparent in-plane motion of the visible surface.

This provides spatially resolved information for mechanical testing, locating strain concentrations and assessing deformation of engineered or biological materials. Its usefulness depends on textured surfaces, sufficient optical contrast and small out-of-plane motion. Inferring motion below the pixel spacing is difficult because an image records intensities rather than material coordinates, and noise or illumination changes violate ideal brightness conservation.

## Problem Description

Recover the two-component displacement at every supplied mesh node from the reference and deformed speckle images. The geometric observation equation is

```math
I_1\!\left(\boldsymbol{X}+\boldsymbol{u}(\boldsymbol{X})\right)
= I_0(\boldsymbol{X})+\epsilon(\boldsymbol{X}),\qquad
\boldsymbol{u}(\boldsymbol{X})=\sum_{a=1}^{4}N_a(\xi,\eta)\boldsymbol{d}_a.
```

Here `I0` and `I1` are the measured reference and deformed grayscale intensities, respectively; the subscripted symbols in the formula denote the same images. The vector `X` is a reference image coordinate in pixels, `u` is the apparent in-plane displacement in pixels, and epsilon describes photometric mismatch and measurement error. Each element has four nodes with unknown displacement vectors `d`. The local coordinates xi and eta lie in [0,1]; its bilinear shape functions are `(1-xi)(1-eta)`, `xi(1-eta)`, `xi*eta`, and `(1-xi)*eta`, ordered as the supplied connectivity. Reference coordinates are interpolated from the known nodes using those same functions.

The two images are repository-supplied speckle observations from a digital image correlation example, cropped identically without resampling. No independent displacement measurement or physical length calibration accompanies this bound pair. Texture gradients constrain different directions unequally, interpolation is required between pixels, and the image-warp inverse problem is nonlinear.

Save `output/reconstruction.npz`, relative to the task workspace, with key `reconstruction`: a real float32 or float64 array of shape `(1, 15, 2)` and finite values. Axis order is batch, supplied node index, component `(ux, uy)`; displacement units are pixels. Positive ux points right and positive uy points up. Values are increments relative to the supplied reference nodes, not absolute coordinates. Preserve image pixel scale; no translation, rotation, intensity, sign or magnitude alignment is applied. Evaluation checks the RMS displacement error across all nodes and both components against a validated standard-method result. Cosine similarity and dynamic-range normalized RMS error are also reported as diagnostics.

## Data Description

All visible arrays are in `data/raw_data.npz`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reference_image | (1, 420, 260) | float64 | Reference grayscale intensity, original 8-bit digital counts represented exactly as float64; axes batch, y, x. |
| deformed_image | (1, 420, 260) | float64 | Deformed grayscale intensity in digital counts, with the same orientation and crop. |
| nodes | (1, 15, 2) | float64 | Reference node positions in pixels; last axis x right, y up. |
| elements | (1, 8, 4) | int64 | Dimensionless zero-based node indices, ordered lower-left, lower-right, upper-right, upper-left. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Image dimensions | 420 by 260 | pixel | Height by width of the bound crop. |
| Pixel pitch | 1 | pixel | Coordinate spacing; conversion to metres is unavailable. |
| Origin | (0, 0) | pixel | Centre of the lower-left stored pixel. |
| Axis convention | x right; y up | dimensionless | Stored row index increases upward. |
| Reference mesh bounds | x=50 to 210; y=50 to 370 | pixel | Common field of interest in both images. |
| Mesh topology | 2 columns by 4 rows | element | Structured Q4 mesh, 80 by 80 pixels per element. |
| Optical geometry | single-camera 2D projection | dimensionless | Only apparent in-plane displacements are observable. |
| Illumination convention | approximate brightness conservation | digital count | No brightness normalization is prescribed. |
| Boundary conditions | all mesh nodes free | dimensionless | No imposed zero displacement or known target motion. |
| Noise calibration | unavailable | digital count | No calibrated statistical noise model is supplied. |
| Exposure and frame interval | unavailable | second | Output is displacement, not velocity. |

## Method Hints

Finite-element digital image correlation couples adjacent image regions through shared nodal displacement variables. Bilinear shape functions make the displacement continuous across element edges while allowing strain to vary between elements. Minimize the summed squared intensity mismatch over reference pixels, using interpolated deformed intensities. A modified Newton method can keep the reference-image gradient and its normal matrix fixed while updating the deformed node positions. Strong speckle texture makes this local linearization informative. Absolute physical strain requires a justified camera and surface geometry model; apparent pixel displacements alone do not identify out-of-plane motion.

## References

- S. N. Olufsen, M. E. Andersen and E. Fagerholt, *muDIC: An open-source toolkit for digital image correlation*, SoftwareX 11 (2020), 100391. https://doi.org/10.1016/j.softx.2019.100391
- Repository-provided `ut_151.tif` and `ut_156.tif` example images distributed with the muDIC toolkit; MIT notice retained in this task package. Their specific specimen identity and acquisition timing are not independently established here.
