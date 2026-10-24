# Optical Gas Density and Uncertainty Mapping

> Reconstruct a complete gas-density field and its propagated uncertainty from background-pattern displacements whose spatially varying noise makes gradient integration unreliable.
> Domain: mechanics | Keywords: schlieren imaging, density reconstruction, uncertainty propagation | Difficulty: Hard

## Background

A gas changes the direction of a light ray when its refractive index varies along the ray. Viewing a dot pattern through such a flow makes the dots appear displaced. Comparing the distorted and reference patterns therefore provides information about refractive-index gradients without inserting a probe into the flow.

For a gas of known composition, refractive index is related to density by the Gladstone–Dale relation. Background-oriented schlieren measurements can reveal heated gas, shock waves and compressible mixing. Converting a displacement field into density requires spatial integration and boundary information. Uncertainty varies across the image and propagates through that integration, so a local displacement error can affect density elsewhere.

## Problem Description

The observations are the complete32×32 displacement field from an existing author simulation, together with observation-derived displacement uncertainty. They have already undergone correlation processing. The required target is depth-averaged volume density, under a thin-slab optical approximation, and its conditional standard uncertainty.

For signed sensor displacement components U,V in pixels, the forward relations are

```math
U=\frac{M Z K L}{n_0 p}\frac{\partial\rho}{\partial x}+\epsilon_x,
\qquad
V=\frac{M Z K L}{n_0 p}\frac{\partial\rho}{\partial y}+\epsilon_y.
```

Hereρ is density in kg/m³; x,y are transverse physical coordinates; M is dot-pattern magnification; Z is the pattern-to-flow distance; L is flow thickness; K is the Gladstone–Dale constant; n0 is ambient refractive index; p is sensor pixel pitch; andεx,εy are displacement errors. The slab approximation replaces the viewing-direction integral of density gradient by thickness times its depth average. Spatial integration has an additive ambiguity removed by the supplied ambient perimeter conditions. No independent density truth is available, so reference agreement measures the specified conditional inverse, not absolute physical accuracy.

Save `output/reconstruction.npz` relative to the task workspace with both required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,32,32) | float64 | Depth-averaged density in kg/m³; axes(batch,y,x), absolute density including ambient offset. |
| density_sigma | (1,32,32) | float64 | Nonnegative conditional standard uncertainty in kg/m³; same axes and physical scale. |

All1024 samples of both quantities are evaluated using cosine similarity and RMS error divided by that quantity's reference dynamic range. A zero estimate has cosine0. Evaluation also checks the complete measured gradient-edge and known boundary residual, standardized by their propagated uncertainties. There is no crop, fitted alignment, mean removal or rescaling. The very small boundary uncertainty requires float64 values consistent with the known perimeter condition.

## Data Description

`data/raw_data.npz` contains only processed observations and their legitimate correlation-plane uncertainty estimates.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| positions | (1,32,32,2) | float64 | Sensor-plane grid positions in pixels; final component order(x,y). |
| displacement | (1,32,32,2) | float64 | Signed pattern displacement in pixels; final component order(U,V), positive displacement corresponds to positive density gradient. |
| displacement_sigma | (1,32,32,2) | float64 | Positive MC random standard displacement uncertainty in pixels, component order(U,V); estimated correlation-peak bias variance has been removed. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Native support | 32×32, all nodes | samples | No selected quality mask. Rows increase y; columns increase x. |
| Ambient density | 1.225 | kg/m³ | Known constant density on the complete perimeter, including corners once. |
| Boundary standard uncertainty | 1e-10 | kg/m³ | Conditional independent uncertainty of each perimeter value. |
| Pixel pitch p | 1e-5 | m/pixel | Sensor pitch. |
| Focal length | .105 | m | Known lens focal length. |
| Object distance | 1 | m | Lens-to-pattern distance. |
| Magnification M | .105/(1−.105) | dimensionless | Dot-pattern magnification. |
| Magnification uncertainty | .1 | dimensionless | Source conditional first-order standard uncertainty. |
| Flow distance Z | .25 | m | Pattern to density-gradient midplane. |
| Distance standard uncertainty | .001 | m | Conditional uncertainty of Z. |
| Thickness L | .01 | m | Known slab thickness. |
| Gladstone–Dale constant K | .000225 | m³/kg | Fixed gas optical coefficient. |
| Ambient refractive index n0 | 1+K×1.225 | dimensionless | Fixed ambient refractive index. |
| Gradient-plane magnification Mg | M/.75 | dimensionless | Grid conversion factor. |
| Coordinate origin | (256,256) | pixel | x=(X−256)p/Mg, y=(Y−256)p/Mg, where X,Y are supplied positions. |

The selected metadata follows the author's released sample program; it is not claimed identical to every paper simulation. The uncertainty model is a conditional approximation: it treats the supplied displacement errors and calibration contributions at distinct nodes as independent. A common calibration error would instead create spatial correlations, and the large magnification uncertainty makes first-order propagation an approximation.

## Method Hints

Use uncertainty-weighted least-squares integration and propagate covariance through its linear solution. Let c=n0p/(MZKL), gradients gx=cU,gy=cV, and displacement standard uncertainties sU,sV. The gradient variance for the x component is

```math
s_{g_x}^2=(c s_U)^2+
\left(\frac{c U}{M}s_M\right)^2+
\left(\frac{c U}{Z}s_Z\right)^2,
```

where sM and sZ are the supplied magnification and distance uncertainties; the y component replaces U,sU by V,sV. This expression uses derivatives of the same optical law as the mean gradient.

For the selected conditional inverse, stack forward density differences on all horizontal and vertical edges, followed by perimeter density divided by horizontal grid spacing, into Dρ. Form b=Bg+Cd, where g stacks nodal x and y gradients, B averages the two endpoint gradients at each edge, and C inserts the perimeter density observations d with the same spacing factor. Boundary rows of B and edge rows of C vanish. Let Q be the covariance of b obtained from these same mappings and the stated independent input variances; set W to the diagonal matrix of reciprocal diagonal entries of Q. The conditional objective is

```math
\frac12(D\rho-b)^{\mathsf T}W(D\rho-b).
```

The density covariance follows by applying the linear solution operator to Q on both sides. Retain the off-diagonal covariance induced by shared endpoints during this propagation, even though W uses only marginal variances. A centered-divergence Poisson integration provides a useful comparison for how weighting changes noise propagation.

## References

Rajendran et al., “Uncertainty-based weighted least squares density integration for background-oriented schlieren,” Experiments in Fluids61,239(2020). [Article](https://doi.org/10.1007/s00348-020-03071-w).

Rajendran et al., “Uncertainty quantification in density estimation from background-oriented Schlieren measurements,” Measurement Science and Technology31,054002(2020). [Article](https://doi.org/10.1088/1361-6501/ab60c8).

Bhattacharya et al., “Particle image velocimetry uncertainty quantification using moment of correlation plane,” Measurement Science and Technology29,115301(2018). [Article](https://doi.org/10.1088/1361-6501/aadfb4).
