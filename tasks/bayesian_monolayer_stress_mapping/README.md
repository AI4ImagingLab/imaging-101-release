# Bayesian Monolayer Stress Mapping

> Reconstruct the spatial internal stress tensor of a confined monolayer from noisy substrate traction forces, despite having more stress unknowns than force-balance equations.
> Domain: mechanics | Keywords: stress imaging, Bayesian inversion, continuum mechanics | Difficulty: Hard

## Background

A thin material sheet transmits internal forces while exerting tractions on its support. At negligible inertia, spatial derivatives of the internal stress balance these external forces. This relationship enables stress microscopy when direct internal stress measurements are difficult.

Applications include collective cell motion, tissue mechanics and active thin films. Measuring the two in-plane traction components does not by itself determine the three components of a symmetric two-dimensional stress tensor. Boundary information and statistical assumptions are needed to resolve the missing spatial information.

## Problem Description

Recover the height-integrated internal stress tensor of a confined square monolayer from the supplied author-existing noisy finite-element simulation. These are numerical observations, not live-cell experimental measurements. No tissue constitutive law or velocity measurement is supplied or required. Stress has units kPa micrometres, while traction has units kPa; no monolayer-height division is applied.

Let x increase along columns from left to right and y along increasing row indices in the upward Cartesian direction. The symmetric physical stress components are sigma(xx),sigma(yy),sigma(xy). Traction t is the force the monolayer exerts on its substrate. With negligible inertia,

```math
\frac{\partial \sigma_{xx}}{\partial x}+\frac{\partial \sigma_{xy}}{\partial y}=t_x,\qquad
\frac{\partial \sigma_{yx}}{\partial x}+\frac{\partial \sigma_{yy}}{\partial y}=t_y,\qquad
\sigma_{xy}=\sigma_{yx}.
```

On square cells of width l, use stresses on cell faces: sigma(xx) and sigma(yx) on vertical faces, sigma(yy) and sigma(xy) on horizontal faces. If the cell row is i and column j, the finite-volume measurement model is

```math
t_x(i,j)=\frac{\sigma_{xx}(i,j+1)-\sigma_{xx}(i,j)+
\sigma_{xy}(i+1,j)-\sigma_{xy}(i,j)}{l}+\epsilon_x(i,j),
```
```math
t_y(i,j)=\frac{\sigma_{yx}(i,j+1)-\sigma_{yx}(i,j)+
\sigma_{yy}(i+1,j)-\sigma_{yy}(i,j)}{l}+\epsilon_y(i,j).
```

The epsilon terms represent noise and finite-element-to-grid discrepancies. There are 5000 force observations and10200 auxiliary face stresses. Physical symmetry equates the interpolated shear components. The free boundary condition is sigma times the outward unit normal equals zero at the outer square boundary. Central normal stresses are averages of the two adjacent normal-face values; central shear is the average of the four surrounding shear-face values. Differentiation noise and divergence-free stress modes make inversion ill-posed.

Save output/reconstruction.npz, relative to the task workspace, with key reconstruction, real float64 array shape(1,3,50,50), axes batch,component,y,x. Component order is xx,yy,xy. Units are kPa micrometres for every component. Cell centers in each coordinate are1,3,...,99 micrometres. Positive stress is tensile and negative stress compressive. Output the symmetric physical shear once; no independent antisymmetric stress is requested. All three full spatial arrays are evaluated separately using cosine NCC and RMS error normalized by the dynamic range of that component. No cropping, centering, shifting, rescaling or alignment is permitted.

## Data Description

The single frame is reused unchanged from the authors' existing finite-element example of a viscous sheet driven by active force dipoles. White noise with5% relative amplitude was added by the original authors. The exact random realization and seed are not supplied. The quoted relative noise does not specify an exact per-component standard deviation for this bundled frame.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| traction | (1,2,50,50) | float64 | Noisy substrate traction in kPa; axes batch,component(tx,ty),y,x. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Grid rows and columns | 50,50 | 1 | One square spatial observation frame. |
| Spatial sample width l | 2 | micrometres | Isotropic finite-volume cell width. |
| Cell-center x and y range | 1 to99 in increments2 | micrometres | x columns increase rightward; increasing row index is positive Cartesian y, upward in plots. |
| Outer square boundaries | x=0,100 and y=0,100 | micrometres | Face coordinates extend one half cell outside center samples. |
| Free boundary condition | sigma times outward normal=0 | kPa micrometres | Both normal and tangential boundary tractions vanish. |
| Traction sign | monolayer force on substrate | 1 | The stress divergence equals traction, not its negative. |
| Stress physical scale | height-integrated two-dimensional stress | kPa micrometres | No assumed monolayer height or conversion to3D stress. |
| Original added relative noise | 5% | 1 | Author-provided simulated measurement noise; component covariance not specified. |
| Temporal samples | 1 | frame | Static spatial reconstruction. |

## Method Hints

Gaussian Bayesian inversion combines the discrete force-balance likelihood with a small-stress prior, equality of interpolated shear and free boundary constraints. If A is the face-stress divergence matrix, B is the prior precision, and lambda controls prior weight, the posterior mean minimizes l² times the force residual squared plus lambda times sigma-transpose-B-sigma. A general sparse linear solve evaluates this quadratic estimator directly.

Define B as the identity plus weighted squared shear-equality and boundary-selector operators. This lifts the divergence nullspace without imposing a viscous or elastic constitutive law. Interpolate the inferred auxiliary face fields to physical central stress before submission. Posterior uncertainty is conditional on the selected noise/prior model; interpolation requires covariance cross terms as well as face variances.

## References

- Nier, Jain, Lim, Ishihara, Ladoux and Marcq (2016), Inference of Internal Stress in a Cell Monolayer, Biophysical Journal110(7),1625–1635. [Paper](https://doi.org/10.1016/j.bpj.2016.03.002), [author manuscript](https://arxiv.org/abs/1603.03694). The distributed simulation and method are provided under GPL-3.0; attribution and license notices are preserved.
