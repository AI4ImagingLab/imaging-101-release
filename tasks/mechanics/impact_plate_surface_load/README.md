# Impact Load Imaging from Optical Plate Deflection

> Reconstruct the transient spatial pressure applied to a real steel plate from optically measured deflection fields despite noise amplification by spatial and temporal differentiation.
> Domain: mechanics | Keywords: inverse dynamics, optical metrology, load imaging | Difficulty: Hard

## Background

A pressure field bends a thin elastic plate and accelerates its mass. Optical deflectometry measures small changes in surface slope by observing a reflected grid; integrating these measurements gives a spatially resolved deflection recording. The surface can therefore act as an extended mechanical sensor without an array of pressure transducers disturbing the load.

Recovering a load distribution is useful in impact mechanics, blast loading and fluid-structure interaction. The deformation spreads the effect of a localized force over a much larger area, and direct differentiation strongly amplifies measurement noise. Elastic bending and inertial effects must both be accounted for during a rapid transient, even when displacements remain small.

## Problem Description

The observations are complete deflection fields measured during one real impact-hammer experiment on a clamped, mirror-finished steel plate. They are displacement estimates derived from the author's original optical grid recordings, with one independently justified boundary-index correction in slope integration, independently of any pressure reconstruction or force target. They are not direct displacement ground truth. Recover the signed pressure field acting on the plate. Under linear isotropic Kirchhoff-Love bending, the strong-form relation is

```math
D\left(\frac{\partial^4 w}{\partial x^4}+2\frac{\partial^4 w}{\partial x^2\partial y^2}+\frac{\partial^4 w}{\partial y^4}\right)+\rho h\frac{\partial^2w}{\partial t^2}=p,
\qquad D=\frac{E h^3}{12(1-\nu^2)}.
```

Here $w(x,y,t)$ is transverse deflection in meters, $p(x,y,t)$ pressure in pascals, $E$ Young's modulus in pascals, $\nu$ Poisson ratio, $h$ plate thickness in meters, $\rho$ density in kilograms per cubic meter, and $D$ bending stiffness in newton meter. Coordinates $x,y$ follow increasing observed row and column, respectively; $t$ is time in seconds. Positive pressure follows the sign of the inertial term for positive deflection. The unknown is a spatial load, not a material stiffness. High-order derivatives and measurement noise make pointwise recovery unstable, while a finite recording and unmeasured exterior region limit boundary information.

The declared output is an effective pressure estimated at every valid moving square measurement patch. It uses the fixed temporal bandwidth and spatial averaging convention described here so that its physical resolution and coordinates are unambiguous: first take a Gaussian temporal average of deflection with standard deviation 80 microseconds (six recorded frames), nearest-value extension and truncation at four standard deviations. Then compute centered interior and first-order one-sided boundary differences for slopes, curvatures and acceleration. The pressure's virtual averaging patch covers 20 by 20 measured samples (physical side length $L=20\Delta$), with separable unit-center Hermite weight $v(x,y)=H(x/L)H(y/L)$ sampled at patch pixel midpoints. Here $\Delta$ is measured-field sample spacing. For $0\leq s\leq1/2$, define $u=2s$ and $H(s)=3u^2-2u^3$; for $1/2<s\leq1$, define $u=2s-1$ and $H(s)=1-3u^2+2u^3$. Both the virtual displacement and its first derivative vanish on the patch perimeter. The weak-work expression in Method Hints defines this effective pressure. These fixed output-resolution conventions should not be interpreted as recovery of an unsmoothed delta-like contact load.

Save **`output/reconstruction.npz`** relative to your task workspace, with required key `reconstruction`, a finite real float64 array of shape **`(1, 250, 60, 78)`** in **Pa**, ordered **batch, time, pressure row, pressure column**. All 250 frames are required. A pressure index $(r,c)$ is located at $((r+9.5)\Delta,(c+9.5)\Delta)$ relative to the origin of the measured deflection field. Do not rescale, take magnitudes, suppress negative estimates or shift the pressure map. Time is $t_i=i/75000$ for zero-based frame index $i$; this is the published example's clock, whose origin is the first processed frame.

Pressure is compared using flattened cosine NCC without mean subtraction and RMS error normalized by reference dynamic range. A separate physical check compares the force obtained by summing pressure rows 20 through 49 and columns 20 through 49 and multiplying by $\Delta^2$ with an independent impact-hammer load-cell recording on the stated clock. Its error is RMS divided by the measured force dynamic range; no time or amplitude fit is applied. This fixed comparison subregion follows the supplied experiment convention, and the entire pressure field remains numerically evaluated. There is no pixelwise experimental pressure truth, so method agreement and the independent integrated-force check have different meanings.

## Data Description

The public observation is the complete derived optical deflection recording: 250 frames at the author's original measured-field resolution, without any pressure-based calibration. All stored values retain float64 precision.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| deflection | (1, 250, 79, 97) | float64 | Measured transverse deflection in meter; axes batch, processed time frame, row-coordinate x, column-coordinate y. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Specimen | Mirror-finished steel plate, nominal 0.3 by 0.3 | m | Outer plate dimensions; measured field covers a subset |
| Young's modulus | 190000000000 | Pa | Selected author implementation's calibrated elastic constant |
| Poisson ratio | 0.3 | dimensionless | Isotropic plate model |
| Material density | 7934.0 | kg/m^3 | Selected author implementation value |
| Plate thickness | 0.00495 | m | Selected author implementation value |
| Field spacing $\Delta$ | 0.0030065919701586503 | m | Uniform spacing along both measured-field axes, preserving source numerical calibration |
| Sampling rate | 75000.0 | Hz | Optical recording rate |
| Exposure | 0.000005 | s | Original high-speed camera exposure |
| Clock origin | First processed frame | s | Frame 0 corresponds to raw frame index 30; local clock resets to zero as in supplied example |
| Processed frames | 250 | frame | All original frames after the 30-frame reference interval |
| Camera | Phantom v2511; Sigma 50-100mm f/1.8 lens | dimensionless | Original optical recording hardware |
| Reference interval | Raw frame indices 0 through 29 | frame | Independently observed unloaded-grid reference |
| Original grid pitch | 0.00588433 | m | Physical printed-grid spacing |
| Original grid pitch in image | 5.0 | pixel | Carrier period in original optical image |
| Selected mirror-grid distance | 1.385 | m | Distance used in author code to derive the bound measured deflection |
| Optical pixel scale factor | 1.02189781 | dimensionless | Source mirror-plane calibration factor |
| Boundary | Plate edges clamped; finite measured field | dimensionless | Load inverse uses only valid internal virtual patches and does not infer exterior loads |
| Measured-field displacement gauge | Mean of two bottom 5 by 5 corner areas set to zero | m | Source optical integration convention, independently of the unknown load |
| Noise | Optical measurement noise, covariance not calibrated | m | Preserve measured fields; no synthetic noise added |
| Coordinates | Row x increasing down, column y increasing right | m | Source kinematic derivative convention; no transpose or half-pixel shift is fitted |

The dataset's descriptive text rounds density to 7900 kg/m^3 and lists a mirror-grid distance of 1.37 m. This task consistently uses the explicitly selected author implementation values above, which were used when deriving its bound observations. The difference is documented rather than silently mixing calibrations. The measured optical deflection is an observation estimate, with finite spatial resolution and integration uncertainty.

## Method Hints

Use the virtual-fields method to move two spatial derivatives onto a smooth, compact virtual displacement. Define measured curvatures $\kappa_{xx}=-\partial^2_x w$, $\kappa_{yy}=-\partial^2_y w$ and $\kappa_{xy}=-\partial_x\partial_y w$, and virtual curvatures by the same negative derivatives of $v$. Let $a=\partial_t^2 w$ and $D_{12}=\nu D$. The effective patch pressure follows the discrete virtual-work identity

```math
\bar p=\frac{
D\sum\left(\kappa_{xx}\kappa^v_{xx}+\kappa_{yy}\kappa^v_{yy}+2\kappa_{xy}\kappa^v_{xy}\right)
+D_{12}\sum\left(\kappa_{xx}\kappa^v_{yy}+\kappa_{yy}\kappa^v_{xx}-2\kappa_{xy}\kappa^v_{xy}\right)
+\rho h\sum a v}{\sum v}.
```

Superscript $v$ labels the virtual quantities, and each sum covers one measurement patch with uniform sampling; the common sample area cancels. This decomposition separates bending and inertia and avoids taking four noisy spatial derivatives directly. The compact Hermite virtual field eliminates unknown work at patch boundaries. The moving-patch estimate trades spatial resolution for noise sensitivity; numerical midpoint sampling also introduces finite-window quadrature error.

## References

- Olufsen, S. N., Kaufmann, R., Fagerholt, E., and Aune, V. *RECOLO: A Python package for the reconstruction of surface pressure loads from kinematic fields using the virtual fields method*. JOSS 7(71), 3980 (2022). [Paper](https://doi.org/10.21105/joss.03980).
- Kaufmann, R., Ganapathisubramani, B., and Pierron, F. *Full-Field Surface Pressure Reconstruction Using the Virtual Fields Method*. Experimental Mechanics 59, 1203-1221 (2019). [Method paper](https://doi.org/10.1007/s11340-019-00530-2).
- *Experimental data for force reconstruction using impact hammer and deflectometry*. [Data deposit](https://doi.org/10.18710/TKS7SI), CC0.
