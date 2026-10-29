# Omnidirectional Pressure Imaging from Velocity-Derived Gradients

> Reconstruct a two-dimensional fluid pressure field from calibrated pressure-gradient images while recovering boundary values and resolving the missing pressure datum.
> Domain: mechanics | Keywords: fluid imaging, pressure reconstruction, variational inversion | Difficulty: Medium

## Background

Pressure transfers momentum through fluids and determines forces on immersed structures. Optical velocity measurements can estimate pressure gradients through the momentum equation, enabling pressure imaging without placing probes throughout the flow. Such measurements support studies of aerodynamic loading, vortices, and biological fluid transport.

Integrating a vector gradient into a scalar image requires consistency across many possible paths. Discretization and measurement errors make those paths disagree. The boundary provides fewer neighboring observations, and gradients contain no information about an additive pressure constant.

## Problem Description

Recover the pressure image from a simulated planar vortex acquisition on the full supplied Cartesian grid. The observed two-component gradient is normalized using known pressure and length scales:

```math
\mathbf{g}(\xi,\eta)=\nabla_{\xi,\eta}p^*(\xi,\eta)+\boldsymbol{\epsilon}(\xi,\eta),\qquad
p^*=p/P_0,\quad (\xi,\eta)=(x/L_0,y/L_0).
```

Here physical pressure $p$ is in Pa, physical coordinates $x,y$ are in m, $P_0$ and $L_0$ are the stated scales, and $\boldsymbol{\epsilon}$ describes deterministic finite-difference and float32 storage error. No independent random noise is added. The gradient was estimated from the planar fluid momentum balance, using convective acceleration calculated from a sampled velocity field. Reconstruction must determine boundary pressure along with interior values; no Dirichlet boundary pressure is supplied.

Save `output/reconstruction.npz`, relative to the task workspace, with exactly one array: `reconstruction`, shape `(1, 1001, 1001)`, real `float64`, in Pa. Axis order is `(batch, y, x)`, with both spatial coordinates increasing. The pressure datum at zero-based spatial index `(1,1)` must equal zero Pa, within $10^{-12}$ Pa. No further translation, rescaling, cropping, or gauge adjustment is applied during evaluation. Evaluation uses cosine NCC and RMS error divided by the reference pressure dynamic range over the entire image. All values must be finite.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| pressure_gradient | (1, 2, 1001, 1001) | float32 | Calibrated normalized gradient; component order x,y; dimensionless, equivalent to physical gradient divided by $P_0/L_0$ |
| coordinate | (1, 1001) | float32 | Common normalized x and y coordinates; dimensionless, physical position is value multiplied by $L_0$ |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Grid points per axis | 1001 | pixel | Full native Cartesian sampling |
| Coordinate domain | -3 to 3 | dimensionless | Increasing normalized coordinate in each direction |
| Nominal coordinate spacing | 0.006 | dimensionless | Stored coordinates are float32; numerical integration uses the first stored difference |
| Length scale $L_0$ | 0.001 | m | Converts normalized coordinates to physical positions |
| Pressure scale $P_0$ | 0.001 | Pa | Converts reconstructed normalized pressure to Pa |
| Density | 1000 | kg/m³ | Fluid density used in velocity-derived gradients |
| Kinematic viscosity | 0.000001 | m²/s | Fluid viscosity in the acquisition model |
| Observation time | 0.3 | s | Instantaneous planar observation |
| Random noise | none | dimensionless | Retained errors arise from differentiation and float32 storage |
| Boundary | no supplied pressure values | Pa | Missing exterior neighbors carry no measurements |
| Pressure datum | zero at pixel (1,1) | Pa | Zero-based spatial index, after reconstruction |

## Method Hints

A boundary-aware gradient-integration formulation can combine trapezoidal edge pressure increments into a weighted graph Laplacian. An edge joining adjacent pixels uses their average measured gradient multiplied by the coordinate separation. Omnidirectional weights account for the available neighboring directions. The constant pressure mode is unobservable; apply the declared datum after inversion. A conjugate-gradient solve can use matrix products without constructing a dense matrix.

## References

Fernando Zigunov and John Charonko. *One-shot omnidirectional pressure integration through matrix inversion*. arXiv:2402.09988v1, 2024. [Paper](https://arxiv.org/abs/2402.09988v1).
