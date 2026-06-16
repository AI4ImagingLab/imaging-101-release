# Differentiable Refractive Deflectometry

> Recover the surface curvatures and thickness of a refractive lens from phase-shifted fringe images captured by two calibrated cameras, made challenging by the nonlinear ray–surface intersection geometry and the high-dimensional pixel-level measurement space.

> Domain: industrial/commercial | Keywords: deflectometry, differentiable rendering, optical metrology | Difficulty: Hard

## Background

Deflectometry is an optical metrology technique that infers the shape of refractive or reflective surfaces by analysing how they distort known display patterns. A display screen shows sinusoidal fringe patterns, and one or more calibrated cameras observe those fringes through the optical element under test. Phase-shifting analysis — acquiring multiple images at known phase offsets — allows each camera pixel to be mapped to a precise location on the display, effectively encoding the ray deflection caused by the element.

In the refractive case the optical path traverses two curved surfaces, refracting twice according to Snell's law. Each ray's exit point on the display depends nonlinearly on the surface curvatures, thickness, position, and tilt of the lens. Because the measurement is dense (every camera pixel provides a constraint), the resulting inverse problem is heavily over-determined in data but nonlinear in parameters, requiring an efficient gradient-based solver.

## Problem Description

**Forward model.** A ray originating from camera pixel $\mathbf{u}$ passes through a lens parameterised by curvatures $(c_0, c_1)$, thickness $d$, origin $\mathbf{o}$, and tilt angles $(\theta_x, \theta_y)$. The ray refracts at each surface via Snell's law and intersects the display screen at point $\mathbf{p}(\mathbf{u}; \boldsymbol{\theta})$, where $\boldsymbol{\theta} = (c_0, c_1, d, \mathbf{o}, \theta_x, \theta_y)$ collects all unknowns. Finding each ray–surface intersection requires solving an implicit equation (e.g. via Newton's method on the sag function of an aspheric surface).

**Measurement equation.** Phase-shifting fringe analysis converts the raw images into a measured screen intersection map $\hat{\mathbf{p}}(\mathbf{u})$ for each camera. The inverse problem minimises the residual

$$\min_{\boldsymbol{\theta}} \sum_{\mathbf{u} \in \mathcal{V}} \left\| \mathbf{p}(\mathbf{u}; \boldsymbol{\theta}) - \hat{\mathbf{p}}(\mathbf{u}) \right\|^2$$

over all valid pixels $\mathcal{V}$ across both cameras.

**Difficulty.** The forward map is nonlinear due to the implicit ray–surface intersection and Snell's law refraction. The Jacobian $\partial \mathbf{p} / \partial \boldsymbol{\theta}$ has dimensions $|\mathcal{V}| \times |\boldsymbol{\theta}|$ (hundreds of thousands of rows, ~10 columns), so an analytic derivation is impractical — automatic differentiation through the ray tracer is essential.

**Input/output.** Input: phase-shifted fringe images from two cameras (with and without the lens). Output: recovered lens parameters $(c_0, c_1, d, \mathbf{o}, \theta_x, \theta_y)$.

## Data Description

### `data/raw_data.npz`

Phase-shifted fringe images from two cameras observing a ThorLabs LE1234-A biconvex lens.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `imgs` | `(1, 3, 8, 2, 2048, 2048)` | `float32` | Measurement images with lens in place; axes: (batch, sinusoid_periods, phase_shifts, cameras, H, W). 3 periods (T = 70, 100, 110 px), 8 shifts per period (4 for x-direction + 4 for y-direction), 2 cameras. Intensity in arbitrary units. |
| `refs` | `(1, 3, 8, 2, 2048, 2048)` | `float32` | Reference images without lens (same axes as `imgs`). Used to extract differential phase. Intensity in arbitrary units. |

### `data/ground_truth.npz`

Manufacturer-specified lens parameters for the ThorLabs LE1234-A.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `surface_0_roc_mm` | `(1,)` | `float64` | Radius of curvature of the first lens surface in mm |
| `surface_1_roc_mm` | `(1,)` | `float64` | Radius of curvature of the second lens surface in mm |
| `thickness_mm` | `(1,)` | `float64` | Centre thickness of the lens in mm |

### Other data files

- `data/calibration/`: Camera intrinsics/extrinsics, rotation calibration, gamma-correction LUTs (MATLAB `.mat` files).
- `data/lenses/ThorLabs/LE1234-A.txt`: Zemax-format lens prescription (surface curvatures, thicknesses, materials).
- `data/sinusoids/`: Display fringe pattern images at period T = 70 px.
- `data/meta_data.json`: Imaging parameters (sensor size, display pixel pitch, sinusoid periods) and calibration settings. Does not contain solver parameters.

## Method Hints

Use a **differentiable ray tracer** implemented in an autodiff framework (e.g. PyTorch). Model each lens surface as an aspheric sag function, find ray–surface intersections with Newton's method, and apply Snell's law for refraction. The entire camera-to-screen path must be differentiable so that the Jacobian of the residual with respect to the lens parameters can be obtained via backpropagation.

Solve the nonlinear least-squares problem with a **Levenberg–Marquardt** (LM) algorithm, which interpolates between gradient descent and Gauss–Newton and handles the near-singular Jacobian structure well. The autodiff Jacobian replaces manual or finite-difference derivatives.

Preprocessing uses **four-step phase shifting** to extract wrapped phase, followed by temporal phase unwrapping across multiple fringe periods to obtain absolute screen displacement in mm.

## References

Wang, C., Chen, N., & Heidrich, W. (2021). Towards self-calibrated lens metrology by differentiable refractive deflectometry. *Optics Express*, 29(19), 30284–30295.

- [Project page](https://vccimaging.org/Publications/Wang2021DiffDeflectometry/)
- [Paper PDF](https://vccimaging.org/Publications/Wang2021DiffDeflectometry/Wang2021DiffDeflectometry.pdf)
- [Original code](https://github.com/vccimaging/DiffDeflectometry)
