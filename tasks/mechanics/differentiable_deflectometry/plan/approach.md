# Approach

## Problem Statement

Recover the surface curvatures $(c_0, c_1)$, centre thickness $d$, 3D origin, and tilt angles $(\theta_x, \theta_y)$ of a biconvex lens from dense phase-shifted fringe images captured by two calibrated cameras in a deflectometry setup.

## Mathematical Formulation

### Forward Model

A ray originating from camera pixel $\mathbf{u}$ is projected into 3D via the calibrated camera model, then propagated through the lens:

1. **Ray–surface intersection.** Each lens surface is parameterised by an aspheric sag function $z(r) = c\,r^2 / (1 + \sqrt{1 - c^2 r^2})$, where $c = 1/R$ is the curvature and $r$ is the radial distance from the optical axis. The intersection of a ray with this implicit surface is found by Newton's method on $f(t) = z_{\text{ray}}(t) - z_{\text{sag}}(r(t))$.

2. **Refraction.** At each surface the ray refracts according to the vector form of Snell's law: $\mathbf{d}_t = (n_1/n_2)\,\mathbf{d}_i + \left[(n_1/n_2)\cos\theta_i - \cos\theta_t\right]\mathbf{n}$, where $\mathbf{n}$ is the surface normal, $n_1, n_2$ are the refractive indices on either side, and $\theta_i, \theta_t$ are the incidence and transmission angles.

3. **Screen intersection.** After two refractions the exit ray intersects the display screen at point $\mathbf{p}(\mathbf{u};\boldsymbol{\theta})$, where $\boldsymbol{\theta} = (c_0, c_1, d, \mathbf{o}, \theta_x, \theta_y)$ collects all 8 unknowns.

The entire path — projection, intersection, refraction, screen hit — is implemented in PyTorch so that $\partial \mathbf{p}/\partial \boldsymbol{\theta}$ is available via autograd.

### Measurement Model

Four-step phase shifting converts raw fringe images into measured screen displacements. For each of 3 sinusoid periods $T \in \{70, 100, 110\}$ px, 8 images are captured (4 x-shifts, 4 y-shifts) per camera, both with and without the lens. The wrapped phase is:

$$\psi = \arctan\!\left(\frac{I_3 - I_1}{I_0 - I_2}\right)$$

where $I_0, \ldots, I_3$ are the four phase-shifted intensity images. Subtracting the reference phase (no lens) and unwrapping gives the displacement in pixels, which is converted to mm via the display pixel pitch (0.115 mm/px). Averaging across periods reduces noise.

### Inverse Problem

The objective minimises the sum-of-squares residual over all valid pixels $\mathcal{V}$ across both cameras:

$$\min_{\boldsymbol{\theta}} \sum_{\mathbf{u} \in \mathcal{V}} \left\|\mathbf{p}(\mathbf{u};\boldsymbol{\theta}) - \hat{\mathbf{p}}(\mathbf{u})\right\|^2$$

where $\hat{\mathbf{p}}$ are the measured intersection points. The problem has $|\boldsymbol{\theta}| = 8$ unknowns and $|\mathcal{V}| \sim 10^5$ constraints (heavily over-determined), but is nonlinear due to the implicit ray–surface intersection and Snell's law.

## Solution Strategy

### Step 1: Preprocessing

Load calibration data (camera intrinsics/extrinsics from MATLAB `.mat` files, display geometry, gamma LUTs). Centre-crop the 2048x2048 raw images to the configured 768x768 filmsize. Estimate the lens mount origin by triangulating rotation-calibration points across the two cameras.

Apply four-step phase shifting to extract wrapped phase maps, unwrap using `skimage.restoration.unwrap_phase`, remove the DC offset by searching over integer-period shifts, and convert to mm. Subtract reference ray-traced intersection points (scene without lens) to obtain the measured displacement map $\hat{\mathbf{p}}$.

Threshold on fringe amplitude and modulation to produce a binary validity mask $\mathcal{V}$ per camera (select the largest connected component near the image centre).

### Step 2: Forward Model Construction

Build a `Scene` object containing two `Camera` instances, a `Screen`, and a `Lensgroup` with two aspheric `Surface` objects. The `Lensgroup` is parameterised by $(c_0, c_1, d, \mathbf{o}, \theta_x, \theta_y)$; calling `scene.trace(with_element=True)` propagates rays through the full optical path and returns predicted screen intersection points.

Newton's method for ray–surface intersection uses implicit differentiation: rather than differentiating through the Newton iterations, the converged intersection is treated as an implicit function of the parameters, giving correct gradients at lower cost.

### Step 3: Optimization

Solve the nonlinear least-squares problem with Levenberg–Marquardt (LM):

1. Compute the $M \times 8$ Jacobian $\mathbf{J}$ via column-wise forward-mode autodiff (perturb each of the 8 parameters independently).
2. Form the normal equations: $(\mathbf{J}^T \mathbf{J} + \lambda\,\mathbf{R})\,\boldsymbol{\delta} = \mathbf{J}^T \mathbf{r}$, where $\mathbf{r}$ is the residual vector.
3. Accept the step if the loss decreases (reduce $\lambda$ by factor $\mu$); reject and increase $\lambda$ otherwise.
4. Terminate when the relative loss change or parameter update falls below $10^{-8}$, or after the iteration limit.

Key choices:
- **Regularisation** $\mathbf{R} = \text{diag}(\mathbf{J}^T \mathbf{J})$ (Marquardt's original form) normalises the vastly different parameter scales (curvatures ~$10^{-2}$ mm$^{-1}$ vs. origin ~$10^1$ mm).
- **Initial curvatures** $c_0 = c_1 = 0.001$ (not zero) to avoid a degenerate flat-surface Jacobian.
- **Damping** $\lambda_0 = 0.1$, $\mu = 2$.
- **`lstsq` fallback** when the normal equation matrix is singular.

### Step 4: Evaluation

Compare recovered parameters against the manufacturer specification (ThorLabs LE1234-A):

| Parameter | Ground Truth | Metric |
|-----------|-------------|--------|
| Surface 0 ROC | $-82.23$ mm | relative error |
| Surface 1 ROC | $-32.14$ mm | relative error |
| Thickness | $3.59$ mm | relative error |

Also report the mean displacement error (RMS of residual in $\mu$m) and the final loss value.

### Step 5: Visualization

Generate:
- **Image comparisons**: measured fringe pattern vs. rendered pattern at initial and optimised parameters (per camera).
- **Spot diagrams**: measured vs. modelled screen intersection points at initial and optimised parameters.
- **Loss curve**: LM loss vs. iteration number showing convergence.

## Expected Results

| Parameter | Recovered | Relative Error |
|-----------|-----------|----------------|
| Surface 0 ROC | ~$-76.5$ mm | ~7% |
| Surface 1 ROC | ~$-31.7$ mm | ~1.5% |
| Thickness | ~$3.25$ mm | ~9.5% |

Mean displacement error: ~43 $\mu$m. The LM solver converges in ~21 iterations. The relatively larger errors on surface 0 ROC and thickness reflect the inherent sensitivity structure of the deflectometry setup — the second surface, being closer to the cameras and having higher curvature, provides stronger constraints.
