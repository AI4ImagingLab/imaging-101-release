# Approach

## Problem Statement

Recover the 2D internal conductivity distribution of a circular domain from boundary voltage measurements collected by a 16-electrode array, using three complementary difference-imaging algorithms (Back-Projection, GREIT, Jacobian Dynamic) that trade off speed, resolution, and robustness.

## Mathematical Formulation

### Forward Model

The governing PDE is the generalized Laplace equation on a 2D unit-circle domain:

$$\nabla \cdot \bigl(\sigma(x,y)\,\nabla \phi(x,y)\bigr) = 0 \quad \text{in } \Omega$$

with Neumann boundary conditions: current density $j$ on electrode surfaces, zero flux elsewhere.

- **sigma** in R^{n_elems}: piecewise-constant conductivity per triangular element (n_elems ~ 686)
- **phi** in R^{n_nodes}: nodal electric potentials (n_nodes ~ 376)
- **K(sigma)** in R^{n_nodes x n_nodes}: global FEM stiffness matrix assembled from element contributions

The FEM discretization converts the PDE into a sparse linear system:

$$K(\sigma)\,\phi = f$$

where $f$ encodes the current injection pattern. For each excitation pair, solving this system yields nodal potentials; extracting voltages at measurement electrodes gives the forward map $\sigma \mapsto V$.

### Jacobian (Sensitivity Matrix)

The Jacobian $J = \partial V / \partial \sigma$ is computed via the adjoint method. For element $e$ with conductivity $\sigma_e$:

$$J_{m,e} = -\phi_\text{exc}^T \, K_e \, \phi_\text{meas}$$

where $\phi_\text{exc}$ and $\phi_\text{meas}$ are the potential fields due to the excitation and (virtual) measurement current patterns, and $K_e$ is the element stiffness sub-matrix. This yields an M x n_elems matrix (M = number of measurements, typically 192-208).

### Inverse Problem

All three methods solve a linearized difference-imaging problem. Given reference voltages $V_0$ (homogeneous background) and anomaly voltages $V_1$:

$$\delta V = V_1 - V_0 \approx J\,\delta\sigma$$

This is severely underdetermined (M << n_elems for BP/GREIT) or ill-conditioned, so regularization is essential. The general regularized solution is:

$$\delta\hat{\sigma} = (J^T J + \lambda R)^{-1} J^T \delta V$$

where $R$ is a regularization matrix and $\lambda$ controls regularization strength.

## Solution Strategy

### Step 1: Data Preprocessing

Load the mesh, protocol, and voltage data from `raw_data.npz`. Each experiment (BP, GREIT, JAC Dynamic) has its own mesh realization and excitation/measurement protocol stored with a prefix (`bp_`, `greit_`, `jac_dyn_`). Construct the FEM mesh data structure (nodes, elements, electrode positions, reference node) and measurement protocol (excitation pairs, measurement pattern, keep mask) for each experiment.

Voltage normalization: for all methods, compute the normalized voltage difference $\delta v = (V_1 - V_0) / |V_0|$ to reduce sensitivity to absolute voltage scale.

### Step 2: Forward Model Construction

For each experiment, assemble the FEM forward model:

1. Build the global stiffness matrix $K(\sigma_0)$ from the background conductivity.
2. Solve the forward problem for each excitation pattern to obtain reference potentials.
3. Compute the Jacobian matrix $J$ via the adjoint method.
4. Optionally normalize the Jacobian rows by the forward voltages (used by GREIT and JAC Dynamic).

The forward solver uses a sparse direct solve (scipy.sparse.linalg) with the reference node eliminated to ensure uniqueness.

### Step 3: Reconstruction

Three algorithms are applied, each suited to different imaging scenarios:

**3a. Back-Projection (BP)**

The simplest and fastest method. It maps voltage differences directly onto mesh nodes via the transpose of the sensitivity matrix $B$ (derived from the Jacobian):

$$\delta\hat{\sigma} = -B^T \, \delta v$$

where $B$ is the back-projection matrix computed from element-to-node mappings of the Jacobian rows. The sign-normalized voltage difference $\delta v = (V_1 - V_0) / \text{sign}(V_0)$ is used. The result is scaled by a factor of 192 (number of measurement channels) following the pyEIT convention.

BP produces blurry reconstructions with limited spatial resolution but runs in real time — useful as an initial estimate or for monitoring applications.

**3b. GREIT (Graz Consensus Reconstruction Algorithm for EIT)**

A linear, one-step algorithm that maps boundary voltages directly to a regular $n \times n$ pixel grid (default $n = 32$). The reconstruction matrix is built in three stages:

1. **Rasterize** the triangular mesh onto a regular grid using barycentric point-in-triangle testing, producing a binary mask for grid points inside the domain.
2. **Compute sigmoid weights** $W$ between element centres and grid points: $w_{e,g} = 1 / (1 + \exp(s \cdot (d_{eg}/d_\text{max} - r_0)))$, column-normalized. Parameters: $s = 20$ (decay rate), $r_0 = 0.1$ (radius ratio).
3. **Build the reconstruction matrix**: $H = W^T J^T (J J^T + \lambda R)^{-1}$ where $R = \text{diag}(\text{diag}(J J^T)^p)$ is the noise covariance regularizer with $p = 0.5$, $\lambda = 0.01$.

At inference: $\delta\hat{\sigma}_\text{grid} = -H\,\delta v$, masked to zero outside the domain. This is a single matrix-vector multiply, trained offline to optimize figures of merit (amplitude uniformity, position error, resolution, shape deformation, ringing).

**3c. JAC Dynamic (Jacobian-based Time-Difference)**

Solves the Tikhonov-regularized least-squares problem with a Kotre prior:

$$\delta\hat{\sigma} = -(J^T J + \lambda R)^{-1} J^T \delta v$$

where $R = \text{diag}(\text{diag}(J^T J)^p)$ is the Kotre regularization matrix with $p = 0.5$, $\lambda = 0.01$. The Kotre prior weights regularization by the diagonal of $J^T J$, which normalizes for varying element sensitivities across the mesh.

This method reconstructs element-wise conductivity changes and is well-suited for monitoring temporal changes (breathing, flow) where a stable reference measurement is available. The `dist_exc=8` excitation pattern (opposite-drive) provides better spatial coverage than adjacent-drive, improving reconstruction quality for the dynamic case.

### Step 4: Evaluation

Compare all three methods using two metrics computed between the reconstruction $\hat{x}$ and the ground truth conductivity change $x_\text{ref}$:

- **NRMSE**: $\text{NRMSE} = \frac{\sqrt{\frac{1}{N}\sum_i (\hat{x}_i - x_{\text{ref},i})^2}}{\max(x_\text{ref}) - \min(x_\text{ref})}$ (lower is better)
- **NCC**: $\text{NCC} = \frac{\hat{x} \cdot x_\text{ref}}{\|\hat{x}\|\,\|x_\text{ref}\|}$ (higher is better, max 1)

For BP, both reconstruction and ground truth are on mesh nodes (element values interpolated to nodes via `sim2pts`). For JAC Dynamic, comparison is on mesh elements. For GREIT, the FEM ground truth is rasterized onto the same pixel grid for comparison.

### Step 5: Visualization

Generate comparison panels showing:
- Ground truth conductivity change for each experiment (on mesh or grid)
- Reconstructed conductivity change for each method
- Metrics table summarizing NRMSE and NCC across methods

## Expected Results

| Method      | NCC   | NRMSE  | Speed    | Notes                                  |
|-------------|-------|--------|----------|----------------------------------------|
| BP          | ~0.56 | ~0.54  | Fastest  | Blurry; good for real-time monitoring  |
| GREIT       | ~0.59 | ~0.09  | Fast     | 32x32 grid; best NRMSE                |
| JAC Dynamic | ~0.61 | ~0.11  | Moderate | Best NCC; opposite-drive excitation    |

All three methods successfully localize the conductivity inclusion(s) within the circular domain. GREIT and JAC Dynamic achieve substantially lower NRMSE than BP due to regularization. JAC Dynamic achieves the highest NCC, benefiting from the opposite-drive excitation pattern which provides more uniform sensitivity coverage. BP trades reconstruction quality for speed and simplicity.
