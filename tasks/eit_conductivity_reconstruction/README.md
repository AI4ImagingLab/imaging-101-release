# EIT Conductivity Reconstruction

> Reconstruct the 2D internal conductivity distribution of a circular domain from boundary voltage measurements collected by an electrode array, where the severe ill-posedness of the inverse problem demands regularization.

> Domain: Medicine | Keywords: impedance imaging, regularized inversion, finite element method | Difficulty: Medium

## Background

Electrical Impedance Tomography (EIT) is a non-invasive imaging technique that
recovers the internal conductivity distribution of a body from electrical
measurements taken at its boundary. An array of electrodes is placed around the
object's surface; small alternating currents are injected through electrode pairs
and the resulting voltages are measured at the remaining electrodes. By cycling
through many excitation patterns, a set of boundary voltage measurements is
collected that encodes information about the internal conductivity.

EIT has applications in medical imaging (lung ventilation monitoring, breast
cancer detection), industrial process tomography (multiphase flow imaging), and
geophysical surveying. Its main advantages are that it is radiation-free,
portable, low-cost, and capable of high temporal resolution.

## Problem Description

The forward model is governed by the generalized Laplace equation on a 2D
domain $\Omega$ (unit circle):

$$\nabla \cdot \bigl(\sigma(x,y)\,\nabla \phi(x,y)\bigr) = 0 \quad \text{in } \Omega$$

with boundary conditions:

$$\sigma \frac{\partial \phi}{\partial n} = j \quad \text{on electrode surfaces}$$
$$\sigma \frac{\partial \phi}{\partial n} = 0 \quad \text{elsewhere on } \partial\Omega$$

where $\sigma(x,y)$ is the conductivity distribution (the unknown), $\phi(x,y)$
is the electric potential field, and $j$ is the injected current density.

The domain is discretized into triangular finite elements, each assigned a
constant conductivity. The FEM discretization converts the PDE into a linear
system $K(\sigma)\,\phi = f$, where $K$ is the global stiffness matrix
assembled from element contributions.

The **inverse problem** is to recover $\sigma$ from boundary voltage
measurements $V$. For small perturbations around a reference conductivity
$\sigma_0$:

$$\delta V \approx J\,\delta\sigma$$

where $J = \partial V / \partial \sigma$ is the Jacobian (sensitivity) matrix
computed via the adjoint method. This linearized system is severely
ill-conditioned: the singular values of $J$ decay rapidly, so small measurement
noise causes large reconstruction errors. Regularization is essential.

## Data Description

### `data/raw_data.npz`

Contains mesh, protocol, and voltage data for three experiments (BP, GREIT,
JAC Dynamic). Each experiment's arrays are prefixed with its name. The table
below uses BP as an example; GREIT arrays have identical structure, while
JAC Dynamic arrays differ slightly in measurement count due to `dist_exc=8`.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `{prefix}_node` | `(1, 376, 3)` | `float64` | FEM mesh node coordinates (x, y, z) in metres |
| `{prefix}_element` | `(1, 686, 3)` | `int32` | Triangle connectivity (node indices per element) |
| `{prefix}_el_pos` | `(1, 16)` | `int64` | Electrode node indices on the boundary |
| `{prefix}_ref_node` | `(1,)` | `int64` | Reference (ground) node index |
| `{prefix}_perm_background` | `(1, 686)` | `float64` | Background conductivity per element (S/m) |
| `{prefix}_perm_anomaly` | `(1, 686)` | `float64` | Anomaly conductivity per element (S/m) |
| `{prefix}_v0` | `(1, 208)` | `float64` | Reference boundary voltages — homogeneous conductivity (V) |
| `{prefix}_v1` | `(1, 208)` | `float64` | Anomaly boundary voltages — with inclusion(s) (V) |
| `{prefix}_ex_mat` | `(1, 16, 2)` | `int64` | Excitation electrode pairs |
| `{prefix}_meas_mat` | `(1, 208, 3)` | `int64` | Measurement pattern matrix |
| `{prefix}_keep_ba` | `(1, 256)` | `bool` | Measurement keep mask |

Prefixes: `bp`, `greit`, `jac_dyn`. For `jac_dyn`, voltage arrays have 192
measurements instead of 208 (due to `dist_exc=8`).

### `data/ground_truth.npz`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `bp_perm_anomaly` | `(1, 686)` | `float64` | BP anomaly conductivity — single inclusion at (0.5, 0.5) with $\sigma = 10$ S/m |
| `bp_perm_background` | `(1, 686)` | `float64` | BP background conductivity — uniform $\sigma = 1$ S/m |
| `greit_perm_anomaly` | `(1, 686)` | `float64` | GREIT anomaly — four inclusions (two high, two low conductivity) |
| `greit_perm_background` | `(1, 686)` | `float64` | GREIT background conductivity — uniform $\sigma = 1$ S/m |
| `jac_dyn_perm_anomaly` | `(1, 686)` | `float64` | JAC Dynamic anomaly — single inclusion with $\sigma = 1000$ S/m |
| `jac_dyn_perm_background` | `(1, 686)` | `float64` | JAC Dynamic background conductivity — uniform $\sigma = 1$ S/m |

### `data/meta_data.json`

Experiment configuration including electrode count (`n_el=16`), mesh resolution
(`h0=0.1`), excitation pattern (`dist_exc`), and anomaly geometry (centre
coordinates, radius, conductivity value) for each of the three experiments.
Does not contain solver or algorithm parameters.

## Method Hints

Three complementary reconstruction algorithms address different trade-offs
between speed, resolution, and robustness:

1. **Back-Projection (BP)**: maps voltage differences onto the mesh via the
   transpose of the Jacobian. Fastest method but produces blurry images with
   limited spatial resolution. Suitable for real-time monitoring.

2. **GREIT (Graz consensus Reconstruction algorithm for EIT)**: a linear,
   one-step method that maps measurements directly to a regular pixel grid.
   A reconstruction matrix is trained offline by optimising figures of merit
   (amplitude response, position error, resolution, shape deformation, ringing)
   over a set of point targets placed throughout the domain. At inference time,
   a single matrix-vector multiply produces the image.

3. **Jacobian Dynamic (time-difference)**: solves the Tikhonov-regularized
   least-squares problem $\delta\hat{\sigma} = (J^T J + \lambda R)^{-1} J^T \delta V$
   where $R$ is a regularization matrix (e.g., Kotre prior). Reconstructs the
   *change* in conductivity between a reference state and the current state.

All three methods are difference-imaging approaches: they reconstruct
$\delta\sigma = \sigma_\text{anomaly} - \sigma_\text{background}$ rather than
absolute conductivity, which avoids the more difficult fully nonlinear inverse
problem.

## References

- Adler, A. and Lionheart, W.R.B. (2006). "Uses and abuses of EIDORS: an extensible software base for EIT." *Physiological Measurement*, 27(5), S25–S42.
- Adler, A. et al. (2009). "GREIT: a unified approach to 2D linear EIT reconstruction of lung images." *Physiological Measurement*, 30(6), S35–S55.
- Liu, B. et al. (2018). "pyEIT: A python based framework for Electrical Impedance Tomography." *SoftwareX*, 7, 304–308.
