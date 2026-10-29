# Seismic Traveltime Tomography: Code Design

## Module Overview

```
src/
├── physics_model.py   # Eikonal solver, traveltime interpolation, gradient
├── preprocessing.py   # Data I/O, residuals, misfit
├── generate_data.py   # Synthetic velocity model and dataset generation
├── solvers.py         # Ray tracing, kernel accumulation, ATT inversion loop
└── visualization.py   # Plotting, NCC/NRMSE metrics
```

---

## `src/physics_model.py`

### `solve_eikonal`
```python
def solve_eikonal(
    slowness: np.ndarray,   # (Nz, Nx) float64, s/km
    dx: float,              # horizontal grid spacing, km
    dz: float,              # vertical grid spacing, km
    src_x: float,           # source x position, km
    src_z: float,           # source z position, km
) -> np.ndarray:            # (Nz, Nx) float32, traveltime in seconds
```
Solves the Eikonal equation from a point source via Fast Marching Method
(`skfmm.travel_time`).  Returns traveltime field T(x, z) in seconds.
Source position is clamped to grid boundary if outside domain.

### `compute_traveltime_at`
```python
def compute_traveltime_at(
    T: np.ndarray,    # (Nz, Nx) traveltime field
    rec_x: float,     # receiver x, km
    rec_z: float,     # receiver z, km
    dx: float,
    dz: float,
) -> float:           # traveltime at (rec_x, rec_z)
```
Bilinear interpolation of T at an arbitrary point.

### `compute_traveltime_gradient`
```python
def compute_traveltime_gradient(
    T: np.ndarray,    # (Nz, Nx)
    dx: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:   # (dTdz, dTdx), both (Nz, Nx)
```
Central finite differences on interior; one-sided on boundaries.

### `compute_all_traveltimes`
```python
def compute_all_traveltimes(
    slowness: np.ndarray,   # (Nz, Nx)
    dx: float, dz: float,
    sources: np.ndarray,    # (N_src, 2)  columns: [x, z]
    receivers: np.ndarray,  # (N_rec, 2)  columns: [x, z]
) -> np.ndarray:            # (N_src, N_rec) float32
```
Solves Eikonal for each source; interpolates at all receivers.

---

## `src/preprocessing.py`

### `load_data`
```python
def load_data(data_dir: str) -> dict:
```
Loads `raw_data.npz`, `ground_truth.npz`, `meta_data.json`.
Strips the leading batch dimension from all arrays.
Returns dict with keys: `traveltime_obs`, `sources`, `receivers`,
`velocity_true`, `meta`.

### `compute_residuals`
```python
def compute_residuals(
    T_syn: np.ndarray,   # (N_src, N_rec)
    T_obs: np.ndarray,   # (N_src, N_rec)
) -> np.ndarray:         # (N_src, N_rec), T_syn - T_obs
```

### `compute_misfit`
```python
def compute_misfit(residuals: np.ndarray) -> float:
    # 0.5 * sum(residuals**2)
```

---

## `src/generate_data.py`

### Constants
```python
DEFAULT_NX, DEFAULT_NZ = 111, 26
DEFAULT_DX_KM, DEFAULT_DZ_KM = 2.0, 2.0
DEFAULT_V0, DEFAULT_V1 = 4.0, 8.0    # km/s, top and bottom
DEFAULT_PERT = 0.05
DEFAULT_N_SRC, DEFAULT_N_REC = 100, 8
DEFAULT_NOISE = 0.1                   # s, Gaussian traveltime noise std
```

### `make_background_velocity`
```python
def make_background_velocity(
    Nx: int, Nz: int,
    dx: float, dz: float,
    v0: float = DEFAULT_V0,
    v1: float = DEFAULT_V1,
) -> np.ndarray:   # (Nz, Nx) float64
```
Linear vertical gradient: v(z) = v0 + (v1 - v0) * z / z_max.

### `make_checkerboard_perturbation`
```python
def make_checkerboard_perturbation(
    Nx: int, Nz: int,
    dx: float, dz: float,
    pert: float = DEFAULT_PERT,
    n_x: int = 2, n_z: int = 2,
) -> np.ndarray:   # (Nz, Nx) float64, fractional perturbation
```
Sinusoidal checkerboard: pert * sin(n_x * π * x / x_max) * sin(n_z * π * z / z_max).

### `make_true_velocity`
```python
def make_true_velocity(
    Nx: int, Nz: int,
    dx: float, dz: float,
    v0: float = DEFAULT_V0, v1: float = DEFAULT_V1,
    pert: float = DEFAULT_PERT,
) -> np.ndarray:   # (Nz, Nx) float64
```
v_true = v_bg * (1 + checkerboard_perturbation).

### `make_sources`
```python
def make_sources(n: int, x_max: float) -> np.ndarray:   # (n, 2)
```
Sources at three depth levels (10, 20, 35 km); 65% in the left third of domain.

### `make_receivers`
```python
def make_receivers(n: int, x_max: float) -> np.ndarray:   # (n, 2)
```
Uniformly spaced at surface (z = 0).

### `generate_data`
```python
def generate_data(data_dir: str) -> None:
```
Generates and saves `raw_data.npz`, `ground_truth.npz`, `meta_data.json`.

---

## `src/solvers.py`

### `_trace_ray_on_grid`
```python
def _trace_ray_on_grid(
    T: np.ndarray,
    dTdz: np.ndarray, dTdx: np.ndarray,
    rec_x: float, rec_z: float,
    src_x: float, src_z: float,
    dx: float, dz: float,
    step_km: float = 1.0,
) -> tuple[list, list, list]:   # (xs, zs, dls)
```
Euler ray tracing from receiver toward source along −∇T / |∇T|.
Terminates when distance to source < step_km.  Uses `scipy.ndimage.map_coordinates`
for bilinear gradient interpolation.  Clamps position to grid boundary.

### `compute_sensitivity_kernel`
```python
def compute_sensitivity_kernel(
    slowness: np.ndarray,
    dx: float, dz: float,
    sources: np.ndarray,
    receivers: np.ndarray,
    T_obs: np.ndarray,
    step_km: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # kernel (Nz,Nx), kernel_density (Nz,Nx), T_syn (N_src,N_rec), misfit
```

### `kernel_density_normalization`
```python
def kernel_density_normalization(
    kernel: np.ndarray,
    kernel_density: np.ndarray,
    zeta: float = 0.5,
    epsilon: float = 1e-4,
) -> np.ndarray:   # normalized kernel
    # kernel / (kernel_density + epsilon) ** zeta
```

### `update_slowness`
```python
def update_slowness(
    slowness: np.ndarray,
    kernel_norm: np.ndarray,
    step_size: float = 0.02,
) -> np.ndarray:   # (Nz, Nx) float32, clipped to [1/15, 1/1]
```

### `ATTSolver`
```python
class ATTSolver:
    def __init__(
        self,
        num_iterations: int = 40,
        step_size: float = 0.02,
        step_decay: float = 0.97,
        zeta: float = 0.5,
        epsilon: float = 1e-4,
        step_km: float = 1.0,
        smooth_sigma: float = 1.5,
    ): ...

    def run(
        self,
        slowness_init: np.ndarray,   # (Nz, Nx)
        dx: float, dz: float,
        sources: np.ndarray,         # (N_src, 2)
        receivers: np.ndarray,       # (N_rec, 2)
        T_obs: np.ndarray,           # (N_src, N_rec)
        verbose: bool = True,
    ) -> dict:
        # Returns: slowness, velocity, misfit_history, kernel_final, T_syn_final
```

---

## `src/visualization.py`

### `compute_ncc`
```python
def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    # cosine similarity: dot(a.flat, b.flat) / (||a|| * ||b||)
```

### `compute_nrmse`
```python
def compute_nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    # rms(pred - ref) / (max(ref) - min(ref))
```

### `evaluate_reconstruction`
```python
def evaluate_reconstruction(
    v_inv: np.ndarray,    # (Nz, Nx) inverted velocity
    v_true: np.ndarray,   # (Nz, Nx) true velocity
    v_init: np.ndarray,   # (Nz, Nx) background velocity
) -> dict:
    # Returns: ncc, nrmse (perturbation), ncc_full, nrmse_full (full model)
```

### `plot_checkerboard_recovery`
```python
def plot_checkerboard_recovery(
    v_true, v_init, v_inv,
    dx, dz,
    sources=None, receivers=None,
    metrics=None,
    save_path=None,
) -> None:
```
4-panel figure: true, initial, inverted models + perturbation comparison.

### `plot_convergence`
```python
def plot_convergence(misfit_history: list, save_path=None) -> None:
```

### `plot_sensitivity_kernel`
```python
def plot_sensitivity_kernel(
    kernel: np.ndarray,
    dx: float, dz: float,
    save_path=None,
) -> None:
```
