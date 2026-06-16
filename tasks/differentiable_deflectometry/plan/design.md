# Code Architecture

## Module Structure

```
src/
├── preprocessing.py    # Raw data -> processed observations
├── physics_model.py    # Differentiable ray tracer (forward model)
├── solvers.py          # LM optimization
└── visualization.py    # Plotting and metrics
```

## Key Classes and Functions

### `src/physics_model.py`

Core differentiable ray tracer combining optics primitives, surfaces, and scene composition.

```python
class Ray:
    """3D ray with origin, direction, wavelength."""
    def __init__(self, o, d, wavelength, device): ...
    def __call__(self, t) -> Tensor: ...  # evaluate ray at parameter t

class Transformation:
    """Rigid body transformation (R, t)."""
    def transform_point(self, o) -> Tensor: ...
    def transform_ray(self, ray) -> Ray: ...
    def inverse(self) -> Transformation: ...

class Material:
    """Optical material with Sellmeier-approximated IOR."""
    def ior(self, wavelength) -> float: ...

class Surface:
    """Base class for implicit optical surfaces g(x,y) + h(z) = 0."""
    def ray_surface_intersection(self, ray, active) -> (valid, point, g): ...
    def newtons_method(self, maxt, o, D, option='implicit') -> (valid, point): ...
    def normal(self, x, y) -> Tensor: ...

class Aspheric(Surface):
    """Aspheric surface: z = c*r^2 / (1 + sqrt(1-(1+k)*c^2*r^2)) + higher_order."""
    # Parameters: c (curvature), k (conic), ai (higher-order coefficients)

class Lensgroup:
    """Multi-surface optical element with pose parameters."""
    def load_file(self, filename, lens_dir): ...
    def trace(self, ray) -> (ray_out, valid, mask): ...
    def update(self, _x, _y): ...  # update transformation from pose params
    # Parameters: origin, shift, theta_x, theta_y, theta_z, surfaces[], materials[]

class Scene:
    """Full imaging scene: cameras + screen + lensgroup."""
    def trace(self, i, with_element, mask) -> (points, valid, mask): ...
    def render(self, i, with_element, mask) -> images: ...

def get_nested_attr(obj, path) -> Any: ...  # 'lensgroup.surfaces[0].c' -> tensor
def set_nested_attr(obj, path, value): ...
def set_texture(scene, texture, device): ...
```

### `src/preprocessing.py`

```python
class Fringe:
    """Four-step phase-shifting fringe analysis."""
    def solve(self, fs) -> (a, b, psi): ...
    def unwrap(self, fs, Ts, valid) -> displacement: ...

def load_calibration(calibration_path, rotation_path, lut_path, scale, device):
    """Load camera/screen geometry from MATLAB .mat files."""
    -> cameras, screen, p_rotation, lut_data

def compute_mount_origin(cameras, p_rotation, device):
    """Estimate lens mount position via ray intersection."""
    -> origin (ndarray[3])

def crop_images(imgs, filmsize, full_filmsize):
    """Center-crop raw images."""

def solve_for_intersections(imgs, refs, Ts, scene, device):
    """Full pipeline: fringe images -> measured intersection points."""
    -> ps_cap (Tensor), valid_cap (Tensor), centers (Tensor)

def prepare_measurement_images(imgs, xs, valid_cap, fringe_a_cap, device):
    """Background-subtract and normalize for visualization."""
    -> I0 (Tensor)
```

### `src/solvers.py`

```python
class LMSolver:
    """Levenberg-Marquardt with autograd Jacobians."""
    def __init__(self, lamb, mu, regularization, max_iter): ...
    def jacobian(self, func, inputs) -> Tensor: ...  # M-by-N via column-wise JVP
    def optimize(self, forward_fn, scene, param_names, residual_fn, device):
        """Main LM loop with adaptive damping."""
        -> loss_history (list[float])

def setup_diff_parameters(scene, param_names, device) -> list[Tensor]: ...
def change_parameters(scene, param_names, deltas, sign) -> list[Tensor]: ...
```

### `src/visualization.py`

```python
def plot_spot_diagram(ps_measured, ps_modeled, valid, camera_count, filename): ...
def plot_image_comparison(I_measured, I_modeled, valid, filename_prefix): ...
def plot_loss_curve(loss_history, filename): ...
def compute_metrics(ps_modeled, ps_measured, valid, loss_history, recovered, gt) -> dict: ...
def save_metrics(metrics, filepath): ...
```

## Data Flow

```
raw_data.npz (imgs, refs)
    |
    v
preprocessing.solve_for_intersections()
    |-- Fringe.solve() -> phase maps
    |-- Fringe.unwrap() -> displacement
    |-- Scene.trace(no_element) -> reference points
    |-- subtract displacement -> ps_measured
    v
LMSolver.optimize()
    |-- forward_fn() -> Scene.trace(with_element) -> ps_modeled
    |-- jacobian() -> J via autograd
    |-- damping loop -> parameter updates
    v
metrics + visualizations
```
