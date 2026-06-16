# Code Design

## File Structure

```text
main.py                  # End-to-end pipeline orchestration
src/
  preprocessing.py       # Metadata loading
  physics_model.py       # Broxton wave-optics engine: dataclasses, LFMSystem
                         # (forward/backward projectors), PSF, MLA, system builder,
                         # and pipeline wrappers (write_wave_model_config,
                         # build_volume_system, compute_conventional_image)
  solvers.py             # Richardson-Lucy deconvolution implementation
  generate_data.py       # Synthetic target and noise helpers
  visualization.py       # Metrics, profiles, and plots
data/
  meta_data.json         # Microscope, benchmark, and volume-demo configuration
  raw_data.npz           # Batch-first light-field observations
  ground_truth.npz       # Batch-first thickness-1 volumes and planar slices
evaluation/
  metrics.json           # NCC/NRMSE evaluation boundaries
  reference_outputs/
    cases.npz            # Benchmark observations, volumes, baselines, reconstructions
    volume_demo.npz      # Five-slice generalized reconstruction demo
    baseline_reference.npz  # Stacked benchmark RL reference reconstruction
    metrics.json         # Detailed benchmark + volume-demo metrics
    volume_demo_metrics.json
    light_field_usaf_comparison.png
    volume_reconstruction_demo.png
notebooks/
  light_field_microscope.ipynb
output/
  reconstruction.npy         # Benchmark reconstruction used for scoring
  volume_demo_reconstruction.npy
```

## Pipeline Flow

```text
data/meta_data.json
    |
    +-- preprocessing.load_metadata()
    |
    +-- physics_model.write_wave_model_config()
    |
    +-- benchmark loop over occupied target depths
    |     |
    |     +-- physics_model.build_volume_system() on a singleton z-grid
    |     +-- generate_data.make_linepair_object()
    |     +-- generate_data.place_object_at_depth()
    |     +-- system.forward_project()
    |     +-- generate_data.add_poisson_noise()
    |     +-- physics_model.compute_conventional_image()
    |     +-- solvers.run_richardson_lucy()
    |     +-- visualization.compute_image_metrics()
    |     +-- visualization.compute_center_line_profile()
    |     +-- visualization.extract_line_profile()
    |     +-- visualization.normalize_profile()
    |
    +-- generalized volume demo
    |     |
    |     +-- physics_model.build_volume_system() on [-100, -75, -50, -25, 0] um
    |     +-- generate_data.make_linepair_object()
    |     +-- generate_data.place_object_at_depth()
    |     +-- system.forward_project()
    |     +-- generate_data.add_poisson_noise()
    |     +-- solvers.run_richardson_lucy()
    |     +-- visualization.compute_volume_slice_energy()
    |     +-- visualization.plot_volume_reconstruction_demo()
    |
    +-- visualization.plot_light_field_usaf_comparison()
    +-- save data/*.npz
    +-- save evaluation/reference_outputs/*
    +-- save output/*.npy
```

## Function Signatures

### preprocessing.py

```python
def load_metadata(path: str | Path = "data/meta_data.json") -> dict:
    """Load microscope and benchmark metadata from JSON."""
```

### physics_model.py

All wave-optics dataclasses (`CameraParams`, `Resolution`, `PatternOp`),
the `LFMSystem` class (with `forward_project` and `backward_project` methods),
all optics functions (`load_camera_params`, `build_resolution`,
`build_forward_kernels`, `build_lfm_system`, `compute_native_plane_psf`, etc.),
and the three pipeline wrappers used by `main.py`:

```python
def write_wave_model_config(metadata: dict, config_path: str | Path) -> None:
    """Write the YAML configuration consumed by load_camera_params."""

def build_volume_system(
    config_path: str | Path,
    n_lenslets: int,
    new_spacing_px: int,
    depth_range_um: tuple[float, float],
    depth_step_um: float,
    theta_samples: int,
    kernel_tol: float,
) -> LFMSystem:
    """Build the wave-model light-field system on the requested z-grid."""

def compute_conventional_image(
    system,
    object_2d: np.ndarray,
    target_depth_um: float,
    theta_samples: int,
) -> np.ndarray:
    """Compute the defocused conventional-microscope baseline image."""
```

### solvers.py

```python
def run_richardson_lucy(
    system,
    observation: np.ndarray,
    iterations: int,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Run Richardson-Lucy deconvolution for the light-field forward model.

    Implements:  v^(q+1) = v^(q) * H^T(y / (H v^(q) + eps)) / H^T(1)
    """
```

### generate_data.py

```python
DEFAULT_RANDOM_SEED = 0
DEFAULT_USAF_TARGET_DEPTH_UM = 0.0
DEFAULT_USAF_SHIFT_PERPENDICULAR_LENSLETS = 0.5

def make_linepair_object(
    tex_shape: tuple[int, int],
    tex_res_xy_um: tuple[float, float],
    lp_per_mm: float,
    window_size_um: float,
    shift_perpendicular_um: float = 0.0,
) -> np.ndarray:
    """Create the square-windowed line-pair target."""

def place_object_at_depth(
    object_2d: np.ndarray,
    tex_shape: tuple[int, int],
    depths: np.ndarray,
    target_depth_um: float,
    background: float = 0.0,
) -> np.ndarray:
    """Embed the 2D target into the nearest depth plane of a 3D volume."""

def compute_shift_perpendicular_um_from_sampling(
    tex_nnum_x: float,
    tex_res_x_um: float,
    lenslet_pitch_fraction: float = DEFAULT_USAF_SHIFT_PERPENDICULAR_LENSLETS,
    shift_um: Optional[float] = None,
) -> float:
    """Resolve the phantom shift in object-space micrometers."""

def resolve_usaf_configuration(metadata: dict) -> dict:
    """Normalize the USAF configuration block from metadata."""

def add_poisson_noise(
    image: np.ndarray,
    scale: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the peak-normalize / Poisson-sample / rescale noise model."""
```

### visualization.py

```python
def normalized_cross_correlation(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Compute cosine-style NCC between two arrays."""

def normalized_root_mean_square_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Compute dynamic-range-normalized RMSE."""

def compute_image_metrics(image_2d: np.ndarray, gt_2d: np.ndarray) -> dict:
    """Return MSE, NRMSE, SSIM, and PSNR."""

def compute_center_line_profile(
    object_2d: np.ndarray,
    tex_res_x_um: float,
    margin_vox: int = 10,
) -> dict:
    """Return the center-row profile specification through the line-pair target."""

def extract_line_profile(image_2d: np.ndarray, profile_spec: dict) -> np.ndarray:
    """Sample one row from an image using the saved profile specification."""

def normalize_profile(profile: np.ndarray) -> np.ndarray:
    """Normalize a profile to unit peak."""

def compute_volume_slice_energy(volume: np.ndarray) -> np.ndarray:
    """Sum the non-negative intensity in each axial slice of a volume."""

def plot_light_field_usaf_comparison(cases: list, rl_iterations: int, title: str) -> plt.Figure:
    """Plot benchmark GT, observation, conventional baseline, RL, and line profiles."""

def plot_volume_reconstruction_demo(
    observation: np.ndarray,
    gt_volume: np.ndarray,
    reconstruction_volume: np.ndarray,
    depths_um: np.ndarray,
    target_depth_um: float,
    title: str,
) -> plt.Figure:
    """Plot all reconstruction slices and the axial energy profile for the volume demo."""
```

## Notes

- `src/physics_model.py` contains the complete wave-optics engine: a Python
  port of the oLaF MATLAB implementation (Broxton et al.), including all
  dataclasses, the `LFMSystem` forward/backward projector, PSF/MLA/kernel
  computation, and the three pipeline wrappers used by `main.py`.
- `src/solvers.py` implements the full Richardson-Lucy loop directly; it does
  not delegate to any method on the system object.
- `main.py` remains the only end-to-end pipeline entry point. The `src/`
  package contains small helpers, not an alternative workflow.
- Solver parameters (`_RL_ITERATIONS`, `_THETA_SAMPLES`, `_KERNEL_TOL`) are
  hardcoded in `main.py` as module-level constants and are not stored in
  `data/meta_data.json` to prevent leaking algorithm information to evaluation
  agents.
- The benchmark used for evaluation scores the singleton-depth cases via
  `output/reconstruction.npy`, but the task narrative and notebook are framed as
  3D volume reconstruction, with the singleton-depth cases treated explicitly as
  a special case.
