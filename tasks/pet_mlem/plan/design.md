# Design: PET MLEM Reconstruction

## Module Architecture

```
src/
├── preprocessing.py     # Data I/O
├── physics_model.py     # PET forward/back projection, Poisson noise, background
├── solvers.py           # MLEM and OSEM algorithms
├── visualization.py     # Plotting utilities and metrics
└── generate_data.py     # Synthetic PET phantom and data generation
```

## Function Signatures

### src/physics_model.py

```python
def pet_forward_project(image, theta) -> np.ndarray:
    """Forward project activity to sinogram via Radon transform."""

def pet_back_project(sinogram, theta, N) -> np.ndarray:
    """Unfiltered back-projection (A^T y)."""

def compute_sensitivity_image(theta, N) -> np.ndarray:
    """Sensitivity image A^T 1."""

def add_poisson_noise(sinogram, scale, rng) -> np.ndarray:
    """Add Poisson counting noise."""

def add_background(sinogram, randoms_fraction, rng) -> (sino_bg, background):
    """Add uniform random coincidence background."""
```

### src/solvers.py

```python
def solve_mlem(sinogram, theta, N, n_iter, background, x_init) -> (x, ll_history):
    """MLEM reconstruction with Poisson log-likelihood tracking."""

def solve_osem(sinogram, theta, N, n_iter, n_subsets, background, x_init) -> (x, ll_history):
    """OSEM (ordered subsets EM) acceleration of MLEM."""
```

### src/generate_data.py

```python
def create_activity_phantom(N) -> np.ndarray:
    """Shepp-Logan phantom with activity levels and hot lesions."""

def generate_synthetic_data(N, n_angles, count_level, ...) -> dict:
    """Generate complete synthetic PET dataset with Poisson noise."""
```

## Pipeline Flow (main.py)

1. Load data (or generate if missing)
2. MLEM reconstruction (50 iterations)
3. OSEM reconstruction (10 iterations, 6 subsets)
4. Evaluate NCC/NRMSE within activity mask
5. Save reference outputs, metrics, visualizations
