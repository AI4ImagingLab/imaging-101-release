# Design: MRI T2 Mapping

## Module Architecture

```
src/
├── preprocessing.py     # Data I/O and signal preparation
├── physics_model.py     # Forward model (signal simulation)
├── solvers.py           # T2 fitting algorithms
├── visualization.py     # Metrics and plotting
└── generate_data.py     # Synthetic phantom generation
```

## Function Signatures

### src/physics_model.py

```python
def mono_exponential_signal(M0, T2, TE) -> np.ndarray:
    """Compute S = M0 * exp(-TE / T2). Shape: (..., N_echoes)."""

def add_rician_noise(signal, sigma, rng=None) -> np.ndarray:
    """Add Rician noise: |S + CN(0, sigma^2)|."""

def simulate_multi_echo(M0, T2, TE, sigma=0.0, rng=None) -> np.ndarray:
    """Full simulation: signal model + optional noise."""
```

### src/solvers.py

```python
def fit_t2_loglinear(signal, TE, mask=None) -> (T2_map, M0_map):
    """Log-linear T2 fit via vectorized least squares."""

def fit_t2_nonlinear(signal, TE, mask=None, T2_init=None, M0_init=None) -> (T2_map, M0_map):
    """Nonlinear least-squares T2 fit with Levenberg-Marquardt."""
```

### src/preprocessing.py

```python
def load_multi_echo_data(task_dir) -> np.ndarray:
    """Load raw_data.npz, return (1, Ny, Nx, N_echoes) float64."""

def load_ground_truth(task_dir) -> (T2_map, M0_map, tissue_mask):
    """Load ground_truth.npz."""

def load_metadata(task_dir) -> dict:
    """Load meta_data.json with echo_times_ms as numpy array."""

def preprocess_signal(signal) -> np.ndarray:
    """Remove batch dim, ensure non-negative. (Ny, Nx, N_echoes)."""
```

### src/visualization.py

```python
def compute_ncc(estimate, reference, mask=None) -> float:
    """Cosine similarity (no mean subtraction)."""

def compute_nrmse(estimate, reference, mask=None) -> float:
    """RMSE / dynamic_range(reference)."""

def plot_t2_maps(T2_gt, T2_est, tissue_mask, ...) -> Figure:
    """Side-by-side T2 maps with error map."""

def plot_signal_decay(signal, TE, T2_gt, T2_est, pixel_coords, ...) -> Figure:
    """Signal decay curves at selected pixels."""
```

### src/generate_data.py

```python
def create_t2_m0_phantom(N=256) -> (T2_map, M0_map, tissue_mask):
    """Shepp-Logan phantom with T2/M0 tissue assignments."""

def generate_synthetic_data(N=256, echo_times_ms=None, sigma=0.02, seed=42) -> dict:
    """Generate full synthetic dataset."""

def save_data(data, task_dir):
    """Save to raw_data.npz, ground_truth.npz, meta_data.json."""
```

## Pipeline Flow (main.py)

1. Load data (or generate if missing)
2. Preprocess: remove batch dim, clip negatives
3. Log-linear fit -> T2_init, M0_init
4. Nonlinear LS fit (initialized from step 3)
5. Compute NCC, NRMSE on tissue-masked T2 maps
6. Save reference outputs and metrics
7. Generate visualization figures
