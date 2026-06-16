# Design: Fan-Beam CT Reconstruction

## Module Architecture

```
src/
├── preprocessing.py     # Data I/O
├── physics_model.py     # Fan-beam forward model, FBP, geometry, Parker weights
├── solvers.py           # TV-PDHG iterative reconstruction
├── visualization.py     # Plotting utilities and metrics
└── generate_data.py     # Synthetic fan-beam data generation
```

## Function Signatures

### src/physics_model.py

```python
def fan_beam_geometry(N, n_det, n_angles, D_sd, D_dd, angle_range) -> dict:
    """Compute fan-beam geometry parameters."""

def fan_beam_forward_vectorized(image, geo) -> np.ndarray:
    """Fan-beam forward projection via ray-tracing. Shape: (n_angles, n_det)."""

def fan_beam_backproject(sinogram, geo) -> np.ndarray:
    """Fan-beam back-projection (adjoint). Shape: (N, N)."""

def ramp_filter(n_det, det_spacing, filter_type, cutoff) -> (filt, pad_len):
    """Construct ramp filter in frequency domain."""

def parker_weights(angles, det_pos, D_sd) -> np.ndarray:
    """Parker weights for short-scan. Shape: (n_angles, n_det)."""

def fan_beam_fbp(sinogram, geo, filter_type, cutoff, short_scan) -> np.ndarray:
    """Full fan-beam FBP reconstruction. Shape: (N, N)."""

def add_gaussian_noise(sinogram, sigma, rng) -> np.ndarray:
    """Add Gaussian noise to sinogram."""
```

### src/solvers.py

```python
def _gradient_2d(x) -> np.ndarray:
    """Discrete gradient (forward differences). Shape: (2, N, N)."""

def _divergence_2d(p) -> np.ndarray:
    """Discrete divergence (negative adjoint of gradient). Shape: (N, N)."""

def _prox_l1_norm(p, sigma) -> np.ndarray:
    """Proximal of sigma * ||p||_1 (pointwise L2 projection on gradient)."""

def solve_tv_pdhg(sinogram, geo, lam, n_iter, positivity, x_init) -> (x, loss):
    """TV-regularized reconstruction via Chambolle-Pock PDHG."""
```

### src/visualization.py

```python
def compute_ncc(estimate, reference, mask=None) -> float:
    """Cosine similarity (no mean subtraction)."""

def compute_nrmse(estimate, reference, mask=None) -> float:
    """RMSE / dynamic_range(reference)."""

def centre_crop_normalize(image, crop_fraction=0.8) -> np.ndarray:
    """Centre-crop and normalize to [0, 1]."""

def plot_reconstructions(phantom, recon_fbp, recon_tv, ...) -> Figure:
    """4-panel: GT, FBP, TV, error map."""

def plot_sinogram(sinogram, title, ...) -> Figure:
    """Single sinogram visualization."""
```

### src/preprocessing.py

```python
def load_sinogram_data(task_dir) -> (sino_full, sino_short, angles_full, angles_short, det_pos):
    """Load raw_data.npz."""

def load_ground_truth(task_dir) -> phantom:
    """Load ground_truth.npz."""

def load_metadata(task_dir) -> dict:
    """Load meta_data.json."""

def preprocess_sinogram(sinogram) -> np.ndarray:
    """Remove batch dim."""
```

### src/generate_data.py

```python
def create_phantom(N) -> np.ndarray:
    """Shepp-Logan phantom."""

def generate_synthetic_data(...) -> dict:
    """Generate full and short-scan fan-beam sinograms."""

def save_data(data, task_dir):
    """Save to npz/json."""
```

## Pipeline Flow (main.py)

1. Load data (or generate if missing)
2. FBP reconstruction: full-scan (no Parker) and short-scan (with Parker)
3. TV-PDHG iterative reconstruction on short-scan sinogram
4. Centre-crop and normalize, compute NCC/NRMSE
5. Save reference outputs and metrics
6. Generate figures (sinograms, reconstructions, error maps)
