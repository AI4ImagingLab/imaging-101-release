# Design: Sparse-View CT Reconstruction

## Module Architecture

```
src/
├── preprocessing.py     # Data I/O
├── physics_model.py     # Radon transform, FBP, noise
├── solvers.py           # TV-PDHG reconstruction
├── visualization.py     # Metrics and plotting
└── generate_data.py     # Synthetic phantom + sinogram generation
```

## Function Signatures

### src/preprocessing.py

```python
def load_ground_truth(data_dir: str) -> np.ndarray:
    """Returns phantom (256, 256)."""

def load_raw_data(data_dir: str) -> dict:
    """Returns dict with sinogram_sparse, sinogram_full, angles_sparse, angles_full."""

def load_metadata(data_dir: str) -> dict:
    """Returns imaging parameters from meta_data.json."""
```

### src/physics_model.py

```python
def radon_transform(image: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Forward Radon transform. Returns sinogram (n_detectors, n_angles)."""

def filtered_back_projection(sinogram: np.ndarray, angles_deg: np.ndarray,
                              output_size: int = None,
                              filter_name: str = "ramp") -> np.ndarray:
    """FBP reconstruction. Returns image (output_size, output_size)."""

def add_gaussian_noise(sinogram: np.ndarray, noise_std: float,
                       rng=None) -> np.ndarray:
    """Add relative Gaussian noise to sinogram."""
```

### src/solvers.py

```python
def tv_reconstruction(sinogram: np.ndarray, angles_deg: np.ndarray,
                       output_size: int, lam: float = 0.01,
                       n_iter: int = 300,
                       positivity: bool = True) -> tuple[np.ndarray, list]:
    """TV-PDHG reconstruction. Returns (image, loss_history)."""
```

Internal helpers: `_gradient_2d`, `_divergence_2d`, `_prox_tv`.

### src/visualization.py

```python
def compute_nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
def compute_ncc(estimate: np.ndarray, reference: np.ndarray) -> float:
def compute_ssim(estimate: np.ndarray, reference: np.ndarray) -> float:
def centre_crop(image: np.ndarray, crop_fraction: float = 0.8) -> np.ndarray:
def plot_reconstruction_comparison(phantom, fbp_recon, tv_recon, save_path=None):
def plot_sinograms(sinogram_full, sinogram_sparse, angles_full, angles_sparse, save_path=None):
def plot_loss_history(loss_history, save_path=None):
```

### src/generate_data.py

```python
def generate_phantom(image_size: int = 256) -> np.ndarray:
def generate_data(task_dir: str, image_size: int = 256, ...):
```

## Data Flow

```
generate_data.py
    → data/ground_truth.npz (phantom)
    → data/raw_data.npz (sinograms, angles)
    → data/meta_data.json

main.py
    → preprocessing.load_*() → phantom, sinograms, angles
    → physics_model.filtered_back_projection() → fbp_recon
    → solvers.tv_reconstruction() → tv_recon, loss_history
    → visualization.compute_*() → metrics
    → visualization.plot_*() → figures
    → evaluation/reference_outputs/reconstructions.npz
    → evaluation/metrics.json
```
