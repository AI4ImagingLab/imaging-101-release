# Design: Ultrasound Speed-of-Sound Tomography

## Module Architecture

```
src/
├── preprocessing.py     # Data I/O
├── physics_model.py     # Radon forward model, FBP, noise
├── solvers.py           # SART + TV-PDHG reconstruction
├── visualization.py     # Metrics and plotting
└── generate_data.py     # Synthetic phantom + sinogram generation
```

## Function Signatures

### src/preprocessing.py

```python
def load_ground_truth(data_dir: str) -> dict:
    """Returns dict with 'sos_phantom' (H,W) and 'slowness_perturbation' (H,W)."""

def load_raw_data(data_dir: str) -> dict:
    """Returns dict with sinogram, sinogram_clean, sinogram_full, angles, angles_full."""

def load_metadata(data_dir: str) -> dict:
    """Returns imaging parameters from meta_data.json."""
```

### src/physics_model.py

```python
def radon_forward(slowness: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Forward Radon transform. Returns sinogram (n_detectors, n_angles)."""

def filtered_back_projection(sinogram: np.ndarray, angles_deg: np.ndarray,
                              output_size: int = None,
                              filter_name: str = "ramp") -> np.ndarray:
    """FBP reconstruction. Returns image (output_size, output_size)."""

def adjoint_projection(sinogram: np.ndarray, angles_deg: np.ndarray,
                       output_size: int) -> np.ndarray:
    """Unfiltered backprojection (adjoint). Returns image (output_size, output_size)."""

def slowness_to_sos(slowness: np.ndarray) -> np.ndarray:
    """Convert slowness (s/m) to speed of sound (m/s)."""

def sos_to_slowness(sos: np.ndarray) -> np.ndarray:
    """Convert speed of sound (m/s) to slowness (s/m)."""

def add_gaussian_noise(sinogram: np.ndarray, noise_std: float,
                       rng=None) -> np.ndarray:
    """Add relative Gaussian noise to sinogram."""
```

### src/solvers.py

```python
def sart_reconstruction(sinogram: np.ndarray, angles_deg: np.ndarray,
                        output_size: int, n_iter: int = 30,
                        relaxation: float = 0.15) -> tuple[np.ndarray, list]:
    """SART reconstruction. Returns (image, loss_history)."""

def tv_pdhg_reconstruction(sinogram: np.ndarray, angles_deg: np.ndarray,
                            output_size: int, lam: float = 1e-6,
                            n_iter: int = 300,
                            positivity: bool = False) -> tuple[np.ndarray, list]:
    """TV-PDHG reconstruction. Returns (image, loss_history)."""
```

Internal helpers: `_gradient_2d`, `_divergence_2d`, `_prox_tv_iso`.

### src/visualization.py

```python
def compute_nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
def compute_ncc(estimate: np.ndarray, reference: np.ndarray) -> float:
def compute_ssim(estimate: np.ndarray, reference: np.ndarray) -> float:
def centre_crop(image: np.ndarray, crop_fraction: float = 0.8) -> np.ndarray:
def plot_sos_comparison(sos_gt, sos_fbp, sos_tv, save_path=None):
def plot_sinograms(sinogram, sinogram_full, angles, angles_full, save_path=None):
def plot_loss_history(loss_history_sart, loss_history_tv, save_path=None):
def plot_sos_profiles(sos_gt, sos_fbp, sos_tv, row=None, save_path=None):
```

### src/generate_data.py

```python
def generate_sos_phantom(image_size: int = 128) -> np.ndarray:
def generate_data(task_dir: str, image_size: int = 128, ...):
```

## Data Flow

```
generate_data.py
    -> data/ground_truth.npz (sos_phantom, slowness_perturbation)
    -> data/raw_data.npz (sinograms, angles)
    -> data/meta_data.json

main.py
    -> preprocessing.load_*() -> phantom, sinograms, angles
    -> physics_model.filtered_back_projection() -> fbp_recon
    -> solvers.sart_reconstruction() -> sart_recon, loss_sart
    -> solvers.tv_pdhg_reconstruction() -> tv_recon, loss_tv
    -> convert slowness perturbation -> SoS (m/s)
    -> visualization.compute_*() -> metrics
    -> visualization.plot_*() -> figures
    -> evaluation/reference_outputs/reconstructions.npz
    -> evaluation/metrics.json
```
