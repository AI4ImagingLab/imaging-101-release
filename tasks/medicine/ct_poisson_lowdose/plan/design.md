# Design: Low-Dose CT Reconstruction with Poisson Noise

## Module Architecture

```
src/
├── __init__.py
├── generate_data.py      # Synthetic data generation
├── preprocessing.py      # Data loading and format conversion
├── physics_model.py      # Forward model and noise simulation
├── solvers.py            # SVMBIR reconstruction wrappers
└── visualization.py      # Metrics and plotting
```

## Function Signatures

### src/generate_data.py

```python
def generate_phantom(image_size: int = 256, scale: float = 0.02) -> np.ndarray:
    """Returns 2D Shepp-Logan phantom, shape (H, W)."""

def generate_angles(num_views: int = 256, angle_range: float = np.pi) -> np.ndarray:
    """Returns 1D angle array in radians, shape (V,)."""

def forward_project(phantom_3d: np.ndarray, angles: np.ndarray,
                    num_channels: int = 367) -> np.ndarray:
    """Returns sinogram shape (V, 1, C) via SVMBIR projector."""

def simulate_poisson_sinogram(sino_clean: np.ndarray, I0: float,
                               rng: np.random.RandomState) -> tuple:
    """Returns (sino_noisy, weights, photon_counts), all same shape as sino_clean."""

def generate_all(output_dir: str) -> None:
    """Generate and save all data files to output_dir."""
```

### src/preprocessing.py

```python
def load_ground_truth(data_dir: str) -> np.ndarray:
    """Returns 2D phantom, shape (H, W)."""

def load_raw_data(data_dir: str) -> dict:
    """Returns dict with sinograms, weights, angles (batch dim stripped)."""

def load_metadata(data_dir: str) -> dict:
    """Returns imaging parameter dict from meta_data.json."""

def sinogram_to_svmbir(sino_2d: np.ndarray) -> np.ndarray:
    """(V, C) -> (V, 1, C) for SVMBIR."""

def weights_to_svmbir(weights_2d: np.ndarray) -> np.ndarray:
    """(V, C) -> (V, 1, C) for SVMBIR."""
```

### src/physics_model.py

```python
def radon_forward(image: np.ndarray, angles: np.ndarray,
                  num_channels: int) -> np.ndarray:
    """2D image -> 2D sinogram (V, C) via SVMBIR projector."""

def radon_backproject(sinogram: np.ndarray, angles: np.ndarray,
                      num_rows: int, num_cols: int) -> np.ndarray:
    """2D sinogram -> 2D back-projected image (H, W)."""

def poisson_pre_log_model(sinogram_clean: np.ndarray, I0: float) -> np.ndarray:
    """Clean sinogram -> expected photon counts: I0 * exp(-sino)."""

def simulate_poisson_noise(transmission: np.ndarray,
                           rng: np.random.RandomState) -> np.ndarray:
    """Expected counts -> noisy counts via Poisson sampling."""

def post_log_transform(photon_counts: np.ndarray, I0: float) -> np.ndarray:
    """Photon counts -> post-log sinogram: -log(I / I0)."""

def compute_poisson_weights(photon_counts: np.ndarray) -> np.ndarray:
    """Photon counts -> PWLS weights (w_i = I_i)."""
```

### src/solvers.py

```python
def svmbir_recon_unweighted(sinogram: np.ndarray, angles: np.ndarray,
                             num_rows: int, num_cols: int,
                             verbose: int = 1) -> np.ndarray:
    """Unweighted SVMBIR recon, returns 2D image (H, W)."""

def svmbir_recon_pwls(sinogram: np.ndarray, angles: np.ndarray,
                       weights: np.ndarray, num_rows: int, num_cols: int,
                       verbose: int = 1) -> np.ndarray:
    """PWLS SVMBIR recon with Poisson weights, returns 2D image (H, W)."""
```

### src/visualization.py

```python
def compute_ncc(estimate: np.ndarray, reference: np.ndarray) -> float:
def compute_nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
def centre_crop(image: np.ndarray, fraction: float = 0.8) -> np.ndarray:
def plot_reconstruction_comparison(phantom, recon_uw, recon_pwls, save_path=None):
def plot_sinogram_comparison(sino_clean, sino_low, sino_high, save_path=None):
def plot_dose_comparison(phantom, recon_low_uw, recon_low_pwls,
                          recon_high_pwls, save_path=None):
def plot_error_maps(phantom, recon_uw, recon_pwls, save_path=None):
```

## Data Flow

```
generate_data.py
  phantom -> svmbir.project -> sino_clean
  sino_clean -> Poisson noise -> sino_noisy, weights
  Save: ground_truth.npz, raw_data.npz, meta_data.json

main.py
  load data -> unweighted recon -> PWLS recon -> metrics -> save
```
