# Design: PnP-CASSI Code Architecture

## Module Structure

```
src/
├── __init__.py
├── preprocessing.py      # Data loading and measurement generation
├── physics_model.py      # Forward/transpose model and shift operations
├── solvers.py            # GAP/ADMM solvers + neural network denoiser
└── visualization.py      # Metrics and plotting utilities
```

## Function Signatures

### src/physics_model.py

```python
def A(x: ndarray, Phi: ndarray) -> ndarray:
    """Forward model: (H, W+s*(nC-1), nC) -> (H, W+s*(nC-1))"""

def At(y: ndarray, Phi: ndarray) -> ndarray:
    """Transpose model: (H, W) -> (H, W, nC)"""

def shift(inputs: ndarray, step: int) -> ndarray:
    """Apply spectral dispersion: (H, W, nC) -> (H, W+(nC-1)*step, nC)"""

def shift_back(inputs: ndarray, step: int) -> ndarray:
    """Reverse spectral dispersion: (H, W+(nC-1)*step, nC) -> (H, W, nC)"""
```

### src/preprocessing.py

```python
def load_meta_data(meta_path: str) -> dict:
    """Load imaging parameters from meta_data.json."""

def load_mask(mask_path: str, r: int, c: int, nC: int, step: int) -> ndarray:
    """Load mask and build 3D sensing matrix Phi. Shape: (r, c+step*(nC-1), nC)"""

def load_ground_truth(data_path: str) -> ndarray:
    """Load ground truth HSI. Shape: (r, c, nC)"""

def generate_measurement(truth: ndarray, mask_3d: ndarray, step: int) -> ndarray:
    """Generate compressed measurement. Shape: (r, c+step*(nC-1))"""
```

### src/solvers.py

```python
class HSI_SDeCNN(nn.Module):
    """7-band spectral denoiser. Input: (B, 7, H, W) + sigma -> (B, 1, H, W)"""

def load_denoiser(checkpoint_path: str, device=None) -> tuple[HSI_SDeCNN, torch.device]:
    """Load pretrained denoiser."""

def TV_denoiser(x: ndarray, _lambda: float, n_iter_max: int) -> ndarray:
    """Total variation denoiser."""

def gap_denoise(y, Phi, _lambda, accelerate, iter_max, sigma, tv_iter_max,
                x0=None, X_orig=None, checkpoint_path=None, show_iqa=True
                ) -> tuple[ndarray, list[float], list[float]]:
    """GAP reconstruction. Returns (x_shifted, psnr_list, ssim_list)."""

def admm_denoise(y, Phi, _lambda, gamma, iter_max, sigma, tv_weight, tv_iter_max,
                 x0=None, X_orig=None, checkpoint_path=None, show_iqa=True
                 ) -> tuple[ndarray, list[float], list[float]]:
    """ADMM reconstruction. Returns (theta_shifted, psnr_list, ssim_list)."""
```

### src/visualization.py

```python
def psnr(ref: ndarray, img: ndarray) -> float:
    """Peak signal-to-noise ratio in dB."""

def ssim(img1: ndarray, img2: ndarray) -> float:
    """Single-channel SSIM."""

def calculate_ssim(img1: ndarray, img2: ndarray, border: int = 0) -> float:
    """Mean SSIM across all spectral channels."""

def plot_spectral_bands(data, title, wavelength_start, wavelength_step,
                        vmin=None, vmax=None, save_path=None):
    """Plot all spectral bands in a grid layout."""

def plot_comparison(truth, recon, psnr_val, ssim_val,
                    wavelength_start, wavelength_step, save_path=None):
    """Side-by-side GT vs reconstruction for all bands."""

def plot_measurement(meas, title, save_path=None):
    """Plot compressed measurement."""
```

## Data Flow

```
data/*.mat ──> preprocessing.py ──> (truth, mask_3d, meas)
                                        │
                                        v
                              solvers.gap_denoise()
                                  │
                     ┌────────────┼────────────┐
                     v            v             v
              physics_model   TV_denoiser   HSI_SDeCNN
              (A, At, shift)  (skimage)     (PyTorch)
                     │            │             │
                     └────────────┼─────────────┘
                                  v
                            shift_back(x)
                                  │
                                  v
                          visualization.py ──> plots + metrics.json
```
