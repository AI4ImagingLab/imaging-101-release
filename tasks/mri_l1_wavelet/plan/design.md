# Design: Multi-Coil MRI L1-Wavelet Reconstruction

## Module Architecture

```
src/
├── preprocessing.py    # Data I/O from npz files
├── physics_model.py    # MRI forward/adjoint operators and mask generation
├── solvers.py          # L1-Wavelet reconstruction via FISTA, ported from SigPy
├── visualization.py    # Metrics computation and plotting
└── generate_data.py    # Synthetic multi-coil data generation
```

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir: str = "data") -> dict:
    """Returns {'masked_kspace': (N,C,H,W), 'sensitivity_maps': (N,C,H,W), 'undersampling_mask': (W,)}"""

def load_ground_truth(data_dir: str = "data") -> np.ndarray:
    """Returns phantom images: (N, 1, H, W) complex128"""

def load_metadata(data_dir: str = "data") -> dict:
    """Returns imaging parameters dict from meta_data.json"""

def prepare_data(data_dir: str = "data") -> tuple:
    """Returns (obs_data, ground_truth, metadata)"""
```

### physics_model.py

```python
def fft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2D FFT, ortho norm. (..., H, W) -> (..., H, W)"""

def ifft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2D IFFT, ortho norm. (..., H, W) -> (..., H, W)"""

def forward_operator(image, sensitivity_maps, mask) -> np.ndarray:
    """(H,W) complex image -> (C,H,W) undersampled k-space"""

def adjoint_operator(masked_kspace, sensitivity_maps) -> np.ndarray:
    """(C,H,W) k-space -> (H,W) MVUE estimate"""

def generate_undersampling_mask(total_lines, acceleration_ratio, ...) -> np.ndarray:
    """Generate 1-D binary undersampling mask: (total_lines,)"""
```

### solvers.py

```python
def get_wavelet_filters(wave_name="db4", data_dir="data") -> tuple:
    """Load or compute Daubechies wavelet filter coefficients."""

def wavelet_forward(image, wave_name="db4", level=None) -> (np.ndarray, list):
    """Multi-level 2D DWT. Returns (coeffs_array, coeff_info)."""

def wavelet_inverse(coeffs_array, coeff_info, original_shape, wave_name="db4") -> np.ndarray:
    """Multi-level 2D inverse DWT. Returns (H,W) image."""

def soft_thresh(lamda, x) -> np.ndarray:
    """Soft thresholding (L1 proximal operator)."""

def sense_forward(image, sensitivity_maps, mask) -> np.ndarray:
    """SENSE forward: image -> undersampled multi-coil k-space."""

def sense_adjoint(kspace, sensitivity_maps) -> np.ndarray:
    """SENSE adjoint: k-space -> image."""

def sense_normal(image, sensitivity_maps, mask) -> np.ndarray:
    """Normal operator A^H A."""

def estimate_max_eigenvalue(sensitivity_maps, mask, max_iter=30) -> float:
    """Max eigenvalue of A^H A via power iteration."""

def fista_l1_wavelet(masked_kspace, sensitivity_maps, mask, lamda, ...) -> np.ndarray:
    """FISTA solver for L1-wavelet regularized MRI. (H,W) complex."""

def l1_wavelet_reconstruct_single(masked_kspace, sensitivity_maps, lamda=1e-3, wave_name='db4') -> np.ndarray:
    """Public API: (C,H,W) k-space -> (H,W) complex reconstruction."""

def l1_wavelet_reconstruct_batch(masked_kspace, sensitivity_maps, lamda=1e-3, wave_name='db4') -> np.ndarray:
    """Public API: (N,C,H,W) k-space -> (N,H,W) complex reconstructions."""
```

### visualization.py

```python
def compute_metrics(estimate, reference) -> dict:
    """Returns {'nrmse': float, 'ncc': float, 'psnr': float}"""

def compute_batch_metrics(estimates, references) -> dict:
    """Returns per-sample and average metrics"""

def plot_reconstruction_grid(ground_truths, reconstructions, zero_filled=None, tv_recons=None, ...) -> None:
def plot_error_maps(ground_truths, reconstructions, tv_recons=None, ...) -> None:
def plot_undersampling_mask(mask, ...) -> None:
def print_metrics_table(batch_metrics, method_name='Recon') -> None:
```

### generate_data.py

```python
def generate_gaussian_csm(n_coils, image_shape, sigma=0.4, seed=42) -> np.ndarray:
    """Generate (n_coils, H, W) Gaussian coil sensitivity maps"""

def generate_phantom(image_size=128) -> np.ndarray:
    """Generate (image_size, image_size) Shepp-Logan phantom"""

def generate_data(image_size=128, n_coils=8, acceleration_ratio=8, ...) -> tuple:
    """Generate and save complete synthetic MRI dataset"""
```

## Pipeline Flow

```
generate_data.py ──► data/raw_data.npz + data/ground_truth.npz + data/meta_data.json
                                │
                                ▼
raw_data.npz ──► preprocessing.prepare_data() ──► obs_data, ground_truth, metadata
                                                        │
                                          ┌─────────────┼─────────────┐
                                          ▼             ▼             ▼
                                   adjoint_operator  l1_wavelet_*  tv_*
                                   (zero-filled)     (L1-Wavelet)  (TV comparison)
                                          │             │             │
                                          └─────────────┼─────────────┘
                                                        ▼
                                           visualization.compute_batch_metrics()
                                                        │
                                                        ▼
                                           visualization.plot_*() + save outputs
```
