# Design: Multi-Coil MRI TV Reconstruction

## Module Architecture

```
src/
├── preprocessing.py    # Data I/O from npz files
├── physics_model.py    # MRI forward/adjoint operators and mask generation
├── solvers.py          # TV reconstruction via PDHG (Chambolle-Pock), ported from SigPy
└── visualization.py    # Metrics computation and plotting
```

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir: str = "data") -> dict:
    """Returns {'masked_kspace': (N,C,H,W), 'sensitivity_maps': (N,C,H,W), 'undersampling_mask': (W,)}"""

def load_ground_truth(data_dir: str = "data") -> np.ndarray:
    """Returns MVUE images: (N, 1, H, W) complex64"""

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
def sense_forward(image, sensitivity_maps, mask_weights) -> np.ndarray:
    """SENSE forward: image -> weighted multi-coil k-space. (C,H,W)"""

def sense_adjoint(y, sensitivity_maps, mask_weights) -> np.ndarray:
    """SENSE adjoint: weighted k-space -> image. (H,W)"""

def finite_difference(x) -> np.ndarray:
    """Circular finite differences for TV. (2,H,W)"""

def finite_difference_adjoint(grad) -> np.ndarray:
    """Adjoint of finite difference (negative divergence). (H,W)"""

def stacked_forward(x, sensitivity_maps, mask_weights) -> tuple:
    """Stacked [Sense; G] forward. Returns (y_sense, y_grad)."""

def stacked_adjoint(u_sense, u_grad, sensitivity_maps, mask_weights) -> np.ndarray:
    """Adjoint of stacked operator. (H,W)"""

def soft_thresh(lamda, x) -> np.ndarray:
    """Soft thresholding (L1 proximal)."""

def prox_l2_reg(sigma, u, y) -> np.ndarray:
    """Proximal of data fidelity: (u + sigma*y) / (1+sigma)."""

def prox_l1_conj(sigma, u, lamda) -> np.ndarray:
    """Proximal of conjugate of L1 (Moreau decomposition)."""

def estimate_max_eigenvalue(sensitivity_maps, mask_weights, img_shape, ...) -> float:
    """Max eigenvalue of A^H A via power iteration for step-size selection."""

def pdhg_tv_recon(masked_kspace, sensitivity_maps, lamda, max_iter=100) -> np.ndarray:
    """PDHG (Chambolle-Pock) TV-regularized MRI reconstruction. (H,W) complex."""

def tv_reconstruct_single(masked_kspace, sensitivity_maps, lamda=1e-4) -> np.ndarray:
    """Public API: (C,H,W) k-space -> (H,W) complex reconstruction."""

def tv_reconstruct_batch(masked_kspace, sensitivity_maps, lamda=1e-4) -> np.ndarray:
    """Public API: (N,C,H,W) k-space -> (N,H,W) complex reconstructions."""
```

### visualization.py

```python
def compute_metrics(estimate, reference) -> dict:
    """Returns {'nrmse': float, 'ncc': float, 'psnr': float}"""

def compute_batch_metrics(estimates, references) -> dict:
    """Returns per-sample and average metrics"""

def plot_reconstruction_grid(ground_truths, reconstructions, zero_filled=None, ...) -> None:
def plot_error_maps(ground_truths, reconstructions, ...) -> None:
def plot_undersampling_mask(mask, ...) -> None:
def print_metrics_table(batch_metrics) -> None:
```

## Pipeline Flow

```
raw_data.npz ──► preprocessing.prepare_data() ──► obs_data, ground_truth, metadata
                                                        │
                                                        ▼
                                              solvers.tv_reconstruct_batch()
                                                        │
                                                        ▼
                                           visualization.compute_batch_metrics()
                                                        │
                                                        ▼
                                           visualization.plot_*() + save outputs
```
