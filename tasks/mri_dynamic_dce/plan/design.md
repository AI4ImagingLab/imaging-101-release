# Code Design: DCE-MRI Temporal TV Reconstruction

## Module Architecture

```
src/
├── generate_data.py     # Synthetic phantom + k-space generation
├── preprocessing.py     # Data loading and preparation
├── physics_model.py     # Forward/adjoint MRI operators
├── solvers.py           # Zero-fill and temporal TV solvers
└── visualization.py     # Metrics computation and plotting
```

## Function Signatures

### src/generate_data.py

```python
def gamma_variate(t, A, t_arrival, t_peak, alpha) -> ndarray:
    """Gamma-variate contrast uptake curve."""

def make_dynamic_phantom(N=128, T=20) -> tuple[ndarray, ndarray]:
    """Create (T, N, N) dynamic phantom and (T,) time points."""

def generate_variable_density_mask(N, sampling_rate, center_fraction, seed) -> ndarray:
    """Generate (N, N) 2D variable-density random undersampling mask."""

def generate_dce_data(N, T, sampling_rate, center_fraction, noise_level, seed) -> tuple:
    """Generate complete dataset: phantom, kspace, masks, time_points."""

def save_data(output_dir, phantom, kspace, masks, time_points, **params) -> None:
    """Save in benchmark format (raw_data.npz, ground_truth.npz, meta_data.json)."""
```

### src/preprocessing.py

```python
def load_observation(data_dir) -> dict:
    """Load undersampled_kspace (T,N,N) complex, undersampling_masks (T,N,N)."""

def load_ground_truth(data_dir) -> dict:
    """Load dynamic_images (T,N,N), time_points (T,)."""

def load_metadata(data_dir) -> dict:
    """Load meta_data.json."""

def prepare_data(data_dir) -> tuple[dict, dict, dict]:
    """Load all components."""
```

### src/physics_model.py

```python
def fft2c(x) -> ndarray:
    """Centered 2D FFT, ortho normalized."""

def ifft2c(x) -> ndarray:
    """Centered 2D IFFT, ortho normalized."""

def forward_single(image, mask) -> ndarray:
    """Single-frame forward: (N,N) -> (N,N) complex."""

def adjoint_single(kspace, mask=None) -> ndarray:
    """Single-frame adjoint: (N,N) complex -> (N,N) complex."""

def forward_dynamic(images, masks) -> ndarray:
    """Dynamic forward: (T,N,N) -> (T,N,N) complex."""

def adjoint_dynamic(kspace, masks=None) -> ndarray:
    """Dynamic adjoint: (T,N,N) complex -> (T,N,N) complex."""

def normal_operator_dynamic(images, masks) -> ndarray:
    """Normal operator A^H A: (T,N,N) -> (T,N,N) complex."""
```

### src/solvers.py

```python
def zero_filled_recon(kspace, masks=None) -> ndarray:
    """Per-frame IFFT magnitude: (T,N,N) complex -> (T,N,N) float."""

def temporal_tv_pgd(kspace, masks, lamda, max_iter, tol, verbose) -> tuple[ndarray, dict]:
    """Temporal TV via PGD/ISTA: returns (T,N,N) recon + info dict."""

def temporal_tv_admm(kspace, masks, lamda, rho, max_iter, tol, verbose) -> tuple[ndarray, dict]:
    """Temporal TV via ADMM: returns (T,N,N) recon + info dict."""
```

### src/visualization.py

```python
def compute_nrmse(recon, reference) -> float
def compute_ncc(recon, reference) -> float
def compute_psnr(recon, reference) -> float
def compute_frame_metrics(recon, reference) -> dict
def print_metrics_table(metrics) -> None
def plot_frame_comparison(gt, zero_fill, tv_recon, frames, save_path) -> None
def plot_time_activity_curves(gt, zf, tv, time_points, roi_center, roi_radius, save_path) -> None
def plot_convergence(loss_history, save_path) -> None
def plot_error_maps(gt, tv_recon, frames, save_path) -> None
```

## Data Flow

```
generate_data.py
    └─> data/raw_data.npz       (1, T, N, N) complex kspace + masks
    └─> data/ground_truth.npz   (1, T, N, N) float phantom + time_points
    └─> data/meta_data.json     imaging parameters

main.py
    preprocessing.py  ─> load data
    solvers.py        ─> zero_filled_recon(kspace)
    solvers.py        ─> temporal_tv_pgd(kspace, masks, ...)
    visualization.py  ─> compute_frame_metrics, plots
    └─> evaluation/reference_outputs/
```
