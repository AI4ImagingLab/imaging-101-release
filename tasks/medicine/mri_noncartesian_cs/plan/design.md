# Design: Non-Cartesian MRI with L1-Wavelet CS

## Module Architecture

```
src/
├── preprocessing.py    # Data I/O from npz files
├── physics_model.py    # NUFFT forward/adjoint, density compensation, gridding
├── solvers.py          # L1-wavelet reconstruction via SigPy
├── visualization.py    # Metrics computation and plotting
└── generate_data.py    # Synthetic data generation (Shepp-Logan + radial sampling)
```

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir: str = "data") -> dict:
    """Returns {'kdata': (N,C,M), 'coord': (N,M,2), 'coil_maps': (N,C,H,W)}"""

def load_ground_truth(data_dir: str = "data") -> np.ndarray:
    """Returns phantom: (N, H, W) complex64"""

def load_metadata(data_dir: str = "data") -> dict:
    """Returns imaging parameters dict from meta_data.json"""

def prepare_data(data_dir: str = "data") -> tuple:
    """Returns (obs_data, ground_truth, metadata)"""
```

### physics_model.py

```python
def nufft_forward(image, coord) -> np.ndarray:
    """(H,W) complex image + (M,2) coords -> (M,) k-space samples"""

def nufft_adjoint(kdata, coord, image_shape) -> np.ndarray:
    """(M,) k-space + (M,2) coords -> (H,W) image"""

def multicoil_nufft_forward(image, coil_maps, coord) -> np.ndarray:
    """(H,W) image + (C,H,W) coils + (M,2) coords -> (C,M) multicoil k-space"""

def compute_density_compensation(coord, image_shape, max_iter=30) -> np.ndarray:
    """(M,2) coords -> (M,) density compensation weights"""

def gridding_reconstruct(kdata, coord, coil_maps, dcf=None) -> np.ndarray:
    """(C,M) k-space + (M,2) coords + (C,H,W) coils -> (H,W) reconstruction"""
```

### solvers.py

```python
def l1wav_reconstruct_single(kdata, coord, coil_maps, lamda=5e-5, max_iter=100) -> np.ndarray:
    """(C,M) k-space -> (H,W) complex reconstruction"""

def l1wav_reconstruct_batch(kdata, coord, coil_maps, lamda=5e-5, max_iter=100) -> np.ndarray:
    """(N,C,M) k-space -> (N,H,W) complex reconstructions"""
```

### visualization.py

```python
def compute_metrics(estimate, reference) -> dict:
    """Returns {'nrmse': float, 'ncc': float, 'psnr': float}"""

def compute_batch_metrics(estimates, references) -> dict:
    """Returns per-sample and average metrics"""

def plot_reconstruction_comparison(ground_truth, gridding, l1wav, ...) -> None:
def plot_error_maps(ground_truth, gridding, l1wav, ...) -> None:
def plot_trajectory(coord, n_spokes=None, ...) -> None:
def print_metrics_table(metrics) -> None:
```

### generate_data.py

```python
def shepp_logan_phantom(n=128) -> np.ndarray:
    """(n,n) complex phantom"""

def generate_coil_maps(n_coils, image_shape) -> np.ndarray:
    """(n_coils, H, W) complex coil maps"""

def golden_angle_radial_trajectory(n_spokes, n_readout, image_shape) -> np.ndarray:
    """(n_spokes*n_readout, 2) float trajectory"""

def generate_synthetic_data(...) -> dict:
    """Complete synthetic dataset generation"""

def save_data(data_dir, ...) -> None:
    """Save to npz files"""
```

## Pipeline Flow

```
raw_data.npz ──> preprocessing.prepare_data() ──> obs_data, ground_truth, metadata
                                                        |
                    ┌───────────────────────────────────┤
                    |                                   |
                    v                                   v
    physics_model.gridding_reconstruct()     solvers.l1wav_reconstruct_single()
                    |                                   |
                    v                                   v
            gridding result                     L1-wavelet result
                    |                                   |
                    └───────────────┬───────────────────┘
                                    v
                     visualization.compute_metrics()
                                    |
                                    v
                     visualization.plot_*() + save outputs
```
