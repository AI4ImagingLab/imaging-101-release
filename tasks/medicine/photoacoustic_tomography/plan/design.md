# Code Design: PAT Reconstruction

## Module Architecture

```
src/
├── __init__.py
├── physics_model.py      # Forward model
├── solvers.py             # Reconstruction algorithm
├── preprocessing.py       # Data I/O
├── visualization.py       # Plotting and metrics
└── generate_data.py       # Synthetic data generation
```

## Function Signatures

### src/physics_model.py

```python
def step_function(x: np.ndarray) -> np.ndarray:
    """Heaviside step function for arrays."""

def pa_signal_single_target(
    tar_info: np.ndarray,     # shape (4,): [x, y, z, radius] in metres
    xd: np.ndarray,           # shape (n_det_x,): detector x-positions
    yd: np.ndarray,           # shape (n_det_y,): detector y-positions
    t: np.ndarray,            # shape (n_time,): time vector in seconds
    c: float = 1484.0,        # sound speed m/s
    det_len: float = 2e-3,    # detector side length m
    num_subdet: int = 25,     # sub-elements per detector
) -> np.ndarray:              # shape (n_time, n_det_x, n_det_y)
    """PA signal from one spherical target on detector array."""

def simulate_pa_signals(
    tar_info_array: np.ndarray,  # shape (n_targets, 4)
    xd, yd, t, c, det_len, num_subdet,
) -> np.ndarray:                 # shape (n_time, n_det_x, n_det_y)
    """Summed PA signals from multiple targets."""

def generate_ground_truth_image(
    tar_info_array: np.ndarray,  # shape (n_targets, 4)
    xf: np.ndarray,              # image x-coords
    yf: np.ndarray,              # image y-coords
) -> np.ndarray:                 # shape (nx, ny)
    """Binary target map on reconstruction grid."""
```

### src/solvers.py

```python
def universal_back_projection(
    prec: np.ndarray,    # shape (n_time, n_det_x, n_det_y)
    xd: np.ndarray,      # detector x-positions
    yd: np.ndarray,      # detector y-positions
    t: np.ndarray,       # time vector
    z_target: float,     # reconstruction plane z-coord
    c: float = 1484.0,
    resolution: float = 500e-6,
    det_area: float = 4e-6,
    nfft: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Universal back-projection. Returns (image, xf, yf, zf)."""
```

### src/preprocessing.py

```python
def load_raw_data(data_dir: str) -> tuple:
    """Load signals, detector_x, detector_y, time_vector."""

def load_ground_truth(data_dir: str) -> tuple:
    """Load ground_truth_image, image_x, image_y."""

def load_metadata(data_dir: str) -> dict:
    """Load meta_data.json."""
```

### src/visualization.py

```python
def compute_ncc(estimate, reference) -> float:
def compute_nrmse(estimate, reference) -> float:
def centre_crop(image, fraction=0.8) -> np.ndarray:
def plot_reconstruction(recon, xf, yf, ax=None, ...) -> Axes:
def plot_cross_sections(recon, xf, yf, axes=None) -> tuple:
def plot_signals(signals, t, xd, yd, det_indices=None, ax=None) -> Axes:
```

### src/generate_data.py

```python
def define_targets() -> tuple[np.ndarray, float]:
def define_detector_array() -> tuple[np.ndarray, np.ndarray]:
def define_time_vector() -> tuple[np.ndarray, float]:
def generate_and_save(data_dir="data", c=1484.0) -> tuple:
```

## Data Flow

```
define_targets + define_detector_array + define_time_vector
    → simulate_pa_signals (forward model)
    → save raw_data.npz, ground_truth.npz, meta_data.json
    → load_raw_data + load_metadata
    → universal_back_projection (solver)
    → compute_ncc, compute_nrmse (evaluation)
    → save reference_outputs, metrics.json
```
