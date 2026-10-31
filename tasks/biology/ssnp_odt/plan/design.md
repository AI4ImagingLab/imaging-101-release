# SSNP-ODT: Code Design

## File Structure

```
ssnp_odt/
├── main.py                   # Pipeline entry point
├── src/
│   ├── physics_model.py      # SSNP forward model (complex field output)
│   ├── preprocessing.py      # Load complex measurements and metadata
│   ├── solvers.py            # Gradient-descent reconstructor (complex MSE loss)
│   ├── visualization.py      # Amplitude/phase plots and metrics
│   └── generate_data.py      # Simulate complex ODT measurements from phantom
└── data/
    ├── raw_data.npz          # Complex fields (1, 8, 256, 256) complex128
    ├── ground_truth.npz      # Δn volume (1, 308, 256, 256) float32
    └── meta_data.json        # Imaging parameters
```

## Function Signatures

### `src/physics_model.py`

```python
@dataclass
class SSNPConfig:
    volume_shape: tuple   # (Nz, Ny, Nx)
    res: tuple            # normalised (dx·n0/λ, dy·n0/λ, dz·n0/λ)
    n0: float
    NA: float
    wavelength_um: float
    res_um: tuple
    n_angles: int

    @classmethod
    def from_metadata(cls, metadata: dict) -> SSNPConfig: ...

class SSNPForwardModel:
    def __init__(self, config: SSNPConfig, device: str = "cpu"): ...
    def forward_single(self, dn_volume: Tensor, angle_idx: int) -> Tensor:
        # Returns complex128 (Ny, Nx) — full complex field at detector
    def forward(self, dn_volume: Tensor, n_angles: int = None) -> Tensor:
        # Returns complex128 (n_angles, Ny, Nx)
```

### `src/preprocessing.py`

```python
def load_measurements(data_dir: str = "data") -> np.ndarray:
    # Returns complex128 (n_angles, Ny, Nx)

def load_metadata(data_dir: str = "data") -> dict: ...

def prepare_data(data_dir: str = "data") -> tuple:
    # Returns (measurements: complex128 (n_angles, Ny, Nx), metadata: dict)
```

### `src/solvers.py`

```python
def tv_3d(volume: Tensor, epsilon: float = 1e-8) -> Tensor:
    # 3D isotropic TV, Huber-smoothed, real scalar

class SSNPReconstructor:
    def __init__(self, n_iter: int, lr: float, tv_weight: float,
                 positivity: bool, device: str): ...
    def reconstruct(self, measurements: Tensor, model: SSNPForwardModel) -> tuple:
        # measurements: complex128 (n_angles, Ny, Nx)
        # Returns (dn_recon: ndarray (Nz, Ny, Nx), loss_history: list[float])
        # Loss per angle: sum(|E_pred - E_meas|²) / (Ny * Nx)
```

### `src/visualization.py`

```python
def plot_ri_slices(volume, slice_indices=None, title=..., vmin=None, vmax=None, figsize=None) -> Figure: ...
def plot_xz_cross_section(volume, y_index=None, title=..., vmin=None, vmax=None, figsize=...) -> Figure: ...
def plot_comparison(ground_truth, reconstruction, slice_idx=None, figsize=...) -> Figure: ...
def plot_loss_history(losses, title=..., figsize=...) -> Figure: ...
def plot_measurements(measurements, n_cols=4, figsize=None) -> Figure:
    # measurements: complex128 (n_angles, Ny, Nx)
    # Shows amplitude (row 1) and phase (row 2) for each angle
def compute_metrics(estimate, ground_truth) -> dict:
    # Returns {'nrmse': float, 'ncc': float, 'ssim': float}
    # NCC = cosine similarity (no mean subtraction)
def print_metrics_table(metrics: dict) -> None: ...
```

### `src/generate_data.py`

```python
def generate_measurements(data_dir: str = "data", device: str = "cpu") -> tuple:
    # Returns (measurements: complex128 (n_angles, Ny, Nx),
    #          phantom_dn: float64 (Nz, Ny, Nx),
    #          metadata: dict)
    # No abs() — returns complex field directly
```

## Data Flow

```
sample.tiff → _load_phantom() → phantom_dn (float64, Nz×Ny×Nx)
                                     │
                               SSNPForwardModel.forward()
                                     │
                               complex fields (n_angles×Ny×Nx)  ←── raw_data.npz
                                     │
                               SSNPReconstructor.reconstruct()
                                     │
                               dn_recon (float64, Nz×Ny×Nx)
                                     │
                               compute_metrics() vs ground_truth.npz
```

## Key Differences from IDT

| Aspect | IDT (SSNP_ODT) | ODT (ssnp_odt) |
|--------|----------------|----------------|
| Measurement | `I = \|E\|²` (real) | `E` (complex128) |
| Loss | `‖√I_pred − √I_meas‖²` | `‖E_pred − E_meas‖²` |
| `forward_single` output | real (Ny, Nx) | complex128 (Ny, Nx) |
| `raw_data.npz` dtype | float64 | complex128 |
| Visualization | intensity grid | amplitude + phase grid |
