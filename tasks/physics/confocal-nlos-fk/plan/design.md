# f-k Migration Reconstruction — Design

## Module Layout

```
src/
  __init__.py
  preprocessing.py   — load_nlos_data, preprocess_measurements, volume_axes
  physics_model.py   — nlos_forward_model, define_psf, resampling_operator
  solvers.py         — fk_reconstruction
  visualization.py   — plot_nlos_result, plot_measurement_slice, ...
```

## Function Signatures

### `src/preprocessing.py`

```python
def load_nlos_data(path: str) -> dict:
    """Returns {'meas': (Ny,Nx,Nt), 'tofgrid': (Ny,Nx) or None}"""

def preprocess_measurements(
    meas: np.ndarray,       # (Ny, Nx, Nt)
    tofgrid: np.ndarray | None,
    bin_resolution: float,
    crop: int = 512,
) -> np.ndarray:            # (crop, Ny, Nx)

def volume_axes(Nt, Ny, Nx, wall_size, bin_resolution, c=3e8) -> tuple:
    """Returns (z_axis, y_axis, x_axis) in metres."""
```

### `src/physics_model.py`

```python
def define_psf(N: int, M: int, slope: float) -> np.ndarray:
    """Returns PSF of shape (2*M, 2*N, 2*N). Used by LCT/FBP, not f-k."""

def resampling_operator(M: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mtx, mtxi) each of shape (M, M). Used by LCT/FBP, not f-k."""

def nlos_forward_model(
    rho: np.ndarray,        # (Nz, Ny, Nx)
    wall_size: float,
    bin_resolution: float,
    n_time_bins: int | None = None,
    c: float = 3e8,
) -> np.ndarray:            # (n_time_bins, Ny, Nx)
```

### `src/solvers.py`

```python
def fk_reconstruction(
    meas: np.ndarray,       # (Nt, Ny, Nx)
    wall_size: float,
    bin_resolution: float,
    c: float = 3e8,
) -> np.ndarray:            # (Nt, Ny, Nx), float64, non-negative
```

## Data Flow

```
raw_data.npz (Ny,Nx,Nt)
    │
    ▼ preprocess_measurements  (TOF align + crop + transpose)
meas (Nt, Ny, Nx)
    │
    ▼ fk_reconstruction
      1. amplitude scaling: sqrt(|meas| * z²)
      2. zero-pad to (2M, 2N, 2N)
      3. 3-D FFT + fftshift
      4. Stolt interpolation: map_coordinates(order=1)
      5. Jacobian weight + positive-depth mask
      6. ifftshift + inverse 3-D FFT
      7. |·|², unpad
vol (Nt, Ny, Nx)
    │
    ▼ evaluation/reference_outputs/
       reconstruction.npz, fk_reference.npy, metrics.json, fk.png
```

## Evaluation

Metrics against `evaluation/reference_outputs/fk_reference.npy` (original
authors' reconstruction):
- **NCC** ≈ 1.0: expected since our implementation matches the reference exactly
- **NRMSE** ≈ 0: confirms numerical parity with the reference implementation
