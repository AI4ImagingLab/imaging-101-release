# Code Architecture: X-ray Ptychography Reconstruction

## Module Overview

```
main.py                   -- Entry point: orchestrates the full pipeline
src/
  __init__.py             -- Package marker
  preprocessing.py        -- Data loading and array preparation
  physics_model.py        -- Ptychographic forward model (patch extraction, FFT propagation)
  solvers.py              -- ePIE iterative reconstruction
  visualization.py        -- Metric computation and plotting
```

## Data Flow

```
raw_data.npz ──> preprocessing.load_raw_data()
                     │
                     ├── diffraction_patterns (516, 128, 128) float32
                     ├── scan_positions (516, 2) float32
                     └── probe_guess (1, 1, 1, 128, 128) complex64
                              │
                              v
                   preprocessing.shift_scan_positions(scan, offset=20)
                   preprocessing.add_probe_modes(probe, n_modes=1)
                   preprocessing.initialize_psi(scan, probe.shape)
                              │
                              v
                   solvers.reconstruct(data, scan, probe, psi,
                                       num_iter=64, num_batch=7)
                     │ uses: physics_model.extract_patches()
                     │ uses: physics_model.insert_patches()
                     │ uses: physics_model.validate_inputs()
                     │
                     v
                   result['psi']     (D, H, W) complex64
                   result['probe']   (1, 1, S, W, H) complex64
                   result['costs']   list[list[float]]
                              │
                              v
                   visualization.compute_metrics(est_phase, ref_phase)
                   visualization.plot_phase(psi[0])
                   visualization.plot_cost_curve(costs)
                              │
                              v
                   output/
                     reconstructed_object.npy
                     reconstructed_probe.npy
                     costs.npy
                     metrics.json
                     reconstructed_object.png
                     cost_curve.png
```

## Function Signatures

### src/preprocessing.py

```python
def load_raw_data(data_dir: str) -> dict:
    """Load raw_data.npz, return dict with batch dim removed."""

def load_metadata(data_dir: str) -> dict:
    """Load meta_data.json as a dict."""

def shift_scan_positions(scan: np.ndarray, offset: float = 20.0) -> np.ndarray:
    """Shift scan so min coordinate = offset. Returns (N, 2) float32."""

def initialize_psi(scan: np.ndarray, probe_shape: tuple,
                   n_slices: int = 1, buffer: int = 2,
                   fill_value: complex = 0.5+0j) -> np.ndarray:
    """Create (D, H, W) complex64 object array sized to fit scan + probe."""

def add_probe_modes(probe: np.ndarray, n_modes: int = 1) -> np.ndarray:
    """Initialize additional probe modes with random phase and same amplitude."""
```

### src/physics_model.py

```python
def extract_patches(psi: np.ndarray, scan: np.ndarray,
                    probe_shape: tuple) -> np.ndarray:
    """Extract object patches at scan positions. Returns (N, ph, pw) complex64."""

def insert_patches(grad_patches: np.ndarray, scan: np.ndarray,
                   obj_shape: tuple, probe_shape: tuple) -> np.ndarray:
    """Accumulate patch gradients back into object space."""

def forward(psi: np.ndarray, probe: np.ndarray,
            scan: np.ndarray) -> np.ndarray:
    """Ptychographic forward model: probe * patch -> FFT -> intensity.
    Returns (N, W, H) float32 simulated diffraction intensities."""

def validate_inputs(data, scan, probe, psi) -> None:
    """Check shapes and dtypes; raise ValueError on mismatch."""
```

### src/solvers.py

```python
def reconstruct(data: np.ndarray, scan: np.ndarray,
                probe: np.ndarray, psi: np.ndarray,
                num_iter: int = 64, num_batch: int = 7,
                alpha: float = 0.05) -> dict:
    """Run ePIE reconstruction with Gaussian noise model and
    least-squares preconditioned updates.
    Returns {'psi': (D,H,W) complex64, 'probe': (1,1,S,W,H) complex64,
             'costs': list[list[float]]}"""
```

### src/visualization.py

```python
def compute_ncc(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Cosine similarity between flattened arrays."""

def compute_nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    """RMSE normalized by dynamic range of reference."""

def compute_metrics(estimate_phase: np.ndarray,
                    reference_phase: np.ndarray) -> dict:
    """Return {'ncc': float, 'nrmse': float}."""

def plot_phase(psi_2d, title=..., save_path=None):
    """Plot complex array as amplitude + phase side-by-side."""

def plot_cost_curve(costs, title=..., save_path=None):
    """Semilog plot of per-iteration cost."""
```
