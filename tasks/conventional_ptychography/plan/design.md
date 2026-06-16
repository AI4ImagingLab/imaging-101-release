# Design: Conventional Ptychography Code Architecture

## Module Overview

```
tasks/conventional_ptychography/
├── main.py                   # Pipeline entry point
├── src/
│   ├── generate_data.py      # Synthetic CP dataset generation
│   ├── preprocessing.py      # Data loading and reconstruction initialization
│   ├── physics_model.py      # CP forward model (exit wave, propagation)
│   ├── solvers.py            # mPIE reconstruction solver
│   ├── utils.py              # fft2c, ifft2c, circ, aspw, etc.
│   └── visualization.py      # Complex image display (HSV encoding)
├── data/
│   ├── raw_data.npz          # Generated CP dataset (ptychogram, encoder)
│   ├── meta_data.json        # Physical parameters
│   └── ground_truth.npz      # Ground-truth complex object
└── evaluation/
    └── reference_outputs/
        ├── recon.hdf5        # Saved reconstruction (object, probe, error)
        ├── metrics.json      # Final error and quality metrics
        └── reconstruction_summary.png
```

## Function Signatures

### src/generate_data.py

```python
def generate_probe(wavelength, zo, Nd, dxd, f=8e-3) -> (ndarray[Nd,Nd,complex], float):
    """Return focused probe and dxp."""

def generate_object(No, dxp, d=1e-3, b=33) -> ndarray[No,No,complex]:
    """Return complex spiral-zone-plate object."""

def generate_scan_grid(No, Np, dxp, num_points=100, radius=150, offset=(50,20))
    -> (ndarray[J,2,int], ndarray[J,2,float]):
    """Return pixel positions and physical encoder [m]."""

def generate_ptychogram(obj, probe, positions, Nd, bit_depth=14, seed=42)
    -> ndarray[J,Nd,Nd,float32]:
    """Simulate diffraction data with Poisson noise."""

def save_dataset(filepath, ptychogram, encoder, wavelength, zo, dxd, Nd, No, epd):
    """Write HDF5 file in PtyLab format."""

def main(output_path=None) -> Path:
    """Generate and save complete CP dataset."""
```

### src/preprocessing.py

```python
def load_experimental_data(data_dir) -> PtyData:
    """Load raw_data.npz + meta_data.json → PtyData."""

def setup_reconstruction(data: PtyData, no_scale=1.0, seed=0) -> PtyState:
    """Initialize object (ones + noise) and probe (circ + quadratic phase)."""

def setup_params() -> SimpleNamespace:
    """Return standard CP params: Fraunhofer propagator, probe smoothing, L2 reg."""

def setup_monitor(figure_update_freq=10) -> SimpleNamespace:
    """Return low-verbosity monitor (no interactive GUI)."""

def save_results(state: PtyState, filepath) -> None:
    """Save object, probe, error to HDF5 (PtyLab-compatible 6D schema)."""
```

### src/physics_model.py

```python
def get_object_patch(obj, position, Np) -> ndarray[Np,Np,complex]:
    """Extract object patch at (row, col)."""

def compute_exit_wave(probe, object_patch) -> ndarray[complex]:
    """ψ = P * O_j  (thin-element multiplication)."""

def fraunhofer_propagate(esw) -> ndarray[complex]:
    """Ψ = FT{ψ}  (far-field propagation)."""

def asp_propagate(esw, zo, wavelength, L) -> ndarray[complex]:
    """Ψ = ASP{ψ}  (angular-spectrum propagation)."""

def compute_detector_intensity(detector_field) -> ndarray[float]:
    """I = |Ψ|²."""

def forward_model(probe, obj, position, Np, propagator='Fraunhofer',
                  zo=None, wavelength=None, L=None) -> (ndarray, ndarray):
    """Full pipeline: position → (intensity, esw)."""
```

### src/solvers.py

```python
def run_mpie(state: PtyState, data: PtyData, params, monitor=None,
             num_iterations=350, beta_probe=0.25, beta_object=0.25,
             iterations_per_round=50, seed=0) -> PtyState:
    """mPIE with 7-round alternating L2 regularization schedule."""

def compute_reconstruction_error(ptychogram, ptychogram_est) -> float:
    """Normalized amplitude RMSE."""
```

### src/visualization.py

```python
def complex_to_hsv(arr, max_amp=None) -> ndarray[Ny,Nx,3]:
    """Complex → HSV-encoded RGB (hue=phase, value=amplitude)."""

def plot_complex_image(arr, ax, title='', pixel_size_um=None, max_amp=None):
    """Display complex array as HSV image on given Axes."""

def add_colorwheel(ax, size=64):
    """Add phase colorwheel inset to axes."""

def plot_diffraction_pattern(pattern, ax, title='', log_scale=True):
    """Show single diffraction frame (log scale)."""

def plot_scan_grid(positions, Np, obj_shape, ax, title=''):
    """Overlay scan centers on object FOV."""

def plot_reconstruction_summary(obj, probe, error_history, ptychogram_sample=None,
                                 pixel_size_um=None, figsize=(14,4)) -> Figure:
    """4-panel: object | probe | convergence | diffraction."""
```

## Data Flow

```
raw_data.npz + meta_data.json
    ↓ load_experimental_data()
PtyData (ptychogram, encoder, wavelength, zo, dxd, …)
    ↓ setup_reconstruction()
PtyState (probe=circ+phase, object=ones+noise)
    ↓ run_mpie() [7 rounds × 50 iter]
PtyState (converged probe & object)
    ↓ save_results()
recon.hdf5 + metrics.json
    ↓ plot_reconstruction_summary()
reconstruction_summary.png
```
