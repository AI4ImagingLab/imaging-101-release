# Code Architecture

## File Structure

```
main.py                           # Pipeline orchestration
src/
  __init__.py
  preprocessing.py                # Data loading, flat-field correction, log transform
  physics_model.py                # Radon transform forward/adjoint, center finding
  solvers.py                      # FBP: ramp filter, back-projection, circular mask
  visualization.py                # Plotting utilities and quality metrics
```

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir: str) -> dict:
    """Load raw_data.npz, return dict with projections, flat_field, dark_field, theta
    (batch dimension removed)."""

def load_metadata(data_dir: str) -> dict:
    """Load meta_data.json, return imaging parameters dict."""

def normalize(projections, flat_field, dark_field) -> ndarray:
    """Flat-field correction: (proj - dark_avg) / (flat_avg - dark_avg).
    Returns float64 normalised transmission."""

def minus_log(projections) -> ndarray:
    """Beer-Lambert: -log(clip(proj, 1e-12, None)). Returns sinogram data."""
```

### physics_model.py

```python
class ParallelBeamProjector:
    """Forward and adjoint operators for parallel-beam CT."""

    def __init__(self, n_pixels: int, n_detector: int, theta: ndarray): ...
    def forward(self, image: ndarray) -> ndarray:
        """Radon transform: (n, n) -> (n_angles, n_det)."""
    def adjoint(self, sinogram: ndarray) -> ndarray:
        """Back-projection: (n_angles, n_det) -> (n, n)."""

def find_rotation_center(sinogram, theta, init=None, tol=0.5) -> float:
    """Cross-correlation of opposing projections + variance-based refinement."""
```

### solvers.py

```python
def ramp_filter(n_detector: int) -> ndarray:
    """Ram-Lak filter |omega| in frequency domain, shape (n_fft,)."""

def filter_sinogram(sinogram: ndarray, filt=None) -> ndarray:
    """Apply ramp filter to each projection row via FFT."""

def back_project(sinogram: ndarray, theta: ndarray, n_pixels: int) -> ndarray:
    """Pixel-driven back-projection with linear interpolation."""

def filtered_back_projection(sinogram, theta, n_pixels) -> ndarray:
    """Full FBP: filter_sinogram + back_project."""

def circular_mask(image: ndarray, ratio=0.95) -> ndarray:
    """Zero out pixels outside circle of radius ratio * n/2."""
```

### visualization.py

```python
def compute_metrics(estimate, reference) -> dict:
    """Return {'nrmse': float, 'ncc': float}."""

def plot_sinogram(sinogram, ax=None, title="Sinogram"):
    """Display sinogram as image."""

def plot_reconstruction(image, ax=None, title="Reconstruction"):
    """Display reconstruction as grayscale image."""

def plot_comparison(images, titles, suptitle="Comparison"):
    """Side-by-side comparison of multiple images."""
```

## Data Flow

```
data/raw_data.npz
  -> load_observation() -> projections, flat, dark, theta
  -> normalize() -> transmission
  -> minus_log() -> sinogram
  -> find_rotation_center() -> rot_center
  -> _shift_sinogram() -> centered sinogram
  -> filtered_back_projection() -> reconstruction
  -> circular_mask() -> masked reconstruction
  -> compute_metrics() -> NRMSE, NCC
  -> save to output/ and evaluation/reference_outputs/
```
