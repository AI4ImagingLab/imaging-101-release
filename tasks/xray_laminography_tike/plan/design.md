# Code Architecture

## Module Overview

```
main.py                  -- Entry point: load data, reconstruct, evaluate, visualize
src/
  __init__.py            -- Package marker (empty)
  preprocessing.py       -- Data loading and initial guess creation
  physics_model.py       -- Laminographic forward/adjoint projection via NUFFT
  solvers.py             -- Conjugate gradient reconstruction
  visualization.py       -- Metrics computation and plotting
  generate_data.py       -- Synthetic data generation from phantom
```

## Data Flow

```
ground_truth.npz --[generate_data]--> raw_data.npz
                                          |
                                    [preprocessing]
                                          |
                                          v
                              projections, theta, tilt
                                          |
                                      [solvers]
                                          |
                                          v
                                  reconstructed volume
                                          |
                                   [visualization]
                                          |
                                          v
                              metrics.json, plots (.png)
```

## Function Signatures

### src/preprocessing.py

```python
def load_raw_data(path: str) -> dict:
    """Returns {'projections': (1,128,128,128) complex64, 'theta': (1,128) float32}"""

def load_ground_truth(path: str) -> dict:
    """Returns {'volume': (1,128,128,128) complex64}"""

def load_metadata(path: str) -> dict:
    """Returns dict from meta_data.json"""

def create_initial_guess(volume_shape: tuple, dtype=np.complex64) -> np.ndarray:
    """Returns zeros array of given shape and dtype"""
```

### src/physics_model.py

```python
def make_frequency_grid(theta, tilt, n, xp=None) -> np.ndarray:
    """Compute 3D frequency coordinates for laminographic projection.
    Returns (R*n*n, 3) float32."""

def forward_project(obj: np.ndarray, theta: np.ndarray, tilt: float) -> np.ndarray:
    """Forward laminographic projection using Fourier slice theorem + NUFFT.
    obj: (nz, n, n) complex64
    theta: (n_angles,) float32
    tilt: float (radians)
    Returns: (n_angles, n, n) complex64"""

def adjoint_project(data: np.ndarray, theta: np.ndarray, tilt: float, n: int) -> np.ndarray:
    """Adjoint laminographic operator (backprojection).
    data: (n_angles, n, n) complex64
    Returns: (n, n, n) complex64"""

def cost_function(obj, data, theta, tilt) -> float:
    """Least-squares cost: ||forward(obj) - data||^2"""

def gradient(obj, data, theta, tilt) -> np.ndarray:
    """Gradient of the least-squares cost: adjoint(forward(obj) - data) / (R * n^3)"""
```

### src/solvers.py

```python
def reconstruct(
    data: np.ndarray,       # (n_angles, n, n) complex64
    theta: np.ndarray,      # (n_angles,) float32
    tilt: float,            # radians
    volume_shape: tuple,    # (nz, n, n)
    n_rounds: int = 5,
    n_iter_per_round: int = 4,
) -> dict:
    """Conjugate gradient reconstruction with Dai-Yuan direction and
    backtracking line search. Returns {'obj': (nz, n, n) complex64, 'costs': list[float]}"""
```

### src/visualization.py

```python
def compute_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """Returns {'ncc': float, 'nrmse': float}"""

def plot_complex_slice(volume_slice: np.ndarray, title: str = None) -> Figure:
    """Plot real/imaginary parts of a 2D complex slice"""

def plot_volume_slices(volume: np.ndarray, slice_indices: list) -> Figure:
    """Plot multiple axial slices with real/imaginary columns"""
```

### src/generate_data.py

```python
def generate_projections(
    volume: np.ndarray,     # (nz, n, n) complex64
    n_angles: int,
    tilt: float,
    theta_range: tuple = (0.0, np.pi),
) -> tuple:
    """Returns (projections (n_angles,n,n), theta (n_angles,))"""

def generate_and_save(task_dir: str) -> None:
    """Load ground truth, simulate, save raw_data.npz"""
```
