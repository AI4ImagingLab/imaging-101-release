# Code Architecture

## Module Layout

```
src/
├── preprocessing.py    # Data I/O and gradient estimation
├── physics_model.py    # Forward model (gradient, divergence, DCT solver)
├── solvers.py          # ADMM phase unwrapping algorithm
└── visualization.py    # Plotting and metrics
```

## Function Signatures

### preprocessing.py

```python
def load_data(data_path: str) -> dict:
    """Load raw_data.npz -> dict of arrays."""

def load_metadata(meta_path: str) -> dict:
    """Load meta_data.json -> dict."""

def extract_phase_and_coherence(interferogram: ndarray) -> tuple[ndarray, ndarray]:
    """Complex interferogram -> (wrapped_phase, coherence)."""

def est_wrapped_gradient(arr: ndarray, dtype: str = "float32") -> tuple[ndarray, ndarray]:
    """Wrapped phase -> (phi_x, phi_y) with wrapping adjustment."""
```

### physics_model.py

```python
def apply_gradient_x(arr: ndarray) -> ndarray:
    """Forward difference along columns with Neumann BCs."""

def apply_gradient_y(arr: ndarray) -> ndarray:
    """Forward difference along rows with Neumann BCs."""

def apply_divergence(grad_x: ndarray, grad_y: ndarray) -> ndarray:
    """Divergence (adjoint of gradient) for Neumann BCs."""

def make_laplace_kernel(rows: int, columns: int, dtype: str = "float32") -> ndarray:
    """Inverse eigenvalues of DCT-diagonalized Laplacian."""

def solve_poisson_dct(rhs: ndarray, K: ndarray) -> ndarray:
    """Solve Poisson equation via DCT: (D^T D) Phi = rhs."""
```

### solvers.py

```python
def p_shrink(X: ndarray, lmbda: float = 1, p: float = 0, epsilon: float = 0) -> ndarray:
    """p-shrinkage operator for sparsity penalty."""

def make_congruent(unwrapped: ndarray, wrapped: ndarray) -> ndarray:
    """Snap unwrapped to nearest 2*pi*k offset from wrapped."""

def unwrap_phase(
    f_wrapped: ndarray,
    phi_x: ndarray = None, phi_y: ndarray = None,
    max_iters: int = 500, tol: float = pi/5,
    lmbda: float = 1, p: float = 0, c: float = 1.3,
    dtype: str = "float32", debug: bool = False,
    congruent: bool = False,
) -> tuple[ndarray, int]:
    """ADMM phase unwrapping. Returns (unwrapped_phase, n_iterations)."""
```

### visualization.py

```python
def plot_wrapped_phase_and_coherence(wrapped_phase, coherence, save_path=None) -> Figure:
    """Paper Fig. 1 reproduction."""

def plot_unwrapped_comparison(unwrapped_spurs, unwrapped_snaphu, save_path=None) -> Figure:
    """Paper Fig. 2 reproduction (aligns constant offset)."""

def plot_residuals(unwrapped_spurs, unwrapped_snaphu, wrapped_phase, save_path=None) -> Figure:
    """Paper Fig. 3 reproduction."""

def plot_difference_map(unwrapped_spurs, unwrapped_snaphu, save_path=None) -> Figure:
    """Direct SPURS-SNAPHU difference map."""

def compute_metrics(unwrapped_spurs, unwrapped_snaphu) -> dict:
    """RMSE, fraction within pi/2pi, etc."""
```

## Data Flow

```
raw_data.npz
    |
    v
preprocessing.load_data() -> interferogram (complex64)
    |
    v
preprocessing.extract_phase_and_coherence() -> wrapped_phase, coherence
    |
    v
preprocessing.est_wrapped_gradient() -> phi_x, phi_y
    |
    v
solvers.unwrap_phase()  [uses physics_model internally]
    |                      - apply_gradient_x/y
    |                      - apply_divergence
    |                      - make_laplace_kernel
    |                      - solve_poisson_dct
    |                      - p_shrink
    v
unwrapped_phase
    |
    v
visualization.compute_metrics() -> metrics.json
visualization.plot_*()           -> figure PNGs
```
