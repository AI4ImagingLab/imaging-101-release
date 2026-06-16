# Code Design

## File Structure

```
main.py                    # Pipeline orchestration (single/multi-freq modes)
src/
  preprocessing.py         # Data loading, restriction operator, masking
  physics_model.py         # CBS Helmholtz solver (forward model)
  solvers.py               # FWI objective, gradient, NCG optimizer
  visualization.py         # Plotting utilities and metrics
```

## Function Signatures

### preprocessing.py

```python
def load_metadata(data_dir: str = "data") -> dict:
    """Load meta_data.json. Returns dict with nx, ny, dh_um, frequencies, etc."""

def load_observations(data_dir: str = "data", freq: float = None) -> dict:
    """Load raw_data.npz. Returns dict with receiver_ix, receiver_iy, and dobs tensors."""

def load_baseline_reference(data_dir: str = "data") -> np.ndarray:
    """Load baseline reference velocity. Returns float32 (480, 480)."""

def build_restriction_operator(
    ix: np.ndarray, iy: np.ndarray, nx: int, ny: int
) -> torch.Tensor:
    """Build sparse restriction operator R: (n_rec, nx*ny) on CUDA.
    Maps flattened wavefield to receiver locations."""

def create_dobs_masks(
    dobs: torch.Tensor, ix: np.ndarray, iy: np.ndarray,
    dh: float, mute_dist: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create masked observations and near-source muting masks.
    Returns: dobs_masked, mask_esi, mask_misfit (all on CUDA)."""

def create_initial_slowness(nx: int, ny: int, v0: float) -> torch.Tensor:
    """Create homogeneous initial slowness 1/v0. Returns (nx, ny) on CUDA."""
```

### physics_model.py

```python
def setup_domain(velocity, freq, dh=50, ppw=8, lamb=1,
                 boundary_widths=20, born_max=500, energy_threshold=1e-5) -> dict:
    """Prepare CBS domain: padding, potential, wiggles, propagation kernels.
    Args: velocity (nx,ny) tensor, freq in 100kHz units.
    Returns: dict with grid, wiggles, gamma, mix_weights, roi, etc."""

def cbs_solve(ix, iy, domain) -> torch.Tensor:
    """Solve Helmholtz for one source via CBS iteration.
    Args: 0-indexed source position, domain dict from setup_domain.
    Returns: complex64 (nx, ny) scaled wavefield."""

def solve_all_sources(velocity, ix_arr, iy_arr, freq, **cbs_kwargs) -> torch.Tensor:
    """Solve CBS for all sources. Reuses domain setup.
    Returns: complex64 (nx, ny, n_src)."""
```

### solvers.py

```python
def create_gaussian_kernel(kernel_size: int = 9, sigma: float = 1.0) -> torch.Tensor:
    """2D Gaussian kernel for gradient smoothing."""

def create_objective(all_u_size, freq, dobs_masked, mask_esi, mask_misfit,
                     R, ix, iy, sigma, **cbs_kwargs) -> Callable:
    """Create FWI objective closure.
    Returns: callable(slowness, fscale, gscale) -> (J, G)."""

def ncg(fun, x0, max_iters=3, v_bounds=(1300, 1700), ...) -> torch.Tensor:
    """NCG optimizer (Polak-Ribiere) with More-Thuente line search.
    Returns: optimized slowness."""

def invert_single_frequency(freq_mhz, dobs, sigma, slowness, ix, iy, R,
                            all_u_size, **kwargs) -> torch.Tensor:
    """Run FWI for one frequency. Returns updated slowness."""

def invert_multi_frequency(frequencies_mhz, dobs_dict, slowness, ix, iy, R,
                           all_u_size, **kwargs) -> Tuple[torch.Tensor, list]:
    """Run multi-frequency FWI with bootstrapping.
    Returns: final_slowness, history list."""
```

### visualization.py

```python
def compute_ncc(x: np.ndarray, ref: np.ndarray) -> float:
    """Normalized Cross-Correlation (cosine similarity)."""

def compute_nrmse(x: np.ndarray, ref: np.ndarray) -> float:
    """NRMSE normalized by dynamic range."""

def plot_velocity(vp, domain_size_cm=(24,24), ...) -> plt.Figure:
def plot_comparison(vp_ref, vp_recon, ...) -> plt.Figure:
def plot_dobs(dobs, freq, ...) -> plt.Figure:
def plot_convergence(history, vp_ref, ...) -> plt.Figure:
```

## Data Flow

```
raw_data.npz → preprocessing.load_observations() → dobs (256,256) complex64
                                                  → ix, iy (256,) float32
meta_data.json → preprocessing.load_metadata() → physics params

preprocessing.build_restriction_operator(ix,iy) → R sparse (256, 230400)
preprocessing.create_initial_slowness(480,480,1480) → slowness (480,480)

For each frequency:
  preprocessing.create_dobs_masks(dobs) → masked_dobs, masks
  solvers.create_objective(...) → get_grad closure
  solvers.ncg(get_grad, slowness) → updated slowness
    └─ get_grad(slowness):
       physics_model.solve_all_sources(1/slowness, ix, iy, freq) → all_u (480,480,256)
       restrict to receivers → dsrc (256,256)
       estimate source intensity → alpha (256,)
       compute misfit J and adjoint gradient G (480,480)
       Gaussian blur G → smoothed G
       return J, G

Final slowness → 1/slowness → vp_reconstructed (480,480)
```
