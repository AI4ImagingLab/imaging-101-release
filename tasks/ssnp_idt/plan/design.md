# Code Design

## File Structure

```
main.py                  # Pipeline orchestration
src/
  __init__.py            # Package exports
  preprocessing.py       # Data loading and preparation
  physics_model.py       # SSNP forward model (PyTorch)
  solvers.py             # Reconstruction via gradient descent + TV
  visualization.py       # Plotting utilities and metrics
  generate_data.py       # Synthetic measurement generation
```

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir: str = "data") -> np.ndarray:
    """
    Load phantom TIFF and convert to RI contrast Δn.

    Returns: np.ndarray, shape (Nz, Ny, Nx), float64 — RI contrast volume
    """

def load_metadata(data_dir: str = "data") -> dict:
    """
    Load meta_data JSON.

    Returns: dict with keys
      'volume_shape'      : list[int]  — [Nz, Ny, Nx]
      'res_um'            : list[float] — voxel size [dx, dy, dz] in μm
      'wavelength_um'     : float — illumination wavelength in μm
      'n0'                : float — background RI
      'NA'                : float — objective numerical aperture
      'n_angles'          : int   — number of illumination angles
      'ri_contrast_scale' : float — RI contrast scaling
      'tiff_scale'        : float — TIFF normalization factor
    """

def prepare_data(data_dir: str = "data") -> tuple:
    """
    Combined loader.

    Returns: (phantom_dn, metadata)
      phantom_dn : np.ndarray (Nz, Ny, Nx) — RI contrast Δn
      metadata   : dict — imaging parameters
    """
```

### physics_model.py

```python
@dataclass
class SSNPConfig:
    """Physical parameters for the SSNP model."""
    volume_shape: tuple        # (Nz, Ny, Nx)
    res: tuple                 # Normalised resolution (dx/λ·n0, dy/λ·n0, dz/λ·n0)
    n0: float                  # Background RI
    NA: float                  # Objective NA
    wavelength_um: float       # Wavelength in μm
    res_um: tuple              # Voxel size in μm

    @classmethod
    def from_metadata(cls, metadata: dict) -> "SSNPConfig": ...


class SSNPForwardModel:
    """
    SSNP-based intensity diffraction tomography forward model.

    Implements the split-step non-paraxial propagation through a 3D
    RI distribution and computes the resulting intensity images for
    multiple illumination angles.

    All computation uses PyTorch for GPU acceleration and autograd support.
    """

    def __init__(self, config: SSNPConfig, device: str = "cpu"):
        """Precompute kz grid, propagator components, and pupil."""

    def _compute_kz(self) -> torch.Tensor:
        """Compute kz = sqrt(k0²n0² - kx² - ky²) on 2D frequency grid."""

    def _compute_evanescent_mask(self) -> torch.Tensor:
        """Damping mask for evanescent waves: exp(min((γ-0.2)·5, 0))."""

    def _compute_pupil(self) -> torch.Tensor:
        """Binary pupil with cutoff at k0·n0·NA."""

    def _make_incident_field(self, angle_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Construct tilted plane wave and its z-derivative.
        Returns: (u, ud) each shape (Ny, Nx) complex128
        """

    def _apply_propagation(self, u: torch.Tensor, ud: torch.Tensor,
                           dz: float = 1.0) -> tuple:
        """P operator: FFT → 2×2 rotation → IFFT."""

    def _apply_scattering(self, u: torch.Tensor, ud: torch.Tensor,
                          dn_slice: torch.Tensor, dz: float = 1.0) -> tuple:
        """Q operator: ud -= k0²(2·n0·Δn + Δn²)·Δz · u."""

    def _extract_forward_component(self, u: torch.Tensor,
                                   ud: torch.Tensor) -> torch.Tensor:
        """
        Back-propagate to focal plane, apply pupil, extract forward component.
        Returns: complex field at camera, shape (Ny, Nx)
        """

    def forward_single(self, dn_volume: torch.Tensor,
                       angle_idx: int) -> torch.Tensor:
        """
        Simulate intensity for one illumination angle.
        Returns: intensity image (Ny, Nx)
        """

    def forward(self, dn_volume: torch.Tensor,
                n_angles: int = 8) -> torch.Tensor:
        """
        Simulate intensities for all angles.
        Returns: (n_angles, Ny, Nx)
        """
```

### solvers.py

```python
class SSNPReconstructor:
    """
    Gradient-descent reconstruction with TV regularization.

    Uses PyTorch autograd for gradient computation through the
    SSNP forward model.
    """

    def __init__(self, n_iter: int = 5, lr: float = 350.0,
                 tv_weight: float = 0.0, positivity: bool = True,
                 device: str = "cpu"):
        """Configure reconstruction hyperparameters."""

    def reconstruct(self, measurements: torch.Tensor,
                    model: SSNPForwardModel) -> tuple:
        """
        Reconstruct 3D RI contrast from intensity measurements.

        Parameters
        ----------
        measurements : (n_angles, Ny, Nx) tensor — measured intensities
        model        : SSNPForwardModel — forward model

        Returns
        -------
        dn_recon : np.ndarray (Nz, Ny, Nx) — reconstructed RI contrast
        loss_history : list[float] — loss per iteration
        """

def tv_3d(volume: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    3D isotropic total variation (Huber-smoothed).

    TV(x) = Σ sqrt(|∂x/∂x_i|² + |∂x/∂y_i|² + |∂x/∂z_i|² + ε)
    """
```

### visualization.py

```python
def plot_ri_slices(volume: np.ndarray, slice_indices: list = None,
                   title: str = "RI Slices", vmin=None, vmax=None) -> Figure:
    """Plot XY cross-sections at selected z positions."""

def plot_xz_cross_section(volume: np.ndarray, y_index: int = None,
                          title: str = "XZ Cross Section") -> Figure:
    """Plot XZ cross-section through the volume centre."""

def plot_comparison(ground_truth: np.ndarray, reconstruction: np.ndarray,
                    slice_idx: int = None) -> Figure:
    """Side-by-side comparison: GT vs reconstruction (XY and XZ)."""

def plot_loss_history(losses: list, title: str = "Loss History") -> Figure:
    """Plot training loss convergence."""

def compute_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute reconstruction quality metrics.
    Returns dict with 'nrmse', 'ncc', 'ssim'.
    """

def print_metrics_table(metrics: dict) -> None:
    """Print formatted metrics table."""
```

### generate_data.py

```python
def generate_measurements(data_dir: str = "data",
                          device: str = "cpu") -> tuple:
    """
    Load phantom and simulate IDT measurements.

    Returns: (measurements, phantom_dn, metadata)
      measurements : np.ndarray (n_angles, Ny, Nx) — intensity images
      phantom_dn   : np.ndarray (Nz, Ny, Nx) — ground truth RI contrast
      metadata     : dict
    """
```

### main.py

```python
def main():
    """
    Orchestrate the full SSNP-IDT pipeline:
      1. prepare_data("data")           → phantom, metadata
      2. SSNPForwardModel(config)       → model
      3. generate_measurements(...)     → intensity images
      4. SSNPReconstructor.reconstruct  → 3D RI volume
      5. compute_metrics(...)           → evaluation
      6. Save reference outputs + plots
    """
```

## Data Flow

```
data/sample.tiff + data/meta_data
        │
        ▼
  preprocessing.py  ──→  phantom_dn (Nz,Ny,Nx), metadata dict
        │
        ▼
  physics_model.py  ──→  SSNPForwardModel (P/Q operators, pupil)
        │
        ▼
  generate_data.py  ──→  measurements (n_angles, Ny, Nx)
        │
        ▼
  solvers.py        ──→  reconstructed dn (Nz, Ny, Nx) + loss_history
        │
        ▼
  visualization.py  ──→  comparison plots + metrics
        │
        ▼
  evaluation/reference_outputs/
    ground_truth.npy, measurements.npy, reconstruction.npy,
    loss_history.npy, metrics.json
```
