# Design: Reflection-Mode ODT Code Architecture

## Module Structure

### src/preprocessing.py

```python
def generate_phantom(metadata: dict) -> np.ndarray
    """Generate 4-layer USAF-like synthetic phantom.
    Returns: (Nz, Ny, Nx) array with Δn = -0.07 for structures"""

def load_metadata(data_dir: str = "data") -> dict
    """Load imaging parameters from data/meta_data JSON"""

def prepare_data(data_dir: str = "data") -> tuple[np.ndarray, dict]
    """Load metadata and generate phantom. Returns (phantom, metadata)"""
```

### src/physics_model.py

```python
@dataclass
class ReflectionBPMConfig:
    volume_shape: tuple  # (Nz, Ny, Nx)
    res: tuple           # Normalized (dx·n0/λ, dy·n0/λ, dz·n0/λ)
    n0: float
    NA_obj: float
    wavelength_um: float
    res_um: tuple
    illumination_rings: list  # [{"NA", "n_angles", "type": "BF"|"DF"}]
    n_angles: int             # Total angles over all rings
    dz_layer: float      # BPM step for layer scattering
    dz_gap: float        # BPM step for gaps

    @classmethod
    def from_metadata(cls, metadata: dict) -> ReflectionBPMConfig

class ReflectionBPMForwardModel:
    def __init__(self, config: ReflectionBPMConfig, device: str)

    def _compute_c_gamma(self) -> torch.Tensor
        """cos(gamma) = sqrt(1 - fx² - fy²)"""

    def _compute_kz(self) -> torch.Tensor
        """kz = c_gamma · 2π·res_z"""

    def _compute_evanescent_mask(self) -> torch.Tensor
    def _compute_pupil(self) -> torch.Tensor
        """Objective pupil; blocks DF direct beam when NA_illu > NA_obj"""

    def _make_incident_field(self, na: float, angle_idx: int,
                             n_angles_in_ring: int) -> torch.Tensor
        """Tilted plane wave for one angle within one illumination ring"""

    def _bpm_propagate(self, u: torch.Tensor, dz: float) -> torch.Tensor
        """P operator: fft → exp(i·kz·dz)·eva → ifft"""

    def _bpm_scatter(self, u: torch.Tensor, dn_slice: torch.Tensor, dz: float) -> torch.Tensor
        """Q operator: u *= exp(i·Δn·2π·res_z/n0·dz)"""

    def _reflect(self, u: torch.Tensor) -> torch.Tensor
        """Mirror reflection: u *= -1"""

    def forward_single_ring(self, dn_volume: torch.Tensor, na: float,
                            angle_idx: int, n_angles_in_ring: int) -> torch.Tensor
        """One BF/DF illumination angle. Returns intensity (Ny, Nx)"""

    def forward(self, dn_volume: torch.Tensor) -> torch.Tensor
        """All ring angles in metadata order. Returns (n_angles, Ny, Nx)"""

    def simulate_measurements(self, dn_volume: torch.Tensor) -> torch.Tensor
        """Convenience wrapper for forward()"""
```

### src/solvers.py

```python
def tv_2d_proximal(volume: torch.Tensor, tau: float, n_iter: int = 20) -> torch.Tensor
    """Chambolle dual projection per-slice 2D TV denoising"""

class ReflectionBPMReconstructor:
    def __init__(self, n_iter: int, lr: float, tv_weight: float,
                 positivity: bool, device: str)

    def reconstruct(self, measurements: torch.Tensor, model) -> tuple[np.ndarray, list]
        """FISTA reconstruction. Returns (dn_recon, loss_history)"""
```

### src/visualization.py

```python
def plot_ri_slices(volume: np.ndarray, ...) -> Figure
def plot_comparison(ground_truth: np.ndarray, reconstruction: np.ndarray, ...) -> Figure
def plot_loss_history(losses: list, ...) -> Figure
def plot_measurements(measurements: np.ndarray, ...) -> Figure
def plot_illumination_angles(illumination_rings: list, NA_obj: float, ...) -> Figure
def compute_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> dict
def print_metrics_table(metrics: dict) -> None
```

### src/generate_data.py

```python
def generate_measurements(data_dir: str = "data",
                          device: str = "cpu") -> tuple[np.ndarray, np.ndarray, dict]
    """Load phantom, run forward model, return (measurements, phantom_dn, metadata)"""
```

## Pipeline (main.py)

1. Load metadata and generate phantom
2. Build ReflectionBPMForwardModel
3. Simulate BF+DF measurements (4 rings, 56 angles total)
4. Reconstruct via FISTA (50 iter, lr=5.0, tv=8e-7)
5. Evaluate metrics (NRMSE, NCC, SSIM)
6. Save outputs and visualizations
