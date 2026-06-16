# Design: Dual-Energy CT Material Decomposition

## Module structure

```
src/
├── __init__.py
├── generate_data.py       # Synthetic phantom and polychromatic sinogram generation
├── physics_model.py       # Forward model, spectra, Radon wrappers (MACs loaded from data)
├── solvers.py             # Gauss-Newton decomposition and FBP reconstruction
├── preprocessing.py       # Data loading utilities
└── visualization.py       # Plotting and metric computation
```

## Function signatures

### src/physics_model.py

```python
def get_spectra(energies: ndarray) -> ndarray:
    """Generate low-energy and high-energy X-ray spectra.
    Returns: (2, nE) array [low_spectrum, high_spectrum] in photons/bin."""

def radon_transform(image: ndarray, theta: ndarray) -> ndarray:
    """Compute Radon transform with circle=True.
    Returns: (nBins, nAngles) sinogram."""

def fbp_reconstruct(sinogram: ndarray, theta: ndarray,
                    output_size: int = None, filter_name: str = 'ramp') -> ndarray:
    """Filtered back-projection reconstruction.
    Returns: (output_size, output_size) image."""

def polychromatic_forward(material_sinograms: ndarray, spectra: ndarray,
                          mus: ndarray, dE: float = 1.0) -> ndarray:
    """Polychromatic Beer-Lambert forward model.
    material_sinograms: (nMats, nBins, nAngles) in g/cm^2
    Returns: (nMeas, nBins, nAngles) expected photon counts."""
```

### src/solvers.py

```python
def gauss_newton_decompose(sinograms: ndarray, spectra: ndarray,
                           mus: ndarray, n_iters: int = 20,
                           dE: float = 1.0, eps: float = 1e-6,
                           verbose: bool = True) -> ndarray:
    """Gauss-Newton material decomposition in the sinogram domain.
    sinograms: (nMeas, nBins, nAngles) measured counts
    Returns: (nMats, nBins, nAngles) estimated density line integrals (g/cm^2)."""

def reconstruct_material_maps(material_sinograms: ndarray, theta: ndarray,
                              image_size: int, pixel_size: float = 0.1) -> ndarray:
    """FBP of material sinograms to density maps.
    Returns: (nMats, image_size, image_size) density maps (g/cm^3)."""
```

### src/generate_data.py

```python
def get_attenuation_coefficients(energies: ndarray) -> ndarray:
    """Interpolate NIST XCOM MACs for tissue and bone onto energy grid.
    Returns: (2, nE) array [tissue_mac, bone_mac] in cm^2/g.
    Uses hardcoded NIST reference tables (only needed during data generation)."""

def create_phantom(size: int = 128) -> tuple[ndarray, ndarray]:
    """Create dual-material phantom.
    Returns: (tissue_map, bone_map) each (size, size) in g/cm^3."""

def generate_synthetic_data(size: int = 128, n_angles: int = 180,
                            seed: int = 42) -> dict:
    """Generate complete synthetic dataset with Poisson noise."""

def save_task_data(data: dict, task_dir: str) -> None:
    """Save to raw_data.npz, ground_truth.npz, meta_data.json."""
```

### src/preprocessing.py

```python
def load_raw_data(data_dir: str) -> tuple:
    """Load sinograms, spectra, mus, energies, theta."""

def load_ground_truth(data_dir: str) -> tuple:
    """Load tissue_map, bone_map, tissue_sinogram, bone_sinogram."""

def load_metadata(data_dir: str) -> dict:
    """Load meta_data.json."""
```

### src/visualization.py

```python
def compute_ncc(estimate: ndarray, reference: ndarray) -> float:
    """Cosine similarity (NCC)."""

def compute_nrmse(estimate: ndarray, reference: ndarray) -> float:
    """NRMSE normalised by reference dynamic range."""

def compute_metrics(tissue_est, tissue_ref, bone_est, bone_ref) -> dict:
    """Compute NCC/NRMSE for both materials, masked to body region."""

def plot_material_maps(tissue_est, bone_est, tissue_ref, bone_ref,
                       save_path=None) -> Figure:
    """Side-by-side material map comparison with difference images."""

def plot_sinograms(sino_low, sino_high, save_path=None) -> Figure:
    """Visualise input sinograms."""

def plot_spectra_and_mac(energies, spectra, mus, save_path=None) -> Figure:
    """Plot X-ray spectra and mass attenuation coefficients."""
```

## Pipeline flow (main.py)

1. Generate synthetic data if not present
2. Load sinograms, spectra, MACs, ground truth
3. Run Gauss-Newton decomposition (sinogram domain)
4. FBP reconstruction of material density maps
5. Clip negative densities
6. Compute and save metrics
7. Save reconstructed maps and reference outputs
8. Generate visualisation figures
