# Code Design: MCR Hyperspectral Unmixing

## File Structure

```
mcr_hyperspectral/
├── main.py                    # Pipeline orchestration (7 steps)
├── requirements.txt
├── README.md
├── data/
│   ├── raw_data.npz           # hsi_noisy, wn
│   ├── ground_truth.npz       # concentrations, spectra, hsi_clean
│   └── meta_data.json         # Imaging parameters
├── plan/
│   ├── approach.md
│   └── design.md
├── src/
│   ├── generate_data.py       # Synthetic data generation
│   ├── preprocessing.py       # Data loading + SVD initial guess
│   ├── physics_model.py       # Forward model D = C S^T
│   ├── solvers.py             # MCR-AR with 5 method configs
│   └── visualization.py       # Plotting + metrics
├── notebooks/
│   └── mcr_hyperspectral.ipynb
└── evaluation/
    ├── metrics.json
    ├── reference_outputs/
    ├── fixtures/
    └── tests/
```

## Function Signatures

### src/generate_data.py

```python
def make_spectral_components(wn, centers, widths, amplitude=1e4, baseline=1000):
    """Create Gaussian spectral components.
    Returns: spectra (n_components, n_freq)"""

def make_concentration_maps(M, N, n_components, rng):
    """Create 2D Gaussian concentration maps with random patches.
    Returns: conc (M, N, n_components), sum-to-one normalised"""

def generate_hsi(conc, spectra, noise_std, rng):
    """Generate noisy HSI from bilinear model.
    Returns: hsi_clean (M*N, n_freq), hsi_noisy (M*N, n_freq)"""

def generate_dataset(data_dir, seed=0):
    """Generate full dataset and save to data_dir."""
```

### src/preprocessing.py

```python
def load_observation(data_dir):
    """Load raw_data.npz. Returns: dict with 'hsi_noisy', 'wn'"""

def load_ground_truth(data_dir):
    """Load ground_truth.npz. Returns: dict with concentrations, spectra, etc."""

def load_metadata(data_dir):
    """Load meta_data.json. Returns: dict"""

def estimate_initial_spectra(hsi_noisy, n_components):
    """SVD-based initial spectral estimate.
    Returns: initial_spectra (n_components, n_freq)"""
```

### src/physics_model.py

```python
def forward(C, ST):
    """D = C @ ST. Returns: D (n_pixels, n_freq)"""

def residual(C, ST, D_obs):
    """D_obs - C @ ST. Returns: R (n_pixels, n_freq)"""

def mse(C, ST, D_obs):
    """Mean squared error. Returns: float"""
```

### src/solvers.py

```python
class ConstraintSingleGauss(Constraint):
    """Gaussian shape constraint via lmfit NLLS fitting."""
    def __init__(self, alpha=1.0, copy=False, axis=-1)
    def transform(self, A) -> ndarray

def build_method_configs():
    """Return list of 5 MCR method config dicts."""

def match_components(C_est, conc_ravel, n_components):
    """Greedy component matching by MSE. Returns: select (list of int)"""

def run_mcr(hsi_noisy, initial_spectra, config, mcr_params=None):
    """Run single MCR method. Returns: McrAR object"""

def run_all_methods(hsi_noisy, initial_spectra, conc_ravel, spectra, mcr_params=None):
    """Run all 5 methods. Returns: list of result dicts"""
```

### src/visualization.py

```python
def compute_metrics(estimate, reference):
    """NRMSE and NCC. Returns: dict with 'nrmse', 'ncc'"""

def compute_method_metrics(result, conc_ravel, spectra, hsi_noisy):
    """Per-method detailed metrics. Returns: dict"""

def plot_spectral_components(ax, wn, spectra, title)
def plot_concentration_maps(fig, conc, suptitle)
def plot_comparison_boxplots(results, conc_ravel, spectra, hsi_noisy, method_names)
def plot_method_result(wn, result, conc_shape, method_name)
```

## Data Flow

```
data/raw_data.npz + data/meta_data.json
        ↓
preprocessing.py → hsi_noisy, wn, meta
        ↓
preprocessing.estimate_initial_spectra() → initial_spectra
        ↓
solvers.run_all_methods() → 5 × {McrAR, select, mse, timing}
        ↓
visualization.compute_method_metrics() → per-method NCC, NRMSE, Δ stats
        ↓
output/*.npz, output/*.png, evaluation/reference_outputs/
```
