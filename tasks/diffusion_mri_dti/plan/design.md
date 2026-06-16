# Design: Diffusion MRI DTI

## Module Architecture

```
src/
├── preprocessing.py     # Data I/O and DWI signal preparation
├── physics_model.py     # Stejskal-Tanner forward model and tensor utilities
├── solvers.py           # OLS and WLS tensor fitting
├── visualization.py     # Plotting utilities and metrics
└── generate_data.py     # Synthetic phantom generation
```

## Function Signatures

### src/physics_model.py

```python
def tensor_from_elements(Dxx, Dxy, Dxz, Dyy, Dyz, Dzz) -> np.ndarray:
    """Construct 3x3 symmetric tensor(s) from 6 elements. Shape: (..., 3, 3)."""

def elements_from_tensor(D) -> np.ndarray:
    """Extract 6 independent elements from tensor(s). Shape: (..., 6)."""

def tensor_from_eig(eigenvalues, eigenvectors) -> np.ndarray:
    """Construct tensor from eigendecomposition. D = V diag(evals) V^T."""

def build_design_matrix(bvals, bvecs) -> np.ndarray:
    """Build linearized Stejskal-Tanner design matrix. Shape: (N_volumes, 7)."""

def stejskal_tanner_signal(S0, D, bvals, bvecs) -> np.ndarray:
    """Compute DWI signal: S = S0 * exp(-b * g^T D g). Shape: (..., N_volumes)."""

def add_rician_noise(signal, sigma, rng=None) -> np.ndarray:
    """Add Rician noise: |S + CN(0, sigma^2)|."""

def compute_fa(eigenvalues) -> np.ndarray:
    """Fractional anisotropy from eigenvalues. Shape: (...,)."""

def compute_md(eigenvalues) -> np.ndarray:
    """Mean diffusivity from eigenvalues. Shape: (...,)."""
```

### src/solvers.py

```python
def fit_dti_ols(dwi_signal, bvals, bvecs, mask=None) -> (tensor_elems, S0_map):
    """OLS tensor fit via linearized Stejskal-Tanner. Vectorized over all voxels."""

def fit_dti_wls(dwi_signal, bvals, bvecs, mask=None) -> (tensor_elems, S0_map):
    """WLS tensor fit: OLS init + weighted re-solve."""

def tensor_eig_decomposition(tensor_elems, mask=None) -> (eigenvalues, eigenvectors, fa_map, md_map):
    """Eigendecompose fitted tensors to get FA, MD maps."""
```

### src/preprocessing.py

```python
def load_dwi_data(task_dir) -> (dwi_signal, bvals, bvecs):
    """Load raw_data.npz. Returns (1, Ny, Nx, N_volumes) signal + gradient table."""

def load_ground_truth(task_dir) -> (fa_map, md_map, tensor_elements, tissue_mask):
    """Load ground_truth.npz."""

def load_metadata(task_dir) -> dict:
    """Load meta_data.json."""

def preprocess_dwi(dwi_signal, bvals, bvecs) -> (dwi_2d, S0):
    """Remove batch dim, compute mean S0 from b=0 volumes. (Ny, Nx, N_volumes)."""
```

### src/visualization.py

```python
def compute_ncc(estimate, reference, mask=None) -> float:
    """Cosine similarity (no mean subtraction)."""

def compute_nrmse(estimate, reference, mask=None) -> float:
    """RMSE / dynamic_range(reference)."""

def plot_dti_maps(fa_gt, fa_est, md_gt, md_est, tissue_mask, ...) -> Figure:
    """Side-by-side FA and MD maps with error maps."""

def plot_color_fa(fa_map, eigenvectors, tissue_mask, ...) -> Figure:
    """Directionally-encoded color FA (RGB = |v1| * FA)."""
```

### src/generate_data.py

```python
def generate_gradient_table(n_directions=30, b_value=1000.0, n_b0=1, seed=42) -> (bvals, bvecs):
    """Fibonacci sphere gradient directions + b=0 volumes."""

def create_dti_phantom(N=128) -> (tensor_field, S0_map, tissue_mask):
    """Shepp-Logan phantom with tissue-specific diffusion tensors."""

def generate_synthetic_data(N=128, n_directions=30, ...) -> dict:
    """Generate full synthetic DWI dataset."""

def save_data(data, task_dir):
    """Save to raw_data.npz, ground_truth.npz, meta_data.json."""
```

## Pipeline Flow (main.py)

1. Load data (or generate if missing)
2. Preprocess: remove batch dim, compute S0 from b=0 volumes
3. OLS tensor fit -> tensor_elements_init
4. WLS tensor fit (using OLS weights)
5. Eigendecompose both fits -> FA, MD maps
6. Compute NCC, NRMSE on tissue-masked FA/MD maps
7. Save reference outputs and metrics
8. Generate visualization figures (FA/MD comparison, color FA)
