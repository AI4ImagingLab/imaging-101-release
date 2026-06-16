# Code Architecture

## Module Overview

```
src/
├── preprocessing.py    # Image I/O and shapelet decomposition
├── physics_model.py    # Shapelets, lens models, ray-tracing, PSF, simulation
├── solvers.py          # WLS linear inversion
├── visualization.py    # Plotting utilities
└── generate_data.py    # End-to-end synthetic data generation
```

## Function Signatures

### `src/preprocessing.py`

```python
def load_and_prepare_galaxy(image_path: str, smooth_sigma: float = 5,
                            downsample_factor: int = 25) -> tuple[np.ndarray, ...]:
    """Load galaxy image, subtract background, pad to square, smooth, downsample.
    Returns: (ngc_square, ngc_conv, ngc_resized, numPix_resized)"""

def decompose_shapelets(image_2d: np.ndarray, n_max: int, beta: float,
                        deltaPix: float = 1.0) -> np.ndarray:
    """Decompose 2D image into shapelet coefficients. Returns: coeff array."""
```

### `src/physics_model.py`

```python
# --- Grid utilities ---
def make_grid(numPix: int, deltapix: float) -> tuple[np.ndarray, np.ndarray]:
def make_grid_2d(numPix: int, deltaPix: float) -> tuple[np.ndarray, np.ndarray]:
def image2array(image: np.ndarray) -> np.ndarray:
def array2image(array: np.ndarray) -> np.ndarray:
def re_size(image: np.ndarray, factor: int) -> np.ndarray:

# --- Shapelets ---
def pre_calc_shapelets(x, y, beta, n_max, center_x, center_y) -> tuple:
def iterate_n1_n2(n_max: int) -> Iterator[tuple[int, int, int]]:
def shapelet_function(x, y, amp, n_max, beta, center_x, center_y) -> np.ndarray:
def shapelet_decomposition(image_1d, x, y, n_max, beta, deltaPix, ...) -> np.ndarray:
def shapelet_basis_list(x, y, n_max, beta, center_x, center_y) -> list[np.ndarray]:

# --- Lens models ---
def spep_deflection(x, y, theta_E, gamma, e1, e2, center_x, center_y) -> tuple:
def shear_deflection(x, y, gamma1, gamma2) -> tuple:
def ray_shoot(x, y, kwargs_spemd, kwargs_shear) -> tuple:

# --- PSF ---
def fwhm2sigma(fwhm: float) -> float:
def gaussian_convolve(image, fwhm, pixel_size, truncation) -> np.ndarray:

# --- Image simulation ---
def simulate_image(numPix, deltaPix, supersampling_factor, fwhm, kwargs_source,
                   kwargs_spemd, kwargs_shear, apply_lens, apply_psf) -> np.ndarray:
def add_poisson_noise(image, exp_time) -> np.ndarray:
def add_background_noise(image, sigma_bkg) -> np.ndarray:
```

### `src/solvers.py`

```python
def build_response_matrix(numPix, deltaPix, supersampling_factor, fwhm,
                          n_max_recon, beta_recon, center_x, center_y,
                          kwargs_spemd, kwargs_shear, apply_lens) -> np.ndarray:
    """Build design matrix A (num_basis x num_pix)."""

def linear_solve(A, data_2d, background_rms, exp_time) -> tuple:
    """WLS solve. Returns: (params, model_2d)."""

def reduced_residuals(model, data, background_rms, exp_time) -> np.ndarray:
```

### `src/visualization.py`

```python
def plot_shapelet_decomposition(ngc_square, ngc_conv, ngc_resized,
                                 reconstructed, save_path) -> None:
def plot_lensing_stages(images, labels, save_path) -> None:
def plot_reconstruction(data, model, residuals, source_true,
                        source_recon, source_resid, save_path) -> None:
```

### `src/generate_data.py`

```python
def generate_all_data(image_path, output_dir) -> dict:
    """Generate all synthetic data and save to output_dir. Returns metrics dict."""
```
