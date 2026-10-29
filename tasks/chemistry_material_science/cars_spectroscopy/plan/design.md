# Code Architecture

## Module Overview

```
src/
├── physics_model.py    # Constants, LineStrength, forward_operator
├── preprocessing.py    # load_and_preprocess_data
├── solvers.py          # run_inversion (scipy least_squares)
├── visualization.py    # compute_metrics, plot_inversion_result
└── generate_data.py    # Synthetic data generation
```

## Function Signatures

### src/physics_model.py

```python
def gaussian_line(w: np.ndarray, w0: float, sigma: float) -> np.ndarray
def lorentz_line(w: np.ndarray, w0: float, sigma: float) -> np.ndarray
def asym_Gaussian(w, w0, sigma, k, a_sigma, a_k, offset, power_factor=1.) -> np.ndarray
def asym_Voigt(w, w0, sigma, k, a_sigma, a_k, sigma_L_l, sigma_L_h, offset, power_factor=1.) -> np.ndarray
def downsample(w, w_fine, spec_fine, mode='local_mean') -> np.ndarray

class LineStrength:
    def __init__(self, species: str = 'N2')
    def int_corr(self, j, branch=0) -> tuple[float, float]
    def term_values(self, v, j, mode='sum') -> float | np.ndarray
    def line_pos(self, v, j, branch=0) -> float | np.ndarray
    def pop_factor(self, T, v, j, branch=0, del_Tv=0.0) -> float | np.ndarray
    def doppler_lw(self, T, nu_0=2300.) -> float

def forward_operator(x_params: dict) -> np.ndarray
```

### src/preprocessing.py

```python
def load_and_preprocess_data(
    raw_signal: np.ndarray,
    nu_axis: np.ndarray,
    noise_level: float = 0.0
) -> tuple[np.ndarray, np.ndarray]
```

### src/solvers.py

```python
def run_inversion(
    measured_signal: np.ndarray,
    nu_axis: np.ndarray,
    initial_guesses: dict
) -> dict  # keys: 'best_params', 'y_pred', 'cost', 'nfev', 'success'
```

### src/visualization.py

```python
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    params_pred: dict,
    params_true: dict | None = None
) -> dict

def plot_inversion_result(
    nu_axis, y_measured, y_pred, params_pred,
    y_ground_truth=None, params_true=None,
    save_path='inversion_result.png'
) -> None
```

### src/generate_data.py

```python
def generate_and_save(data_dir: str = 'data', seed: int = 42) -> None
```

## Data Flow

```
raw_data.npz → preprocessing.load_and_preprocess_data
                        ↓
               processed_signal
                        ↓
            solvers.run_inversion  ←  physics_model.forward_operator (called iteratively)
                        ↓
                  best_params, y_pred
                        ↓
            visualization.compute_metrics + plot_inversion_result
                        ↓
              metrics.json + inversion_result.png
```
