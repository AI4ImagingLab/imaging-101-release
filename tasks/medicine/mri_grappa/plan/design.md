# Design: MRI GRAPPA Reconstruction

## Module Architecture

```
main.py
  ├── src/preprocessing.py    → Load k-space, undersample, extract ACS
  ├── src/physics_model.py    → Centered FFT/IFFT, RSS combine
  ├── src/solvers.py          → GRAPPA kernel calibration + interpolation
  └── src/visualization.py    → SSIM, NCC, NRMSE metrics + plots
```

## Data Flow

```
raw_data.npz
    │
    ▼
preprocessing.prepare_data(R=2, acs_width=20)
  ├── kspace_us (128, 128, 8)     # zeros at missing lines
  ├── calib (20, 128, 8)          # fully-sampled ACS
  ├── kspace_full (128, 128, 8)   # for reference
  └── phantom (128, 128)
          │
          ▼
solvers.grappa_reconstruct(kspace_us, calib)
  ├── view_as_windows → unique sampling geometries
  ├── for each geometry: least-squares kernel calibration
  └── apply weights to fill all holes
          │
          ▼
physics_model.centered_ifft2() → sos_combine()
          │
          ▼
visualization.compute_metrics() + plot_*()
```

## Function Signatures

### src/preprocessing.py
```python
def load_observation(data_dir) -> dict:
def load_ground_truth(data_dir) -> ndarray:
def get_full_kspace(obs_data) -> ndarray:   # (Nx, Ny, Nc) complex128
def undersample_kspace(kspace_full, R, acs_width) -> (kspace_us, calib, mask):
def prepare_data(data_dir, R, acs_width) -> (kspace_us, calib, kspace_full, phantom, meta):
```

### src/physics_model.py
```python
def centered_fft2(imspace) -> ndarray:
def centered_ifft2(kspace) -> ndarray:
def sos_combine(imspace) -> ndarray:
def zero_filled_recon(kspace_us) -> ndarray:
def fully_sampled_recon(kspace_full) -> ndarray:
```

### src/solvers.py
```python
def grappa_reconstruct(kspace_us, calib, kernel_size, lamda) -> ndarray:  # returns k-space
def grappa_image_recon(kspace_us, calib, kernel_size, lamda) -> ndarray:  # returns image
```

### src/visualization.py
```python
def compute_metrics(estimate, reference) -> dict:  # {nrmse, ncc, ssim}
def plot_reconstruction_comparison(...):
def plot_error_maps(...):
def plot_kspace(...):
def print_metrics(metrics, label):
```
