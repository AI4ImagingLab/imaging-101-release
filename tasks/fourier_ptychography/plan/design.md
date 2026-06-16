# Design: Fourier Ptychography Code Architecture

## Module Overview

```
tasks/fourier_ptychography/
├── main.py                    # Pipeline entry point
├── src/
│   ├── generate_data.py       # Synthetic FPM dataset generation
│   ├── preprocessing.py       # Data loading and reconstruction initialization
│   ├── physics_model.py       # FPM forward model (pupil bandpass, k-shift)
│   ├── solvers.py             # qNewton reconstruction solver
│   ├── utils.py               # fft2c, ifft2c, circ, etc.
│   └── visualization.py       # Object, pupil, raw-data display
├── data/
│   ├── raw_data.npz           # Generated FPM dataset (ptychogram, encoder)
│   ├── meta_data.json         # Physical parameters
│   └── ground_truth.npz       # Ground-truth real-space USAF phase object
└── evaluation/
    └── reference_outputs/
        ├── recon.hdf5         # Saved reconstruction (object, probe, error)
        ├── metrics.json       # Final error and quality metrics
        └── reconstruction_summary.png
```

## Function Signatures

### src/generate_data.py

```python
def generate_usaf_object(No, dxp, groups=(5,6,7,8), phi_max=pi/2) -> ndarray[No,No,complex]:
    """USAF 1951 phase test object: amplitude=1, binary phase in {0, phi_max}."""

def generate_led_array(n_leds_side=11, led_pitch=4e-3, z_led=60e-3)
    -> (ndarray[J,2,float], ndarray[J,2,float]):
    """LED positions [m] and angles [rad]."""

def compute_pupil(Nd, dxp, wavelength, NA) -> ndarray[Nd,Nd,float]:
    """Binary pupil mask in fftshift convention."""

def simulate_fpm_images(obj, pupil, encoder, z_led, wavelength, dxp, Nd,
                        bit_depth=10, seed=42) -> ndarray[J,Nd,Nd,float32]:
    """Full FPM image stack with Poisson noise."""

def main(output_path=None) -> Path:
    """Generate and save raw_data.npz, ground_truth.npz, meta_data.json."""
```

### src/preprocessing.py

```python
def load_experimental_data(data_dir) -> PtyData:
    """Load raw_data.npz + meta_data.json → PtyData."""

def setup_reconstruction(data: PtyData) -> PtyState:
    """Initialize pupil (circ) and object (upsampled from LR images)."""

def setup_params() -> SimpleNamespace:
    """FPM params: positionOrder='NA', probeBoundary=True, adaptiveDenoising."""

def setup_monitor(figure_update_freq=10) -> SimpleNamespace:
    """Low-verbosity monitor (no interactive GUI)."""

def save_results(state: PtyState, filepath) -> None:
    """Save object, probe, error to HDF5 (PtyLab-compatible 6D schema)."""
```

### src/physics_model.py

```python
def compute_pupil_mask(Nd, dxp, wavelength, NA) -> ndarray[Nd,Nd,float]:
    """Binary pupil in fftshift convention."""

def compute_kspace_shift(led_pos, z_led, wavelength, Nd, dxp) -> ndarray[2]:
    """k-space shift [pixels] for given LED position."""

def fpm_forward_single(obj_spectrum, pupil, shift_px, Nd)
    -> (ndarray[Nd,Nd,float], ndarray[Nd,Nd,complex]):
    """Single LR image: extract sub-spectrum → pupil → IFFT → |.|²."""

def forward_model_stack(obj, pupil, encoder, z_led, wavelength, dxp, Nd)
    -> ndarray[J,Nd,Nd,float]:
    """Full image stack for all LEDs."""
```

### src/solvers.py

```python
def run_qnewton(state: PtyState, data: PtyData, params, monitor=None,
                num_iterations=200, beta_probe=1.0, beta_object=1.0,
                reg_object=1.0, reg_probe=1.0) -> PtyState:
    """qNewton FPM solver: Hessian-scaled PIE update, NA-ordered LEDs."""

def compute_reconstruction_error(ptychogram, ptychogram_est) -> float:
    """Normalized amplitude RMSE."""
```

### src/visualization.py

```python
def complex_to_hsv(arr, max_amp=None) -> ndarray[Ny,Nx,3]:
    """Complex → HSV RGB (hue=phase, value=amplitude)."""

def plot_complex_image(arr, ax, title='', pixel_size_um=None, max_amp=None):
    """HSV display on Axes."""

def get_object_realspace(reconstruction) -> ndarray[No,No,complex]:
    """Extract real-space object O(r) = ifft2c(reconstruction.object)."""

def get_pupil(reconstruction) -> ndarray[Np,Np,complex]:
    """Extract reconstructed pupil P̃(q) from reconstruction.probe."""

def plot_raw_data_mean(ptychogram, ax, title='', log_scale=False):
    """Mean of all LR images."""

def plot_brightfield_image(ptychogram, ax, title=''):
    """Image with highest total intensity (on-axis LED)."""

def plot_reconstruction_summary(reconstruction, experimental_data, error_history,
                                 figsize=(15,5)) -> Figure:
    """6-panel: mean raw | bright-field | obj amp | obj phase | pupil | error."""
```

## Data Flow

```
raw_data.npz + meta_data.json
    ↓ load_experimental_data()
PtyData (ptychogram, encoder, wavelength, NA, magnification, …)
    ↓ setup_reconstruction()
PtyState (pupil=circ, object=upsampled from LR)
    ↓ run_qnewton() [200 iterations, NA-ordered LEDs]
PtyState (converged: k-space object Õ(q) + pupil P̃(q))
    ↓ save_results()
recon.hdf5 + metrics.json
    ↓ plot_reconstruction_summary()
reconstruction_summary.png
```

## FPM vs CP: key differences

| Aspect | CP | FPM |
|---|---|---|
| Probe/pupil location | Real space P(r) | k-space P̃(q) |
| Reconstructed object | Real space O(r) | k-space Õ(q) = FT{O(r)} |
| Forward model | ψ = P·O_j, I=\|FT{ψ}\|² | I=\|IFT{P̃·Õ_j}\|² |
| Default solver | mPIE (ePIE + Nesterov momentum) | qNewton (Hessian-scaled PIE) |
| Position order | random | NA (bright-field first) |
| Metric domain | Real space phase | Real space phase via ifft2c |
