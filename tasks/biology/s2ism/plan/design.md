# Code Architecture

## Module Layout

```
src/
├── preprocessing.py    # Shift vectors, magnification, misalignment
├── physics_model.py    # PSF estimation + forward model
├── solvers.py          # ML-EM reconstruction
├── visualization.py    # Plotting and metrics
└── generate_data.py    # Synthetic data generation
```

## Function Signatures

### `src/preprocessing.py`

```python
def shift_matrix(geometry: str = 'rect') -> np.ndarray
def rotation_matrix(theta: float) -> np.ndarray
def mirror_matrix(alpha: float) -> np.ndarray
def crop_shift(shift_exp: np.ndarray, geometry: str = 'rect') -> np.ndarray
def transform_shift_vectors(param: list, shift: np.ndarray) -> np.ndarray
def loss_shifts(x0, shift_exp: np.ndarray, shift_theor: np.ndarray, mirror: float) -> float
def svm_loss_minimizer(shift_m, shift_t, alpha_0, theta_0, tol, opt, mirror) -> tuple
def find_parameters(shift_exp: np.ndarray, geometry: str = 'rect', name: str = None,
                    alpha_0: float = 2, theta_0: float = 0.5) -> tuple
def calc_shift_vectors(parameters, geometry: str = 'rect') -> np.ndarray

def scalar_psf(r: np.ndarray, wl: float, na: float) -> np.ndarray
def rect(r: np.ndarray, d: float) -> np.ndarray
def scalar_psf_det(r, wl, na, pxdim, pxpitch, m) -> np.ndarray
def shift_value(m, wl_ex, wl_em, pxpitch, pxdim, na) -> float
def find_mag(shift, wl_ex, wl_em, pxpitch, pxdim, na) -> float

def gaussian_2d(params, x, y) -> np.ndarray
def gaussian_fit(image: np.ndarray) -> tuple
def find_misalignment(dset, pxpitch, mag, na, wl) -> tuple
def realign_psf(psf: np.ndarray) -> np.ndarray
```

### `src/physics_model.py`

```python
class GridFinder(psf_sim.GridParameters):
    def estimate(self, dset, wl_ex, wl_em, na) -> None

def psf_width(pxsizex, pxsizez, Nz, simPar, spad_size, stack='positive') -> int
def find_max_discrepancy(correlation, gridpar, mode, graph) -> float
def conditioning(gridPar, exPar=None, emPar=None, stedPar=None,
                 mode='Pearson', stack='positive', input_psf=None) -> tuple
def find_out_of_focus_from_param(pxsizex=None, exPar=None, emPar=None,
                                  grid=None, stedPar=None, mode='Pearson',
                                  stack='symmetrical', graph=False) -> tuple
def find_upsampling(pxsize_exp, pxsize_sim=4) -> int
def psf_estimator_from_data(data, exPar, emPar, grid, downsample=True,
                             stedPar=None, z_out_of_focus='ToFind',
                             n_photon_excitation=1, stack='symmetrical',
                             check_alignment=False) -> tuple
def forward_model(ground_truth: np.ndarray, psf: np.ndarray) -> np.ndarray
```

### `src/solvers.py`

```python
def partial_convolution_rfft(kernel, volume, dim1='ijk', dim2='jkl',
                              axis='jk', fourier=(False, False),
                              padding=None) -> torch.Tensor
def amd_update_fft(img, obj, psf_fft, psf_m_fft, eps) -> torch.Tensor
def amd_stop(o_old, o_new, pre_flag, flag, stop, max_iter,
             threshold, tot, nz, k) -> tuple
def max_likelihood_reconstruction(dset, psf, stop='fixed', max_iter=100,
                                   threshold=1e-3, rep_to_save='last',
                                   initialization='flat',
                                   process='gpu') -> tuple
```

### `src/visualization.py`

```python
def plot_results(ground_truth, ism_sum, reconstruction, save_path=None) -> Figure
def compute_metrics(ground_truth, reconstruction, ism_sum) -> dict
```

### `src/generate_data.py`

The data generation module is split into four small functions so each stage
of the pipeline can be tested independently. The PSF simulation step
specifically takes `pxsizez` as an explicit argument so that a unit test can
verify that the caller passes the optimal background plane distance.

```python
# Defaults
DEFAULT_NX = 201
DEFAULT_NZ = 2
DEFAULT_PXSIZEX_NM = 40
DEFAULT_DETECTOR_N = 5
DEFAULT_SIGNAL = 300

def make_tubulin_phantom(Nx: int = DEFAULT_NX, Nz: int = DEFAULT_NZ,
                          pxsizex: float = DEFAULT_PXSIZEX_NM,
                          signal: float = DEFAULT_SIGNAL,
                          seed: int = 42) -> np.ndarray
def make_psf_settings() -> tuple                     # (exPar, emPar)
def simulate_psfs(pxsizex: float, pxsizez: float, Nz: int,
                   exPar, emPar, normalize: bool = True) -> np.ndarray
def apply_forward_model_with_noise(ground_truth: np.ndarray,
                                    psf: np.ndarray,
                                    seed: int = 43) -> np.ndarray
def generate_data(output_dir: str = 'data', seed: int = 42) -> tuple
```

The orchestrator `generate_data` chains these steps:

```python
ground_truth = make_tubulin_phantom(...)
exPar, emPar = make_psf_settings()
optimal_bkg_plane, _ = find_out_of_focus_from_param(pxsizex, exPar, emPar,
                                                    mode='Pearson', stack='positive')
psf = simulate_psfs(pxsizex, optimal_bkg_plane, Nz, exPar, emPar)
measurements = apply_forward_model_with_noise(ground_truth, psf)
```

The critical contract `pxsizez = optimal_bkg_plane` is enforced both inside
`generate_data` and verified by `test_optimal_bkg_plane_used_for_pxsizez`.

## Data Flow

```
generate_data.py
  ├── make_tubulin_phantom()         → ground_truth (Nz, Ny, Nx)
  ├── make_psf_settings()             → exPar, emPar
  ├── find_out_of_focus_from_param()  → optimal_bkg_plane (nm)
  ├── simulate_psfs(pxsizez=optimal_bkg_plane)  → psf (Nz, Ny, Nx, Nch)
  ├── apply_forward_model_with_noise()→ measurements (Ny, Nx, Nch)
  └── Save:
       ├── raw_data.npz       (measurements, psf)
       ├── ground_truth.npz   (ground_truth)
       └── meta_data.json     (incl. optimal_bkg_plane_nm)

main.py
  ├── If data files missing → call generate_data()
  ├── Load raw_data.npz, ground_truth.npz, meta_data.json
  ├── max_likelihood_reconstruction(measurements, psf)
  ├── compute_metrics(gt, recon, ism_sum)
  └── Save reconstruction.npz, metrics.json, figure
```
