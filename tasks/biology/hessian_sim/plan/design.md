# Hessian-SIM: Code Architecture

## Module layout

```
src/
├── physics_model.py     # OTF, Fourier shift helpers
├── preprocessing.py     # Pattern frequency + modulation/phase estimation
├── solvers.py           # Wiener SIM, Hessian / TV denoising, running avg
└── visualization.py     # Plotting, NCC / NRMSE metrics
```

## `src/physics_model.py`

```python
def generate_otf(n: int, na: float, wavelength: float, pixel_size: float) -> np.ndarray:
    """Synthetic cone-shaped OTF, shape (n, n), max-normalised."""

def pad_to_size(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Centre zero-pad a 2D array."""

def dft_conv(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """FFT-based linear convolution using centred spectra."""

def shift_otf(H_2n: np.ndarray, kx: float, ky: float, n: int) -> np.ndarray:
    """Apply the Fourier shift theorem to move an OTF by (kx, ky)."""
```

## `src/preprocessing.py`

```python
def estimate_sim_parameters(
    raw_frames: np.ndarray,   # (nframes, sy, sx)
    otf: np.ndarray,          # (n, n)
    nangles: int,
    nphases: int,
    n: int,
    pg: int,
    fanwei: int,
    regul: float,
    spjg: Sequence[int],
    beishu_an: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pattern-frequency coordinates (zuobiaox, zuobiaoy),
    each (nangles*nphases, 1) in 2n-grid pixel units."""

def estimate_modulation_and_phase(
    raw_frames: np.ndarray,
    otf: np.ndarray,
    zuobiaox: np.ndarray,
    zuobiaoy: np.ndarray,
    nangles: int,
    nphases: int,
    n: int,
    regul: float,
    spjg: Sequence[int],
    wavelength: float,
    beishu_an: int = 1,
    bg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (c6, angle6), each length nangles*(nphases-1)."""
```

Internal helpers: `_phase_matrix`, `_compute_merit`, `_emd_decompose`.

## `src/solvers.py`

```python
def wiener_sim_reconstruct(
    raw_frames: np.ndarray,   # (total_frames, sy, sx)
    otf: np.ndarray,
    zuobiaox: np.ndarray,
    zuobiaoy: np.ndarray,
    c6: np.ndarray,
    angle6: np.ndarray,
    nangles: int,
    nphases: int,
    n: int,
    weilac: float,
    regul: float,
    spjg: Sequence[int],
    beishu_re: int = 1,
    starframe: int = 0,
    bg: np.ndarray | None = None,
) -> np.ndarray:
    """Return a (n_timepoints, 2*sy, 2*sx) float32 stack."""

def hessian_denoise(stack: np.ndarray, mu: float = 150.0,
                    sigma_z: float = 1.0, n_iter: int = 100) -> np.ndarray:
    """Split-Bregman Hessian denoiser; input and output shape match."""

def tv_denoise(stack: np.ndarray, mu: float = 150.0,
               n_iter: int = 100) -> np.ndarray:
    """Split-Bregman anisotropic TV denoiser."""

def running_average(stack: np.ndarray, window: int = 3) -> np.ndarray:
    """Centred three-frame temporal boxcar."""

def forward_diff(x: np.ndarray, step: float, dim: int) -> np.ndarray: ...
def back_diff(x: np.ndarray, step: float, dim: int) -> np.ndarray: ...
```

## `src/visualization.py`

```python
def compute_widefield(raw_frames: np.ndarray, nangles: int, nphases: int) -> np.ndarray: ...
def upsample_widefield(widefield: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray: ...
def plot_comparison(wf, wiener, hessian, tv, out_path: str) -> None: ...
def plot_line_profiles(wiener, hessian, tv, out_path: str) -> None: ...
def ncc(a: np.ndarray, b: np.ndarray) -> float: ...
def nrmse(est: np.ndarray, ref: np.ndarray) -> float: ...
```

## Data flow

```
raw_data.npz (1, 72, 256, 256)
        │
        ▼
raw[0]  (72, 256, 256) ─────────────► compute_widefield ──► wf (256, 256)
        │
        ▼
generate_otf ──► otf (512, 512)
        │
        ▼
estimate_sim_parameters        ─► zuobiaox, zuobiaoy (6, 1)
        │
        ▼
estimate_modulation_and_phase  ─► c6 (4,), angle6 (4,)
        │
        ▼
wiener_sim_reconstruct         ─► wiener (12, 512, 512)
        │                           │
        ├──► hessian_denoise ─────► hessian (12, 512, 512)
        └──► tv_denoise ──────────► tv (12, 512, 512)
                                    │
                                    ▼
                           running_average → temporal stack
```

## main.py

Drives every stage, hard-codes all solver hyperparameters as top-level
constants (`_WIENER`, `_HESSIAN`, `_TV`, `_PARAM_EST`), and writes the
full set of reference outputs to `evaluation/reference_outputs/`:

- `wiener_sim.npz`, `hessian_sim.npz`, `tv_sim.npz`
- `running_avg_wiener.npz`, `running_avg_hessian.npz`
- `sim_parameters.npz` (zuobiaox, zuobiaoy, c6, angle6)
- `comparison.png`, `line_profiles.png`

Evaluation metrics are dumped to `evaluation/metrics.json`.
