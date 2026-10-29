# Code Architecture: Shack-Hartmann Wavefront Reconstruction

## Directory Layout

```
tasks/shack-hartmann/
├── main.py                        # Pipeline entry point
├── data/
│   ├── raw_data.npz               # WFS images, response matrix, DM modes, aperture
│   ├── ground_truth.npz           # True wavefront phase at each WFE level
│   └── meta_data.json             # Optical and sensor parameters
├── src/
│   ├── preprocessing.py           # Data loading (strips batch dim)
│   ├── physics_model.py           # Centroid estimator, reconstruction matrix, NCC/NRMSE
│   ├── solvers.py                 # Per-level pipeline: centroid → Tikhonov → phase
│   ├── visualization.py           # WFS image, wavefront maps, metrics plots
│   └── generate_data.py           # HCIPy simulation (data creation)
└── evaluation/
    ├── metrics.json               # Per-level NCC/NRMSE and timing baselines/boundaries
    └── reference_outputs/
        └── reconstruction.npz    # Output of main.py
```

## Module Signatures

### `src/preprocessing.py`

```python
def load_raw_data(npz_path: str) -> dict:
    """Load raw_data.npz and strip batch dimension.
    Returns dict with keys:
        response_matrix    (N_slopes, N_modes)          float32
        wfs_images         (N_levels, H, W)             float32  [photons]
        ref_image          (H, W)                       float32  [photons]
        detector_coords_x  (H, W)                       float32  [m]
        detector_coords_y  (H, W)                       float32  [m]
        subap_map          (H, W)                       int32
        dm_modes           (N_modes, N_pupil_px)        float32
        aperture           (N_pupil_px,)                float32
    """

def load_ground_truth(npz_path: str) -> dict:
    """Load ground_truth.npz and strip batch dimension.
    Returns dict with keys:
        wavefront_phases  (N_levels, N_pupil_px) float32  [rad at lambda_wfs]
    """
```

### `src/physics_model.py`

```python
def estimate_slopes(
    wfs_image: np.ndarray,         # (N_det,) or (H, W) float
    ref_image: np.ndarray,         # (N_det,) or (H, W) float
    detector_coords_x: np.ndarray, # (N_det,) or (H, W) float [m]
    detector_coords_y: np.ndarray, # (N_det,) or (H, W) float [m]
    subap_map: np.ndarray,         # (N_det,) or (H, W) int32  (-1=invalid, 0..N_valid-1)
    n_valid_subaps: int,
) -> np.ndarray:                   # (2*n_valid_subaps,) [all Δcx | all Δcy]
    """Weighted-centroid slope estimation.
    Output ordering: [Δcx_0,…,Δcx_{N-1}, Δcy_0,…,Δcy_{N-1}] (x first, then y).
    """

def compute_reconstruction_matrix(
    response_matrix: np.ndarray,   # (N_slopes, N_modes)
    rcond: float = 1e-3,
) -> np.ndarray:                   # (N_modes, N_slopes)
    """Tikhonov pseudo-inverse M = R⁺ via truncated SVD."""

def reconstruct_wavefront(
    slopes: np.ndarray,                 # (N_slopes,)  slope differences
    reconstruction_matrix: np.ndarray,  # (N_modes, N_slopes)
    dm_modes: np.ndarray,               # (N_modes, N_pupil_px)
    wavelength: float,                  # [m]
) -> np.ndarray:                        # (N_pupil_px,) [rad]
    """Phase = 4pi/lambda * dm_modes.T @ (M @ slopes)."""

def compute_ncc(estimate, reference, mask=None) -> float
def compute_nrmse(estimate, reference, mask=None) -> float
```

### `src/solvers.py`

```python
def reconstruct_all_levels(
    wfs_images:          np.ndarray,   # (N_levels, N_det)
    ref_image:           np.ndarray,   # (N_det,)
    detector_coords_x:   np.ndarray,   # (N_det,)
    detector_coords_y:   np.ndarray,   # (N_det,)
    subap_map:           np.ndarray,   # (N_det,) int32
    response_matrix:     np.ndarray,   # (N_slopes, N_modes)
    dm_modes:            np.ndarray,   # (N_modes, N_pupil_px)
    aperture:            np.ndarray,   # (N_pupil_px,)
    wavelength:          float,
    n_valid_subaps:      int,
    rcond:               float = 1e-3,
    ground_truth_phases: np.ndarray = None,  # (N_levels, N_pupil_px)
) -> dict:
    """Reconstruct wavefront phase for every WFE level.
    Returns dict with keys:
        reconstructed_phases    (N_levels, N_pupil_px)  float32
        reconstruction_time_s   float   (centroid + reconstruction, all levels)
        ncc_per_level           (N_levels,)  float64  (if GT given)
        nrmse_per_level         (N_levels,)  float64  (if GT given)
    """
```

### `src/visualization.py`

```python
def plot_wfs_image(image, det_shape, title=None, output_path=None) -> Figure
    """Raw SH-WFS detector image (log scale)."""

def plot_wavefront_comparison(
    gt_phases, recon_phases, aperture, pupil_shape,
    wfe_levels_nm, ncc_arr, nrmse_arr, output_path) -> Figure
    """N_levels × 3 grid: Ground truth | Reconstructed | Error."""

def plot_metrics_vs_wfe(wfe_levels_nm, ncc_arr, nrmse_arr, ..., output_path) -> Figure
def plot_dm_modes(dm_modes, aperture, pupil_shape, n_show=6, output_path) -> Figure
def plot_response_singular_values(response_matrix, rcond=1e-3, output_path) -> Figure
```

## Data Flow

```
raw_data.npz
  ├─ wfs_images ────────┐
  ├─ ref_image ─────────┼──→ estimate_slopes()  ──→ slopes (N_slopes,)
  ├─ detector_coords_x  ┤
  ├─ detector_coords_y  ┤
  └─ subap_map ─────────┘

  ├─ response_matrix ──→ compute_reconstruction_matrix() ──→ M (N_modes × N_slopes)
                                                              ↓
                                                    reconstruct_wavefront()
                                                              ↓
                                              reconstructed_phases (N_levels × N_px²)
                                              ncc_per_level, nrmse_per_level
                                              reconstruction_time_s

ground_truth.npz
  └─ wavefront_phases ──→ compute_ncc / compute_nrmse (vs reconstructed)

                                ↓
                      evaluation/reference_outputs/reconstruction.npz
```
