# Design: Code Architecture for SMLFM Task

## Directory Layout

```
tasks/single_molecule_light_field/
├── README.md
├── requirements.txt
├── main.py                     # Entry point: runs full pipeline
├── data/
│   ├── raw_data.npz            # Bundled raw localisations (PeakFit-style columns)
│   └── meta_data.json          # All microscope and fitting parameters
├── plan/
│   ├── approach.md             # Algorithm description
│   └── design.md               # This file
├── src/
│   ├── preprocessing.py        # Load NPZ / extract PeakFit cols, scale, centre
│   ├── physics_model.py        # FourierMicroscope, MLA, lens assignment, alpha
│   ├── solvers.py              # 3D fitting (aberration pass + full pass)
│   └── visualization.py        # Plot helpers (MLA alignment, 3D, histograms)
├── notebooks/
│   └── single_molecule_light_field.ipynb
└── evaluation/
    ├── reference_outputs/
    │   ├── locs_3d.csv         # 3D localisation table (24,778 rows)
    │   ├── metrics.json        # Summary metrics
    │   ├── fig_3d_locs.png
    │   ├── fig_histograms.png
    │   └── fig_occurrences.png
    ├── fixtures/               # Per-function I/O fixtures (.npz)
    └── tests/                  # Unit, parity, integration tests
```

## Module Interfaces

### `src/preprocessing.py`

```python
def load_localizations(npz_path: Path, meta: dict) -> np.ndarray:
    """
    Accepts either:
    - a standardised (N, 8) localisation array in `raw_data.npz`, or
    - the bundled raw PeakFit-style array, from which the required columns are extracted

    Returns: locs_2d_csv (N, 8) — columns:
        [0] frame, [1] X µm, [2] Y µm, [3] sigma_X µm,
        [4] sigma_Y µm, [5] intensity ph, [6] background ph, [7] precision µm
    """

def center_localizations(locs_2d_csv: np.ndarray) -> np.ndarray:
    """Subtract mean X and Y. Returns copy."""
```

### `src/physics_model.py`

```python
def build_microscope(meta: dict) -> smlfm.FourierMicroscope:
    """
    Key attributes of returned object:
        .bfp_radius       — back focal plane radius (µm)
        .magnification    — total magnification to camera
        .pixel_size_sample — pixel size in sample space (µm)
        .rho_scaling      — converts image µm to normalised pupil coords
        .mla_to_uv_scale  — converts MLA lattice spacing to pupil coords
        .mla_to_xy_scale  — converts MLA lattice spacing to image µm
    """

def build_mla(meta: dict) -> smlfm.MicroLensArray:
    """
    Generates hexagonal lattice, rotates by mla_rotation degrees.
    Key attributes:
        .lens_centres (K, 2) — X, Y in lattice-spacing units
        .centre (2,) — origin in lattice-spacing units
    """

def assign_to_lenses(
    locs_2d_csv: np.ndarray,
    mla: smlfm.MicroLensArray,
    lfm: smlfm.FourierMicroscope,
) -> smlfm.Localisations:
    """
    Returns Localisations with locs_2d (N, 13):
        [0] frame, [1] U, [2] V, [3] X µm, [4] Y µm,
        [5] sigma_X, [6] sigma_Y, [7] intensity, [8] background,
        [9] precision, [10] alpha_U, [11] alpha_V, [12] lens_index
    """

def compute_alpha_model(
    lfl: smlfm.Localisations,
    lfm: smlfm.FourierMicroscope,
    model: str = "INTEGRATE_SPHERE",
    worker_count: int = 0,
) -> None:
    """Populates filtered_locs_2d[:, 10:12] (alpha_U, alpha_V) in-place."""
```

### `src/solvers.py`

```python
def fit_aberrations(
    lfl: smlfm.Localisations,
    lfm: smlfm.FourierMicroscope,
    meta: dict,
    worker_count: int = 0,
) -> np.ndarray:
    """
    Returns correction (V, 5):
        [0] U, [1] V, [2] dx µm, [3] dy µm, [4] n_points_used
    """

def fit_3d_localizations(
    lfl: smlfm.Localisations,
    lfm: smlfm.FourierMicroscope,
    meta: dict,
    worker_count: int = 0,
    progress_func=None,
) -> np.ndarray:
    """
    Returns locs_3d (M, 8):
        [0] X µm, [1] Y µm, [2] Z µm (z_calib applied),
        [3] lateral_err µm, [4] axial_err µm,
        [5] n_views, [6] photons, [7] frame
    """
```

### `src/visualization.py`

```python
def plot_mla_alignment(fig, lfl, mla, lfm) -> Figure
def plot_3d_locs(fig, locs_3d, max_lateral_err=None, min_views=None) -> Figure
def plot_histograms(fig, locs_3d, max_lateral_err=None, min_views=None) -> Figure
def plot_occurrences(fig, locs_3d, max_lateral_err=None, min_views=None) -> Figure
```

## Data Formats

### Input: `raw_data.npz`
- Key `localizations_2d` contains the bundled raw PeakFit-style array
- Raw columns used by preprocessing:
  col 0: frame, col 7: background, col 8: intensity, col 9: X (pixels),
  col 10: Y (pixels), col 12: sigma (pixels), col 13: precision (nm)
- `src.preprocessing.load_localizations` converts these columns into the
  internal standardised `(N, 8)` array in physical units

### Output: `locs_3d.csv`
- No header lines, comma-separated
- 8 columns: x, y, z, lateral_err, axial_err, n_views, photons, frame (all in µm)

## External Dependencies

All algorithm logic is implemented directly in `src/` using standard scientific Python:
- `numpy` — array operations, OLS fitting (`np.linalg.lstsq`)
- `scipy` — hexagonal lattice generation, spatial utilities
- `scikit-learn` — k-NN search for lens assignment
- `matplotlib` — visualisation

No task-specific research packages are required.
