# Design: Code Architecture

## Module Overview

```
src/
├── preprocessing.py   Data loading and masking
├── physics_model.py   ADI rotation model + KL basis
├── solvers.py         KLIP-ADI reconstruction pipeline
└── visualization.py   Plotting utilities + SNR computation
```

---

## src/preprocessing.py

```python
def load_raw_data(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cube, angles, psf from raw_data.npz.
    Returns cube (N,H,W), angles (N,), psf (Hpsf,Wpsf)."""

def create_circular_mask(
    ny: int, nx: int, center: tuple,
    iwa: float, owa: float = None, device: str = 'cpu'
) -> torch.BoolTensor:
    """2D boolean mask: True where dist < iwa or dist > owa."""

def apply_circular_mask(
    cube: np.ndarray, center: tuple, iwa: float, owa: float = None
) -> np.ndarray:
    """Apply IWA/OWA mask to cube (set masked pixels to NaN)."""

def mean_subtract_frames(cube: np.ndarray) -> np.ndarray:
    """Per-frame spatial mean subtraction (nanmean, ignores NaN)."""
```

---

## src/physics_model.py

```python
def rotate_frames(
    images: torch.Tensor,  # (B, N, H, W)
    angles: torch.Tensor,  # (N,) degrees
) -> torch.Tensor:         # (B, N, H, W)
    """Batched bilinear rotation via affine_grid + grid_sample.
    Rotation angle convention: positive = counter-clockwise.
    Derotation uses negative angles internally."""

def compute_kl_basis_svd(reference_flat: torch.Tensor, K_max: int) -> torch.Tensor:
    """Top-K right singular vectors of reference_flat (N, n_pix)."""

def compute_kl_basis_pca(reference_flat: torch.Tensor, K_max: int) -> torch.Tensor:
    """Low-rank PCA via torch.pca_lowrank (center=False)."""

def compute_kl_basis_eigh(reference_flat: torch.Tensor, K_max: int) -> torch.Tensor:
    """KL basis via eigendecomposition of covariance R R^T (original KLIP)."""

def compute_kl_basis(
    reference_flat: torch.Tensor,
    K_max: int,
    method: str = 'svd',  # 'svd' | 'pca' | 'eigh'
) -> torch.Tensor:        # (n_pix, K_max)
    """Dispatch to the selected basis computation method."""
```

---

## src/solvers.py

```python
def compute_psf_residuals(
    cube: np.ndarray,             # (N, H, W)
    K_klip: int | list[int],
    method: str = 'svd',
    device: str = 'cpu',
) -> torch.Tensor:                # (n_K, N, H, W)
    """Core KLIP: mean-subtract, build KL basis, project, subtract."""

def derotate_cube(
    residuals: torch.Tensor,      # (n_K, N, H, W)
    angles: np.ndarray,           # (N,)
) -> torch.Tensor:                # (n_K, N, H, W)
    """Derotate each frame by its parallactic angle."""

def combine_frames(
    derotated: torch.Tensor,      # (n_K, N, H, W)
    statistic: str = 'mean',      # 'mean' | 'median'
) -> torch.Tensor:                # (n_K, H, W)
    """Temporal combination (nanmean or nanmedian)."""

def klip_adi(
    cube: np.ndarray,
    angles: np.ndarray,
    K_klip: int | list[int],
    iwa: float = None,
    center: tuple = None,
    method: str = 'svd',
    statistic: str = 'mean',
    device: str = 'cpu',
) -> np.ndarray:                  # (H,W) or (n_K,H,W)
    """End-to-end KLIP+ADI: mask → PSF-subtract → derotate → combine."""
```

---

## src/visualization.py

```python
def plot_raw_frame(
    frame: np.ndarray, center=None, iwa=None,
    vmin=None, vmax=None, log_scale=True,
    scalebar_length=None, scalebar_label=None,
    title=None, output_path=None, figsize=(5,5),
) -> tuple[Figure, Axes]:
    """Plot single ADI frame with IWA circle and scalebar."""

def plot_klip_result(
    image: np.ndarray, center=None, iwa=None,
    vmin=None, vmax=None,
    scalebar_length=None, scalebar_label=None,
    planet_xy=None,
    title=None, output_path=None, figsize=(5,5),
    xlim_half=None, ylim_half=None,
) -> tuple[Figure, Axes]:
    """Plot KLIP detection map with companion cross-hair."""

def compute_snr(
    image: np.ndarray,
    planet_x: float, planet_y: float,
    fwhm: float,
    exclude_nearest: int = 0,
) -> float:
    """Mawet et al. (2014) two-sample t-test SNR for a companion."""
```

---

## Data flow

```
raw_data.npz
    │
    ▼
load_raw_data()         cube (N,H,W), angles (N,), psf (Hpsf,Wpsf)
    │
    ▼
klip_adi()
  ├─ apply_circular_mask()     NaN-mask inner working angle
  ├─ compute_psf_residuals()
  │   ├─ mean_subtract_frames()
  │   ├─ compute_kl_basis()    KL eigenvectors from full-frame library
  │   └─ project + subtract    residuals (n_K, N, H, W)
  ├─ derotate_cube()           align companions to common sky frame
  └─ combine_frames()          mean/median → detection map (n_K, H, W)
    │
    ▼
compute_snr()           SNR of Beta Pic b
```

---

## Tensor conventions

- All PyTorch tensors use `float32`.
- Image axes: `(rows, cols)` = `(y, x)`.
- Pixel coordinate origin: `(0,0)` at bottom-left (`origin='lower'` in imshow).
- Angles: degrees, positive = counter-clockwise.
- NaN pixels are excluded from PCA and combination (`nan_to_num` before SVD,
  `nanmean`/`nanmedian` when combining).
