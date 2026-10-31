# Design: Electron Ptychography

## Data Flow

```
data/raw_data.h5 + data/probe.h5 + data/meta_data.json
        |
        v
  preprocessing.py
    load_data() → datacube (48,48,192,192), probe (192,192)
    load_metadata() → dict
    calibrate_datacube() → calibrated datacube
    compute_dp_mean() → (192,192)
    estimate_probe_size() → radius, center
    compute_virtual_images() → BF (48,48), DF (48,48)
    compute_bf_mask() → (192,192) bool
        |
        v
  physics_model.py
    compute_com() → com_x (48,48), com_y (48,48)
    ptychographic_forward() → predicted intensities
        |
        v
  solvers.py
    solve_dpc() → DPC phase (48,48)
    solve_parallax() → parallax phase (upsampled)
    solve_ptychography() → complex object, probe
        |
        v
  visualization.py
    compute_metrics() → {ncc, nrmse}
    plot_phase_comparison() → figure
    plot_probe() → figure
    plot_convergence() → figure
```

## Module Specifications

### src/preprocessing.py

```python
def load_data(data_dir: str) -> tuple:
    """Load 4D-STEM datacube and vacuum probe from HDF5 files.

    Parameters
    ----------
    data_dir : str
        Path to data/ directory containing raw_data.h5 and probe.h5.

    Returns
    -------
    datacube : py4DSTEM.DataCube
        4D-STEM dataset, shape (Rx, Ry, Qx, Qy).
    probe : np.ndarray
        Vacuum probe intensity, shape (Qx, Qy).
    """

def load_metadata(data_dir: str) -> dict:
    """Load imaging parameters from meta_data.json.

    Returns
    -------
    meta : dict
        Keys: energy_eV, R_pixel_size_A, convergence_semiangle_mrad,
        scan_shape, detector_shape, defocus_A, dp_mask_threshold,
        com_rotation_deg.
    """

def calibrate_datacube(
    datacube,
    probe: np.ndarray,
    R_pixel_size: float,
    convergence_semiangle: float,
    thresh_upper: float = 0.7,
) -> tuple:
    """Estimate probe size and set calibration on the datacube.

    Returns
    -------
    probe_radius_pixels : float
    probe_center : tuple[float, float]
        (qx0, qy0) center of the probe.
    """

def compute_dp_mean(datacube) -> np.ndarray:
    """Compute the mean diffraction pattern, shape (Qx, Qy)."""

def compute_virtual_images(
    datacube,
    center: tuple,
    radius: float,
    expand: float = 2.0,
) -> tuple:
    """Compute bright-field and dark-field virtual images.

    Returns
    -------
    bf : np.ndarray, shape (Rx, Ry)
    df : np.ndarray, shape (Rx, Ry)
    """

def compute_bf_mask(dp_mean: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """Threshold mean DP to create bright-field disk mask, shape (Qx, Qy)."""
```

### src/physics_model.py

```python
def compute_com(
    datacube,
    mask: np.ndarray = None,
) -> tuple:
    """Compute center-of-mass of each diffraction pattern.

    Parameters
    ----------
    datacube : py4DSTEM.DataCube
    mask : np.ndarray, optional
        Binary mask for the BF disk, shape (Qx, Qy).

    Returns
    -------
    com_x : np.ndarray, shape (Rx, Ry)
    com_y : np.ndarray, shape (Rx, Ry)
    """

def ptychographic_forward(
    obj: np.ndarray,
    probe: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Ptychographic forward model: compute predicted diffraction intensities.

    I_j(k) = |F{P(r - r_j) * O(r)}|^2

    Parameters
    ----------
    obj : np.ndarray, complex, shape (Nx, Ny)
    probe : np.ndarray, complex, shape (Np, Np)
    positions : np.ndarray, int, shape (J, 2)

    Returns
    -------
    intensities : np.ndarray, shape (J, Np, Np)
    """
```

### src/solvers.py

```python
def solve_dpc(
    datacube,
    energy: float,
    dp_mask: np.ndarray,
    com_rotation: float,
) -> np.ndarray:
    """DPC phase reconstruction.

    Parameters
    ----------
    datacube : py4DSTEM.DataCube
    energy : float
        Beam energy in eV.
    dp_mask : np.ndarray
        BF disk mask.
    com_rotation : float
        Rotation angle in degrees (including 180-degree flip).

    Returns
    -------
    phase : np.ndarray, shape (Rx, Ry)
    """

def solve_parallax(
    datacube,
    energy: float,
    com_rotation: float,
    fit_aberrations_max_order: int = 3,
) -> tuple:
    """Parallax phase reconstruction with CTF correction.

    Returns
    -------
    phase : np.ndarray
        CTF-corrected upsampled phase image.
    aberrations : dict
        Fitted aberration coefficients (C1, rotation_Q_to_R_rads, transpose).
    """

def solve_ptychography(
    datacube,
    probe: np.ndarray,
    energy: float,
    defocus: float,
    com_rotation: float,
    transpose: bool,
    max_iter: int = 10,
    step_size: float = 0.5,
    batch_fraction: int = 4,
) -> tuple:
    """Single-slice ptychographic reconstruction.

    Returns
    -------
    object_phase : np.ndarray
        Phase of the reconstructed complex object (cropped).
    object_complex : np.ndarray
        Full complex object (cropped).
    probe_complex : np.ndarray
        Reconstructed complex probe.
    error_history : list[float]
        NMSE per iteration.
    """
```

### src/visualization.py

```python
def compute_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """Compute NCC and NRMSE between estimate and reference.

    Returns
    -------
    dict with keys 'ncc' and 'nrmse'.
    """

def plot_phase_comparison(
    dpc_phase: np.ndarray,
    parallax_phase: np.ndarray,
    ptycho_phase: np.ndarray,
    save_path: str = None,
) -> None:
    """Side-by-side comparison of DPC, parallax, and ptychographic phase images."""

def plot_reconstruction(
    object_phase: np.ndarray,
    probe_complex: np.ndarray,
    error_history: list,
    save_path: str = None,
) -> None:
    """Plot ptychographic reconstruction: object phase, probe, and convergence."""

def plot_virtual_images(
    bf: np.ndarray,
    df: np.ndarray,
    save_path: str = None,
) -> None:
    """Plot bright-field and dark-field virtual images."""
```
