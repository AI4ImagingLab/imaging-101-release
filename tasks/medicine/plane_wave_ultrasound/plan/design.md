# Design: Plane Wave Ultrasound f-k Migration

## Module Layout

```
src/
  preprocessing.py   # DC removal, data loading helpers
  physics_model.py   # ERM velocity, Stolt mapping coordinates
  solvers.py         # fkmig() — core f-k migration; compound()
  visualization.py   # B-mode display, PSF/CNR metrics
```

## Function Signatures

### `src/preprocessing.py`

```python
def remove_dc(RF: np.ndarray) -> np.ndarray:
    """Subtract the global mean from the RF array.

    Parameters
    ----------
    RF : np.ndarray, shape (...) float
        Raw RF signals in ADC counts.

    Returns
    -------
    np.ndarray
        DC-removed RF signals, same shape and dtype as input.
    """

def load_dataset(npz_path: str, meta_path: str, dataset: str = 'fibers'):
    """Load and return RF signals plus acquisition parameters.

    Parameters
    ----------
    npz_path : str
        Path to data/raw_data.npz.
    meta_path : str
        Path to data/meta_data.json.
    dataset : {'fibers', 'cysts'}
        Which phantom to load.

    Returns
    -------
    RF : np.ndarray, shape (N_t, N_x, N_angles), float64
        DC-removed RF signals.
    params : dict
        Keys: c, fs, pitch, TXangle_rad (list), t0.
    """
```

### `src/physics_model.py`

```python
def erm_velocity(c: float, TXangle: float) -> float:
    """Compute the ERM (exploding reflector model) effective velocity.

    v = c / sqrt(1 + cos(theta) + sin(theta)^2)

    Parameters
    ----------
    c : float
        Speed of sound (m/s).
    TXangle : float
        Steering angle (rad).

    Returns
    -------
    float
        ERM velocity (m/s).
    """

def stolt_fkz(f: np.ndarray, Kx: np.ndarray,
               c: float, TXangle: float) -> np.ndarray:
    """Compute the Stolt-mapped frequency f_kz.

    f_kz = v * sqrt(Kx^2 + 4*f^2 / (beta^2 * c^2))

    Parameters
    ----------
    f : np.ndarray, shape (nf, nkx)
        Temporal frequency grid (Hz).
    Kx : np.ndarray, shape (nf, nkx)
        Lateral spatial frequency grid (cycles/m).
    c : float
        Speed of sound (m/s).
    TXangle : float
        Steering angle (rad).

    Returns
    -------
    np.ndarray, shape (nf, nkx)
        Migrated frequency grid f_kz (Hz).
    """

def steering_delay(nx: int, pitch: float, c: float,
                   TXangle: float, t0: float = 0.0) -> np.ndarray:
    """Per-element transmit delay for steering compensation.

    t_shift[e] = sin(theta) * ((nx-1)*(theta<0) - e) * pitch / c + t0

    Parameters
    ----------
    nx : int
        Number of array elements.
    pitch : float
        Element pitch (m).
    c : float
        Speed of sound (m/s).
    TXangle : float
        Steering angle (rad).
    t0 : float
        Acquisition start time (s).

    Returns
    -------
    np.ndarray, shape (nx,)
        Delay in seconds for each element.
    """
```

### `src/solvers.py`

```python
def fkmig(SIG: np.ndarray, fs: float, pitch: float,
          TXangle: float = 0.0, c: float = 1540.0,
          t0: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stolt f-k migration for a single plane-wave angle.

    Parameters
    ----------
    SIG : np.ndarray, shape (N_t, N_x), float or complex
        RF signals; rows = time samples, columns = array elements.
    fs : float
        Sampling frequency (Hz).
    pitch : float
        Element pitch (m).
    TXangle : float
        Steering angle (rad), positive = first element fires first.
    c : float
        Speed of sound (m/s).
    t0 : float
        Acquisition start time (s).

    Returns
    -------
    x : np.ndarray, shape (N_x,)
        Lateral positions (m), centered on array midpoint.
    z : np.ndarray, shape (N_t,)
        Depth positions (m).
    migSIG : np.ndarray, shape (N_t, N_x), complex128
        Migrated (focused) complex RF image.
    """

def coherent_compound(RF: np.ndarray, fs: float, pitch: float,
                      TXangles: list[float], c: float = 1540.0,
                      t0: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Migrate and coherently compound all steering angles.

    Parameters
    ----------
    RF : np.ndarray, shape (N_t, N_x, N_angles), float
        RF signals for all steering angles.
    fs, pitch, c, t0 : float
        Acquisition parameters (see fkmig).
    TXangles : list of float
        Steering angles in radians, length N_angles.

    Returns
    -------
    x : np.ndarray, shape (N_x,)
    z : np.ndarray, shape (N_t,)
    compound : np.ndarray, shape (N_t, N_x), complex128
        Mean of per-angle migrated images.
    """
```

### `src/visualization.py`

```python
def envelope_bmode(migSIG: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Compute envelope-detected, power-law-compressed B-mode image.

    Parameters
    ----------
    migSIG : np.ndarray, shape (N_t, N_x), complex
        Coherently compounded migrated RF image.
    gamma : float
        Power-law exponent for dynamic range compression (default 0.5).

    Returns
    -------
    np.ndarray, shape (N_t, N_x), float
        B-mode image (non-negative, in [0, max]).
    """

def plot_bmode(bmode: np.ndarray, x: np.ndarray, z: np.ndarray,
               title: str = '', ax=None):
    """Display B-mode image with correct physical axes.

    Parameters
    ----------
    bmode : np.ndarray, shape (N_t, N_x)
    x : np.ndarray, shape (N_x,)   lateral positions (m)
    z : np.ndarray, shape (N_t,)   depth positions (m)
    title : str
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib.image.AxesImage
    """

def measure_psf_fwhm(bmode: np.ndarray, x: np.ndarray, z: np.ndarray,
                     z_targets: list[float]) -> list[float]:
    """Measure lateral FWHM (mm) of point-spread function at each target depth.

    Parameters
    ----------
    bmode : np.ndarray, shape (N_t, N_x)
    x, z : np.ndarray
    z_targets : list of float
        Approximate depths of wire targets (m).

    Returns
    -------
    list of float
        Lateral FWHM in mm for each target.
    """

def measure_cnr(bmode: np.ndarray, x: np.ndarray, z: np.ndarray,
                cyst_centers: list[tuple], cyst_radius: float = 2e-3,
                shell_inner: float = 2.5e-3, shell_outer: float = 4e-3) -> list[float]:
    """Measure contrast-to-noise ratio (CNR) for each cyst.

    CNR = |mean_inside - mean_outside| / std_outside

    Parameters
    ----------
    bmode : np.ndarray
    x, z : np.ndarray
    cyst_centers : list of (x_c, z_c) tuples in meters
    cyst_radius, shell_inner, shell_outer : float (m)

    Returns
    -------
    list of float
    """
```

## Data Flow

```
raw_data.npz  ──► preprocessing.load_dataset()
                      │
                      ▼
              RF (N_t, N_x, 7)  +  params
                      │
             for each angle θ:
                      ▼
              solvers.fkmig(RF[:,:,i], fs, pitch, TXangle=θ, c, t0)
                      │
                      ▼
              migSIG_i (N_t, N_x, complex)
                      │
          coherent_compound()  ──► compound (N_t, N_x, complex)
                      │
        visualization.envelope_bmode()
                      │
                      ▼
              bmode (N_t, N_x, float)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   plot_bmode()         measure_psf_fwhm() / measure_cnr()
```
