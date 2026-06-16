# Code Design

## File Structure

```
main.py                  # Pipeline orchestration
src/
  preprocessing.py       # Data loading, closure quantity computation
  physics_model.py       # DFT forward model + closure chi-squared terms
  solvers.py             # RML solvers (visibility, amplitude+CP, closure-only)
  visualization.py       # Plotting utilities and metrics
  generate_data.py       # Synthetic data generation with gain corruption
```

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir: str = "data") -> dict:
    """
    Load raw_data.npz from data directory.

    Returns: dict with keys
      'vis_cal'       : np.ndarray, shape (M,), complex128 -- calibrated visibilities
      'vis_corrupt'   : np.ndarray, shape (M,), complex128 -- gain-corrupted visibilities
      'uv_coords'     : np.ndarray, shape (M, 2), float64  -- baseline positions [wavelengths]
      'sigma_vis'     : np.ndarray, shape (M,), float64    -- per-baseline noise sigma (Jy)
      'station_ids'   : np.ndarray, shape (M, 2), int64    -- station pair indices
    """

def load_metadata(data_dir: str = "data") -> dict:
    """
    Load meta_data JSON from data directory.

    Returns: dict with keys
      'N'              : int   -- image size
      'pixel_size_uas' : float -- pixel size in microarcseconds
      'pixel_size_rad' : float -- pixel size in radians
      'noise_std'      : float -- median noise standard deviation sigma
      'freq_ghz'       : float -- observing frequency
      'n_baselines'    : int   -- number of baselines M
      'n_stations'     : int   -- number of stations
      'station_names'  : list  -- station name strings
      'gain_amp_error' : float -- fractional gain amplitude error
      'gain_phase_error_deg' : float -- gain phase error in degrees
    """

def find_triangles(station_ids: np.ndarray, n_stations: int) -> np.ndarray:
    """
    Enumerate closure-phase triangles from station connectivity.

    Parameters
    ----------
    station_ids : (M, 2) int -- station pair indices
    n_stations  : int -- number of stations

    Returns
    -------
    triangles : (N_tri, 3) int -- station index triples (i, j, k)
    """

def find_quadrangles(station_ids: np.ndarray, n_stations: int) -> np.ndarray:
    """
    Enumerate closure-amplitude quadrangles from station connectivity.

    Parameters
    ----------
    station_ids : (M, 2) int
    n_stations  : int

    Returns
    -------
    quadrangles : (N_quad, 4) int -- station index quads (i, j, k, l)
    """

def compute_closure_phases(
    vis: np.ndarray, station_ids: np.ndarray, triangles: np.ndarray,
) -> np.ndarray:
    """
    Compute closure phases from complex visibilities.

    Closure phase for triangle (i, j, k):
        phi_C = arg(V_ij * V_jk * V_ki)

    Returns: (N_tri,) float -- closure phases in radians
    """

def compute_log_closure_amplitudes(
    vis: np.ndarray, station_ids: np.ndarray, quadrangles: np.ndarray,
) -> np.ndarray:
    """
    Compute log closure amplitudes from complex visibilities.

    For quadrangle (i, j, k, l):
        log CA = log|V_ij| + log|V_kl| - log|V_ik| - log|V_jl|

    Returns: (N_quad,) float
    """

def closure_phase_sigma(
    sigma_vis: np.ndarray, vis: np.ndarray,
    station_ids: np.ndarray, triangles: np.ndarray,
) -> np.ndarray:
    """
    Noise propagation for closure phases.

    sigma_CP = sqrt(sigma_1^2/|V_1|^2 + sigma_2^2/|V_2|^2 + sigma_3^2/|V_3|^2)

    Returns: (N_tri,) float -- closure phase sigma in radians
    """

def closure_amplitude_sigma(
    sigma_vis: np.ndarray, vis: np.ndarray,
    station_ids: np.ndarray, quadrangles: np.ndarray,
) -> np.ndarray:
    """
    Noise propagation for log closure amplitudes.

    sigma_logCA = sqrt(1/SNR_1^2 + 1/SNR_2^2 + 1/SNR_3^2 + 1/SNR_4^2)

    Returns: (N_quad,) float
    """

def prepare_data(data_dir: str = "data") -> tuple:
    """
    Combined loader: returns (obs, closure, meta).

    obs     : dict -- raw observation arrays
    closure : dict -- closure phases, log closure amps, sigmas, triangles, quadrangles
    meta    : dict -- imaging parameters
    """
```

### physics_model.py

```python
def _triangle_pulse_F(omega: float, pdim: float) -> float:
    """Fourier-domain triangle pulse response (matches ehtim)."""

def _ftmatrix(psize: float, N: int, uv_coords: np.ndarray) -> np.ndarray:
    """
    Build DFT matrix matching ehtim's ftmatrix exactly.

    Sign convention: +2*pi*i. Pixel grid matches ehtim.

    Parameters
    ----------
    psize     : float -- pixel size in radians
    N         : int -- image size (N x N)
    uv_coords : (M, 2) -- baseline (u, v) in wavelengths

    Returns
    -------
    A : (M, N^2) complex DFT matrix
    """

class ClosureForwardModel:
    """
    VLBI forward model with closure quantity chi-squared support.

    Attributes:
      A           : (M, N^2) complex -- DFT measurement matrix
      N           : int -- image side length
      triangles   : (N_tri, 3) int -- closure phase triangles
      quadrangles : (N_quad, 4) int -- closure amplitude quadrangles
      station_ids : (M, 2) int -- station pair indices
    """

    def __init__(self, uv_coords, N, pixel_size_rad, triangles, quadrangles,
                 station_ids=None):
        """Build DFT matrix from (u,v) positions and pixel grid."""

    def forward(self, image: np.ndarray) -> np.ndarray:
        """image (N,N) -> visibilities (M,) complex. Computes y = A @ x."""

    def adjoint(self, vis: np.ndarray) -> np.ndarray:
        """visibilities (M,) complex -> image (N,N) real. Computes A^H @ y."""

    def dirty_image(self, vis: np.ndarray) -> np.ndarray:
        """Normalized back-projection: A^H y / M."""

    def psf(self) -> np.ndarray:
        """Point spread function (dirty beam)."""

    def visibility_chisq(self, image, vis_obs, sigma) -> float:
        """Normalized visibility chi-squared: (1/M) sum |V_model - V_obs|^2 / sigma^2."""

    def visibility_chisq_grad(self, image, vis_obs, sigma) -> np.ndarray:
        """Gradient of visibility chi-squared w.r.t. image."""

    @staticmethod
    def chisq_cphase_from_uv(imvec, N, psize, uv1, uv2, uv3,
                              clphase_deg, sigma_deg) -> float:
        """
        Closure phase chi-squared (Chael 2018 Eq. 11).

        chi2 = (2/N_CP) sum (1 - cos(phi_obs - phi_model)) / sigma^2
        """

    @staticmethod
    def chisqgrad_cphase_from_uv(imvec, N, psize, uv1, uv2, uv3,
                                  clphase_deg, sigma_deg) -> np.ndarray:
        """
        Gradient of closure phase chi-squared.

        d_chi2/dI = (-2/N_CP) Im[ sum sin(phi_obs - phi_model)/sigma^2 * (A_k^T / i_k) ]
        """

    @staticmethod
    def chisq_logcamp_from_uv(imvec, N, psize, uv1, uv2, uv3, uv4,
                               log_clamp, sigma) -> float:
        """
        Log closure amplitude chi-squared (Chael 2018 Eq. 12).

        chi2 = (1/N_CA) sum ((logCA_obs - logCA_model) / sigma)^2
        """

    @staticmethod
    def chisqgrad_logcamp_from_uv(imvec, N, psize, uv1, uv2, uv3, uv4,
                                   log_clamp, sigma) -> np.ndarray:
        """Gradient of log closure amplitude chi-squared."""

    def model_closure_phases(self, image: np.ndarray) -> np.ndarray:
        """Compute model closure phases from image (radians)."""

    def model_log_closure_amplitudes(self, image: np.ndarray) -> np.ndarray:
        """Compute model log closure amplitudes from image."""
```

### solvers.py

```python
class GullSkillingRegularizer:
    """
    Gull-Skilling entropy: S(I) = sum(I - P - I*log(I/P))

    Parameters
    ----------
    prior   : (N, N) ndarray or None -- prior image P
    epsilon : float -- floor to avoid log(0)
    """
    def __init__(self, prior=None, epsilon=1e-30): ...
    def value_and_grad(self, x: np.ndarray) -> tuple:
        """Returns (float, ndarray) -- value and gradient of S(x)."""

class SimpleEntropyRegularizer:
    """
    Simple entropy: S(I) = -sum(I * log(I/P))

    Parameters
    ----------
    prior   : (N, N) ndarray or None
    epsilon : float
    """
    def __init__(self, prior=None, epsilon=1e-30): ...
    def value_and_grad(self, x: np.ndarray) -> tuple:
        """Returns (float, ndarray) -- value and gradient of S(x)."""

class TVRegularizer:
    """
    Total Variation (Huber-smoothed): TV(x) = sum sqrt(dx^2 + dy^2 + eps^2) - eps

    Parameters
    ----------
    epsilon : float -- Huber smoothing parameter
    """
    def __init__(self, epsilon: float = 1e-6): ...
    def value_and_grad(self, x: np.ndarray) -> tuple:
        """Returns (float, ndarray) -- value and gradient of TV(x)."""

class ClosureRMLSolver:
    """
    RML solver using closure quantity chi-squared (Chael 2018).

    Objective: sum alpha_d * (chi2_d - 1) + sum alpha_r * reg_r(x)

    Parameters
    ----------
    data_terms : dict  -- keys: 'cphase', 'logcamp', 'vis'; values: weight alpha
    reg_terms  : dict  -- keys: 'gs', 'simple', 'tv'; values: weight alpha
    prior      : (N, N) ndarray -- Gaussian prior image
    n_iter     : int -- max L-BFGS-B iterations per round
    n_rounds   : int -- number of imaging rounds
    """
    def __init__(self, data_terms=None, reg_terms=None, prior=None,
                 n_iter=300, n_rounds=3): ...

    def reconstruct(self, model, obs_data, x0=None) -> np.ndarray:
        """
        Run closure-based RML imaging.

        Parameters
        ----------
        model    : ClosureForwardModel
        obs_data : dict with closure quantity arrays and UV coordinates
        x0       : (N, N) initial image, defaults to prior

        Returns
        -------
        image : (N, N) reconstructed image
        """

class VisibilityRMLSolver:
    """
    Traditional visibility-based RML (comparison baseline).

    Minimizes: (1/M) sum |A*x - y|^2/sigma^2 + sum lambda_r R_r(x)
    """
    def __init__(self, regularizers=None, n_iter=500, positivity=True): ...
    def reconstruct(self, model, vis, noise_std=1.0, x0=None) -> np.ndarray:
        """Optimization-based reconstruction. Returns (N,N) image."""
```

### visualization.py

```python
def compute_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """
    Compute image-quality metrics after flux normalization.
    Returns dict with 'nrmse', 'ncc', 'dynamic_range'.
    """

def print_metrics_table(metrics_dict: dict) -> None:
    """Print formatted comparison table."""

def plot_uv_coverage(uv_coords: np.ndarray, title: str = "UV Coverage",
                     ax=None) -> Axes:
    """Plot (u,v)-plane sampling pattern with conjugate points."""

def plot_image(image: np.ndarray, pixel_size_uas: float = 2.0,
               title: str = "", cmap: str = "afmhot", ax=None) -> Axes:
    """Display 2D image with physical axis labels (uas)."""

def plot_comparison(images: dict, ground_truth: np.ndarray,
                    metrics_dict: dict = None,
                    pixel_size_uas: float = 2.0) -> Figure:
    """Side-by-side comparison of multiple reconstruction methods."""

def plot_gain_robustness(cal_metrics: dict, cor_metrics: dict) -> None:
    """Print calibrated vs corrupted metrics comparison table."""
```

### generate_data.py

```python
EHT_TELESCOPES = {...}
"""EHT 2017 station properties (ECEF coords, SEFD) for 8 stations."""

def make_ring_image(N=64, ring_radius_frac=0.22, ring_width_frac=0.055,
                    asymmetry=0.5, asymmetry_angle_deg=220.0) -> np.ndarray:
    """Synthetic M87*-like ring image, normalized to unit total flux."""

def simulate_eht_uv_coverage(
    source_ra_deg=187.7059, source_dec_deg=12.3911,
    obs_start_utc="2017-04-06T00:00:00", obs_duration_hours=6.0,
    n_time_steps=15, freq_ghz=230.0,
) -> tuple:
    """
    Simulate EHT uv-coverage using astropy for proper coordinate transforms.

    Returns: (uv_coords, station_ids, n_stations, timestamps, sefds, station_names)
    """

def compute_sefd_noise(station_ids, sefds, eta=0.88, bandwidth_hz=2e9,
                       tau_int=10.0) -> np.ndarray:
    """Per-baseline thermal noise from station SEFDs: sigma_ij = (1/eta) sqrt(SEFD_i SEFD_j) / sqrt(2 Bw tau)."""

def apply_station_gains(vis_clean, station_ids, n_stations,
                        amp_error=0.2, phase_error_deg=30.0, rng=None) -> tuple:
    """
    Apply station-based gain errors: V_ij^corr = g_i * conj(g_j) * V_ij^true.
    Returns (vis_corrupted, gains).
    """

def generate_dataset(N=64, pixel_size_uas=2.0, seed=42, save_dir="data") -> dict:
    """Generate and save complete synthetic EHT dataset with gain corruption."""
```

### main.py

```python
def main():
    """
    Orchestrate the full reconstruction pipeline:
      1. prepare_data("data")         -> obs, closure, meta
      2. ClosureForwardModel(...)     -> model
      3. Build obs_data dicts for each solver configuration
      4. Reconstruct with 3 methods x 2 data states (cal + corrupt)
      5. compute_metrics(...)         -> metrics dict
      6. plot_comparison(...)         -> visualization
      7. Save output/reconstruction.npy (best closure-only result)
    """
```

## Data Flow

```
data/raw_data.npz + data/meta_data
        |
        v
  preprocessing.py  -->  obs dict + closure dict (CP, logCA, sigmas) + meta
        |
        v
  physics_model.py  -->  ClosureForwardModel (DFT matrix A + closure chi-sq methods)
        |
        v
  solvers.py        -->  3 methods: Vis RML, Amp+CP, Closure-only
        |                 x 2 data: calibrated vs. corrupted
        v
  visualization.py  -->  comparison plots + gain robustness table
        |
        v
  output/reconstruction.npy
```
