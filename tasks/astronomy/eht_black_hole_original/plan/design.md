# Code Design

## File Structure

```
src/
  main.py                # Public-data pipeline orchestration
  preprocessing.py       # Data loading, closure quantity computation
  physics_model.py       # DFT forward model + closure chi-squared terms
  solvers.py             # ClosureRMLSolver and a separate VisibilityRMLSolver helper
  visualization.py       # Plotting utilities and metrics
```

The generator design is preserved below for upstream context; `generate_data.py` is not bundled in this format-only task package. The hidden notebook restores the calibrated/corrupted comparisons independently of the public-data entry point.

The signatures below match the current source, except the explicitly historical generator section. Amp+CP is an archived result, not an implemented data term. The notebook uses `ClosureRMLSolver` for both visibility and closure-only comparisons; it does not instantiate `VisibilityRMLSolver`.

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir: str='data') -> dict:
    """
    Load batch-first raw_data.npz with allow_pickle=False and remove its batch axis.
    Calibrated observations are loaded separately from hidden notebook reference outputs.

    Returns: dict including
      'vis_corrupt'   : np.ndarray, shape (M,), complex128 -- gain-corrupted visibilities
      'uv_coords'     : np.ndarray, shape (M, 2), float64  -- baseline positions [wavelengths]
      'sigma_vis'     : np.ndarray, shape (M,), float64    -- per-baseline noise sigma (Jy)
      'station_ids'   : np.ndarray, shape (M, 2), int64    -- station pair indices
      Plus corrupted closure values, uncertainties, and their per-leg UV arrays.
    """

def load_metadata(data_dir: str='data') -> dict:
    """
    Return acquisition constants documented in README; no metadata JSON is loaded.
    Preserve pixel_size_rad = 9.696273622188272e-12 from the original acquisition.
    data_dir is retained for call compatibility and is not read by this function.

    Returns: dict with keys
      'N'              : int   -- image size
      'pixel_size_uas' : float -- pixel size in microarcseconds
      'pixel_size_rad' : float -- pixel size in radians
      'total_flux'     : float -- integrated source flux in Jy
      'freq_ghz'       : float -- observing frequency
      'n_baselines'    : int   -- number of baselines M
      'n_stations'     : int   -- number of stations
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

def compute_closure_phases(vis: np.ndarray, station_ids: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """
    Compute closure phases from complex visibilities.

    Closure phase for triangle (i, j, k):
        phi_C = arg(V_ij * V_jk * V_ki)

    Returns: (N_tri,) float -- closure phases in radians
    """

def compute_log_closure_amplitudes(vis: np.ndarray, station_ids: np.ndarray, quadrangles: np.ndarray) -> np.ndarray:
    """
    Compute log closure amplitudes from complex visibilities.

    For quadrangle (i, j, k, l):
        log CA = log|V_ij| + log|V_kl| - log|V_ik| - log|V_jl|

    Returns: (N_quad,) float
    """

def closure_phase_sigma(sigma_vis: np.ndarray, vis: np.ndarray, station_ids: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """
    Noise propagation for closure phases.

    sigma_CP = sqrt(sigma_1^2/|V_1|^2 + sigma_2^2/|V_2|^2 + sigma_3^2/|V_3|^2)

    Returns: (N_tri,) float -- closure phase sigma in radians
    """

def closure_amplitude_sigma(sigma_vis: np.ndarray, vis: np.ndarray, station_ids: np.ndarray, quadrangles: np.ndarray) -> np.ndarray:
    """
    Noise propagation for log closure amplitudes.

    sigma_logCA = sqrt(1/SNR_1^2 + 1/SNR_2^2 + 1/SNR_3^2 + 1/SNR_4^2)

    Returns: (N_quad,) float
    """

def prepare_data(data_dir: str='data') -> tuple:
    """
    Combined loader: returns (obs, closure, meta).

    obs     : dict -- raw observation arrays
    closure : dict -- closure phases, log closure amps, sigmas, triangles, quadrangles
    meta    : dict -- imaging parameters

    This legacy helper forms one connectivity graph and looks up the first
    matching baseline per station pair. It is not scan-aware and is not called
    by src/main.py or the notebook's reconstruction. They use supplied per-scan
    closure values and UV legs instead. Enumeration does not ensure statistical
    independence of the resulting triangles or quadrangles.
    """
```

### physics_model.py

```python
def _triangle_pulse_F(omega, pdim):
    """Fourier-domain triangle pulse response (matches ehtim)."""

def _ftmatrix(psize, N, uv_coords):
    """
    Build the DFT matrix using the preserved ehtim sign, grid, and pulse conventions.

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

    def __init__(self, uv_coords, N, pixel_size_rad, triangles, quadrangles, station_ids=None):
        """Build DFT matrix from (u,v) positions and pixel grid."""

    def forward(self, image: np.ndarray) -> np.ndarray:
        """image (N,N) -> visibilities (M,) complex. Computes y = A @ x."""

    def adjoint(self, vis: np.ndarray) -> np.ndarray:
        """visibilities (M,) complex -> image (N,N) real. Computes real(A^H @ y)."""

    def dirty_image(self, vis: np.ndarray) -> np.ndarray:
        """Normalized back-projection: A^H y / M."""

    def psf(self) -> np.ndarray:
        """Point spread function (dirty beam)."""

    def visibility_chisq(self, image: np.ndarray, vis_obs: np.ndarray, sigma: np.ndarray) -> float:
        """Normalized visibility chi-squared: (1/M) sum |V_model - V_obs|^2 / sigma^2."""

    def visibility_chisq_grad(self, image: np.ndarray, vis_obs: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Flat (N^2,) gradient: (2/M) * real(A^H @ ((A @ x - y) / sigma^2))."""

    @staticmethod
    def chisq_cphase_from_uv(imvec, N, psize, uv1, uv2, uv3, clphase_deg, sigma_deg):
        """
        Closure phase chi-squared (Chael 2018 Eq. 11).

        clphase_deg and sigma_deg are converted from degrees to radians internally.
        chi2 = (2/N_CP) sum (1 - cos(phi_obs - phi_model)) / sigma^2
        """

    @staticmethod
    def chisqgrad_cphase_from_uv(imvec, N, psize, uv1, uv2, uv3, clphase_deg, sigma_deg):
        """
        Gradient of closure phase chi-squared.

        d_chi2/dI = (-2/N_CP) Im[ sum sin(phi_obs - phi_model)/sigma^2 * (A_k^T / i_k) ]
        """

    @staticmethod
    def chisq_logcamp_from_uv(imvec, N, psize, uv1, uv2, uv3, uv4, log_clamp, sigma):
        """
        Log closure amplitude chi-squared (Chael 2018 Eq. 12).

        chi2 = (1/N_CA) sum ((logCA_obs - logCA_model) / sigma)^2
        """

    @staticmethod
    def chisqgrad_logcamp_from_uv(imvec, N, psize, uv1, uv2, uv3, uv4, log_clamp, sigma):
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
    """
    def __init__(self, prior=None): ...
    def value_and_grad(self, x: np.ndarray):
        """Returns (float, ndarray) -- value and gradient of S(x)."""

class SimpleEntropyRegularizer:
    """
    Simple entropy: S(I) = -sum(I * log(I/P))

    Parameters
    ----------
    prior   : (N, N) ndarray or None
    """
    def __init__(self, prior=None): ...
    def value_and_grad(self, x: np.ndarray):
        """Returns (float, ndarray) -- value and gradient of S(x)."""

class TVRegularizer:
    """
    Total Variation (Huber-smoothed): TV(x) = sum(sqrt(dx^2 + dy^2 + eps^2) - eps)
    Differences wrap periodically via np.roll.

    Parameters
    ----------
    epsilon : float -- Huber smoothing parameter
    """
    def __init__(self, epsilon: float=1e-06): ...
    def value_and_grad(self, x: np.ndarray):
        """Returns (float, ndarray) -- value and gradient of TV(x)."""

class ClosureRMLSolver:
    """
    RML solver using closure quantity chi-squared (Chael 2018).

    Objective: sum alpha_d * (chi2_d - 1) - sum alpha_r * raw_regularizer_r(x)
    Entropy helpers return raw S values; TVRegularizer returns positive TV.
    The optional tv branch therefore subtracts TV. The standard configuration
    uses only gs and simple and does not enable this optional branch.

    Parameters
    ----------
    data_terms : dict  -- keys: 'cphase', 'logcamp', 'vis'; values: weight alpha
    reg_terms  : dict  -- keys: 'gs', 'simple', 'tv'; values: weight alpha
    prior      : (N, N) ndarray -- Gaussian prior image
    n_iter     : int -- max L-BFGS-B iterations per round
    n_rounds   : int -- number of imaging rounds
    """
    def __init__(self, data_terms=None, reg_terms=None, prior=None, n_iter=300, n_rounds=3): ...

    def reconstruct(self, model, obs_data, x0=None):
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

    Minimizes: 0.5 * sum |A*x - y|^2/sigma^2 + sum lambda_r R_r(x)
    No division by M. This helper optimizes image pixels directly, optionally
    with bounds x >= 0, and is not the notebook's visibility implementation.
    """
    def __init__(self, regularizers=None, n_iter=500, positivity=True): ...
    def reconstruct(self, model, vis, noise_std=1.0, x0=None):
        """Optimization-based reconstruction. Returns (N,N) image."""
```

### visualization.py

```python
def compute_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """
    Compute dynamic-range NRMSE, cosine NCC, and peak/residual-RMS dynamic range
    after normalizing the estimate to the reference flux. Results round to four
    decimal places; this function does not use the cached notebook's L2 NRMSE.
    Returns dict with 'nrmse', 'ncc', 'dynamic_range'.
    """

def print_metrics_table(metrics_dict: dict):
    """Print formatted comparison table."""

def plot_uv_coverage(uv_coords: np.ndarray, title: str='UV Coverage', ax=None):
    """Plot (u,v)-plane sampling pattern with conjugate points."""

def plot_image(image: np.ndarray, pixel_size_uas: float=2.0, title: str='', cmap: str='afmhot', ax=None, vmin=0, vmax=None):
    """Display 2D image with physical axis labels (uas)."""

def plot_comparison(images: dict, ground_truth: np.ndarray, metrics_dict: dict=None, pixel_size_uas: float=2.0, suptitle: str='Reconstruction Comparison'):
    """Side-by-side comparison of multiple reconstruction methods."""

def plot_gain_robustness(cal_metrics: dict, cor_metrics: dict):
    """Print calibrated vs corrupted metrics comparison table."""
```

### generate_data.py (Historical Upstream Design)

These signatures preserve the original simulation design for maintainer reference. They are not callable APIs in the current package; existing observations are reused rather than regenerated.

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

### src/main.py

```python
def make_gaussian_prior(n, pixel_size_rad, total_flux, fwhm_uas=20.0):
    """Return a centered Gaussian prior with the known integrated flux."""

def run_reconstruction(data_dir=TASK_DIR / 'data', n_iter=300, n_rounds=3):
    """Load public observations, construct the forward model, and run closure-only RML."""

def main():
    """
    Orchestrate the public-data reconstruction pipeline:
      1. load_observation and load_metadata -> obs and acquisition constants
      2. ClosureForwardModel(...)           -> model
      3. Build a Gaussian prior and the corrupted closure-data dictionary
      4. ClosureRMLSolver.reconstruct(...)   -> closure-only reconstruction
      5. Normalize to the known 0.6 Jy total flux
      6. Save output/reconstruction.npz, key reconstruction,
         shape (1, 64, 64), dtype float64, units Jy/pixel

    Run from the task directory with: python -m src.main
    The hidden notebook separately runs the original four Vis RML / closure-only
    comparisons (calibrated and corrupted); Amp+CP remains an archived comparison.
    """
```

## Data Flow

The public-data entry point follows this path; it does not call `prepare_data` or visualization utilities:

```
data/raw_data.npz + README acquisition constants
        |
        v
  load_observation + load_metadata --> obs + acquisition constants
        |
        v
  src/main.py       --> Gaussian prior + supplied corrupted closure values / UV legs
        |
        v
  physics_model.py  --> ClosureForwardModel and DFT matrices
        v
  solvers.py        --> ClosureRMLSolver.reconstruct
        |
        v
  src/main.py       --> finite / positive-flux checks + 0.6 Jy normalization
        |
        v
  output/reconstruction.npz (batch-first reconstruction array)
```

The independent evaluation command reads the submission, hidden truth and thresholds, then writes `output/judge_report.json`. The notebook separately combines public observations with hidden calibrated comparison arrays, runs four `ClosureRMLSolver` configurations, computes metrics through `visualization.py`, and plots the archived and fresh comparisons. Its Amp+CP panels use archived results.

## Objective and Optimization Conventions

| Implementation | Visibility data term | Optimization |
| --- | --- | --- |
| `ClosureForwardModel.visibility_chisq` | sum(abs(residual / sigma)^2) / M | Evaluation helper only. |
| `ClosureRMLSolver` with `vis` | sum(abs(residual / sigma)^2) / (2*M); contributes alpha * (chi2 - 1) | Log brightness, three warm-started rounds by default. Used by notebook visibility comparisons. |
| `VisibilityRMLSolver` | 0.5 * sum(abs(residual / sigma)^2), without division by M | Direct image pixels, optional nonnegative bounds, one solve. Not used by the standard pipeline or notebook. |

`ClosureRMLSolver` precomputes the enabled closure-leg DFT matrices once per reconstruction. Its static `ClosureForwardModel` loss/gradient helpers rebuild these matrices per call. The log-space solver uses `ftol=1e-14`, `gtol=1e-10`, and `maxiter=n_iter`; it accepts each round's `result.x` without checking optimizer success. The direct-pixel `VisibilityRMLSolver` uses `ftol=1e-14`, `gtol=1e-9`, and `maxiter=n_iter`. The public entry point selects closure weights 50/50 and entropy weights 1/10; neither it nor the notebook enables the optional TV branch.

## Hidden Review and Evaluation Assets

- `notebooks/reference_outputs/reference_results.npz` contains the six archived comparison images and calibrated comparison measurements, including their uncertainties.
- `notebooks/reference_outputs/reference_metrics.json` records the archived reference metrics. Historical cached notebook metrics and current metric definitions are distinguished in `evaluation/validation.md`.
- `evaluation/ground_truth.npz` contains the hidden simulated truth. `evaluation/metrics.json` defines thresholds; lowercase `evaluation/judge.py` validates and scores submissions without importing solver code.
- `evaluation/fixtures/` and `evaluation/tests/` preserve numerical tests and submission-contract checks; `test/validation.md` records validation evidence.

## Format Migration Provenance

This document restores the upstream module-by-module design at local `origin/main` commit `dc2f668939b21e8312e22529615def610f8611df`, adapting storage, paths, metadata access, and the public pipeline entry point. Historical generator and comparison designs are identified explicitly. Existing entropy-constructor signatures are corrected to match the preserved source. Remote freshness could not be verified while the remote endpoint was unavailable.
