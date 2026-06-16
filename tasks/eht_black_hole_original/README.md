# EHT Black Hole Imaging with Closure Quantities

> Recover the radio image of a supermassive black hole from gain-corrupted
> interferometric data using closure phases and log closure amplitudes --
> gain-invariant observables that bypass the need for antenna-based calibration.

**Domain:** Astronomy
**Keywords:** radio interferometry, compressed sensing
**Difficulty:** Hard

---

## Background

The Event Horizon Telescope (EHT) is a Very Long Baseline Interferometry (VLBI) array
that links radio dishes across the globe, forming a virtual Earth-sized telescope
operating at 230 GHz (1.3 mm wavelength) with ~20 microarcsecond angular resolution.

Each pair of telescopes measures one **complex visibility** -- a Fourier coefficient of
the sky brightness distribution. In practice, the measured visibilities are corrupted by
**station-based complex gains** arising from atmospheric turbulence, instrumental
instabilities, and calibration errors. A station gain g_i multiplies every visibility
involving station i:

$$V_{ij}^{\text{corr}} = g_i \, g_j^* \, V_{ij}^{\text{true}}$$

Direct fitting of corrupted visibilities produces nonsensical images unless the gains are
first calibrated -- a non-trivial problem when there are no bright calibrator sources
in the field (as with EHT observations of M87*).

**Closure quantities** are algebraic combinations of visibilities that cancel the
station-based gains:

- **Closure phase** (triangle of stations i-j-k):
  $$\phi_C = \arg(V_{ij} \cdot V_{jk} \cdot V_{ki})$$
  The station gains cancel: $\phi_C = \phi_{ij} + \phi_{jk} + \phi_{ki}$.

- **Log closure amplitude** (quadrangle of stations i-j-k-l):
  $$\log CA = \log|V_{ij}| + \log|V_{kl}| - \log|V_{ik}| - \log|V_{jl}|$$
  Again gain-invariant since each station appears equally in numerator and denominator.

Imaging directly from closure quantities avoids calibration entirely, at the cost of
reduced information (fewer independent constraints than raw visibilities).

---

## Problem Description

The van Cittert-Zernike theorem relates the measured visibility to the sky brightness:

$$V(u, v) = \iint I(l, m)\, e^{-2\pi i (ul + vm)}\, dl\, dm$$

Discretized on an N x N pixel grid with the DFT matrix A:

$$\mathbf{y} = \mathbf{A}\,\mathbf{x}$$

The DFT matrix follows the ehtim sign convention (+2*pi*i) and uses a triangle pulse
pixel response function:

$$A_{m,n} = P(u_m, v_m) \, \exp\!\bigl[+2\pi i\,(u_m l_n + v_m m_n)\bigr]$$

where P is the Fourier-domain triangle pulse.

| Symbol | Description | Size |
|--------|-------------|------|
| **x** | Sky brightness image (vectorized, non-negative) | N^2 = 4096 |
| **A** | Measurement matrix (DFT with triangle pulse) | M x N^2 |
| **y_corr** | Gain and phase corrupted complex visibilities | M = 421 |
| **y_cal** | Calibrated complex visibilities (ground truth) | M = 421 |
| **g** | Station-based complex gains | N_s = 7 |

The gain corruption model is:

$$y_{ij}^{\text{corr}} = g_i \, g_j^* \, y_{ij}^{\text{cal}}$$

with gain amplitudes |g_i| ~ 1 +/- 20% and gain phases arg(g_i) ~ U(-30, +30) degrees.

---

## Data Description

### `data/raw_data.npz`

Calibrated and gain-corrupted interferometric visibilities and derived closure quantities from a synthetic M87*-like observation.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `vis_cal` | (421,) | complex128 | Calibrated complex visibilities with thermal noise (Jy) |
| `vis_corrupt` | (421,) | complex128 | Gain and phase corrupted complex visibilities (Jy) |
| `uv_coords` | (421, 2) | float64 | Baseline UV coordinates (u, v) in wavelengths |
| `sigma_vis` | (421,) | float64 | Per-baseline thermal noise σ (Jy) |
| `station_ids` | (421, 2) | int64 | Station index pair for each baseline |
| `cp_values_deg` | (269,) | float64 | Closure phases from calibrated data (°) |
| `cp_sigmas_deg` | (269,) | float64 | Closure phase uncertainties (°) |
| `cp_u1` | (269, 2) | float64 | UV for baseline 1 of each closure triangle |
| `cp_u2` | (269, 2) | float64 | UV for baseline 2 of each closure triangle |
| `cp_u3` | (269, 2) | float64 | UV for baseline 3 of each closure triangle |
| `lca_values` | (233,) | float64 | Log closure amplitudes from calibrated data |
| `lca_sigmas` | (233,) | float64 | Log closure amplitude uncertainties |
| `lca_u1` | (233, 2) | float64 | UV for baseline 1 of each closure quad |
| `lca_u2` | (233, 2) | float64 | UV for baseline 2 of each closure quad |
| `lca_u3` | (233, 2) | float64 | UV for baseline 3 of each closure quad |
| `lca_u4` | (233, 2) | float64 | UV for baseline 4 of each closure quad |
| `cp_corrupt_values_deg` | (269,) | float64 | Closure phases from corrupted data (°) |
| `cp_corrupt_sigmas_deg` | (269,) | float64 | Corrupted closure phase uncertainties (°) |
| `lca_corrupt_values` | (233,) | float64 | Log closure amplitudes from corrupted data |
| `lca_corrupt_sigmas` | (233,) | float64 | Corrupted log closure amplitude uncertainties |

### `data/ground_truth.npz`

Ground-truth sky brightness image of the synthetic M87*-like ring source.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `image` | (64, 64) | float64 | Ground-truth sky brightness image (Jy/pixel) |
| `image_jy` | (64, 64) | float64 | Ground-truth image in absolute Jy units |

### `data/meta_data.json`

JSON file with array configuration and gain corruption parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N` | 64 | Image size (N × N pixels) |
| `pixel_size_uas` | 2.0 | Pixel size (μas) |
| `total_flux` | 0.6 | Source total flux (Jy) |
| `freq_ghz` | 230.0 | Observing frequency (GHz) |
| `n_baselines` | 421 | Number of measured baselines |
| `n_stations` | 7 | Number of EHT stations |
| `station_names` | [...] | Station list (ALMA, APEX, JCMT, LMT, PV, SMA, SMT) |
| `gain_amp_error` | 0.2 | Fractional amplitude gain error (20%) |
| `gain_phase_error_deg` | 30.0 | Max phase gain error (°) |

---

## Method Hints

Reconstruct using **Regularized Maximum Likelihood (RML)** imaging directly from closure quantities, following Chael et al. (2018). The objective minimizes a weighted sum of closure chi-squared data terms (closure phase and log closure amplitude) and image regularizers (Gull-Skilling entropy, TV) subject to non-negativity. Use L-BFGS-B with multi-round optimization (3 rounds of 300 iterations) matching ehtim's imaging workflow.

To demonstrate the robustness of closure quantities, implement three solvers: (1) **Visibility RML** — fits corrupted visibilities directly, catastrophically fails under gain corruption; (2) **Amplitude + Closure Phase RML** — partially robust; (3) **Closure-only RML** — both closure phases and log closure amplitudes, fully gain-invariant, the main result.

## References

- Chael, A.A., Johnson, M.D., Bouman, K.L., et al. (2018). *Interferometric Imaging Directly with Closure Phases and Closure Amplitudes*. ApJ, 857, 23.
- EHT Collaboration et al. (2019). *First M87 Event Horizon Telescope Results. Paper I-VI*. ApJL, 875.
- Thompson, A.R., Moran, J.M., & Swenson, G.W. (2017). *Interferometry and Synthesis in Radio Astronomy* (3rd ed.). Springer.
- ehtim: https://github.com/achael/eht-imaging
