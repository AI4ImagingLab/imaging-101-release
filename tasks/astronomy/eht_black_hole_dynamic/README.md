# EHT Black Hole Dynamic Imaging

> Reconstruct a time-varying black hole video from sparse per-frame interferometric measurements using the Kalman filtering approach with EM optimization.

> Domain: Astronomy | Keywords: radio interferometry, dynamic imaging | Difficulty: Hard

---

## Background

Reconstruct a time-varying video of a black hole (e.g., SgrA*) from sparse
interferometric measurements obtained by the Event Horizon Telescope (EHT).
Each time frame has a different set of (u,v) baseline measurements due to
Earth rotation, and individual frames are severely under-determined.

This task implements the **StarWarps** algorithm from Bouman et al. (2017),
which uses a Gaussian Markov model with forward-backward message passing
and EM optimization to jointly reconstruct all frames while enforcing
temporal coherence.

## Problem Description

### Interferometric Imaging

The EHT measures complex visibilities — samples of the Fourier transform of
the sky brightness distribution at spatial frequencies determined by baseline
geometry:

    V(u,v) = ∫∫ I(l,m) e^{-2πi(ul + vm)} dl dm

For N×N images, this is discretized as a DFT:

    y_t = A_t x_t + n_t

where A_t is the per-frame measurement matrix (M_t × N² complex), x_t is the
vectorized image, and n_t is thermal noise with per-baseline variance from
station SEFDs.

### Gaussian Markov Model (StarWarps)

The video is modeled as a first-order Gaussian Markov process:

    x_t = A(θ) x_{t-1} + w_t,    w_t ~ N(0, Q)

where A(θ) is a warp matrix parameterized by affine motion θ, and Q controls
the allowed intensity variation between frames.

The inverse problem is to recover the video sequence $\{x_t\}$ and warp parameters $\theta$ from the per-frame visibility measurements $\{y_t\}$. Each frame is severely under-determined (M_t ≪ N²), making the problem ill-posed without the temporal prior.

**Input:** T = 12 sets of complex visibilities (28 baselines per frame).
**Output:** a sequence of T brightness images (N × N pixels, Jy/pixel).

## Data Description

### `data/raw_data.npz`

Per-frame interferometric visibilities from 8 EHT stations (T = 12 frames, t = 0…11).
Per-frame keys are stored with a numeric suffix (e.g. `vis_0`, `vis_1`, …, `vis_11`).

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `n_frames` | () | int64 | Number of time frames (12) |
| `frame_times` | (12,) | float64 | Frame mid-times in UTC fractional hours |
| `vis_0` … `vis_11` | (28,) | complex128 | Complex visibilities at frame t (Jy) |
| `sigma_0` … `sigma_11` | (28,) | float64 | Thermal noise σ per baseline at frame t (Jy) |
| `uv_0` … `uv_11` | (28, 2) | float64 | UV coordinates (u, v) per baseline at frame t (wavelengths) |
| `station_ids_0` … `station_ids_11` | (28, 2) | int64 | Station index pair for each baseline at frame t |

### `data/ground_truth.npz`

Ground-truth brightness video of a synthetic rotating crescent (SgrA*-like) source.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `images` | (12, 30, 30) | float64 | Ground-truth brightness frames (Jy/pixel) |

### `data/meta_data.json`

JSON file with observation and reconstruction parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N` | 30 | Image size (N × N pixels) |
| `pixel_size_uas` | 3.4 | Pixel size (μas) |
| `pixel_size_rad` | 1.648e-11 | Pixel size (radians) |
| `n_frames` | 12 | Number of time frames |
| `obs_duration_hours` | 6.0 | Total observation window (hours) |
| `obs_start_utc` | "2017-04-06T00:00:00" | Observation start time (UTC) |
| `total_flux` | 2.0 | Source total flux (Jy) |
| `rotation_deg` | 90.0 | Bright-spot rotation over full observation (°) |
| `base_angle_deg` | 220.0 | Initial bright-spot position angle (°) |
| `freq_ghz` | 230.0 | Observing frequency (GHz) |
| `source_name` | "SgrA*" | Source name |
| `source_ra_deg` | 266.417 | Source right ascension (°) |
| `source_dec_deg` | -28.992 | Source declination (°) |
| `n_stations` | 8 | Number of EHT stations |
| `station_names` | [...] | Station list (ALMA, APEX, JCMT, SMA, SMT, LMT, PV, SPT) |
| `eta` | 0.88 | Antenna efficiency |
| `bandwidth_hz` | 2.0e9 | Receiver bandwidth (Hz) |
| `tau_int` | 10.0 | Integration time per scan (s) |
| `baselines_per_frame` | 28 | Baselines per frame |

## Method Hints

**State-space model for dynamic imaging.** StarWarps treats the image sequence as a first-order Gaussian Markov process, coupling adjacent frames through a warp matrix that models affine motion. This temporal prior regularizes the severely under-determined per-frame reconstruction, allowing information to propagate across frames.

**Expectation-Maximization.** The warp parameters θ and per-frame posteriors are jointly optimized via EM. The E-step runs forward-backward Kalman filtering to compute posterior means and covariances of each frame given all measurements. The M-step updates θ by gradient descent on the expected complete-data log-likelihood.

**Static baseline.** Before running the dynamic reconstruction, independently reconstruct each frame with a Gaussian MAP prior. This serves as both a sanity check and an initialization for the Markov model.

## References

- Bouman et al. 2017, arXiv:1711.01357 ("StarWarps")
- Event Horizon Telescope Collaboration, Paper IV (2019), ApJL 875, L4
- Chael et al. 2018, ApJ 857:23 (closure-only imaging)
