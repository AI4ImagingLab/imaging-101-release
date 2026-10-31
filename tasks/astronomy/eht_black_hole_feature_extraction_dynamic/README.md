# Dynamic EHT Black Hole Feature Extraction (α-DPI)

> Infer posterior distributions of black hole crescent geometry from time-varying EHT closure quantities using α-DPI normalizing flows with importance sampling.

> Domain: Astronomy | Keywords: radio interferometry, variational inference, feature extraction, dynamic imaging | Difficulty: Hard

---

## Background

Per-frame geometric parameter inference from time-varying EHT closure quantities using α-DPI (α-divergence variational inference + importance sampling). This task applies the α-DPI method independently to each time snapshot of a dynamic black hole observation, producing posterior distributions of geometric crescent parameters at each epoch.

The output is a **ridge plot** showing how parameter posteriors evolve over time, similar to Figure 13 of the EHT Sgr A* Paper IV (2022, ApJL 930, L15).

## Problem Description

Sagittarius A* (Sgr A*) exhibits rapid structural variability — the bright emission region orbits the black hole on timescales of minutes to hours. Unlike M87*, whose morphology is essentially static during a single observation, Sgr A* requires time-resolved analysis.

This task simulates this scenario with a **rotating simple crescent**: the position angle of the bright side rotates linearly over the observation window while diameter, width, and asymmetry remain constant. The EHT 2017 array (8 stations: ALMA, APEX, JCMT, SMA, SMT, LMT, PV, SPT) observes each time snapshot independently.

The SimpleCrescent geometric model has 4 parameters:

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Diameter | d | [20, 80] μas | Outer ring diameter |
| Width | w | [1, 40] μas | Ring thickness (Gaussian σ) |
| Asymmetry | A | [0, 1] | Brightness contrast |
| Position angle | φ | [-181, 181] deg | Bright side orientation (E of N) |

## Data Description

### `data/raw_data.npz`

Per-frame interferometric visibilities from 8 EHT stations (T = 10 frames, t = 0…9).
Per-frame keys are stored with a numeric suffix (e.g. `vis_0`, `vis_1`, …, `vis_9`).

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `n_frames` | () | int64 | Number of time frames (10) |
| `frame_times` | (10,) | float64 | Frame mid-times in hours from observation start |
| `vis_0` … `vis_9` | (28,) | complex128 | Complex visibilities at frame t (Jy) |
| `sigma_0` … `sigma_9` | (28,) | float64 | Thermal noise σ per baseline at frame t (Jy) |
| `uv_0` … `uv_9` | (28, 2) | float64 | UV coordinates (u, v) per baseline at frame t (wavelengths) |
| `station_ids_0` … `station_ids_9` | (28, 2) | int64 | Station index pair for each baseline at frame t |

### `data/ground_truth.npz`

Ground-truth brightness frames and geometric parameters of the rotating crescent.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `images` | (10, 64, 64) | float64 | Ground-truth brightness frames (Jy/pixel) |
| `position_angle_deg` | (10,) | float64 | True bright-side position angle per frame (°, E of N) |
| `diameter_uas` | (10,) | float64 | True outer ring diameter per frame (μas) |
| `width_uas` | (10,) | float64 | True ring width per frame (μas) |
| `asymmetry` | (10,) | float64 | True brightness asymmetry per frame [0, 1] |

### `data/meta_data.json`

JSON file with observation, source, and inference parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `npix` | 64 | Image size (npix × npix pixels) |
| `fov_uas` | 120.0 | Field of view (μas) |
| `obs_duration_hr` | 8.0 | Total observation duration (hours) |
| `frame_duration_hr` | 0.8 | Duration of each time frame (hours) |
| `n_frames` | 10 | Number of time frames |
| `geometric_model` | "simple_crescent" | Crescent model used for generation |
| `total_flux` | 0.6 | Source total flux (Jy) |
| `freq_ghz` | 230.0 | Observing frequency (GHz) |
| `n_stations` | 8 | Number of EHT stations |
| `station_names` | [...] | Station list (ALMA, APEX, JCMT, SMA, SMT, LMT, PV, SPT) |
| `n_flow` | 16 | Number of normalizing flow layers |
| `n_epoch` | 5000 | Training epochs per frame |
| `lr` | 1e-4 | Learning rate |
| `alpha_divergence` | 1.0 | α-divergence parameter |
| `data_product` | "cphase_logcamp" | Observables used (closure phases + log closure amplitudes) |

## Method Hints

For each time frame independently, extract closure phases and log closure amplitudes (gain-invariant observables), then train a Real-NVP normalizing flow to minimize the α-divergence between the approximate posterior and the true posterior over the 4 crescent parameters. Apply importance sampling reweighting for accurate posterior calibration. Repeat across all 10 frames to produce ridge plots of the evolving parameter posteriors.

## References

- Sun et al. (2022), ApJ 932:99 — α-DPI method
- EHT Collaboration (2022), ApJL 930:L15 — Sgr A* Paper IV (Variability, Figure 13)
- EHT Collaboration (2022), ApJL 930:L12 — Sgr A* Paper I
