# Seismic Full Waveform Inversion (From-Scratch PyTorch Implementation)

> Recover the subsurface P-wave velocity model from surface seismograms using a from-scratch PyTorch implementation of acoustic wave propagation with C-PML absorbing boundaries.

> Domain: Earth Science | Keywords: wave imaging, inverse scattering | Difficulty: Medium

---

2D acoustic Full Waveform Inversion on the Marmousi model — **all three
components implemented from scratch**, without importing `deepwave`.

This task is a companion to `seismic_FWI`.  The algorithms are identical but
every deepwave call is replaced by a pure PyTorch implementation so the full
wave-propagation stack is transparent and auditable.

## Background

Recover the subsurface P-wave velocity field $v(y, x)$ from surface
seismograms recorded over the Marmousi model.  The inverse problem minimizes:

$$J(v) = \tfrac{1}{2}\sum_s \|\mathcal{T}[F_s(v)] - \mathcal{T}[d_s^{\text{obs}}]\|_2^2$$

where $F_s$ is the acoustic wave propagation operator and $\mathcal{T}$ is a
5-sample cosine taper.

## Data Description

### `data/raw_data.npz`

Synthetic shot gathers and initial velocity model from the Marmousi benchmark, subsampled to 20 m grid spacing.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `v_init` | (461, 151) | float32 | Smoothed initial velocity model (m/s) |
| `observed_data` | (10, 93, 1350) | float32 | Synthetic shot gathers (n_shots × n_receivers × n_time) |
| `dx` | () | float32 | Grid spacing (20.0 m) |
| `dt` | () | float64 | Time step (0.004 s) |
| `freq` | () | float64 | Ricker wavelet peak frequency (5.0 Hz) |
| `n_shots` | () | int64 | Number of shots (10) |
| `n_receivers` | () | int64 | Receivers per shot (93) |
| `nt` | () | int64 | Time samples per trace (1350) |
| `source_depth` | () | int64 | Source depth in grid cells (1) |
| `receiver_depth` | () | int64 | Receiver depth in grid cells (1) |

### `data/ground_truth.npz`

True Marmousi P-wave velocity model used to generate the synthetic shot gathers.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `v_true` | (461, 151) | float32 | Marmousi P-wave velocity (m/s) at 20 m grid |

### `data/meta_data.json`

JSON file with nested acquisition, inversion, and grid parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `preprocessing.reduced_shape` | [461, 151] | Grid size after 5× subsampling (Ny × Nx) |
| `preprocessing.dx_m` | 20.0 | Grid spacing (m) |
| `velocity.v_true_min_m_s` | 1500.0 | True model minimum velocity (m/s) |
| `velocity.v_true_max_m_s` | 5500.0 | True model maximum velocity (m/s) |
| `acquisition.n_shots` | 10 | Number of seismic shots |
| `acquisition.n_receivers_per_shot` | 93 | Receivers per shot |
| `acquisition.source_depth_m` | 20.0 | Source depth (m) |
| `wavelet.frequency_hz` | 5.0 | Ricker wavelet peak frequency (Hz) |
| `time.nt` | 1350 | Time samples per trace |
| `time.dt_s` | 0.004 | Time step (s) |
| `inversion.optimizer` | "Adam" | Optimizer |
| `inversion.lr` | 100.0 | Learning rate |
| `inversion.lr_milestones` | [75, 300] | Epochs for LR reduction |
| `inversion.n_epochs` | 800 | Total training epochs |
| `inversion.v_min_bound_m_s` | 1480.0 | Velocity lower bound (m/s) |
| `inversion.v_max_bound_m_s` | 5800.0 | Velocity upper bound (m/s) |

## Method Hints

1. **Wave solver**: 4th-order finite differences in space, 2nd-order Verlet in time, with Convolutional PML (C-PML) absorbing boundaries on all four sides — implemented from scratch in PyTorch without deepwave.

2. **Adjoint gradients**: PyTorch autograd handles the adjoint-state method implicitly. Apply Gaussian smoothing and percentile clipping to the gradient before the optimizer step (same post-processing as seismic_FWI).

3. **Cosine taper**: Apply to the last few samples of each trace to suppress end-of-record artifacts before computing the MSE loss.

## References

The wave propagation algorithm follows the deepwave Python backend
(Alan Richardson, https://github.com/ar4/deepwave) and the C-PML derivation
in Pasalic & McGarry (2010) SEG.
