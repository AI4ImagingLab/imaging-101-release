# EHT Black Hole Tomography (BH-NeRF)

> Recover the 3D volumetric emission structure of a black hole accretion flow from time-series images using a NeRF with Keplerian orbital dynamics and gravitational lensing.

> Domain: Astronomy | Keywords: radio interferometry, tomography, neural rendering | Difficulty: Hard

---

## Background

The Event Horizon Telescope (EHT) captures sparse radio interferometric measurements of supermassive black holes. Orbiting hot plasma ("hotspots") near the black hole create time-varying emission that is gravitationally lensed and Doppler-boosted. The challenge is to recover the full 3D emission field from these sparse 2D projections, leveraging the physical constraint that the emission follows Keplerian orbital dynamics.

## Problem Description

### Emission representation
The static 3D emission field is represented as a continuous function parameterized by an MLP neural network with positional encoding:

$$e_0(\mathbf{x}) = \text{sigmoid}(\text{MLP}_\theta(\gamma(\mathbf{x})) - 10)$$

where $\gamma(\mathbf{x})$ is a positional encoding with $L=3$ frequency bands.

### Keplerian dynamics
The emission at time $t$ is obtained by inverse-rotating the query coordinates according to Keplerian angular velocity:

$$e(t, \mathbf{x}) = e_0(R_{\xi}(-\Omega(r) \cdot t) \cdot \mathbf{x})$$

where $\Omega(r) = \sqrt{M}/r^{3/2}$ is the Keplerian angular velocity and $\xi$ is the (unknown) rotation axis.

### Gravitational lensing
Light rays near the black hole follow curved paths in the Schwarzschild metric. Pre-computed ray paths $\Gamma_n$ for each image pixel trace through the 3D volume. The pixel intensity is:

$$I_n(t) = \sum_i g^2(\mathbf{x}_i) \cdot e(t, \mathbf{x}_i) \cdot \Delta\tau_i \cdot \Sigma_i$$

where $g$ is the Doppler factor, $\Delta\tau$ is the proper time step, and $\Sigma = r^2$ for Schwarzschild.

### Measurement model
The EHT measures complex visibilities via discrete Fourier transform:

$$\mathbf{y}(t) = \mathbf{A}(t) \cdot \mathbf{I}(t) + \boldsymbol{\varepsilon}$$

### Optimization
Joint optimization over MLP weights $\theta$ and rotation axis $\xi$:

$$\mathcal{L}(\theta, \xi) = \sum_t \left\| \mathbf{y}(t) - \mathbf{A}(t) \cdot \mathbf{I}_{\theta,\xi}(t) \right\|^2_{\Sigma}$$

## Data Description

### `data/raw_data.npz`

Pre-computed gravitationally lensed ray paths and noisy observed image frames for a synthetic hotspot orbiting a Schwarzschild black hole.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `ray_x` | (48, 48, 80) | float32 | Cartesian x coordinate along each ray (units of M) |
| `ray_y` | (48, 48, 80) | float32 | Cartesian y coordinate along each ray (units of M) |
| `ray_z` | (48, 48, 80) | float32 | Cartesian z coordinate along each ray (units of M) |
| `ray_r` | (48, 48, 80) | float32 | Radial coordinate r along each ray (units of M) |
| `ray_theta` | (48, 48, 80) | float32 | Polar angle θ along each ray (radians) |
| `ray_t_geo` | (48, 48, 80) | float32 | Coordinate time along ray for slow-light correction (units of M) |
| `ray_dtau` | (48, 48, 80) | float32 | Mino time differential Δτ for volume integration |
| `ray_Sigma` | (48, 48, 80) | float32 | Metric factor Σ = r² (Schwarzschild) |
| `Omega` | (48, 48, 80) | float32 | Keplerian angular velocity Ω(r) = √M / r^{3/2} |
| `g_doppler` | (48, 48, 80) | float32 | Doppler boosting factor g at each ray point |
| `t_frames` | (40,) | float32 | Frame observation times (units of M) |
| `images_noisy` | (40, 48, 48) | float32 | Observed noisy image frames |
| `fov_M` | () | float32 | Image field of view (units of M) |
| `t_start_obs` | () | float32 | Observation start time (units of M) |
| `t_injection` | () | float32 | Hotspot injection time (units of M) |
| `r_observer` | () | float32 | Observer radial distance (units of M) |

### `data/ground_truth.npz`

Ground-truth noise-free image frames, 3D emission field, and Keplerian rotation axis.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `images` | (40, 48, 48) | float32 | Ground-truth noise-free image frames |
| `emission_3d` | (64, 64, 64) | float32 | Ground-truth 3D emission field at t=0 |
| `rot_axis` | (3,) | float32 | True Keplerian rotation axis (unit vector) |

### `data/meta_data.json`

JSON file with black hole geometry, observation, and NeRF training parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `spin` | 0.0 | Black hole spin (0 = Schwarzschild) |
| `inclination_deg` | 60.0 | Observer inclination angle (°) |
| `fov_M` | 24.0 | Image field of view (units of M) |
| `num_alpha` | 48 | Image pixels along α axis |
| `num_beta` | 48 | Image pixels along β axis |
| `ngeo` | 80 | Ray integration steps per pixel |
| `emission_resolution` | 64 | 3D emission volume resolution (cube side) |
| `orbit_radius_M` | 8.0 | Hotspot orbital radius (units of M) |
| `n_frames` | 40 | Number of observed time frames |
| `t_obs_M` | 2000.0 | Total observation duration (units of M) |
| `noise_std` | 0.01 | Per-pixel image noise σ |
| `net_depth` | 4 | NeRF MLP depth (layers) |
| `net_width` | 128 | NeRF MLP width (hidden units) |
| `posenc_deg` | 3 | Positional encoding frequency bands |
| `n_iters` | 2000 | Training iterations |
| `lr_init` | 1e-4 | Initial learning rate |
| `lr_final` | 1e-6 | Final learning rate |

## Method Hints

Use a coordinate-based neural network (NeRF) with:
- Positional encoding (L=3 frequency bands)
- 4-layer MLP with 128 hidden units and skip connections
- Keplerian velocity warp for temporal dynamics
- Volume rendering along pre-computed gravitational lensed ray paths
- Adam optimizer with polynomial learning rate schedule
- Multi-initialization of the rotation axis to avoid symmetric local minima

## References

1. Levis, A., Bharadwaj, P., Dugan, K.P., Vardhan, R., Dahlbom, D., Bouman, K.L. (2022). "Gravitationally Lensed Black Hole Emission Tomography." CVPR 2022. [arXiv:2204.03715]
2. Levis, A., et al. (2024). "Orbital polarimetric tomography of a flare near the Sagittarius A* supermassive black hole." Nature Astronomy.
