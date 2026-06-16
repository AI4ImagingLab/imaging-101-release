# ERA5 Tensor-Var 4D-Var

> One-sentence summary: Reconstruct a 5-step sequence of ERA5 atmospheric state fields (geopotential, temperature, humidity, zonal/meridional wind) on a 64×32 lat–lon grid from a rolling 10-step history of noisy observations by running 4D-Var as a quadratic program in a learned deep-kernel feature space.

> Domain: Earth Science | Keywords: data assimilation, weather, deep kernel features, variational inference | Difficulty: Hard

---

## Background

Numerical weather prediction systems estimate the atmospheric state by
*assimilating* a continuous stream of sparse, noisy observations into a model
forecast. The standard tool is **four-dimensional variational data
assimilation** (4D-Var), which jointly fits a physics model and an
observation operator over a sliding time window. For modern global models
this involves $\sim 10^7$ state variables and a nonlinear adjoint, which
makes a single assimilation step extremely expensive.

**Tensor-Var** (Yang et al., 2025) sidesteps both costs by lifting the state
into a finite-dimensional feature space using a learned encoder–decoder pair
($\phi_S$ / $\phi_S^{-1}$) and a transformer-based inverse-observation
network ($\phi_O$). In that latent space, the dynamics become a single
linear matrix multiplication ($z_{t+1} \approx F^\top z_t$), and the entire
4D-Var cost reduces to a convex **quadratic program** that off-the-shelf QP
solvers handle in about a second per assimilation window.

This task ships **one** representative test sample drawn from the upstream
ERA5 split (start index $s = 750$, seed $0$) and lets the agent run a single
assimilation window of length 5 with $\mathrm{num\_mc} = 1$.

## Problem Description

Let $x_t \in \mathbb{R}^{C \times H \times W}$ denote the atmospheric state
at time $t$, with $C = 5$ channels (geopotential, temperature, humidity,
zonal wind, meridional wind), $H = 64$ longitudes and $W = 32$ latitudes.
The forward measurement model is

$$
o_t = h(x_t) + \eta_t,
$$

where $h$ corrupts every channel with noise and (in the upstream dataset)
random masks; $\eta_t$ is per-channel additive noise. The bundled
$o_t$ tensor is already the noisy, normalised observation as produced by the
upstream `ERA5_DA_Dataset` pipeline. At each time step the assimilation
problem receives a 10-step *rolling history*

$$\mathbf{o}_t = (o_{t-9}, \ldots, o_{t})$$

— the inverse-observation network
$\phi_O$ accepts this whole window stacked along the channel axis as a
$( \text{history-len} \cdot C, H, W)$ tensor.

The inverse problem is to recover the analysis trajectory
$\hat x_{1:T}$ for $T = 5$ steps that best fits both the observation history
and a learned dynamical prior, while staying near the climatological mean
$x_b$. It is ill-posed because $h$ destroys information channel-by-channel
and per-pixel, and because the dynamics are only weakly informative over a
five-step window.

**Inputs and outputs**

* Input  : `obs_history` of shape $(T, H_o C, H, W) = (5, 50, 64, 32)$,
           plus $T = 5$ observation windows worth of context.
* Output : Analysis trajectory $\hat x_{1:T}$ of shape $(T, C, H, W) = (5, 5, 64, 32)$
           in the same normalised space as the bundled ground truth.

## Data Description

All arrays follow the imaging-101 batch-first convention: the leading axis
has length 1 (single sample).

### `data/raw_data.npz`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `obs_history` | (1, 5, 50, 64, 32) | float32 | Normalised noisy observation history; axis order `(sample, T, history_len*C, H, W)`. The 50 channels are 10 successive observation timesteps stacked over the 5 physical channels (geopotential, temperature, humidity, wind_u, wind_v). Per-channel min–max normalised to ≈ [0, 1]. |
| `max_val` | (1, 5) | float32 | Per-channel maximum used for de-normalisation. Order matches `meta_data.json::channels`. Units: `m^2/s^2`, `K`, `kg/kg`, `m/s`, `m/s`. |
| `min_val` | (1, 5) | float32 | Per-channel minimum used for de-normalisation. Same channel order and units as `max_val`. |
| `lat_weight_matrix` | (1, 5, 64, 32) | float32 | Cosine-latitude weighting tiled over the 5 physical channels and the 64×32 grid (dimensionless). Used by the weighted-NRMSE metric. |

### `data/ground_truth.npz`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `state` | (1, 5, 5, 64, 32) | float32 | True state sequence in the same normalised space as the observations. Axis order `(sample, T, C, H, W)`. Per-channel range ≈ [0, 1]. |

### `data/meta_data.json`

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_channels` | 5 | Number of physical channels per state field. |
| `channels` | `["geopotential", "temperature", "humidity", "wind_u", "wind_v"]` | Channel order for `state`, `max_val`, `min_val`, `lat_weight_matrix`. |
| `channel_units` | `["m^2/s^2", "K", "kg/kg", "m/s", "m/s"]` | Physical units (post-denormalisation). |
| `grid_height` | 64 | Number of longitude grid points. |
| `grid_width` | 32 | Number of latitude grid points. |
| `history_len` | 10 | Number of observation timesteps stacked into one inverse-obs network input. |
| `ass_T` | 5 | Number of timesteps assimilated per sample. |
| `longitude_resolution_deg` | 5.625 | 360 / 64. |
| `latitude_resolution_deg` | 5.625 | 180 / 32. |
| `dataset_source` | string | Provenance: ERA5 reanalysis via the Tensor-Var release. |

`meta_data.json` contains imaging geometry only — solver hyperparameters
(assimilation window length, seed, sample index, QP backend) live as named
constants in `main.py` to avoid leaking algorithm details to the evaluation
agent.

### Pretrained checkpoints

The four pretrained `.pt` files used by the forward / inverse networks
(`forward_model.pt`, `C_forward.pt`, `inverse_model.pt`, `z_b.pt`, ~30 MB
total) are **not** bundled with this task. `main.py` downloads them on first
run via `src.preprocessing.download_pretrained_weights`, which fetches the
upstream `ERA5_model_weights.zip` from the Tensor-Var Google Drive release
(`gdown` ID `1Rmj18sLiHk6OmOMOoHKHS6Gpi_ZRKhlY`) and extracts them into
`evaluation/checkpoints/`. The same download is performed lazily by the
parity tests.

## Method Hints

This task is naturally solved by **Tensor-Var**, a 4D-Var formulation that
uses three learned operators to lift the assimilation problem into a
finite-dimensional feature space:

1. A convolutional encoder $\phi_S\colon \mathbb{R}^{C\times H\times W}
   \to \mathbb{R}^D$ that maps a state field to a $D = 512$ dimensional
   feature vector, with a matching U-Net-style decoder $\phi_S^{-1}$.
2. A transformer-based inverse-observation network $\phi_O$ that maps a
   stack of 10 historical observations to a single state field.
3. A linear feature-space dynamics matrix $F \in \mathbb{R}^{D\times D}$
   pretrained so that $\phi_S(x_{t+1}) \approx F^\top \phi_S(x_t)$.

In feature space, the 4D-Var cost is a sum of three quadratic terms (a
background term anchored at the climatological feature mean $z_b$, an
observation term, and a dynamics-consistency term with a linearly decaying
window weight). Solving it is therefore a single convex quadratic program
that any QP backend handles in seconds. The optimal feature trajectory is
then mapped back to physical state space with $\phi_S^{-1}$, fed the encoder
skip-connection feature maps captured during the encode of $\phi_O(o)$.

`plan/approach.md` gives the full mathematical derivation of the cost,
including the linear penalty schedule and the identity-fallback covariances.

## References

* Yang Y., Cui T., Tang X., Tao J., Yu Z., Wang Q., Bouman K. L., Wang H.,
  *Tensor-Var: Efficient four-dimensional variational data assimilation*,
  2025. https://github.com/YimingYang2000/TensorVar
* Hersbach H. et al., *The ERA5 global reanalysis*, Q. J. R. Meteorol. Soc.,
  146(730):1999–2049, 2020.
* Bocquet M., *Introduction to the principles and methods of data
  assimilation in the geosciences*, ENPC Lecture Notes, 2014.
