# Approach: Tensor-Var 4D-Var on ERA5

## Problem statement

Estimate a sequence of global atmospheric state fields $\{x_t\}_{t=1}^{T}$ on a
$64 \times 32$ latitude–longitude grid (5 channels: geopotential $z$,
temperature $T$, specific humidity $q$, zonal wind $u$, meridional wind $v$),
given a 10-step rolling history of noisy observations
$\{o_{t-9}, \ldots, o_{t}\}$ for each timestep, by running 4-dimensional
variational data assimilation (4D-Var) over an assimilation window of length
5.

## Mathematical formulation

### Standard 4D-Var

Classical 4D-Var minimises the cost

$$
J(x_{0:T}) = (x_0 - x_b)^\top B_x^{-1} (x_0 - x_b)
           + \sum_{t=0}^{T-1} (h(x_t) - o_t)^\top R_x^{-1} (h(x_t) - o_t)
           + \sum_{t=1}^{T-1} (x_t - M(x_{t-1}))^\top Q_x^{-1} (x_t - M(x_{t-1}))
$$

where $M$ is the dynamics model, $h$ the observation operator, and $B_x, R_x,
Q_x$ are the background, observation and process noise covariances. For
high-dimensional nonlinear $M$ (a global atmospheric model has $\sim 10^7$
state variables) this is computationally prohibitive, and an analytic
adjoint is rarely available.

### Tensor-Var: linear 4D-Var in a learned feature space

Tensor-Var (Yang et al., 2025) learns three encoders:

* $\phi_S\colon \mathbb{R}^{C \times H \times W} \to \mathbb{R}^{D}$
  — a convolutional encoder mapping the **state field** to a feature vector
  of dimension $D = 512$ (`K_S` in the code).
* $\phi_S^{-1}\colon \mathbb{R}^{D} \to \mathbb{R}^{C \times H \times W}$
  — a convolutional decoder reconstructing the state from features (with
  encoder skip connections in the U-Net style; `K_S_preimage`).
* $\phi_O\colon \mathbb{R}^{H_o \cdot C \times H \times W} \to
  \mathbb{R}^{C \times H \times W}$
  — a transformer that takes a window of $H_o = 10$ stacked observations and
  outputs a single state field (`K_O`, an inverse-observation network).

A *linear* feature-space dynamics matrix $F \in \mathbb{R}^{D \times D}$
(`C_forward`) is fitted offline so that
$\phi_S(x_{t+1}) \approx F^\top \phi_S(x_t)$.

Define the latent state $z_t = \phi_S(x_t)$ and the observation feature
$\tilde z_t = \phi_S(\phi_O(o_{t-H_o+1:t}))$ — i.e. the inverse-obs network
first predicts a state field from the observation history, and then the same
encoder $\phi_S$ projects that field into the latent space. The Tensor-Var
4D-Var cost in the feature space is

$$
J(z_{0:T}) =
(z_0 - z_b)^\top B \, (z_0 - z_b) \;+\;
\sum_{t=0}^{T-1} (z_t - \tilde z_t)^\top R \, (z_t - \tilde z_t) \;+\;
\sum_{t=1}^{T-1} w_t \,(z_t - F^\top z_{t-1})^\top Q \,(z_t - F^\top z_{t-1})
$$

where $z_b \in \mathbb{R}^{D}$ is the background mean of the latent
distribution (precomputed offline by averaging $\phi_S$ over the training
set; shipped as `z_b.pt`); $B, R, Q \in \mathbb{R}^{D \times D}$ are the
information (precision) matrices in feature space; and
$w_t = \mathrm{linspace}(1, 0.1, T-1)_{T-1-t}$ is a linear penalty
schedule that down-weights the dynamics-consistency term toward the end of
the window.

Because every term is quadratic in $z_t$, this is a **convex quadratic
program** of size $TD = 5 \times 512 = 2560$ that any QP solver dispatches in
about a second. The optimal feature trajectory $z_{0:T}^\star$ is then mapped
back to physical state space with the encoder skip connections that were
captured during the encode of $\phi_O(o)$ — i.e.
$\hat x_t = \phi_S^{-1}(z_t^\star;\, \mathrm{enc}(\phi_O(o_t)))$.

### Covariance fall-back

The released checkpoint zip does not include calibrated information matrices,
so this task uses the same fall-back as the upstream notebook:
$B = 0.1\,I_D$, $Q = 0.1\,I_D$, $R = I_D$.
Reproducing the calibrated $B/Q/R$ matrices would require running the full
training pipeline on the train split (~6 GB) and is out of scope.

## Solution strategy

### Step 1 — Bundled data

The task ships **one** representative sample taken at start index `s = 750`
with `seed = 0` from the upstream `test_seq_obs.h5 / test_seq_state.h5`
files. The observation history (already noisy and normalised by per-channel
min/max), the latitude weighting matrix, and the per-channel min/max
constants live in `data/raw_data.npz`. The true state sequence (in normalised
space) lives in `data/ground_truth.npz`. No solver hyperparameter is stored
in `meta_data.json`.

### Step 2 — Pretrained checkpoints

The four pretrained `.pt` files (`forward_model.pt`, `C_forward.pt`,
`inverse_model.pt`, `z_b.pt`, ~30 MB total) are **not** bundled with the
benchmark. `src.preprocessing.download_pretrained_weights` fetches them from
the upstream Tensor-Var Google Drive release on first use via `gdown` and
extracts them into `evaluation/checkpoints/`.

### Step 3 — Forward and inverse networks

`src.physics_model.ERA5ForwardModel` is a thin wrapper over the convolutional
encoder $\phi_S$ (`ERA5_K_S`) and decoder $\phi_S^{-1}$ (`ERA5_K_S_preimage`).
Layer names exactly mirror the upstream Tensor-Var repo so the released
state-dict loads with no key remapping. `ERA5InverseModel` wraps the
transformer-based $\phi_O$ network (`ERA5_K_O`, plus frozen copies of $\phi_S$
and $\phi_S^{-1}$ to match the checkpoint).

### Step 4 — 4D-Var loop

`src.solvers.tensor_var_4dvar` runs the assimilation in a single pass:

1. Apply $\phi_O$ to the 10-step observation history for each timestep:
   $s_t = \phi_O(o_{t-9:t})$.
2. Apply $\phi_S$ to that pseudo-state to obtain features $\tilde z_t$, while
   capturing the encoder skip-connection feature maps for the decode step.
3. Solve the QP with `cvxpy` (default backend) over the window.
4. Decode the optimal $z^\star_t$ with $\phi_S^{-1}$, supplying the encoder
   skip features so the decoder mirrors the encoder's spatial pooling.

For this task `assimilation_window = total_steps = 5`, so the outer loop runs
exactly once and the analysis trajectory has shape `(5, 5, 64, 32)`.

### Step 5 — Evaluation

`src.visualization.compute_metrics_per_channel` reports per-channel NCC,
range-normalised NRMSE, and the latitude-weighted relative L2 error
$\mathrm{wNRMSE}_c = \overline{\| (\hat x_t - x_t)\odot w \|_2 / \| x_t \odot w \|_2}$
that the upstream paper uses to compare against operational baselines. All
five channels are scored.

### Hyperparameters

| Constant     | Value | Where |
|--------------|-------|-------|
| `_ASS_W`     | 5     | `main.py` |
| `_ASS_T`     | 5     | `main.py` |
| `_SEED`      | 0     | `main.py` |
| `_SAMPLE_INDEX` | 0  | `main.py` |
| QP backend   | `cvxpy` default (SCS / OSQP) | `solvers.qp_solver_latent` |

## Expected results

Reproducing `main.py` should yield (for the bundled sample, identity-fallback
covariances, CPU run):

| Channel       | NCC    | NRMSE  | wNRMSE |
|---------------|--------|--------|--------|
| geopotential  | 0.9999 | 0.0129 | 0.0087 |
| temperature   | 0.9998 | 0.0173 | 0.0127 |
| humidity      | 0.9988 | 0.0221 | 0.0450 |
| wind_u        | 0.9993 | 0.0268 | 0.0329 |
| wind_v        | 0.9992 | 0.0290 | 0.0326 |
| **mean**      | 0.9994 | 0.0216 | 0.0264 |

Boundaries in `evaluation/metrics.json` use the standard
$0.9 \times \text{baseline\_ncc}$ / $1.1 \times \text{baseline\_nrmse}$ rule
applied to each metric.

## References

* Yang Y., Cui T., Tang X., Tao J., Yu Z., Wang Q., Bouman K. L., Wang H.,
  *Tensor-Var: Efficient four-dimensional variational data assimilation*,
  2025. https://github.com/YimingYang2000/TensorVar
* Bocquet M., *Introduction to the principles and methods of data
  assimilation in the geosciences*, Lecture Notes, ENPC, 2014.
* ERA5 reanalysis: Hersbach H. et al., *The ERA5 global reanalysis*,
  Q. J. R. Meteorol. Soc., 146(730):1999–2049, 2020.
