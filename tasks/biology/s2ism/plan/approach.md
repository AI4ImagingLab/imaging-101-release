# Approach

## Problem

Image-scanning microscopy (ISM) uses a SPAD detector array to capture spatially
resolved fluorescence data that encodes both in-focus and out-of-focus
information across multiple detector channels. The goal is to jointly deconvolve
all detector channels to recover a 3-D fluorescent object with improved
resolution and optical sectioning compared to confocal microscopy.

## Algorithm: Multi-Plane Maximum-Likelihood Expectation-Maximisation (ML-EM)

The reconstruction uses a Richardson–Lucy-style ML-EM algorithm extended to
handle the multi-channel, multi-plane ISM forward model.

### Forward Model

Given a 3-D object $x$ (dimensions $N_z \times N_y \times N_x$) and a set of
channel-specific PSFs $h_c$ (one per SPAD detector element, per z-plane):

$$y_c = \sum_z h_{z,c} * x_z$$

where $*$ denotes spatial convolution and $y_c$ is the measurement at
detector channel $c$.

### ML-EM Update Rule

At each iteration $k$:

1. **Forward project**: $\hat{y}_c = \sum_z h_{z,c} * x_z^{(k)}$
2. **Compute ratio**: $r_c = y_c / \hat{y}_c$ (with epsilon floor to avoid division by zero)
3. **Back project**: $u_z = \sum_c \tilde{h}_{z,c} * r_c$ where $\tilde{h}$ is the flipped PSF
4. **Multiplicative update**: $x_z^{(k+1)} = x_z^{(k)} \cdot u_z$

### Implementation Details

- Convolutions are performed in the Fourier domain using `torch.fft.rfftn` for
  efficiency, with Einstein summation (`torch.einsum`) for the multi-channel
  partial convolution.
- GPU acceleration is attempted automatically; falls back to CPU if CUDA OOM.
- Stopping criterion: either fixed iteration count or automatic convergence
  detection based on photon count change in the focal plane.

### Hyperparameters

| Parameter     | Value | Rationale                                      |
|---------------|-------|-------------------------------------------------|
| `max_iter`    | 50    | Sufficient convergence for simulation data      |
| `stop`        | fixed | Predictable runtime; auto-stop also available   |
| `initialization` | flat | Uniform initial estimate scaled to total counts |

## Synthetic Data Generation

Because no real ISM acquisition is bundled with the task, a synthetic dataset
is produced by `src/generate_data.py`. The pipeline has four stages, each
exposed as a separate function so that individual steps can be unit-tested.

### Stage 1 — Phantom (`make_tubulin_phantom`)

A 2-plane tubulin phantom is sampled with `brighteyes_ism.simulation.Tubulin_sim`:

- `Nx = 201`, `Nz = 2`, `pxsizex = 40 nm`
- 10 random filaments per plane, radius 80 nm, intensity in [0.6, 1]
- Plane $z=1$ is scaled by a factor of 3 relative to $z=0$ to simulate a
  thicker out-of-focus background
- Final phantom is multiplied by `signal_level = 300` to convert to photon counts

### Stage 2 — Optimal background plane (`find_out_of_focus_from_param`)

The out-of-focus plane distance is **not** a free hyperparameter — it is
chosen so that the in-focus and out-of-focus PSFs are maximally distinguishable.
Concretely, with $h_z(\cdot)$ the (channel-summed) PSF at axial offset $z$:

$$z^* = \arg\min_{z\,>\,0}\; \rho\!\bigl(h_0,\, h_z\bigr)$$

where $\rho$ is Pearson correlation. The function scans $z$ over the depth of
field, generates a candidate PSF at each step, and returns the first local
minimum of the correlation curve. For the parameters of this task this gives
$z^* = 720$ nm.

This step is the most error-prone for an agent to reproduce: the temptation is
to use a default small spacing (e.g. 10–100 nm), but doing so makes the two
plane PSFs nearly identical and destroys the multi-plane reconstruction.

### Stage 3 — PSF simulation (`simulate_psfs`)

A multi-plane, 25-channel SPAD PSF stack is generated with
`brighteyes_ism.simulation.PSF_sim.SPAD_PSF_3D`, with the critical assignment

```python
gridPar.pxsizez = optimal_bkg_plane   # NOT a default value
```

The PSF for each axial plane is normalised to unit sum so that the forward
model preserves total photon count.

### Stage 4 — Forward model + noise (`apply_forward_model_with_noise`)

For each detector channel $c$ and each axial plane $z$, the phantom is
convolved with the corresponding PSF; results are summed over $z$ to obtain
the noiseless measurement, then corrupted by Poisson noise:

$$y_c = \mathrm{Poisson}\!\Bigl(\textstyle\sum_z h_{z,c} * x_z\Bigr).$$

### Output files

| File              | Contents                                                |
|-------------------|----------------------------------------------------------|
| `raw_data.npz`    | `measurements` (noisy ISM stack), `psf` (PSF stack)      |
| `ground_truth.npz`| `ground_truth` (clean phantom in photon counts)          |
| `meta_data.json`  | Imaging parameters, including `optimal_bkg_plane_nm`     |

## PSF Estimation Pipeline

When the PSF is not directly provided, it can be estimated from the ISM data:

1. **Shift vector estimation**: cross-correlate each detector channel with the
   central channel to find sub-pixel shifts, then fit rotation/scaling/mirror
   parameters via Nelder-Mead minimisation.
2. **Magnification estimation**: match the measured inter-channel shift to a
   scalar PSF model to estimate the effective magnification.
3. **Out-of-focus plane**: scan z-depths using Pearson correlation (or KL
   divergence) between PSFs at different planes to find the optimal background
   plane depth.
4. **PSF simulation**: use the estimated parameters with `brighteyes-ism` to
   generate the full multi-channel, multi-plane PSF stack.
