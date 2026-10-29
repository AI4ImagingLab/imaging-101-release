# Hessian-SIM: Solution Approach

## Overview

The goal is to reconstruct a 2x super-resolved, temporally stable image
sequence from six phase-shifted 2-beam SIM frames per time point. The solver
runs in six sequential stages; each stage consumes the output of the
previous one and emits explicit intermediate quantities so that every stage
can be independently validated.

## Stage 1 — Pattern frequency estimation

Given the six calibration frames `[a0p0, a0p1, a0p2, a1p0, a1p1, a1p2]`:

1. Average repeated acquisitions (`beishu_an = 1` for this task, so a no-op).
2. Band-separate with a nominal phase matrix built from the `spjg = [4, 4, 3]`
   phase-step ratios (cumulative fractions 0, 4/11, 8/11 of `2π`).
3. Normalise each separated band by its own magnitude — only the phase is
   used for the cross-correlation.
4. Cross-correlate the 0th-order band with a flipped / conjugated 1st-order
   band using `dft_conv`, mask to a ring of radius ``pg ± fanwei`` around the
   expected pattern frequency, and read the peak as a coarse
   ``(k_x, k_y)`` estimate.
5. Refine sub-pixel with a greedy coordinate-descent merit search that
   halves the step whenever the merit fails to improve, down to tolerance
   `1e-4`.

Expected pattern wavevector magnitude: ``pg ≈ 122`` for NA=0.9 at the 512-pixel
working grid.

## Stage 2 — Modulation depth & phase offset

For each sideband order:

1. Use two OTF overlap regions — a wide one (`fc_ang = 120`) for phase
   estimation and a narrower one (`fc_con = 105`) for modulation depth.
2. Compute the complex modulation map
   ``\mathrm{cm}(k) = \mathrm{youhua}(k) / \mathrm{chongdie}(k)`` where
   `youhua` is the product of the shifted sideband and the OTF support and
   `chongdie` is the overlap of the centred and shifted OTFs.
3. Histogram `angle(cm)` (phase) and `|cm|` (depth) over the overlap region.
4. Empirical-mode decompose each histogram, drop the first few high-frequency
   IMFs (the first 3 or 4 depending on peak height), and take the argmax of
   the smoothed histogram as the peak.

## Stage 3 — Wiener SIM reconstruction

For each time point (every six raw frames):

1. Sigmoid-window the raw frames to suppress edge ringing before FFT.
2. Band-separate using a per-angle phase matrix rebuilt from `deph` offsets.
3. Zero-pad to `2n` and Fourier-shift each order to its true
   ``(k_x, k_y)`` location.
4. Apply an estimated-modulation-depth weight per order
   ``w_{a,p} = 1 / m_a`` (except the 0th order which keeps weight 1).
5. Combine with the Wiener formula
   ``\hat x(k) = \sum w_{a,p} H^{*}_{a,p} \tilde y_{a,p} / (\sum |H_{a,p}|^2 + 0.005 \cdot n_a \cdot \mathrm{weilac}^2)``.
6. Apodise by a cosine window at the extended frequency support
   ``k_\max = p_\text{long} + f_c`` and inverse-transform.
7. Crop to ``(2s_y, 2s_x)`` and clip negatives to zero.

## Stage 4 — Hessian denoising (split-Bregman)

Minimise
```
argmin_x   λ Σ_k ||Φ_k x||_1   +   (μ/2) ||x - y||_2^2
```
with `Φ_k ∈ {d_xx, d_yy, d_zz, d_xy, d_xz, d_yz}`, cross-terms and z-axis
terms carrying an additional `σ_z` weight.

- `μ (siranu) = 150`, `λ = 20 μ = 3000`, `σ_z = 1`, 100 iterations.
- Precompute the FFT-domain operator
  ``\mathrm{divide} = μ/λ + Σ_k |\tilde\Phi_k|^2``.
- Each iteration: (i) FFT-domain solve for `x`, (ii) soft-threshold each
  `Φ_k x + b_k` with threshold `1/λ`, (iii) update Bregman variable
  `b_k += Φ_k x - d_k`.
- First iteration special-case: divide only by `μ/λ` because the derivative
  terms are zero.
- Un-normalise at the end by multiplying by `ymax` (the pre-normalisation
  maximum).

## Stage 5 — TV denoising (baseline)

Same split-Bregman with just `{d_x, d_y}` (2D anisotropic TV). `z_bei = 0`
drops the temporal coupling. Used to illustrate the staircase artefacts
avoided by the Hessian prior.

## Stage 6 — Temporal running average

Three-frame centred boxcar over the denoised stack.

## Hyperparameters (not in `meta_data.json`)

| Parameter | Value | Stage |
|---|---|---|
| `weilac` | 2.0 | Wiener |
| `μ` | 150 | Hessian / TV |
| `σ_z` | 1.0 | Hessian |
| `n_iter` | 100 | Hessian / TV |
| `fanwei` | 50 | param-est search band |
| `beishu_an`, `beishu_re` | 1 | frame-averaging (no-op) |

## Validation

After each stage, the reference Wiener-SIM output saved in
`evaluation/reference_outputs/wiener_sim.npz` is the target. A successful
reconstruction achieves
`NCC(hessian, wiener_ref) >= 0.9 * baseline_ncc` and
`NRMSE(hessian, wiener_ref) <= 1.1 * baseline_nrmse` (see
`evaluation/metrics.json`).
