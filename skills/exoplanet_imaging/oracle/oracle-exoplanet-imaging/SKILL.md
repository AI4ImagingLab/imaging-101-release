---
name: oracle-exoplanet-imaging
description: "Reconstruct a KLIP-ADI exoplanet detection map from a temporal high-contrast coronagraphic image cube: build a Karhunen-Loeve PSF subspace, subtract it, derotate by parallactic angle, and combine into a single 2-D detection map written to the required output schema."
---

# oracle-exoplanet-imaging Oracle Skill

This skill reconstructs a faint companion detection map from an Angular Differential Imaging (ADI) cube using Karhunen-Loeve Image Processing (KLIP). The pipeline masks the inner working angle, mean-subtracts each frame, builds a truncated KL eigenbasis over the full frame library, projects each frame onto that basis and subtracts the reconstruction, derotates the residuals by the negative parallactic angle, and combines them into one detection map. Each operation section below is independently removable; follow the enabled sections in dependency order to reproduce the standard behavior. Use the run-local interpreter `python` resolved from the run environment PATH (for example the runtime venv under `public/envs/`); do not hardcode absolute interpreter paths. Read inputs only from run-local `public/`/case input directories and write only under `outputs/`, `workspace/`, and `logs/agent/`.

<!-- ORACLE_OP_START op_010_public_contract -->
## Task contract and submission schema

## Task contract and submission schema

Goal: given one ADI observation case, produce a single 2-D detection map in which a companion appears as a positive point source above residual speckle noise.

Input per case: an ADI image cube of shape `(N, H, W)` with `N` temporal frames and one parallactic angle (degrees) per frame. For the public case `N=61`, `H=W=100`.

Output contract (must match exactly):
- Write one file per case to `outputs/{case_id}.npz` relative to the run root.
- The archive must contain the array key `klip_image` with shape `(100, 100)` and dtype `float32` or `float64`.
- Validation requires **all values finite** and does **not** allow implicit casting. Because the reconstruction masks some pixels to NaN internally (see the inner-working-angle operation), you MUST replace every non-finite value with a finite number (use `0.0`) before saving, otherwise the submission fails the finite-only check.
- The map must be aligned to the reference (derotated, north-up) orientation.

Enumerate cases from the run-local cases manifest and write exactly one npz per case id. Do not write anywhere except `outputs/`.
<!-- ORACLE_OP_END op_010_public_contract -->

<!-- ORACLE_OP_START op_020_data_loading -->
## Load ADI cube, angles, PSF, and metadata

## Load ADI cube, angles, PSF, and metadata

Each case input directory contains a compressed numpy archive with three arrays. Load it and squeeze the leading singleton axis:
- `cube`: stored shape `(1, N, H, W)` float32 -> take `[0]` to get `(N, H, W)`. This is the ADI science cube in detector counts.
- `angles`: stored shape `(1, N)` float32 -> take `[0]` to get `(N,)`. Parallactic angles in degrees (public case spans about -14.3 to +67.1 deg).
- `psf`: stored shape `(1, Hpsf, Wpsf)` float32 -> take `[0]`. Normalized off-axis PSF, used only for FWHM reference; not required for the detection map.

Also read the case params JSON (instrument/observation parameters only) for the fields used downstream:
- `center_px`: stellar centroid as `(cx, cy)` in `(x, y)` pixel order (public case `[50, 50]`).
- `iwa_px`: inner working angle radius in pixels (public case `4`).
- `fwhm_px`: measured PSF FWHM in pixels (public case ~4.80).
- `pixel_scale_arcsec`: plate scale (public case 0.02719).

Cast `cube`, `angles`, `psf` to `float32`. Read arrays and JSON only from the run-local case input directory.
<!-- ORACLE_OP_END op_020_data_loading -->

<!-- ORACLE_OP_START op_030_coordinate_conventions -->
## Pixel/axis conventions and ADI forward model

## Pixel/axis conventions and ADI forward model

Hold these conventions fixed throughout; they define how coordinates, rotations, and masks are applied.

Forward model: each frame is `T_k(x) = I_psi(x) + A(R_{-theta_k} x)`, where `I_psi` is the (approximately static) speckle PSF and `A` is the sky scene rotated by the negative parallactic angle. KLIP estimates and removes `I_psi`; derotation then re-aligns `A` across frames.

Conventions:
- Image axes are `(rows, cols) = (y, x)`; array index `[i, j]` is `y=i, x=j`.
- `center_px` from metadata is `(cx, cy)` in `(x, y)` order. If center is unavailable, default to `((W-1)/2, (H-1)/2)`.
- Display/orientation origin is bottom-left (`origin='lower'`); a companion at sky position `(r, PA)` sits at detector pixel `R_{theta_k}(r sin PA, r cos PA) + c`.
- Angles are in degrees; positive angle is counter-clockwise. Derotation uses the negative angle.
- All working tensors are `float32`. NaN marks excluded pixels and must be handled with NaN-aware ops (fill zero before linear algebra; nan-mean/median when combining).
<!-- ORACLE_OP_END op_030_coordinate_conventions -->

<!-- ORACLE_OP_START op_040_iwa_masking -->
## Inner working angle circular masking

## Inner working angle circular masking

Before PCA, mask the coronagraph-dominated core so it does not contaminate the KL basis or the noise statistics. Build a 2-D boolean mask over `(H, W)`:
- Grid: `y, x = meshgrid(arange(H), arange(W), indexing='ij')` as float.
- Squared distance from center `(cx, cy)`: `dist2 = (x - cx)**2 + (y - cy)**2`.
- Masked where `dist2 <= iwa**2` (optionally also `dist2 >= owa**2` if an outer working angle is used; the standard run uses only the inner radius `iwa=4`).

Apply the mask to every frame by setting masked pixels to `NaN` (broadcast the 2-D mask across the `N` frames). These NaN pixels are excluded from the KL fit and from combination. Keep the same mask to re-apply to the final combined map (see output operation).
<!-- ORACLE_OP_END op_040_iwa_masking -->

<!-- ORACLE_OP_START op_050_mean_subtraction -->
## Per-frame spatial mean subtraction

## Per-frame spatial mean subtraction

KLIP builds the KL basis from mean-subtracted frames. For each frame independently, subtract its spatial mean computed while ignoring NaN pixels:
- Flatten to `(N, H*W)`.
- `means = nanmean(flat, axis=1, keepdims=True)` -> shape `(N, 1)`.
- `cube_ms = (flat - means).reshape(N, H, W)`.

This centers each frame so the covariance/SVD captures speckle structure rather than a DC offset. Do this after IWA masking and before forming the reference library.
<!-- ORACLE_OP_END op_050_mean_subtraction -->

<!-- ORACLE_OP_START op_060_kl_basis -->
## Karhunen-Loeve PSF subspace construction

## Karhunen-Loeve PSF subspace construction

Model the stellar PSF as the subspace spanned by the top-`K` KL eigenimages of the full-frame reference library (full-frame ADI KLIP; no annuli/subsections, all frames used as references).

Form the reference matrix `R` of shape `(N, n_pix)` from the mean-subtracted cube, replacing NaN with `0.0` (`nan_to_num`) so the linear algebra is well-defined. Compute the basis `Z` of shape `(n_pix, K_max)` where columns are eigenvectors. Compute once at `K_max = max(K_list)` and slice `[:, :k]` for each requested `k`.

Three numerically-equivalent methods (default `svd`):
- svd: `U, S, Vt = svd(R, full_matrices=False)`; basis = `Vt.T[:, :K_max]` (top-K right singular vectors).
- pca: `torch.pca_lowrank(R, q=K_max, center=False)` -> returned `V`.
- eigh (original Soummer 2012 form): `cov = R @ R.T` (`(N,N)`); `eigvals, eigvecs = eigh(cov)`; flip to descending; `basis = R.T @ (eigvecs[:, :K_max] * (1/sqrt(eigvals[:K_max])))`.

See the bundled resource for exact formulas and shapes. Truncation `K` is the regularizer: small `K` under-fits (speckle residual), large `K` over-fits (companion self-subtraction). Use `K` in the range ~5-20 for `N`~60 frames; the standard values are `K=10` and `K=20`.
<!-- ORACLE_OP_END op_060_kl_basis -->

<!-- ORACLE_OP_START op_070_psf_residuals -->
## KL projection and PSF subtraction

## KL projection and PSF subtraction

For each frame, project the mean-subtracted, NaN-filled frame onto the truncated KL basis and subtract the reconstruction. With `ref_flat` of shape `(N, n_pix)` (NaN replaced by 0) and basis `Z` of shape `(n_pix, K_max)`:

For each `k` in the requested truncation list:
- `proj = ref_flat @ Z[:, :k]`            # `(N, k)` KL coefficients
- `ihat = proj @ Z[:, :k].T`              # `(N, n_pix)` PSF reconstruction
- `residual_k = (ref_flat - ihat).reshape(N, H, W)`

Stack over `k` to a residual cube of shape `(n_K, N, H, W)`. This equals `A_k = T_k - sum_{j<=k} <T_k, Z_j> Z_j`, the PSF-subtracted frame series. After subtraction, restore the NaN mask: set residual pixels back to `NaN` wherever the original masked frame was NaN (broadcast the `(N,H,W)` NaN mask across the `n_K` axis).

If only a single `K` is needed for the output map, choose `K=10` (robust) or `K=20`; either is standard.
<!-- ORACLE_OP_END op_070_psf_residuals -->

<!-- ORACLE_OP_START op_080_derotation -->
## Parallactic-angle derotation of residuals

## Parallactic-angle derotation of residuals

Rotate each residual frame `k` by `-theta_k` (negative parallactic angle) so a fixed-sky companion aligns across frames. Implement as a single batched bilinear resample.

For a residual tensor of shape `(B, N, H, W)` (`B = n_K`) and angles `(N,)` in degrees:
- Flatten to `(B*N, 1, H, W)`; repeat angles across the `B` batch (`angles.repeat(B)`).
- Convert to radians with the derotation sign: `rad = deg2rad(-angles_repeated)`.
- Build per-image affine matrices `theta` of shape `(B*N, 2, 3)` with `[[cos, -sin, 0], [sin, cos, 0]]` (translation zero).
- `grid = affine_grid(theta, (B*N,1,H,W), align_corners=False)`.
- `rotated = grid_sample(flat, grid, align_corners=False)` (bilinear), reshape back to `(B, N, H, W)`.

Use `align_corners=False` and bilinear sampling to match the standard interpolation. NaN pixels may spread under interpolation; keep NaN-aware handling in the combination step.
<!-- ORACLE_OP_END op_080_derotation -->

<!-- ORACLE_OP_START op_090_combination -->
## Temporal combination into detection map

## Temporal combination into detection map

Collapse the derotated residual cube `(n_K, N, H, W)` along the frame axis `N` to a per-K detection map `(n_K, H, W)` using a NaN-aware statistic:
- mean (standard): `nanmean(derotated, dim=1)`.
- median (more robust to outlier frames): `nanmedian(derotated, dim=1)`.

Use `mean` by default. The result is the aligned (north-up) detection map where companions appear as positive point sources over approximately Gaussian residual noise. If a single output map is required, select the map for your chosen `K` (10 or 20).
<!-- ORACLE_OP_END op_090_combination -->

<!-- ORACLE_OP_START op_100_output_writing -->
## Final masking, finite fill, and npz writing

## Final masking, finite fill, and npz writing

Assemble the single submission map and write it:
1. Re-apply the inner-working-angle mask to the combined map (set pixels within `iwa` of center to NaN) so the masked core is consistent with the PCA exclusion.
2. Select the chosen `K` map (shape `(H, W) = (100, 100)`).
3. Because the output validator requires all-finite values and forbids casting, replace every non-finite value (NaN/inf, including the masked core) with `0.0`, and cast to `float32`.
4. Save to `outputs/{case_id}.npz` with the single array key `klip_image` of shape `(100, 100)`.

Use a self-contained writer; see the bundled helper for a reference implementation. Write only under `outputs/`.
<!-- ORACLE_OP_END op_100_output_writing -->

<!-- ORACLE_OP_START op_110_snr_diagnostic -->
## Companion SNR aperture t-test diagnostic

## Companion SNR aperture t-test diagnostic

Optional quality check (does not affect the written map). Measure companion signal-to-noise with the Mawet et al. (2014) small-sample two-sample t-test on the final detection map:
- Compute companion offset from image center `((W-1)/2, (H-1)/2)`: `dx = px - cx`, `dy = py - cy`, radius `r = hypot(dx, dy)`, `pa = degrees(atan2(dy, dx)) - 90`.
- Aperture radius = `fwhm/2`; number of resolution elements in the ring `n = max(1, int(2*pi*r/fwhm))`; angular step `delta = 2*pi/n`; first aperture at angle `pa+90` (companion), remaining apertures sample noise (skip `exclude_nearest` on each side of the companion).
- Aperture statistic = `nanmedian` of pixels within the aperture.
- `SNR = (signal - mean(noise)) / (std(noise, ddof=1) * sqrt(1 + 1/n_noise))`.

For the public target the companion sits near derotated pixel `(x, y) ~ (62, 63)` (separation ~16-17 px, ~0.44"); `fwhm ~ 4.8 px`. Expect SNR of order >=5 with `K=10-20`. Use as a sanity signal only; see the bundled helper.
<!-- ORACLE_OP_END op_110_snr_diagnostic -->

<!-- ORACLE_OP_START op_120_smoke_probe -->
## Shape and finiteness smoke probe

## Shape and finiteness smoke probe

Cheap non-mutating check to catch schema violations before submission. For each written `outputs/{case_id}.npz`, confirm:
- the archive contains key `klip_image`;
- its shape is exactly `(100, 100)`;
- its dtype is `float32` or `float64`;
- all values are finite (no NaN/inf).

Run with the run-local interpreter (`python scripts/op_120_smoke_probe.py`). This is a diagnostic guardrail and does not change outputs.
<!-- ORACLE_OP_END op_120_smoke_probe -->
