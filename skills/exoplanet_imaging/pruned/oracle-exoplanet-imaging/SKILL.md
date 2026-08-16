---
name: oracle-exoplanet-imaging
description: "Reconstruct a KLIP-ADI exoplanet detection map from a temporal high-contrast coronagraphic image cube: build a Karhunen-Loeve PSF subspace, subtract it, derotate by parallactic angle, and combine into a single 2-D detection map written to the required output schema."
---

# oracle-exoplanet-imaging Oracle Skill

This skill reconstructs a faint companion detection map from an Angular Differential Imaging (ADI) cube using Karhunen-Loeve Image Processing (KLIP). The pipeline masks the inner working angle, mean-subtracts each frame, builds a truncated KL eigenbasis over the full frame library, projects each frame onto that basis and subtracts the reconstruction, derotates the residuals by the negative parallactic angle, and combines them into one detection map. Each operation section below is independently removable; follow the enabled sections in dependency order to reproduce the standard behavior. Use the run-local interpreter `python` resolved from the run environment PATH (for example the runtime venv under `public/envs/`); do not hardcode absolute interpreter paths. Read inputs only from run-local `public/`/case input directories and write only under `outputs/`, `workspace/`, and `logs/agent/`.

## KL projection and PSF subtraction

## KL projection and PSF subtraction

For each frame, project the mean-subtracted, NaN-filled frame onto the truncated KL basis and subtract the reconstruction. With `ref_flat` of shape `(N, n_pix)` (NaN replaced by 0) and basis `Z` of shape `(n_pix, K_max)`:

For each `k` in the requested truncation list:
- `proj = ref_flat @ Z[:, :k]`            # `(N, k)` KL coefficients
- `ihat = proj @ Z[:, :k].T`              # `(N, n_pix)` PSF reconstruction
- `residual_k = (ref_flat - ihat).reshape(N, H, W)`

Stack over `k` to a residual cube of shape `(n_K, N, H, W)`. This equals `A_k = T_k - sum_{j<=k} <T_k, Z_j> Z_j`, the PSF-subtracted frame series. After subtraction, restore the NaN mask: set residual pixels back to `NaN` wherever the original masked frame was NaN (broadcast the `(N,H,W)` NaN mask across the `n_K` axis).

If only a single `K` is needed for the output map, choose `K=10` (robust) or `K=20`; either is standard.
