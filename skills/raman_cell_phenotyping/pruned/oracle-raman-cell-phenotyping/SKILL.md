---
name: oracle-raman-cell-phenotyping
description: "Reconstructs biomolecular abundance maps of THP-1 cells from a volumetric Raman hyperspectral cube by preprocessing spectra, running N-FINDR endmember extraction with FCLS abundance estimation under a linear mixing model, then writing four canonical abundance maps at a fixed z-layer. Each operation is independently removable for ablation."
---

# oracle-raman-cell-phenotyping Oracle Skill

This skill unmixes a (40, 40, 10, B') Raman volume into K=5 endmember spectra and their per-voxel fractional abundances, then selects and names the required output maps. Work proceeds in layers: fix the public output contract and load the case arrays; apply the fixed preprocessing chain (fingerprint crop, cosmic-ray despike, Savitzky-Golay smoothing, asymmetric-least-squares baseline removal, global min-max normalization); extract endmembers by maximizing simplex volume (N-FINDR) and solve constrained per-pixel abundances (FCLS) under a linear mixing forward model; finally reshape abundances to the spatial grid, apply the canonical component-to-key mapping at z-layer 5, and validate the written npz. Determinism of the endmember ordering depends on seeding, called out as its own operation. Constants, axis orders, reshape rules, and the canonical mapping are preserved exactly because the evaluator compares by output key without permuting or relabeling.

## Load raw Raman volume and parameters

Load the case with run-local `python`.

From `public/visible_data/cases/case_000/input_data/raw_data.npz` read two keys:
- `spectral_volume`: shape (1, 40, 40, 10, 1600), float32, raw Raman counts. Drop the leading batch dimension immediately: `volume = raw['spectral_volume'][0]` giving (40, 40, 10, 1600).
- `spectral_axis`: shape (1600,), float64, wavenumbers in cm^-1 spanning roughly -136 to 3671 cm^-1.

From `public/visible_data/cases/case_000/params_data.json` read: `spatial_shape=[40,40,10]`, `n_spectral_points=1600`, `fingerprint_region_cm1=[700,1800]`, `n_endmembers=5`, and the `output_convention` block (z_layer_index and canonical_component_to_output_key). Use `n_endmembers` and `fingerprint_region_cm1` from params rather than hardcoding when present.

The last axis of `spectral_volume` is the spectral/band axis for every downstream step; the first three axes are the (X, Y, Z) voxel grid.

## Crop to the fingerprint region

First preprocessing step. Crop the spectral (last) axis to the fingerprint region [700, 1800] cm^-1.

Build a boolean mask `(spectral_axis >= lo) & (spectral_axis <= hi)` with lo=700, hi=1800 (swap if lo>hi). Apply as `data = intensity[..., mask]` and `axis = spectral_axis[mask]`. This keeps ~426 channels (B') out of 1600, retaining the diagnostically informative bands and discarding noisy edges and the silent region.

All later preprocessing and unmixing operate on the cropped B' channels. Endmember spectra therefore have length B', not 1600.

## Whitaker-Hayes cosmic-ray despiking

Second preprocessing step: remove cosmic-ray spikes per spectrum along the spectral axis using the Whitaker-Hayes modified z-score method. Parameters: kernel_size=3, threshold=8.

Procedure per 1-D spectrum: compute first differences `diff = np.diff(spectrum)`; modified z-score of the diffs is `0.6745*(d - median(d))/MAD` where `MAD = median(|d - median(d)|)` (return zeros if MAD==0). Flag channels where `|z| > threshold` as spikes. Iteratively replace each flagged channel with the mean of non-spike neighbours within radius kernel_size (indices `[i-kernel_size, i+kernel_size]`), marking it clean once replaced; repeat until no further changes. Apply along the last axis of the volume.

See the resource for exact pseudocode of the neighbour selection and iteration guard.

## Asymmetric least-squares baseline correction

Fourth preprocessing step: remove broad fluorescence background per spectrum via Eilers-Boelens asymmetric least squares (ASLS), then subtract the fitted baseline. Parameters: lam=1e6, p=1e-2, max_iter=50, tol=1e-3.

Per 1-D spectrum of length m: build 2nd-order finite-difference sparse matrix D of shape (m-2, m) with diagonals [1, -2, 1] at offsets [0, 1, 2]; precompute `DTD = lam * D.T @ D`. Initialize weights w=ones(m), z=spectrum. Each iteration solves the sparse system `(diag(w) + DTD) z_new = w * spectrum` (scipy.sparse.linalg.spsolve); check convergence `||z_new - z|| / (||z|| + 1e-30) < tol` and break if met; otherwise update asymmetric weights `w = p where spectrum > z else (1-p)`. The baseline is z; corrected spectrum is `spectrum - z`. Apply along the last axis of the volume.

See resource for exact matrix construction and loop.

## Global min-max normalization

Fifth and final preprocessing step: scale the whole volume to [0, 1] using a single GLOBAL minimum and maximum across all voxels and all bands (pixelwise=False).

Compute scalar `mins = data.min()` and `maxs = data.max()` over the entire array; `denom = max(maxs - mins, 1e-30)`; return `(data - mins) / denom`. Do not normalize per pixel or per band — global scaling preserves relative intensity differences between voxels, which N-FINDR relies on to locate pure-pixel extremes in the spectral cloud.

The result of this step is the preprocessed volume fed to unmixing.

## N-FINDR endmember extraction

Extract K=5 endmember spectra with N-FINDR (Winter 1999): find the K data pixels whose spectra span the maximum-volume simplex in spectral space.

Flatten the preprocessed volume to `flat` of shape (N=40*40*10=16000, B') via `volume.reshape(-1, B')`. Steps:
1. Reduce to K-1 dimensions with sklearn PCA: `data_reduced = PCA(n_components=K-1).fit_transform(flat)` -> (N, K-1).
2. Randomly initialize K distinct pixel indices as candidate endmembers (uses the seeded RNG; see the reproducibility operation).
3. Repeatedly (up to max_iter=50 full passes): for every pixel and every endmember slot k, try replacing slot k with that pixel; if the simplex volume increases, accept the replacement and update the current set/volume. Stop when a full pass makes no replacement.
4. Return `flat[endmember_indices]` -> (K, B').

Simplex volume of K reduced points (K, K-1): augment with a leading column of ones to form M (K, K), volume = `abs(det(M))`.

The endmember row order is the discovery order of the indices; the canonical output mapping assumes this order under the fixed seed. See resource for exact pseudocode.

## FCLS constrained abundance estimation

Given endmembers E (K, B'), estimate per-pixel fractional abundances with Fully-Constrained Least Squares (Heinz 2001). For each pixel spectrum y, solve the convex QP:

    min_a 0.5 * ||y - a @ E||^2   subject to  a >= 0  and  sum(a) = 1

Solve each pixel independently with scipy.optimize.minimize, method='SLSQP':
- objective(a) = 0.5 * dot(r, r) where r = y - a @ E
- gradient(a) = -E @ r
- equality constraint fun = sum(a) - 1, jac = ones(K)
- bounds = [(0, None)] * K
- initial guess a0 = ones(K)/K
- options: maxiter=200, ftol=1e-12

Collect result.x into an abundance matrix of shape (N, K). Loop over all N=16000 pixels of the flattened preprocessed volume. See resource for the exact per-pixel setup.

## Deterministic seeding for stable endmember order

N-FINDR starts from a random pixel selection, so the row order of the returned endmembers (and therefore which endmember index means lipids/nucleus/cytoplasm/background) is only reproducible under a fixed seed. Seed once at program start before unmixing:

    import random, numpy as np
    random.seed(12345)
    np.random.seed(12345)

The canonical component-to-output-key mapping assumes endmembers were produced with this seed and the default N-FINDR initialization (numpy global RandomState). Do not reseed between preprocessing and unmixing. If you change the seed or RNG source, the fixed index mapping in the canonical-output operation may no longer align.

## Canonical component selection and output writing

Reshape and select the required maps, then write the submission.

1. Reshape the (N, K) abundance matrix back to the spatial grid: build K maps each of shape (40, 40, 10) via `abundance_matrix[:, k].reshape(40, 40, 10)`. This yields `abundance_maps[k]` indexed by endmember k.
2. Select z-layer index 5 (0-based) from each chosen map, giving 2-D (40, 40) slices: `abundance_maps[k][..., 5]`.
3. Apply the authoritative canonical component -> output key mapping (from params `output_convention`):
   - `abundance_lipids`      = abundance_maps[0][..., 5]
   - `abundance_nucleus`     = abundance_maps[3][..., 5]
   - `abundance_cytoplasm`   = abundance_maps[4][..., 5]
   - `abundance_background`  = abundance_maps[2][..., 5]
   Component index 1 is computed but NOT submitted.
4. Write `outputs/case_000.npz` with exactly those four keys using `np.savez`. Keep dtype float32 or float64 (no cast is applied by the checker). Ensure all values are finite.

This mapping is fixed by the task, not by any peak-based interpretation; the evaluator compares by output key and does not permute endmembers or relabel from Raman peaks.
