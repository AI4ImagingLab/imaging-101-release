---
name: oracle-ct-fan-beam
description: "Reconstruct a 2D attenuation image from fan-beam CT sinograms (full-scan and short-scan divergent geometry) and write it as outputs/{case_id}.npz with key phantom of shape (128,128). Covers fan-beam geometry conventions, distance-weighted filtered back-projection with Parker weighting for short-scan, and TV-regularized Chambolle-Pock iterative reconstruction."
---

# oracle-ct-fan-beam Oracle Skill

This skill reconstructs the attenuation map f(x,y) from divergent fan-beam CT sinograms. It provides the geometry conventions (source-to-isocenter D_sd, isocenter-to-detector D_dd, centered detector positions, centered pixel grid, and the rotated coordinates t and s), the fan-beam forward and adjoint operators with magnification mapping and 1/U^2 distance weighting, and two reconstruction routes: analytical filtered back-projection (pre-weighting, ramp filtering, distance-weighted back-projection, plus Parker weighting for short-scan) and TV-regularized Chambolle-Pock (PDHG) iterative reconstruction seeded by FBP. It also covers final packaging to the required NPZ schema, submission validation, and a geometry smoke probe. Each capability is an independently removable section so variants can be tested. Use the run-local python interpreter and only run-local public/, workspace/, and outputs/ paths.

## Output contract and submission schema

## Output contract and submission schema

Reconstruct one 2D attenuation image per case and write it to a run-local NPZ file.

Hard invariants (must all hold or the submission is rejected):
- Write to `outputs/{case_id}.npz` relative to the run root. For the single public case the id is `case_000`, so the file is `outputs/case_000.npz`.
- The NPZ must contain array key `phantom` with shape exactly `(128, 128)`.
- dtype must be `float32` or `float64`. No implicit casting is allowed, so create the array in one of those dtypes explicitly.
- All values must be finite (no NaN/Inf). Apply a non-negativity clip and replace any non-finite values with 0 before writing.
- The image is the reconstructed attenuation map `f(x, y)` (the phantom), not a sinogram.

Recommended save call:
```python
import numpy as np, os
os.makedirs('outputs', exist_ok=True)
np.savez('outputs/case_000.npz', phantom=recon.astype(np.float32))
```

There is exactly one visible case; iterate over the case list if more are present, always keying the output filename by the case id. Do not add extra required arrays; only `phantom` is scored.

## Fan-beam back-projection (adjoint) with 1/U^2 weighting

## Fan-beam back-projection (adjoint) with 1/U^2 weighting

Maps a sinogram `(n_angles, n_det)` to an image `(N,N)`. Used both as the FBP back-projection stage and as the adjoint operator `A^T` in the iterative solver.

Per angle `theta` over the centered pixel grid `(xx, yy)`:
1. `t = xx*cos_t + yy*sin_t`, `s = -xx*sin_t + yy*cos_t`.
2. `U = D_sd - s` (source-to-pixel axial distance).
3. Magnified detector coordinate: `det_coord = t * (D_sd + D_dd) / U`.
4. Distance weight: `weight = D_sd**2 / U**2` (normalized by `D_sd^2`).
5. Interpolate the sinogram row onto `det_coord` with linear interpolation, out-of-range filled with 0: `scipy.interpolate.interp1d(det_pos, sino[i], kind='linear', bounds_error=False, fill_value=0.0)`.
6. Accumulate: `image += weight * f(det_coord) * d_angle`, where `d_angle = angles[1]-angles[0]` (1.0 if a single angle).

The `1/U^2` distance weighting and the same magnification mapping as the forward operator are essential — they encode the divergent fan geometry.
