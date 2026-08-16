---
name: oracle-eht-black-hole-tomography
description: "Reconstruct the static 3D volumetric emission field of a black hole accretion flow from sparse time-series interferometric images by fitting a coordinate-based neural radiance field (BH-NeRF) that is warped by Keplerian orbital dynamics and rendered through pre-computed gravitationally lensed ray paths. The single required output is a 64x64x64 emission cube written to outputs/{case_id}.npz under key emission_3d."
---

# oracle-eht-black-hole-tomography Oracle Skill

This skill encodes a faithful BH-NeRF tomography pipeline. Each operation is an independently removable unit covering one convention or step: the public output contract, run-local case loading, the geometry/units conventions, positional encoding, the MLP emission network with its sigmoid offset, the Keplerian inverse-rotation velocity warp, the radial/z support mask, the radiative-transfer volume render, the L2 image loss, the Adam training loop with a polynomial learning-rate schedule, the final regular-grid emission sampling, output serialization, and a finiteness/shape validator. A solver enabling all operations can reproduce the standard reconstruction; ablating individual operations yields meaningful behavior variants.

## Keplerian inverse-rotation velocity warp with pre-injection masking

## Keplerian velocity warp (temporal dynamics)

The static field is shared across time by inverse-rotating query coordinates so that orbital shearing is undone before the NeRF lookup.

### Time in M units
`t_M = (t_frame - t_start_obs) / GM_c3 + t_geo - t_injection`, with `GM_c3 = 1.0`. `t_geo` is the per-sample slow-light coordinate time.

### Rotation angle and masking
- `theta_rot = t_M * Omega` per ray sample (radius-dependent because `Omega = sqrt(M)/r^{3/2}`).
- Where `t_M < 0` (before hotspot injection), set `theta_rot = NaN`; these samples are later masked to zero emission.

### Rotation axis (critical)
- Use a FIXED rotation axis `[0, 0, 1]` (z-axis). The axis is NOT optimized in the standard solver, despite multi-init hints in the background text. Keeping it fixed at z is what reproduces the reference behavior.

### Applying the rotation
- Build the inverse rotation `R(axis, -theta_rot)` via the Euler-Rodrigues formula (per-sample, batched over `theta_rot`).
- Apply `warped = einsum('ij...,j...->...i', R, coords)` where `coords` is `(3, *spatial)`; result is `(*spatial, 3)`.

At the final grid-sampling step, `Omega` and `t_geo` are set to zero and `t_frame = t_start_obs = t_injection = 0`, so `t_M = 0`, `theta_rot = 0`, and the warp is identity (no masking). See resource for the rotation-matrix formula.
