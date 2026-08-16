---
name: oracle-ssnp_odt
description: "Reconstruct a 3D refractive-index-contrast volume from intensity-only multi-angle LED measurements using a Split-Step Non-Paraxial (SSNP) multiple-scattering forward model and autograd gradient-descent inversion with amplitude-domain MSE loss and non-negativity projection."
---

# oracle-ssnp_odt Oracle Skill

This skill reconstructs the 3D refractive-index contrast (delta_n) of a sample from 8 intensity images captured under annular LED illumination (NA=0.65), following the SSNP intensity diffraction tomography pipeline.

Data flow: load `measurements` (1,8,256,256) and `params_data.json`; build normalized resolution `res = res_um * n0 / wavelength`; precompute Fourier grids (direction cosine c_gamma, axial frequency kz), evanescent damping mask, and NA pupil; construct per-angle tilted plane-wave incident states (field u and z-derivative ud) with integer-frequency truncation; sweep through all Nz=308 slices alternating the P propagation operator (Fourier-space 2x2 rotation with evanescent damping) and the Q scattering operator (real-space ud update by the local RI contrast); back-propagate to the focal plane, split off the forward-traveling component, apply the pupil, and take intensity. Invert by torch autograd gradient descent on the amplitude-domain MSE between sqrt(predicted) and sqrt(measured) intensities, initialized at delta_n=0, with a large fixed learning rate, per-iteration non-negativity clamping, and optional (off-by-default) 3D TV regularization. Output delta_n as (1,308,256,256) to outputs/{case_id}.npz.

The operations are ablatable: critical physics conventions (normalized resolution, frequency grids, P/Q operators, incident field, forward-component extraction, amplitude loss, gradient descent, output schema) are low deletion priority; scaffolding (TV regularizer which is disabled by default, evanescent mask, smoke probe) is higher priority.

<!-- ORACLE_OP_START op_010_public_contract -->
## Public task contract and output schema

## op_010_public_contract

Reconstruct a 3D refractive-index-contrast volume from intensity-only multi-angle LED measurements using a Split-Step Non-Paraxial (SSNP) multiple-scattering forward model and gradient-descent inversion.

**Run-local inputs (per case `case_id`, e.g. `case_000`):**
- `public/cases/{case_id}/input_data/raw_data.npz` — key `measurements`, shape `(1, 8, 256, 256)`, float64. Layout = `batch x n_angles x Ny x Nx`.
- `public/cases/{case_id}/params_data.json` — imaging parameters (see op_030).
- The case list is `public/cases.json`; iterate every entry's `id`.

**Required output (per case):**
- Write `outputs/{case_id}.npz` with array key `delta_n`, shape `(1, 308, 256, 256)` = `batch x Nz x Ny x Nx`, dtype float32 or float64.
- Values must be finite. Do not rely on automatic casting; produce the exact shape.

**Pipeline overview:** load measurements -> build SSNP forward model from params -> initialize `delta_n = 0` -> gradient descent on amplitude-domain MSE (autograd through forward model) with non-negativity projection -> save `delta_n`.

Work only under run-local `public/`, `workspace/`, `outputs/`, and `logs/agent/`. Use `${CLAUDE_SKILL_DIR}` for bundled assets.
<!-- ORACLE_OP_END op_010_public_contract -->

<!-- ORACLE_OP_START op_030_params_config -->
## Imaging parameters and normalized resolution convention

## op_030_params_config

Load `params_data.json` and build the physical configuration. Keys and case values:

- `volume_shape = [Nz, Ny, Nx] = [308, 256, 256]`
- `res_um = [dz, dy, dx] = [0.1, 0.1, 0.1]` (voxel size in micrometers; in this case isotropic)
- `wavelength_um = 0.5`
- `n0 = 1.0` (background refractive index)
- `NA = 0.65`
- `n_angles = 8`
- `ri_contrast_scale = 0.01` (phantom contrast scale; informational for reconstruction)

**Critical normalized-resolution convention.** The SSNP propagator uses a dimensionless sampling, NOT micrometers:

```
res = tuple(size * n0 / wavelength_um for size in res_um)   # (res_x, res_y, res_z)
```

Index the normalized `res` as `res[0]=x`, `res[1]=y`, `res[2]=z` throughout the model. For this case all `res` entries equal `0.1 * 1.0 / 0.5 = 0.2`. Every frequency grid, phase ramp, `kz`, and scattering phase factor below uses this normalized `res`, never raw micrometers.
<!-- ORACLE_OP_END op_030_params_config -->

<!-- ORACLE_OP_START op_040_frequency_grids -->
## Frequency grid: direction cosine c_gamma and axial frequency kz

## op_040_frequency_grids

Precompute the 2D `(Ny, Nx)` Fourier-space grids used by every propagation step. Use `float64`/`complex128` throughout.

```
fx = torch.fft.fftfreq(Nx, dtype=float64) / res[0]
fy = torch.fft.fftfreq(Ny, dtype=float64) / res[1]
FY, FX = torch.meshgrid(fy, fx, indexing='ij')   # shape (Ny, Nx)
eps = 1e-8
c_gamma = torch.sqrt(torch.clamp(1.0 - FX**2 - FY**2, min=eps))   # cos(gamma), direction cosine
kz = c_gamma * (2.0 * pi * res[2])               # normalized axial spatial frequency
```

`c_gamma` is the cosine of the propagation angle on the frequency grid; `kz` is the axial frequency entering the P-operator rotation. Note `fftfreq` ordering (not fftshifted) — keep all FFTs unshifted and consistent. Cache `c_gamma` and `kz` on the device; they are reused by op_050, op_060, op_070, op_090.
<!-- ORACLE_OP_END op_040_frequency_grids -->

<!-- ORACLE_OP_START op_045_evanescent_mask -->
## Evanescent-wave damping mask

## op_045_evanescent_mask

Damp evanescent / high-angle frequency components inside the P operator to keep propagation stable.

```
eva_mask = torch.exp(torch.clamp((c_gamma - 0.2) * 5.0, max=0.0))   # (Ny, Nx) float64
```

For `c_gamma >= 0.2` the exponent clamps to 0 so `eva_mask = 1` (pass band). For `c_gamma < 0.2` (near-evanescent, high transverse frequency) it decays exponentially, attenuating those components. This mask multiplies the cosine/sine propagation factors in op_060. Removing it lets evanescent components grow and corrupt the field.
<!-- ORACLE_OP_END op_045_evanescent_mask -->

<!-- ORACLE_OP_START op_048_pupil -->
## Microscope pupil function with smooth NA cutoff

## op_048_pupil

Build the objective pupil (aperture) applied at the camera plane, a smooth sigmoid cutoff at the NA limit expressed in `c_gamma`:

```
na_norm = NA / n0
cutoff  = sqrt(1.0 - na_norm**2)            # c_gamma value at the NA edge
mask = torch.exp(torch.clamp(c_gamma - cutoff, max=0.01) * 10000.0)
pupil = mask / (1.0 + mask)                 # smooth 0/1 transition, cast to complex128
```

Frequencies with `c_gamma >= cutoff` (within NA, i.e. small transverse frequency) pass; those below are suppressed. The `clamp(..., max=0.01)*10000` makes a very sharp but differentiable logistic edge. Store as `complex128` `(Ny, Nx)`; it multiplies the forward field in Fourier space in op_090.
<!-- ORACLE_OP_END op_048_pupil -->

<!-- ORACLE_OP_START op_070_q_operator -->
## Q operator: real-space scattering by one RI slice

## op_070_q_operator

The scattering operator updates only the z-derivative `ud` from the local refractive-index contrast slice `dn_slice` `(Ny, Nx)` real, leaving `u` unchanged:

```
phase_factor = (2*pi*res[2] / n0)**2 * dz        # normalized physical thickness, dz=1.0
scatter = phase_factor * dn_slice * (2.0*n0 + dn_slice)   # = phase_factor*(2 n0 dn + dn^2)
ud_new = ud - scatter * u
```

This implements `ud -= k0^2 (n0^2 - n^2) dz * u` with `n = n0 + dn`, expanded as `n0^2 - n^2 = -(2 n0 dn + dn^2)`. The `dn^2` term is retained (non-linear, full scattering). Apply this once per axial slice in the forward sweep (op_080). The factor `(2*pi*res[2]/n0)**2` is the exact constant used — do not drop the `1/n0^2` or the `(2*pi*res_z)^2`.
<!-- ORACLE_OP_END op_070_q_operator -->

