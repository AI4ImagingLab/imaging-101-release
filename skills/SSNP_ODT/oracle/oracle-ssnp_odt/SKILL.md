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

<!-- ORACLE_OP_START op_020_load_measurements -->
## Load intensity measurements and convert to amplitude

## op_020_load_measurements

Load the measured intensities and convert them to amplitudes for the data-fidelity term.

- Read `measurements = np.load(raw_path)['measurements']`. Shape is `(1, 8, 256, 256)` float64.
- Drop the leading batch axis: `meas_np = measurements[0].astype(np.float64)` -> shape `(n_angles=8, Ny=256, Nx=256)`.
- Move to a torch tensor `float64` on the compute device.
- The forward model predicts **intensity**; the loss compares **amplitudes**. Precompute measured amplitude as `meas_amp = torch.sqrt(measurements)` (elementwise sqrt of intensity). Keep this `(8, 256, 256)` amplitude tensor for the solver.

These are intensity images (non-negative); no background subtraction or normalization is applied. Do not reorder the angle axis — angle index `m` must match the illumination geometry in op_050.
<!-- ORACLE_OP_END op_020_load_measurements -->

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

<!-- ORACLE_OP_START op_050_incident_field -->
## Tilted plane-wave incident field per illumination angle

## op_050_incident_field

For illumination index `m = 0..n_angles-1`, build the incident state vector `(u, ud)` = (field, z-derivative), each `(Ny, Nx)` complex128.

```
na = NA / n0
theta = 2*pi*m / n_angles
c_a = na*cos(theta)     # x direction cosine
c_b = na*sin(theta)     # y direction cosine
# Truncate to nearest integer Fourier mode for periodicity (matches trunc=True):
norm_x = Nx*res[0];  norm_y = Ny*res[1]
c_a = int(c_a*norm_x)/norm_x
c_b = int(c_b*norm_y)/norm_y
x = arange(Nx, float64);  y = arange(Ny, float64)
phase_x = 2j*pi*c_a*res[0]*x
phase_y = 2j*pi*c_b*res[1]*y
tilt = exp(phase_y)[:,None] * exp(phase_x)[None,:]    # (Ny, Nx) complex128
c_gamma_in = sqrt(max(1 - c_a**2 - c_b**2, 1e-8))
kz_in = c_gamma_in * (2*pi*res[2])
u  = tilt.clone()
ud = tilt * (1j*kz_in)
```

The 8 LEDs sit on a ring at NA=0.65: `kx_in = NA*cos(2*pi*m/8)`, `ky_in = NA*sin(2*pi*m/8)`. The integer-frequency truncation is essential so the tilted wave is exactly periodic on the grid (otherwise FFT wrap-around artifacts appear). The derivative `ud = i*kz_in*u` encodes the forward-going axial slope.
<!-- ORACLE_OP_END op_050_incident_field -->

<!-- ORACLE_OP_START op_060_p_operator -->
## P operator: free-space propagation (Fourier-space 2x2 rotation)

## op_060_p_operator

The diffraction/propagation operator advances `(u, ud)` by `dz` slices in Fourier space using a 2x2 rotation matrix, with evanescent damping folded into the cos/sin factors.

```
cos_kz = torch.cos(kz*dz) * eva_mask
sin_kz = torch.sin(kz*dz) * eva_mask
a   = fft2(u)
a_d = fft2(ud)
a_new   =  cos_kz * a + (sin_kz / kz) * a_d
a_d_new = (-kz * sin_kz) * a + cos_kz * a_d
u_new  = ifft2(a_new)
ud_new = ifft2(a_d_new)
```

This is the matrix `[[cos(kz dz), sin(kz dz)/kz], [-kz sin(kz dz), cos(kz dz)]]` applied per Fourier mode. Use unshifted `fft2`/`ifft2`. `dz=1.0` for the per-slice sample propagation; `dz` can be negative for back-propagation (op_090). All FFTs must stay inside the autograd graph (use torch.fft).
<!-- ORACLE_OP_END op_060_p_operator -->

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

<!-- ORACLE_OP_START op_080_forward_sweep -->
## Per-angle forward sweep: alternate P then Q over all slices

## op_080_forward_sweep

Simulate the intensity image for a single illumination angle `m`:

```
u, ud = make_incident_field(m)                 # op_050
for iz in range(Nz):                           # Nz = 308
    u, ud = apply_propagation(u, ud, dz=1.0)   # P (op_060)
    u, ud = apply_scattering(u, ud, dn_volume[iz], dz=1.0)   # Q (op_070)
phi_out = extract_forward_component(u, ud)     # op_090
intensity = torch.abs(phi_out)**2              # (Ny, Nx) real
```

Ordering is **P then Q within each slice**, iterating `iz` from 0 to Nz-1 along the first (z) axis of `dn_volume` shape `(Nz, Ny, Nx)`. The full multi-angle forward returns `torch.stack([forward_single(dn, m) for m in range(n_angles)], dim=0)` -> `(n_angles, Ny, Nx)`. Keep the whole sweep differentiable (the RI volume is the autograd leaf).
<!-- ORACLE_OP_END op_080_forward_sweep -->

<!-- ORACLE_OP_START op_090_extract_forward -->
## Camera-plane field: back-propagate, split forward, apply pupil

## op_090_extract_forward

After the sample sweep, map the exit-plane state `(u, ud)` to the forward field at the camera/focal plane:

```
# 1. Back-propagate from exit plane to focal plane (center of volume), no scattering:
u_bp, ud_bp = apply_propagation(u, ud, dz=-Nz/2.0)     # negative dz, op_060
# 2. Split into forward/backward in Fourier space, keep forward:
a   = fft2(u_bp);  a_d = fft2(ud_bp)
a_f = (a - 1j * a_d / kz) * 0.5
uf  = ifft2(a_f)
# 3. Apply pupil in Fourier space:
phi_out = ifft2( fft2(uf) * pupil )                    # op_048
```

The focal plane is at the volume center, hence back-propagation by `dz = -Nz/2`. The forward-component formula `a_f = (a - i a_d / kz)/2` isolates the +z traveling wave from the state vector; the backward part is discarded. Final image intensity is `|phi_out|^2` (op_080).
<!-- ORACLE_OP_END op_090_extract_forward -->

<!-- ORACLE_OP_START op_100_amplitude_loss -->
## Amplitude-domain MSE data-fidelity loss

## op_100_amplitude_loss

Compare predicted and measured **amplitudes** (square-roots of intensity), per angle, normalized by pixel count:

```
n_pixels = Ny * Nx
pred_intensity = forward_single(dn, m)            # op_080
pred_amp = torch.sqrt(pred_intensity + 1e-12)     # +1e-12 for sqrt stability
loss_m = torch.sum((pred_amp - meas_amp[m])**2) / n_pixels
```

Total data loss = sum over all `n_angles` of `loss_m`. The amplitude domain (sqrt of intensity) is more robust to signal-dependent noise than the intensity domain. The `+1e-12` inside the sqrt avoids NaN gradients at zero. Accumulate gradients angle-by-angle via `loss_m.backward()` (see op_120) rather than building one giant graph, to bound memory.
<!-- ORACLE_OP_END op_100_amplitude_loss -->

<!-- ORACLE_OP_START op_110_tv_reg -->
## Optional 3D total-variation regularization (Huber-smoothed)

## op_110_tv_reg

3D isotropic Huber-smoothed total variation, available as a piecewise-smoothness regularizer:

```
dz = vol[1:,:,:] - vol[:-1,:,:]
dy = vol[:,1:,:] - vol[:,:-1,:]
dx = vol[:,:,1:] - vol[:,:,:-1]
dz = F.pad(dz, (0,0,0,0,0,1));  dy = F.pad(dy, (0,0,0,1,0,0));  dx = F.pad(dx, (0,1,0,0,0,0))
TV = torch.sum(torch.sqrt(dz**2 + dy**2 + dx**2 + 1e-8))
```

Added to the objective as `tv_weight * TV(dn) / (Nz*Ny*Nx)` and backpropagated. **In the standard configuration `tv_weight = 0.0`, so TV is effectively disabled** — the data term alone drives the reconstruction. Keep the helper available for ablation/tuning but default the weight to 0.
<!-- ORACLE_OP_END op_110_tv_reg -->

<!-- ORACLE_OP_START op_120_gradient_descent -->
## Gradient-descent reconstruction with non-negativity projection

## op_120_gradient_descent

Reconstruct the RI contrast volume by autograd gradient descent.

**Hyperparameters (standard):** `n_iter=100`, `lr=50.0`, `tv_weight=0.0`, `positivity=True`, `float64`.

```
dn = torch.zeros(Nz, Ny, Nx, dtype=float64, device=device, requires_grad=True)   # init = background
for step in range(n_iter):
    total = 0.0
    for m in range(n_angles):
        loss_m = amplitude_loss(dn, m)     # op_100
        loss_m.backward()                  # accumulate grad across angles
        total += loss_m.item()
    if tv_weight > 0:
        (tv_weight * tv_3d(dn) / (Nz*Ny*Nx)).backward()   # op_110
    with torch.no_grad():
        dn -= lr * dn.grad                 # fixed-step gradient descent
        if positivity:
            dn.clamp_(min=0.0)             # project Dn >= 0 after each step
        dn.grad.zero_()
dn_recon = dn.detach().cpu().numpy()       # (Nz, Ny, Nx)
```

Key conventions: initialize `dn = 0` (uniform background), accumulate per-angle gradients then take **one** step per iteration, use a large fixed learning rate (50.0), and enforce non-negativity (`Dn >= 0`) by clamping after every step. No momentum/Adam — plain gradient descent. Prefer GPU (`cuda`) when available for the 308-slice sweep.
<!-- ORACLE_OP_END op_120_gradient_descent -->

<!-- ORACLE_OP_START op_130_write_output -->
## Assemble and write delta_n output npz

## op_130_write_output

Write the reconstruction to the required schema.

- Reconstruction `dn_recon` has shape `(Nz, Ny, Nx) = (308, 256, 256)`.
- Add the leading batch axis: `delta_n = dn_recon[None, ...]` -> `(1, 308, 256, 256)`.
- Cast to `float32` or `float64`; ensure all values are finite (no NaN/Inf).
- Save `np.savez(f'outputs/{case_id}.npz', delta_n=delta_n)`.
- Create `outputs/` if missing. One file per case id from `public/cases.json`.

Use `scripts/op_130_write_output.py` as a validator/writer helper that enforces shape, dtype, finiteness, and the `delta_n` key before saving.
<!-- ORACLE_OP_END op_130_write_output -->

<!-- ORACLE_OP_START op_140_smoke_probe -->
## Cheap forward-model smoke test on a reduced volume

## op_140_smoke_probe

Before the full 100-iteration / 308-slice run, validate the forward model and one gradient step on a tiny synthetic problem to catch shape/dtype/FFT-convention bugs cheaply.

Checks to run (see `scripts/op_140_smoke_probe.py`):
- Build the frequency grids, incident field, P and Q operators on a small grid (e.g. `Nz=8, Ny=32, Nx=32`, `n_angles=2`).
- Confirm `forward_single` returns a real non-negative `(Ny, Nx)` intensity with finite values.
- Confirm one autograd step reduces the amplitude loss on a planted phantom (gradient flows to `dn`).
- Confirm output assembly yields shape `(1, Nz, Ny, Nx)`.

This probe is diagnostic scaffolding; it must not alter the final reconstruction.
<!-- ORACLE_OP_END op_140_smoke_probe -->
