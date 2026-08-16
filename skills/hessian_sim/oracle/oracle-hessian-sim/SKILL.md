---
name: oracle-hessian-sim
description: "Reconstruct a 2x super-resolved, denoised fluorescence time series from 2-beam structured-illumination-microscopy (SIM) raw frames. Covers the full sequential pipeline: pattern-frequency calibration, modulation/phase estimation, generalized-Wiener band-unmixing reconstruction, split-Bregman Hessian denoising, and edge-preserving temporal averaging, ending in a strict npz output contract."
---

# oracle-hessian-sim Oracle Skill

This skill reconstructs each SIM case from a raw stack of 12 time points x 6 frames (2 angles x 3 phases) at 256x256 into a 12x512x512 super-resolved denoised stack. It bundles the physical OTF model, centred-FFT conventions, band separation, sub-pixel wavevector search, modulation-depth/phase estimation, Wiener recombination, a Hessian split-Bregman denoiser, temporal running average, and output validation. Each operation is independently removable so its contribution can be ablated. Enable operations in dependency order: contract and data loading first, then physics/FFT conventions, then calibration, then reconstruction and denoising, then postprocess and validation. All work uses only run-local public inputs, workspace, and outputs directories, and the run-local python runtime.

<!-- ORACLE_OP_START op_010_public_contract -->
## Task input/output contract and submission protocol

Establish the fixed public contract before writing any code.

Inputs per case (run-local, under `public/`):
- `raw_data.npz` with array key `raw_frames`, shape `(1, 72, 256, 256)`, dtype float32. The leading axis is a batch of size 1. The 72 axis is 12 time points x 6 frames per time point, ordered per time point as `[a0p0, a0p1, a0p2, a1p0, a1p1, a1p2]` (2 illumination angles x 3 phases). Values are detector counts.
- `params_data.json` with fields: `nangles`=2, `nphases`=3, `frames_per_timepoint`=6, `wavelength_nm`=488, `numerical_aperture`=0.9, `pixel_size_nm`=65, `spjg`=[4,4,3] (integer phase-step ratios), `synthetic_otf`=true, plus `n_timepoints`=12 and `image_size_px`=[256,256] when present.

Case discovery: read `cases.json` for the list of case ids and their `input_dir`/`params` paths. There is one public case `case_000` but the solver must loop over all listed cases.

Output per case: write `outputs/{case_id}.npz` (relative to the run root) with a single required array key `data`, shape exactly `(12, 512, 512)`, dtype float32 or float64. Validation requires `finite_only` (no NaN/Inf) and `allow_cast=false` (produce the exact declared dtype/shape; do not rely on the harness to cast).

Hard invariants: output first two spatial dims are 2x the raw spatial dims (256 -> 512); time dimension equals number of raw time points (12). The submitted stack is the temporally-averaged Hessian-denoised reconstruction (see the temporal-averaging and output-writing operations). Do not open task directories, parent directories, or any private/reference files; use only `public/`, `workspace/`, `outputs/`, `logs/agent/`.
<!-- ORACLE_OP_END op_010_public_contract -->

<!-- ORACLE_OP_START op_020_case_loading -->
## Load raw stack, parameters, and derive working grid constants

Load one case and set up the shared constants used by every downstream stage.

Steps:
1. `raw_batch = np.load(raw_data_path)['raw_frames']` then `raw = raw_batch[0].astype(np.float64)`. Result shape `(72, 256, 256)`. Convert to float64 up front; the whole pipeline runs in float64/complex128 and only casts to float32 at output.
2. Read params: `nangles`, `nphases`, `wavelength=wavelength_nm`, `na=numerical_aperture`, `pixel_size=pixel_size_nm`, `spjg`. Do NOT expect solver hyperparameters in params; they are fixed constants (see below).
3. Grid size: `sy, sx = raw.shape[1], raw.shape[2]` (256, 256); `n = max(sy, sx, 512)` = 512. This `n` is the padded single-band working grid; the reconstruction uses a `2*n` extended grid.
4. Frame bookkeeping: `nframes = nangles*nphases` = 6; `n_timepoints = 72 // nframes` = 12.
5. Fixed hyperparameters (hard-coded, not from JSON):
   - `regul = 2*np.pi` (total phase range across a full pattern period).
   - `beishu_an = 1`, `beishu_re = 1` (frame-averaging multipliers for calibration and reconstruction).
   - Parameter-search: `pg_base = 512 * 2 * na * pixel_size / wavelength`; `pg = int(np.ceil(pg_base * (n/512)))` (expected pattern radius in 2n-grid px, ~123 here); `fanwei = int(np.ceil(50 * (n/512)))` (radial half-width of the search annulus, ~50).
   - Wiener: `weilac = 2.0`.
   - Hessian: `mu = 150.0`, `sigma_z = 1.0`, `n_iter = 100`.
6. The calibration stages use only the first `nframes*beishu_an` = 6 frames (`est_frames = raw[:6]`); the reconstruction consumes all 72.

See the resource for the exact constant table and the frame-order layout.
<!-- ORACLE_OP_END op_020_case_loading -->

<!-- ORACLE_OP_START op_030_otf_model -->
## Synthetic cone optical transfer function (OTF)

Regenerate the OTF synthetically from NA, wavelength, and pixel size on an `n x n` grid (`n=512`). No measured OTF is available; `synthetic_otf=true`.

Formula (ideal linearly-tapered cone, max-normalised to 1):
```
cutoff_freq    = 2*na / (wavelength * 1e-3)        # 1/um
freq_per_pixel = 1.0 / (n * pixel_size * 1e-3)     # 1/um
fc_pixels      = cutoff_freq / freq_per_pixel      # cutoff radius in px
ky, kx = np.mgrid[-n//2:n//2, -n//2:n//2]
kr = np.sqrt(kx**2 + ky**2)
otf = np.clip(1.0 - kr/fc_pixels, 0, 1)
otf = otf / otf.max()                              # float64, max=1
```
The OTF is centred (zero frequency at index `n//2`), consistent with the centred-FFT convention used everywhere else. Downstream code builds hard support masks from thresholded, radius-limited copies of this OTF (e.g. `H[kr>fc]=0`, then `H1 = (H!=0)`), so the taper shape and normalisation matter. Keep it float64.
<!-- ORACLE_OP_END op_030_otf_model -->

<!-- ORACLE_OP_START op_040_fourier_conventions -->
## Centred-FFT conventions: padding, convolution, Fourier shift theorem

All Fourier operations use a single centred convention. Getting this wrong mislocates every reconstructed band. Use exactly these helpers.

Spectrum of a centred image: `F = fftshift(fft2(ifftshift(img)))`. Inverse: `img = fftshift(ifft2(ifftshift(F)))`. Zero frequency sits at the array centre.

1. Centre zero-pad (used to go n -> 2n): place the source array centred in a zeros array of `target_shape`, offsets `py=(ty-sy)//2`, `px=(tx-sx)//2`.
2. Linear FFT convolution `dft_conv(h, g)`: pad each to `N = shape(h)+shape(g)` via centred spectra, multiply spectra, inverse-transform, then centre-crop to `shape(h)+shape(g)-1`.
3. Fourier shift theorem `shift_otf(H_2n, kx, ky, n)`: move a spectrum by `(kx, ky)` (radians/px) by multiplying its real-space version by a linear-phase ramp:
```
xx, yy = np.meshgrid(np.arange(2*n), np.arange(2*n))
Ir = np.exp(1j*(kx*xx + ky*yy))
h_real = fftshift(ifft2(ifftshift(H_2n)))
shifted = fftshift(fft2(ifftshift(h_real*Ir)))
```
Wavevector convention: a pattern coordinate stored as `(zx, zy)` in 2n-grid pixel units maps to angular frequencies `ky = 2*pi*(zx - n)/(2*n)` and `kx = 2*pi*(zy - n)/(2*n)`. Note the swap: the first coordinate array feeds `ky`, the second feeds `kx`. Reuse this mapping identically in calibration and reconstruction. The bundled script provides drop-in implementations.
<!-- ORACLE_OP_END op_040_fourier_conventions -->

<!-- ORACLE_OP_START op_050_band_separation -->
## Phase matrix and band separation of raw frames

Separate each angle's 3 phase-shifted raw frames into the three frequency-band components (0th order and +/-1 sidebands) using a known phase matrix. This is the front end for both calibration and reconstruction.

Phase matrix (per angle, size `nphases x nphases`), built from `spjg` and `regul`:
```
total_spjg = sum(spjg)                 # spjg=[4,4,3] -> 11
pm[j,0] = 1
cum_j   = sum(spjg[:j]) / total_spjg   # 0, 4/11, 8/11 for j=0,1,2
pm[j,1] = exp(+1j*regul*cum_j)
pm[j,2] = exp(-1j*regul*cum_j)
inv_pm  = np.linalg.inv(pm)
```
Build frame averages over the `beishu_an` (calibration) or `beishu_re` (reconstruction) repeats, per phase slot `i`: `fd[:,:,i] = mean over indices range(i, nframes*beishu, nframes)`.

Centre-pad each averaged frame from `256x256` into the `n x n` grid, take its centred spectrum `DIbar[:,:,i]`, apply the OTF support mask, then unmix per angle:
```
for itheta in range(nangles):
  for j in range(nphases):
    for k in range(nphases):
      idx_in  = itheta*nphases + k
      idx_out = itheta*nphases + j
      sp[:,:,idx_out] += inv_pm[j,k] * DIbar[:,:,idx_in]
```
For pattern-frequency estimation only, normalise amplitude so only phase survives: `sp = sp / (np.abs(sp) + 1e-12)`. (The reconstruction stage does NOT phase-normalise; it uses a phase matrix rebuilt from estimated phases - see the Wiener operation.) The output `sp[:,:,idx_out]` for `idx_out = itheta*3` is the 0th-order band, `+1` and `+2` are the two sidebands.
<!-- ORACLE_OP_END op_050_band_separation -->

<!-- ORACLE_OP_START op_060_freq_estimation -->
## Pattern wavevector estimation with sub-pixel merit search

Locate each illumination wavevector by cross-correlating the phase-normalised 0th-order band with a sideband, then refine sub-pixel by a greedy coordinate-descent merit search. Returns `zuobiaox, zuobiaoy`, each shape `(nframes,1)` in 2n-grid pixel units; defaults are `(n, n)`.

Support masks: copy the OTF, zero it beyond `fc = ceil(220*n/512)`, take magnitude `H`, binary `H1=(H!=0)`, and `H2=H1`. Pad `H`, `H1` to `2n x 2n` as `H_2n`, `H1_2n`.

Coarse peak (per angle pair, loop `spi` over `range(0, nangles*(nphases-1), 2)`, `spi_m=spi+1`):
- pick center band `idx_center` and sideband `idx_shifted` (see resource for exact index formulas),
- autocorrelate support: `cishu = dft_conv(H2, H2[::-1,::-1])`; binarise `ci` where `|cishu|>=0.9`,
- cross-correlate bands: `jieguo = dft_conv(sp_center, conj(sp_shifted)[::-1,::-1]) * ci`; `lihe = |jieguo/(cishu+1e-12)|`,
- restrict to the annulus `pg-fanwei <= kr <= pg+fanwei` (radial grid centred on the `dft_conv` output size), zero elsewhere,
- peak = argmax -> `(maxx, maxy)`.

Sub-pixel refinement: convert `(maxx,maxy)` to `(kx0,ky0)` via the 2n mapping, evaluate the merit `_compute_merit` (cross-correlation overlap of shifted center band vs shifted sideband, normalised by support overlap). Do greedy coordinate descent on `maxx` then `maxy`: probe merit gradient with `eps=1e-5`, take steps of size 1.0 in the improving direction, halving the step on failure, until both steps drop below `1e-4` (accept a trial only if merit strictly increases).

Store the refined coordinate at `idx1 = spi_m + 1 + floor((spi_m-1)/2) - 1` and its conjugate partner `idx2=idx1+1` as `(2n-maxx, 2n-maxy)`. Keep the exact merit formula from the resource; it is the correctness-critical part.
<!-- ORACLE_OP_END op_060_freq_estimation -->

<!-- ORACLE_OP_START op_070_modulation_phase -->
## Modulation depth and phase-offset estimation via EMD-smoothed histograms

For each sideband order, estimate the modulation depth `c6` and phase offset `angle6` (each length `nangles*(nphases-1)` = 4). Build two OTF overlap regions: a wide one (`fc_ang = ceil(120*n/512)`) for phase and a narrow one (`fc_con = ceil(105*n/512)`, or `ceil(80*n/512)` if `wavelength==647`) for depth.

Rescale coordinates to the working grid: `scale = n / floor((zuobiaox[0]+zuobiaoy[0])/2)`; `zx512 = zuobiaox*scale`, `zy512 = zuobiaoy*scale`. Band-separate the calibration frames as in the band-separation operation but WITHOUT amplitude normalisation, and without the support pre-mask (raw separated `sp`).

Per order `spi` (0..3, `spi_m=spi+1`): pick `idx_center`, `idx_shifted`, and coordinate index `coord_idx = spi_m + 1 + floor((spi_m-1)/2) - 1`; form `kxtest, kytest` from the 2n mapping using `zx512/zy512`. Build shifted binary masks `replcHtest_ang/con` (threshold `>0.9`), shifted magnitude OTFs `replch_ang/con`, shift the sideband via `Ir`, and form the complex modulation maps:
```
cm_ang = (replctest * replcHtest_ang * H_ang_2n) / (spzhongxin*replch_ang + 1e-12)
cm_con = (replctest * replcHtest_con * H_con_2n) / (spzhongxin*replch_con + 1e-12)
angcm = angle(cm_ang);  abscm = |cm_con|
```
Collect `angcm` where the wide overlap `replch_ang*H_ang_2n != 0` (phase samples) and `abscm` where the narrow overlap != 0 (depth samples). Histogram phase over bins `arange(-pi, pi+0.02, 0.02)` and depth over `arange(0, 0.6+0.02, 0.02)`. Smooth each histogram with EMD (empirical mode decomposition, PyEMD) and pick the peak:
- phase: sum the higher-order IMFs (rows 4: if >=5 IMFs and max<50, else rows 3:), `angle6[spi] = -pi + (argmax+1)*0.02`.
- depth: sum all IMFs, `c6[spi] = 0.02 * mean(argmax_indices + 1)`.
If EMD fails or histograms are too short (<=10 bins), fall back to the raw histogram. See resource for exact index/bin formulas.
<!-- ORACLE_OP_END op_070_modulation_phase -->

<!-- ORACLE_OP_START op_080_wiener_reconstruct -->
## Generalized-Wiener band-unmixing super-resolution reconstruction

Produce the super-resolved stack `(n_timepoints, 2*sy, 2*sx)` = `(12, 512, 512)`. Loop over time points, each consuming `nframes*beishu_re`=6 raw frames.

Working grid: `otfx=512` for `image_size<=512`; `n_work=otfx`; rescale coords `zuobiaox_r = zuobiaox*(otfx/n)`, same for y. Resample OTF to `n_work` (scipy.ndimage.zoom order=1) if needed; apply Wiener cutoff `fc=ceil(200*n_work/512)` via `jiequ` mask; `H = otf_r*jiequ` normalised to max=1; `H1=(H!=0)`. Pad both to `2*n_work`.

Per-frame geometry (precompute once): for each of 6 frames, `kytest=2*pi*(zx_r-n_work)/(2n_work)`, `kxtest=2*pi*(zy_r-n_work)/(2n_work)`, phase ramp `Irtest`, binary shifted support `replcHtest = |shift_otf(H1_2n,...)|` binarised at `>0.9`, and shifted OTF `replch = shift_otf(Hk,...)*replcHtest`. Build `re = replch/|replch|` (phase-only, with zeros mapped to 1e19 then normalised), `reh=|replch|`, `hs = sum(reh**2, axis=2)`.

Modulation weights `xishu` (length 6): 0th-order slots = 1; each sideband slot = `mean(1/(c6_pair+1e-12))` for that angle. Per-angle phase offset `deph[itheta] = sign(a6_pair[0])*mean(|a6_pair|)`.

Rebuild the reconstruction phase matrix per angle from `deph` (NOT phase-normalised): `phi_tmp = linspace(deph, regul+deph, total_spjg+1)`, take `phi=[phi_tmp[0], phi_tmp[spjg0], phi_tmp[spjg0+spjg1]]`, form `pm[j]=[1, exp(1j*phi[j]), exp(-1j*phi[j])]`, invert per angle.

Apodization: `plong = floor(sum of sideband radii /(2*nangles))`, `k_max = plong+fc`, `bhs = cos(pi*kr_2n/(2*k_max))` zeroed beyond `k_max` (triangle-like cosine window over the extended grid).

Edge-taper mask (per time point, applied in real space before FFT): a separable sigmoid window `mask` raised to the 3rd power, tiled over frames (`sig=0.25`). Subtract background (`bg=0` here).

Per time point: average repeats into `fd[:,:,i]`, apply `mask9`, centre-pad to `n_work`, FFT -> `DIbar`, unmix with the rebuilt `inv_phase_matrices` per angle, apply `jiequ` mask, pad each band to `2*n_work`, shift by `Irtest`, apply `replcHtest`, phase-correct by `/re`, then combine with the generalized Wiener filter:
```
tmprc[:,:,t] = xishu[t]*retirff[:,:,t]*conj(reh[:,:,t]) / (hs + 0.005*nangles*weilac**2)
dr = sum(tmprc, axis=2); drr = dr*bhs
fimage = fftshift(ifft2(ifftshift(drr)))
result = real(fimage[n_work-sy:n_work+sy, n_work-sx:n_work+sx])  # centred 512x512
result[result<0] = 0
```
Stack results as float32 -> `(12,512,512)`. The full excerpt is in the resource; reproduce it faithfully (weight application, phase correction `/re`, and the `0.005*nangles*weilac**2` denominator floor are all correctness-critical).
<!-- ORACLE_OP_END op_080_wiener_reconstruct -->

<!-- ORACLE_OP_START op_090_hessian_denoise -->
## Split-Bregman second-order (Hessian) denoising

Denoise the Wiener stack jointly in x, y, t with an L1 penalty on the six second-order (Hessian) derivatives, solved by split-Bregman. Input/output shape identical: `(12, 512, 512)`. Parameters: `mu=150.0` (data-fidelity, siranu), `sigma_z=1.0` (z weight, zbei), `n_iter=100`, splitting `lamda = mu*20`.

Objective:
```
min_x (mu/2)||x-y||_2^2 + ||Dxx x||_1 + ||Dyy x||_1 + sigma^2||Dzz x||_1
      + 2||Dxy x||_1 + 2*sigma||Dxz x||_1 + 2*sigma||Dyz x||_1
```
Axis convention: axis 0 = z/time, axis 1 = y, axis 2 = x. Finite differences use periodic BC: `fdiff(x,dim)=roll(x,-1,dim)-x`, `bdiff(x,dim)=x-roll(x,1,dim)`.

Setup: if `nz<3`, set `sigma_z=0`, pad by repeating the last frame to `nz=3` (not needed here, nz=12). Normalise `y = y/ymax` (float32), remember `ymax`. Precompute the denominator spectrum `divide = real((mu/lamda) + Frefft)` where `Frefft` sums `fftn(kernel)*conj(fftn(kernel))` for the six kernels with weights `{1,1,sigma^2,2,2*sigma,2*sigma}` (kernels in resource).

CRITICAL iteration structure - `frac` must carry across iterations:
```
b = [zeros]*6; x = zeros; frac = (mu/lamda)*y
for it in range(n_iter):
    fF = fftn(frac)
    x = real(ifftn(fF/divide)) if it>0 else real(ifftn(fF/(mu/lamda)))
    frac = (mu/lamda)*y                 # reset to data term
    for each component i with adjoint weight alpha_i and operator D_i:
        u = D_i(x); s = max(|u+b_i|-1/lamda, 0); d = s*sign(u+b_i)
        b_i += u - d
        frac += alpha_i * D_i^adjoint(d - b_i)   # accumulate for NEXT iter's x-update
```
The six operator/adjoint pairs and their alpha weights are: Dxx `bdiff(fdiff(x,2),2)` w=1; Dyy `bdiff(fdiff(x,1),1)` w=1; Dzz `bdiff(fdiff(x,0),0)` adjoint scaled by `sigma^2`; Dxy `fdiff(fdiff(x,2),1)` -> adjoint `2*bdiff(bdiff(.,1),2)`; Dxz `fdiff(fdiff(x,2),0)` -> `2*sigma*bdiff(bdiff(.,0),2)`; Dyz `fdiff(fdiff(x,1),0)` -> `2*sigma*bdiff(bdiff(.,0),1)`. After the loop: clip negatives to 0, slice back to original nz, multiply by `ymax`. Common bug: resetting `frac` at the top of the loop discards the regularization adjoints and yields x~=y. Reproduce exactly from the resource excerpt.
<!-- ORACLE_OP_END op_090_hessian_denoise -->

<!-- ORACLE_OP_START op_100_running_average -->
## Edge-preserving three-frame temporal running average

Smooth residual temporal flicker with a centred 3-frame boxcar over the Hessian-denoised stack, preserving the first and last frames unchanged.

Apply to the Hessian output cast to float32 (`window=3`):
```
half_before = (window-1)//2   # 1
half_after  = window//2       # 1
result = stack.copy()
for i in range(half_before, nz - half_after):        # i = 1 .. nz-2
    result[i] = mean(stack[i-half_before : i+half_after+1], axis=0)
# frames 0 and nz-1 remain the original Hessian-denoised frames
```
If `nz < window`, return the stack unchanged. This averaged Hessian stack is the final submission array.
<!-- ORACLE_OP_END op_100_running_average -->

<!-- ORACLE_OP_START op_110_output_writing -->
## Assemble and write the submission npz

Write the final per-case result to `outputs/{case_id}.npz` under the run root, array key `data`.

Requirements:
- Shape exactly `(12, 512, 512)`; dtype float32 (float64 also accepted). Cast explicitly (`arr.astype(np.float32)`); do not depend on harness casting.
- The array is the edge-preserving running-average of the Hessian-denoised stack from the temporal-averaging operation (the full super-resolved denoised time series, NOT a single frame and NOT the batch-expanded 4D form).
- Ensure all values are finite (no NaN/Inf) and non-negative (negatives already clipped upstream; clip again defensively).
- Create the `outputs/` directory if missing. Save with `np.savez(path, data=arr)`.
- Repeat for every case listed in the case index.
<!-- ORACLE_OP_END op_110_output_writing -->

<!-- ORACLE_OP_START op_120_output_validation -->
## Validate submission shape, dtype, and finiteness

After writing each `outputs/{case_id}.npz`, verify it against the contract before considering the case done. Load the file, confirm key `data` exists, shape is exactly `(12, 512, 512)`, dtype is float32 or float64, and every value is finite. Fail loudly (nonzero exit) if any check fails so the reconstruction can be re-run. Use the bundled checker script from the run root.
<!-- ORACLE_OP_END op_120_output_validation -->

<!-- ORACLE_OP_START op_130_smoke_probe -->
## Cheap input-shape and parameter smoke probe

Before running the full pipeline, run a fast non-mutating probe that loads a case, prints the raw stack shape, confirms `raw_frames` reshapes to 12 time points x 6 frames, echoes the key params (nangles, nphases, wavelength, NA, pixel_size, spjg), and derives the working grid `n` and constants. This catches path, key-name, and shape mistakes in seconds without doing any reconstruction. Point it at a case's `raw_data.npz` and `params_data.json` under `public/`.
<!-- ORACLE_OP_END op_130_smoke_probe -->
