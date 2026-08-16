---
name: oracle-hessian-sim
description: "Reconstruct a 2x super-resolved, denoised fluorescence time series from 2-beam structured-illumination-microscopy (SIM) raw frames. Covers the full sequential pipeline: pattern-frequency calibration, modulation/phase estimation, generalized-Wiener band-unmixing reconstruction, split-Bregman Hessian denoising, and edge-preserving temporal averaging, ending in a strict npz output contract."
---

# oracle-hessian-sim Oracle Skill

This skill reconstructs each SIM case from a raw stack of 12 time points x 6 frames (2 angles x 3 phases) at 256x256 into a 12x512x512 super-resolved denoised stack. It bundles the physical OTF model, centred-FFT conventions, band separation, sub-pixel wavevector search, modulation-depth/phase estimation, Wiener recombination, a Hessian split-Bregman denoiser, temporal running average, and output validation. Each operation is independently removable so its contribution can be ablated. Enable operations in dependency order: contract and data loading first, then physics/FFT conventions, then calibration, then reconstruction and denoising, then postprocess and validation. All work uses only run-local public inputs, workspace, and outputs directories, and the run-local python runtime.

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

## Assemble and write the submission npz

Write the final per-case result to `outputs/{case_id}.npz` under the run root, array key `data`.

Requirements:
- Shape exactly `(12, 512, 512)`; dtype float32 (float64 also accepted). Cast explicitly (`arr.astype(np.float32)`); do not depend on harness casting.
- The array is the edge-preserving running-average of the Hessian-denoised stack from the temporal-averaging operation (the full super-resolved denoised time series, NOT a single frame and NOT the batch-expanded 4D form).
- Ensure all values are finite (no NaN/Inf) and non-negative (negatives already clipped upstream; clip again defensively).
- Create the `outputs/` directory if missing. Save with `np.savez(path, data=arr)`.
- Repeat for every case listed in the case index.
