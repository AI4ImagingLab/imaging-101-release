---
name: oracle-fpm-inr-reconstruction
description: "Reconstruct a high-resolution 3D complex field (amplitude/phase) from 68 low-resolution intensity-only Fourier ptychographic microscopy measurements using a factored implicit neural representation trained against a differentiable FPM forward model, and emit the required amplitude z-stack."
---

# oracle-fpm-inr-reconstruction Oracle Skill

This skill captures the concrete conventions, optical formulas, network design, training discipline, and output packaging needed to solve a single-case Fourier Ptychographic Microscopy (FPM) reconstruction with an implicit neural representation (INR). The unknown 3D complex field O(x,y,z)=A·exp(iφ) is not stored as an explicit voxel grid; instead amplitude and phase are each parameterized by a factored representation (a learnable 2D feature grid times a small learnable 1D depth-feature array, decoded by a tiny MLP). The representation is optimized by matching predicted low-resolution amplitudes to measured amplitudes through an FPM forward operator (sub-spectrum cropping, pupil mask, and angular-spectrum defocus propagation). Each operation below is independently removable so its contribution can be ablated. Use the run-local task runtime interpreter at `public/envs/runtime/.venv-posix/bin/python` for all execution; read inputs only from run-local `public/`, write intermediates under `workspace/`/`logs/agent/`, and write the final artifact under `outputs/`. Bundled assets live under `${CLAUDE_SKILL_DIR}`. Preserve every numeric constant, axis order, FFT convention, and reshape rule exactly as stated in each section — they are load-bearing.

## Public task contract, case discovery, and submission schema

## Public task contract

Single-case FPM reconstruction. Enumerate cases from `public/cases.json` (one case, id `case_000`). Each case provides:
- `public/cases/<case_id>/input_data/raw_data.npz` — raw measurements + calibration.
- `public/cases/<case_id>/params_data.json` — optical/system parameters.

### Optical/system parameters (params_data.json)
`wavelength_um=0.5126`, `NA=0.256`, `magnification=10`, `pixel_size_um=3.45`, `D_led_um=4000`, `num_leds=68`, `raw_size=1024`, `num_modes=512`, `upsample_ratio=2`, `z_min_um=-20.0`, `z_max_um=20.0`, `z_stack_slices=161`.

### Required submission
Write exactly one file per case to run-root-relative `outputs/<case_id>.npz` (for the public case: `outputs/case_000.npz`). It must contain array key `amplitude_stack`:
- shape **(1024, 1024, 161)** = (H, W, Z), axis order H, W, then z-slice index;
- dtype `float32` or `float64`;
- all values finite (no NaN/Inf); casting is not allowed, so create the array in an accepted dtype directly.

The 161 z-slices correspond to depths linearly spaced from `z_min_um` to `z_max_um` inclusive (`numpy.linspace(-20, 20, 161)`). Only `amplitude_stack` is graded; any other artifacts (figures, metrics) are optional and never required for submission.

### Hard invariants
- Do not transpose or flip the stack away from (H, W, Z).
- Downsample the model's native high-resolution output to H=W=`raw_size`=1024 before saving (see the output-writing section).
- Use the run-local interpreter `public/envs/runtime/.venv-posix/bin/python`.

## Load raw measurements, squeeze batch dim, crop ROI

## Raw data loading and ROI crop

Load `raw_data.npz` with numpy. Keys, shapes, dtypes:
- `I_low`  : (1, 1024, 1024, 68) float32 — axes (batch, H, W, LED). Units: arbitrary intensity counts.
- `na_calib`: (1, 68, 2) float32 — axes (batch, LED, [NAx, NAy]), dimensionless.
- `mag`    : scalar float32 — magnification (10).
- `dpix_c` : scalar float32 — camera pixel pitch in µm (3.45).
- `na_cal` : scalar float32 — objective NA (0.256).

Processing:
1. Squeeze the leading batch dim: `I_low = raw['I_low'][0].astype('float32')` → (1024, 1024, 68); `na_calib = raw['na_calib'][0].astype('float32')` → (68, 2). Cast the three scalars with `.astype('float32')`.
2. Read `num_modes` and `upsample_ratio` and `wavelength_um` from `params_data.json`.
3. Crop the ROI to the first `2*num_modes` rows and columns: `I = I_low[0:2*num_modes, 0:2*num_modes, :]`. With `num_modes=512` this keeps the full 1024×1024 image, so `M=N=1024`, `ID_len=68`.

Keep both the cropped intensity `I` (H, W, LED) and `na_calib` for the optical-parameter step. Do not normalize here (normalization happens with LED reordering downstream).

## Optical constants, frequency grid, LED frequency positions, measurement normalization

## Derived optical parameters and LED geometry

All arrays are float64 numpy unless noted. Let `M=N=1024`, `MAGimg=upsample_ratio=2`, `MM=NN=M*MAGimg=2048`.

### Scalars
- `k0 = 2*pi / wavelength_um`  (≈ 12.2575 with λ=0.5126).
- `D_pixel = dpix_c / mag`  (image-plane pixel size, ≈ 0.345 µm).
- `kmax = na_cal * k0`  (≈ 3.1379).

### Spatial-frequency coordinate axes (upsampled grid)
```
Fxx1, Fyy1 = np.meshgrid(np.arange(-NN/2, NN/2), np.arange(-MM/2, MM/2))
Fxx1 = Fxx1[0, :] / (N * D_pixel) * (2*np.pi)   # length NN
Fyy1 = Fyy1[:, 0] / (M * D_pixel) * (2*np.pi)   # length MM
```

### LED illumination NA and ordering
```
NAx = na_calib[:, 0]; NAy = na_calib[:, 1]
u = -NAx; v = -NAy                # note the sign flip
NAillu = np.sqrt(u**2 + v**2)
order = np.argsort(NAillu)        # sort LEDs by distance from optical axis
u = u[order]; v = v[order]
```

### LED positions in frequency space (integer pixel indices)
For each LED `idx` in `range(ID_len)` find the nearest grid index:
```
ledpos_true = np.zeros((ID_len, 2), dtype=int)
ledpos_true[idx, 0] = np.argmin(np.abs(Fxx1 - k0 * u[idx]))
ledpos_true[idx, 1] = np.argmin(np.abs(Fyy1 - k0 * v[idx]))
```

### Measurement normalization + reorder
`Isum = I[:, :, order] / np.max(I)` → (M, N, ID_len) float32, normalized by the GLOBAL maximum over all LEDs and reordered to match the sorted LED order. This `Isum` and `ledpos_true` feed the forward model; the LED sort order must be identical everywhere it is used.

Expose `M, N, MM, NN, MAGimg, k0, kmax, D_pixel, ID_len, Fxx1, Fyy1, ledpos_true, Isum, NA=na_cal, wavelength`.

## Binary pupil support and angular-spectrum propagation kernel

## Pupil mask and angular-spectrum kernel (kzz)

Built at raw resolution (M×N = 1024×1024) from the optical parameters.

### Angular-spectrum z-kernel (kzz), complex (M, N)
```
kxx, kyy = np.meshgrid(Fxx1[:M], Fxx1[:N])   # reuse the frequency axis, first M/N samples
kxx, kyy = kxx - np.mean(kxx), kyy - np.mean(kyy)
krr = np.sqrt(kxx**2 + kyy**2)
mask_k = (k0**2 - krr**2) > 0
kzz_ampli = mask_k * np.abs(np.sqrt(k0**2 - krr.astype('complex64')**2))
kzz_phase = np.angle(np.sqrt(k0**2 - krr.astype('complex64')**2))
kzz = kzz_ampli * np.exp(1j * kzz_phase)
```
This is the real, evanescent-masked k_z = sqrt(k0^2 - kx^2 - ky^2). The defocus phase for depth z is `exp(1j * kzz * z)` (applied later in the forward model).

### Binary pupil support (M, N)
```
Fx1, Fy1 = np.meshgrid(np.arange(-N/2, N/2), np.arange(-M/2, M/2))
Fx2 = (Fx1 / (N * D_pixel) * (2*np.pi))**2
Fy2 = (Fy1 / (M * D_pixel) * (2*np.pi))**2
Pupil0 = np.zeros((M, N)); Pupil0[Fx2 + Fy2 <= kmax**2] = 1.0
```
Pupil0 = 1 inside the NA disk (radius kmax = NA·k0), else 0.

### Tensor layout for the forward model
Convert to torch on the compute device:
- `Pupil0` reshaped to (1, 1, M, N).
- `kzz` as (1, M, N) complex (`unsqueeze(0)`).
- `Isum` as a torch tensor (M, N, ID_len) on device.

## Depth-of-field-based training z-plane count

## Training z-sampling parameters

Derive the number of z-planes used *during training* (distinct from the 161 evaluation slices):
```
DOF     = 0.5 / NA**2        # depth of field, ≈ 7.629 µm with NA=0.256
delta_z = 0.8 * DOF          # ≈ 6.104 µm
num_z   = int(np.ceil((z_max - z_min) / delta_z))   # ≈ 7 with z in [-20, 20]
```
Use `z_min=z_min_um=-20.0`, `z_max=z_max_um=20.0`. `num_z` (≈7) sets how many training depth planes are sampled per epoch; `num_z//2` (≈3) is the count used on the random-sampling epochs. These values are consumed by the training loop and the alternating z-sampling strategy.

## FPM forward operator: sub-spectrum crop, pupil-defocus mask, iFFT amplitude

## Differentiable FPM forward model

Maps a predicted complex field at depths `dz` to predicted low-resolution amplitudes for a set of LEDs, and returns the matching measured amplitudes. All FFTs are torch centered FFTs. See `${CLAUDE_SKILL_DIR}/resources/resource_001.md` for the full reference excerpt.

Inputs held by the model: `Pupil0` (1,1,M,N), `kzz` (1,M,N complex), `ledpos_true` (ID_len,2 int), `M`, `N`, `MAGimg`.

### 1. Pupil × defocus spectrum mask — for depths `dz` (n_z,) and LED indices `led_num`
```
dfmask = torch.exp(1j * kzz.repeat(n_z,1,1) * dz[:,None,None].repeat(1,M,N))
dfmask = dfmask.unsqueeze(1).repeat(1, len(led_num), 1, 1)          # (n_z, n_led, M, N)
mask_a = Pupil0.repeat(n_z, len(led_num),1,1) * torch.abs(dfmask)
mask_p = Pupil0.repeat(n_z, len(led_num),1,1) * (torch.angle(dfmask) + 0)
spectrum_mask = mask_a * torch.exp(1j * mask_p)                      # complex (n_z,n_led,M,N)
```

### 2. Sub-spectrum extraction coordinates (per LED)
```
uo = ledpos_true[led_num,0]; vo = ledpos_true[led_num,1]
x_0 = vo - M//2; x_1 = vo + M//2      # rows use vo (second calibration index)
y_0 = uo - N//2; y_1 = uo + N//2      # cols use uo (first calibration index)
```
Note the deliberate cross-assignment: x (rows) from `vo`, y (cols) from `uo`.

### 3. Predicted amplitudes from complex field `img_complex` (n_z, MM, MM)
```
O = torch.fft.fftshift(torch.fft.fft2(img_complex))               # centered spectrum
pad_x = (M*MAGimg - O.shape[-2])//2; pad_y = (N*MAGimg - O.shape[-1])//2
O = F.pad(O, (pad_x,pad_x, pad_y,pad_y, 0,0), 'constant', 0)       # zero pad (0 here since MM=M*MAGimg)
O_sub = torch.stack([O[:, x_0[i]:x_1[i], y_0[i]:y_1[i]] for i in range(len(led_num))], dim=1)
O_sub = O_sub * spectrum_mask
o_sub = torch.fft.ifft2(torch.fft.ifftshift(O_sub))
oI_sub = torch.abs(o_sub)                                          # (n_z, n_led, M, N) real
```

### 4. Measured amplitudes for the same LEDs
```
oI_cap = torch.sqrt(Isum[:, :, led_num])          # amplitude = sqrt(intensity)
oI_cap = oI_cap.permute(2,0,1).unsqueeze(0).repeat(n_z,1,1,1)   # (n_z, n_led, M, N)
```
The training loss compares `oI_cap` (measured amplitude) against `oI_sub` (predicted amplitude). Comparison is on amplitudes (sqrt of intensity), not intensities.

## Factored INR: 2D feature grid x 1D depth features, dual amplitude/phase, upsample

## Implicit neural representation architecture

The 3D field is O = A·exp(iφ) with A and φ each an independent factored representation. Full module code in `${CLAUDE_SKILL_DIR}/resources/resource_002.md`.

### Per-component representation (`G_Tensor3D`)
- Spatial feature grid: `data` shape `(x_mode, y_mode, num_feats)` = (512, 512, 32), init `2e-4 * randn`.
- Depth features: `z_data` shape `(z_dim, num_feats)` = (5, 32), init `randn`.
- Sampling coordinates for an output of size `(x_dim, y_dim)` = (MM, MM) = (2048, 2048) are precomputed in `create_coords`: pixel centers `linspace(0.5/dim, 1-0.5/dim, dim)`, scaled by `x_mode/y_mode`, split into integer base indices `x0,y0` (clamped) plus `x1=x0+1`, `y1=y0+1`, and bilinear `lerp_weights`.
- **Bilinear** sample of the 2D grid at those coords → `xy_feat` (num_pixels, num_feats).
- Depth `z` normalized to `[0, z_dim-1]` via `normalize_z(z) = (z_dim-1)*(z - z_min)/(z_max - z_min)`; **linear** interpolation of `z_data` between `z0=floor` and `z1=z0+1` → `z_feat`.
- **Fusion**: element-wise product `feat = xy_feat[None].repeat(n_z,1,1) * z_feat` (broadcast depth over spatial).
- **Renderer MLP** (`G_Renderer`): Linear(32→32) → ReLU → Linear(32→32) → ReLU → Linear(32→1); `use_layernorm=False`. Output reshaped to `(n_z, 1, x_dim, y_dim)`.

### FullModel
- Holds two `G_Tensor3D`: `img_real` (→ amplitude) and `img_imag` (→ phase). Each is built with `x_mode=y_mode=num_modes=512`, `z_dim=5`, `num_feats=32`, `z_min=-20`, `z_max=20`.
- `w=h=MM=2048`, `ds_factor=1`; `create_coords` uses `x_dim=w//ds_factor`, `y_dim=h//ds_factor`.
- `us_module = nn.Upsample(scale_factor=ds_factor, mode='bilinear')` (identity at ds_factor=1).
- `forward(dz)` returns `(img_real, img_imag)`, each `(n_z, MM, MM)` after upsample+`squeeze(1)`.
- The solver forms the complex field as `img_complex = img_ampli * torch.exp(1j * img_phase)` where `img_ampli=img_real`, `img_phase=img_imag`.

## Optimization: Adam+StepLR, smooth-L1 amplitude loss, bfloat16, per-LED steps

## Training loop and objective

### Reproducibility / config
- Set `torch.manual_seed(0)`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`.
- Hyperparameters: `num_epochs=15`, `lr=1e-3`, `lr_decay_step=6`, `lr_decay_gamma=0.1`, `num_feats=32`, `z_dim=5`, `led_batch_size=1`, `use_amp=True` (bfloat16), `loss=smooth_l1`, `use_layernorm=False`. `torch.compile(model, backend='inductor')` is optional throughput-only (Linux); skip it gracefully if unavailable.
- Device: `cuda:0` if available else `cpu`.

### Optimizer / scheduler
```
optimizer = torch.optim.Adam(lr=1e-3, params=filter(lambda p: p.requires_grad, model.parameters()))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
```

### Loop (per epoch, 15 epochs)
1. `led_indices = list(range(ID_len))` (ID_len=68).
2. Choose training depths `dzs` via the alternating z-sampling strategy (separate section).
3. For each depth `dz` in `dzs`: `dz = dz.unsqueeze(0)` (shape (1,)). For `it in range(ID_len // led_batch_size)`:
   - `model.zero_grad()`; `led_num = led_indices[it*led_batch_size:(it+1)*led_batch_size]` (one LED).
   - `spectrum_mask = forward_model.compute_spectrum_mask(dz, led_num)`.
   - Under `torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16)`:
     - `img_ampli, img_phase = model(dz)`; `img_complex = img_ampli * torch.exp(1j*img_phase)`.
     - `oI_cap = forward_model.get_measured_amplitudes(Isum, led_num, len(dz))`.
     - `oI_sub = forward_model.get_sub_spectrum(img_complex, led_num, spectrum_mask)`.
     - `loss = F.smooth_l1_loss(oI_cap, oI_sub)`; also track `F.mse_loss` for a PSNR readout `10*-log10(mse)`.
   - `loss.backward(); optimizer.step()`.
4. After all depths/LEDs in the epoch: `scheduler.step()`.

Note there is no explicit AMP GradScaler (bfloat16 autocast only). One LED per gradient step is intentional. Training touches all 68 LEDs at every sampled depth each epoch.

## Alternating uniform/random z-plane sampling per epoch

## Alternating z-plane sampling

Convergence strategy that varies which depths are supervised each epoch. Uses `num_z` and `z_min,z_max` from the z-parameter section.

```
# random subset (used on ODD epochs)
dzs = ((torch.randperm(num_z-1)[: num_z//2] + torch.rand(num_z//2))
       * ((z_max - z_min) // (num_z - 1))).to(device) + z_min
# uniform grid overrides on EVEN epochs
if epoch % 2 == 0:
    dzs = torch.linspace(z_min, z_max, num_z).to(device)
```

- Even epochs (0,2,4,...): uniform grid of `num_z` (~7) planes across the full depth range — aligned to the depth-feature discretization.
- Odd epochs (1,3,...): `num_z//2` (~3) randomly jittered planes.

Ablation note: if removed, fall back to always using the uniform grid `torch.linspace(z_min, z_max, num_z)` every epoch. The alternation is a regularizer, not a correctness requirement.

## Chunked inference to produce the high-resolution amplitude z-stack

## Inference over evaluation z-positions

After training, evaluate the model at the target depths to obtain the amplitude stack.

1. Build evaluation depths: `z_eval = np.linspace(z_min_um, z_max_um, z_stack_slices)` = `linspace(-20, 20, 161)`.
2. `model.eval()`; convert `z_eval` to a float tensor on device.
3. Run in chunks under `torch.no_grad()` to bound memory: iterate `torch.chunk(z_positions, max(1, len(z_positions)//chunk_size))` with `chunk_size=8`; for each chunk call `ampli, phase = model(z_chunk)`; move `ampli.float().cpu()` (and phase) to CPU lists.
4. Concatenate along dim 0 → `pred_ampli_full` shape `(161, MM, MM)` = (161, 2048, 2048) float; keep only amplitude (`img_real` branch) for the submission.

The phase output is not needed for the required submission but is produced by the same forward call. Do not normalize here; normalization/downsampling happens in the output-writing step.

## Downsample to raw resolution, transpose to (H,W,Z), write outputs npz

## Assemble and write the submission

The model's native amplitude stack is `(161, MM, MM)` = (161, 2048, 2048), but the required `amplitude_stack` is `(1024, 1024, 161)` = (H, W, Z) at `raw_size`. Steps:

1. **Downsample** each slice from MM=2048 to `raw_size`=1024 with bilinear interpolation:
```
import torch, torch.nn.functional as F
t = torch.from_numpy(pred_ampli_full).unsqueeze(1)          # (161,1,2048,2048)
t = F.interpolate(t, size=(1024,1024), mode='bilinear', align_corners=False)
pred = t.squeeze(1).numpy()                                  # (161,1024,1024) = (Z,H,W)
```
(If the native size already equals `raw_size`, skip interpolation.)
2. **Transpose** to (H, W, Z): `amplitude_stack = np.transpose(pred, (1, 2, 0))` → (1024, 1024, 161).
3. **Cast** to `float32` and confirm all-finite (do not clip; the reconstructed amplitude values are the raw model amplitudes).
4. **Write** to run-root-relative `outputs/<case_id>.npz` (public case: `outputs/case_000.npz`) with `np.savez(path, amplitude_stack=amplitude_stack.astype('float32'))`. Ensure the `outputs/` directory exists first.

Do not normalize the saved stack; the required array is the reconstructed amplitude, and the grader may apply its own normalization.

## Optional Normal-Variance all-in-focus composite (not required for submission)

## Optional all-in-focus composite (Normal Variance)

Not part of the graded submission (which is only `amplitude_stack`). Provided as an optional analysis artifact and to document the intended focus-stacking convention. Helper: `${CLAUDE_SKILL_DIR}/scripts/script_001.py`.

Method (patch_size=64, patch_pace=16 stride):
1. Normalize the (H,W,Z) amplitude stack to [0,1].
2. Estimate background = histogram peak bin edge of `|mean over z|` (`np.histogram(bins='auto')`). If background>0.1, clip each patch's values above background to suppress FPM artifacts.
3. For each overlapping 64×64 patch, compute Normal Variance per z-plane `NV_k = sum((I-mu)^2) / (patch_size^2 * mu)` and pick the plane with max NV (sharpest).
4. Fuse overlapping patches, weighting by an inverse-overlap balance map (`1 / overlap_count`).

Run as a standalone helper against `outputs/<case_id>.npz` if a composite image is desired; it does not modify the submission file.

## Validate output npz shape, dtype, finiteness, and key

## Submission validation gate

Before finishing, verify each `outputs/<case_id>.npz`. Run `${CLAUDE_SKILL_DIR}/scripts/script_002.py outputs/case_000.npz`. Checks:
- File exists and loads with numpy.
- Contains key `amplitude_stack`.
- Shape exactly `(1024, 1024, 161)`.
- Dtype is `float32` or `float64`.
- All values finite (no NaN/Inf).

A nonzero exit indicates the submission would be rejected; fix the output-writing step rather than the validator.

## Cheap pipeline smoke probe (data shapes, FFT roundtrip, forward-op sanity)

## Runtime smoke probe

Optional non-mutating diagnostic to confirm the environment and data conventions before the full ~7-minute training run. Run `${CLAUDE_SKILL_DIR}/scripts/script_003.py` (pass the case input dir, e.g. `public/cases/case_000/input_data`). It:
- imports torch/numpy and reports the compute device;
- loads `raw_data.npz`, checks `I_low` is (1,1024,1024,68) and `na_calib` is (1,68,2);
- verifies a centered FFT/iFFT roundtrip on a small complex tensor;
- confirms `k0=2*pi/wavelength`, `D_pixel=dpix_c/mag`, and `kmax=NA*k0` compute to finite positive values.

Use it to catch shape/env problems early; it does not train or write any submission.
