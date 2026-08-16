---
name: oracle-fourier-ptychography
description: "Reconstruct a high-resolution complex object from a stack of low-resolution Fourier-ptychography (FPM) intensity images captured under angularly varying LED illumination. Covers reading the run-local public contract and inputs, the FFT/coordinate conventions, the LED k-space position mapping, pupil and object initialization, the intensity-constrained quasi-Newton (qNewton) spectrum/pupil update, LED ordering, optional constraints, reference tuning defaults, and writing the required output array."
---

# oracle-fourier-ptychography Oracle Skill

This skill guides an iterative phase-retrieval reconstruction for Fourier ptychographic microscopy. Cameras record only intensity, so the complex object is recovered iteratively by exploiting the redundancy of overlapping k-space patches selected by each LED angle under a fixed objective pupil bandpass.

The reconstruction maintains the object as a high-resolution spatial-frequency spectrum on an No x No grid (centered/fftshift convention, DC at center) and a pupil function on an Nd x Nd grid. For each LED, an Nd x Nd sub-patch of the spectrum is windowed by the pupil to form an exit wave, propagated to the detector, corrected to match the measured amplitude, and back-propagated to drive quasi-Newton updates of both the object spectrum patch and the pupil. After iterating over all LEDs for a number of outer sweeps, the real-space complex object is obtained by an inverse centered FFT of the recovered spectrum and written as the required output array.

Read every case-specific value (array shapes, pixel sizes, wavelength, NA, magnification, LED-to-sample distance, No, Nd, LED positions) from the run-local public files rather than assuming any fixed numbers. Operation sections below give the formulas, conventions, initialization, update rules, and validation needed to implement and tune the method for any case in this family.

All solver-facing work stays under run-local `public/`, `workspace/`, `outputs/`, and `logs/agent/`, using run-local `python`; bundled helpers live under `${CLAUDE_SKILL_DIR}`.

## Public task contract and submission schema

## Public contract and submission schema

Produce one complex high-resolution object array per case. Read the contract at runtime; do not hard-code case values.

- Discover cases from the run-local cases listing (`public/.../cases.json` or the provided cases file). Each case has an id and an `input_dir`.
- For each case, write the result to `outputs/{case_id}.npz` relative to the run root (path template comes from the run-local output schema).
- The output archive must contain the array under key `object`.
  - Shape: `(No, No)` where `No` is read from the case params (do not assume a literal size).
  - Dtype: complex (`complex64` or `complex128`).
  - All entries must be finite (no NaN/Inf). Casting is disallowed by the validator, so create the array already complex and finite.
- The object is the high-resolution complex transmission function of the sample in real space (amplitude and phase), on an `No x No` grid.

Read the exact `format`, `path_template`, required keys, shape, allowed dtypes, and validation flags from the run-local `output_schema.json` at solve time; treat those as authoritative if they differ from any value quoted here.

## Load raw images, LED encoder, and optical parameters

## Load images, LED positions, and parameters

For each case, load the inputs from its `input_dir` (run-local `public/` tree):

- `input_data/raw_data.npz` contains:
  - `ptychogram`: float array, shape `(J, Nd, Nd)` — one low-resolution intensity image per LED. `J` is the number of LEDs, `Nd` the low-res image size.
  - `encoder`: float array, shape `(J, 2)` — LED positions in meters `[x, y]` in the array plane at distance `z_led` from the sample.
- `params_data.json` (case params) provides optical constants. Read these keys when present: `wavelength_m`, `NA`, `magnification`, `dxd_m` (camera pixel size), `dxp_m` (object pixel size), `z_led_m`, `led_pitch_m`, `n_leds_side`, `num_leds`, `Nd`, `No`.

Derived quantities:
- Object/spectrum pixel size `dxp = dxd_m / magnification` (equivalently read `dxp_m`).
- `Nd = ptychogram.shape[-1]` (cross-check against params).
- Per-frame energy `energy_at_pos[j] = sum(ptychogram[j])` (used for the error metric and power normalization).
- `max_probe_power = sqrt(max_j energy_at_pos[j])`.

If `No` is missing from params, compute it from LED geometry (see the LED k-space position operation): take the full pixel span of all LED shifts, add `2*Nd`, round up to an even integer, and use `max(Nd, that span)`.

Cast intensities to float64 for the reconstruction math to reduce roundoff.

## Centered unitary FFT and coordinate conventions

## FFT and coordinate conventions (must match everywhere)

All spectra and pupils use the centered (fftshift) convention with DC at the array center, and unitary (energy-preserving) transforms. Define once and reuse:

- Centered forward FFT: `fft2c(x) = fftshift(fft2(ifftshift(x), norm='ortho'))` over the last two axes.
- Centered inverse FFT: `ifft2c(x) = fftshift(ifft2(ifftshift(x), norm='ortho'))` over the last two axes.

Consequences and rules:
- The object spectrum, pupil, and all k-space patches are stored with DC at index `N//2` (fftshift layout). Patch extraction, pupil masks, and position offsets all assume this layout.
- Use the SAME transform pair for the forward step and the back-propagation; mixing `norm` or shift conventions breaks the amplitude constraint and the updates.
- Real-space object O(r) and its spectrum are related by `O_spectrum = fft2c(O)` and `O = ifft2c(O_spectrum)`.
- A circular aperture on a coordinate grid is `circ(x,y,D) = (x^2 + y^2) < (D/2)^2`.

Keep pixel-size bookkeeping consistent: object/spectrum pixel size `dxp = dxd/magnification`; the pupil and each low-res patch live on the `Nd x Nd` grid.

## Map LED encoder positions to k-space patch offsets

## LED positions -> k-space patch offsets

Each LED angle shifts which Nd x Nd region of the No x No object spectrum falls under the pupil. Compute integer upper-left corners of the pupil window for every LED.

Given encoder positions `encoder[j] = (x_j, y_j)` in meters, LED-to-sample distance `z_led`, wavelength `lambda`, object pixel size `dxo = dxp`, and pupil/window size `Np = Nd`:

```
conv   = -(1.0/lambda) * dxo * Np
dist_j = sqrt(x_j^2 + y_j^2 + z_led^2)         # exact (not paraxial)
shift_j = round(conv * encoder[j] / dist_j)     # per-axis pixel shift, shape (2,)
pos_j   = shift_j + No//2 - Np//2               # integer upper-left corner
```

- `pos_j` are integer row/col indices into the No x No spectrum; the LED patch is `spectrum[pos_row:pos_row+Np, pos_col:pos_col+Np]`.
- The `-(1/lambda)*dxo*Np` factor converts sin(theta) ≈ encoder/dist into pixel units on the Np-grid embedded in the No-grid; the sign and the `No//2 - Np//2` centering must be preserved so the DC LED lands at the spectrum center.
- Keep `encoder` axis order consistent with the `(row, col)` indexing you use for slicing; apply the same order in the forward model and updates.
- If `No` must be derived, the full span `max_j pos_j - min_j pos_j` plus a `2*Nd` margin (rounded to even) sizes the spectrum so all patches fit.

A reference helper implementing this mapping is in the operation's resource.

## Pupil bandpass and boundary window

## Pupil (coherent bandpass) and boundary window

The objective transmits only spatial frequencies inside the NA-limited circle. Build the pupil on the `Nd x Nd` grid in the centered convention.

Two equivalent constructions (use one consistently):
- Frequency-grid form: with `fx = fftshift(fftfreq(Nd))` (cycles/pixel) and cutoff `f_cutoff = NA*dxp/lambda`, pupil `= (FX^2 + FY^2) <= f_cutoff^2`.
- Aperture-diameter form on the probe coordinate grid: with `xp = linspace(-Np/2, Np/2, Np)*dxp`, `Xp,Yp = meshgrid(xp,xp)`, and effective diameter `epd = 2*dxp^2*Np*NA/lambda`, pupil `= circ(Xp,Yp,epd)`.

Also define a slightly larger boundary window `probeWindow = circ(Xp,Yp, epd*1.2)` (about 20% larger). This window is used to null the pupil outside support during iterations (see the constraints operation).

The pupil is the coherent transfer support; the maximum recoverable spatial frequency grows with the outermost LED angle: `k_max ∝ (NA + sin(theta_max))/lambda`, which is why dark-field LEDs extend resolution. Keep the pupil in the same fftshift layout as the spectrum patches.

## Initialize object spectrum and pupil estimate

## Initialize the reconstruction state

Create the initial high-resolution object spectrum and pupil.

Object spectrum (No x No, complex, centered):
- Form a real-space low-resolution estimate from the mean measured amplitude: `low_res = ifft2c(sqrt(mean_j ptychogram[j]))` on the Nd grid.
- Zero-pad it symmetrically to No x No: `pad = (No - Np)//2`, constant zeros, then crop/force to exactly `(No, No)`.
- Use this padded field as the initial object. (Because the object is maintained as a spectrum patchwork with DC at center, this padded low-res field seeds the central patch; the outer spectrum fills in during iteration.)

Pupil (Nd x Nd, complex, centered):
- Start from the binary pupil mask and add a tiny multiplicative perturbation to break symmetry: `probe = pupil_mask * (1 + 0.001*rand(Np,Np))`.
- Keep a copy of `probeWindow` for the boundary constraint.

Use a fixed RNG seed for reproducibility of the small perturbation; its magnitude (~1e-3) is deliberately negligible relative to unity. Store `dxo = dxp`, coordinate grids `Xp,Yp`, and `positions` in the state for the update loop.

## Per-LED forward model and exit-wave propagation

## Forward model per LED

The measured image for LED j is the bandpass-limited, angle-shifted spectrum imaged to the detector:

`I_j(r) = | ifft2c{ P(q) * O_spectrum(q - q_j) } |^2`

Discrete per-LED forward step used inside the loop (centered convention):

```
sy, sx = slice(pos_j_row, pos_j_row+Np), slice(pos_j_col, pos_j_col+Np)
patch  = object_spectrum[sy, sx]           # Nd x Nd complex
esw    = patch * pupil                     # windowed exit-surface wave (k-space)
ESW    = fft2c(esw)                         # propagate to detector plane
Iest   = |ESW|^2                            # estimated intensity
```

Rules:
- Patch extraction uses the integer `pos_j` from the LED-position operation, in the same fftshift layout as the pupil.
- FPM propagation from the (windowed) spectrum patch to the detector uses the centered forward FFT `fft2c` (the exit wave here is already in the frequency domain of the object). Back-propagation uses `ifft2c`.
- The full stack simulator (for testing) loops this over all LEDs; a reference implementation is provided in the resource.

## Amplitude (modulus) constraint and residual wave

## Intensity (modulus) projection

Enforce that the estimated detector amplitude matches the measurement while keeping the estimated phase. This is the core phase-retrieval step.

Given `ESW = fft2c(esw)`, `Iest = |ESW|^2`, and measured `Imeas = ptychogram[j]`:

```
frac    = sqrt( Imeas / (Iest + eps) )      # eps ~ 1e-10
ESW_new = ESW * frac                        # replace magnitude, keep phase
esw_upd = ifft2c(ESW_new)                    # back-propagate to k-space patch domain
DELTA   = esw_upd - esw                       # residual exit-wave update
```

`DELTA` drives both the object-spectrum and pupil updates. Keep `eps` small and strictly positive to avoid division blow-up where `Iest` is near zero (dark-field pixels). Use the same centered transforms as the forward step so magnitudes and phases stay consistent.

## Quasi-Newton object-spectrum and pupil updates

## Quasi-Newton (qNewton) updates

For each LED, after computing `DELTA`, update the object-spectrum patch and the pupil with Hessian-scaled (quasi-Newton) PIE-type steps. The `|P|/Pmax` and `|O|/Omax` prefactors are the quasi-Newton scaling that speeds convergence over constant-step PIE without momentum buffers.

Object-spectrum patch update:
```
Pmax   = max(|pupil|)
absP   = |pupil|
frac_O = (absP / (Pmax + eps)) * conj(pupil) / (absP^2 + reg_O)
patch  = patch + beta_O * frac_O * DELTA
object_spectrum[sy, sx] = patch
```

Pupil update (uses the just-updated patch):
```
Omax   = max(|object_spectrum|)
absO   = |patch|
frac_P = (absO / (Omax + eps)) * conj(patch) / (absO^2 + reg_P)
pupil  = pupil + beta_P * frac_P * DELTA
```

Rules and rationale:
- Use PIE-type (not ePIE) updates: the `|P|/Pmax` weighting distributes updates across dark-field regions rather than concentrating only near the pupil maximum, which is what extends resolution.
- `reg_O`, `reg_P` are additive Tikhonov denominators preventing division by small magnitudes; `beta_O`, `beta_P` are step sizes. Their reference values live in the reference-defaults operation.
- `eps` (~1e-20) guards the max normalizers. Update object first, then pupil, using the same `DELTA`.
- Iterate this over all LEDs (in the chosen order) per outer sweep, repeating for the configured number of sweeps.

## NA-based LED update ordering

## LED update ordering (bright-field first)

Within each outer sweep, visit LEDs in order of increasing distance from the k-space center (NA order): central/bright-field LEDs first, outer dark-field LEDs last.

```
rows = positions[:,0] - mean(positions[:,0])
cols = positions[:,1] - mean(positions[:,1])
order = argsort(sqrt(rows^2 + cols^2))
```

Rationale: bright-field images have far higher SNR and quickly establish a reliable central spectrum estimate; processing them first stabilizes convergence before the noisier dark-field measurements refine the high-frequency content. This ordering is a convergence aid, not a correctness requirement — the same LEDs are used either way.

## Optional pupil constraints and adaptive denoising

## Optional constraints and adaptive denoising

These stabilizers improve robustness on noisy data; they are guardrails, not the core solver.

Adaptive denoising (per LED, applied to the measurement before projection):
```
Ameas = sqrt(|Imeas|); Aest = sqrt(|Iest|)
noise = |mean(Ameas - Aest)|
Ameas = max(0, Ameas - noise); Imeas = Ameas^2
```
This clips a spatially-uniform amplitude noise floor, helping dark-field frames.

Per-sweep constraints (applied after each outer iteration):
- Probe boundary: `pupil *= probeWindow` to force zero support outside the (slightly enlarged) NA circle.
- Probe power correction (optional, off by default): renormalize `pupil` to a fixed power `max_probe_power` via `pupil *= max_probe_power / (||pupil||_2 + eps)`.
- Center-of-mass stabilization (optional, off by default): if enabled every k sweeps, compute the pupil intensity centroid `(xc,yc)` on the `Xp,Yp` grid and `roll` the pupil by `(-yc,-xc)` when it drifts more than one pixel from center.

Enable the probe boundary and adaptive denoising by default; leave power correction and CoM stabilization off unless the pupil drifts or blows up. Each is independently removable for testing.

## Reference tuning defaults for the solver

## Reference defaults (tunable)

Starting values that reproduce the reference behavior. All are tunable; adjust from data diagnostics, not from any known answer.

- Outer iterations (full LED sweeps): `200`. Increase if the error metric is still decreasing; decrease for speed. Monitor convergence rather than fixing blindly.
- Object step size `beta_O = 1.0`; pupil step size `beta_P = 1.0`.
- Quasi-Newton regularizers `reg_O = 1.0`, `reg_P = 1.0` (additive denominators). Larger values damp updates and add stability on noisy/low-SNR frames; smaller values speed convergence but risk instability.
- Projection epsilon `~1e-10`; max-normalizer epsilon `~1e-20`.
- Pupil perturbation magnitude `~1e-3`; RNG seed fixed for reproducibility.
- Boundary window scale `epd*1.2` (20% larger than pupil support).
- Default switches: probe boundary ON, adaptive denoising ON, probe power correction OFF, CoM stabilization OFF.

Applicability/failure signatures:
- If the error metric diverges or the pupil amplitude grows without bound, reduce `beta_O`/`beta_P` or raise `reg_*`, and enable the probe boundary.
- If convergence stalls far from a clean reconstruction, increase iterations or verify the k-space positions and FFT conventions first (those are correctness issues, not tuning).
- These constants are dimensionless step/regularization scales, not answer values; recompute nothing from expected outputs.

## Convert spectrum to real-space object and write output

## Produce and write the output object

The reconstruction maintains the object as a high-resolution spectrum `O_spectrum` (No x No, centered). The required output is the real-space complex object:

```
O = ifft2c(O_spectrum)        # No x No complex real-space object
```

Write it per case:
- Save to `outputs/{case_id}.npz` (path template from the run-local output schema), with the array under key `object`.
- Ensure dtype is complex (`complex64` or `complex128`) and every value is finite; replace any non-finite entries before writing (the validator forbids casting and non-finite values).
- Shape must be exactly `(No, No)` from the case params.

Notes on evaluation-consistent form:
- The quality metrics assess the real-space phase up to an additive global constant and an overall scale (mean-removed phase / normalized correlation), so a global phase offset or amplitude scale does not need correction. Do not attempt to match any specific target magnitude.
- Do not apply cropping or resizing; keep the full No x No field.

## Validate output archive and monitor convergence

## Validate outputs and monitor convergence

Before finishing, verify each written archive and optionally track solver progress.

Submission checks (per case):
- File `outputs/{case_id}.npz` exists and contains key `object`.
- `object.shape == (No, No)` with `No` from case params.
- dtype is complex64 or complex128 and `np.isfinite(object).all()` is True.

Convergence monitoring (diagnostic only, not required for output):
- Normalized amplitude error per sweep: `E = sum_j |Imeas_j - Iest_j| / sum_j energy_at_pos[j]` (or the mean-based normalization). A monotonically decreasing `E` that flattens indicates convergence; a rising `E` signals instability (reduce step sizes / raise regularizers).

Use the provided validation script as a run-local check; it reads the run-local output and params only and prints PASS/FAIL. It never reads private or reference data.
