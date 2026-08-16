---
name: oracle-differentiable-deflectometry
description: "Recover a refractive lens's two surface radii of curvature and center thickness from phase-shifted deflectometry fringe images captured by two calibrated cameras, by building a differentiable ray tracer and fitting lens parameters with Levenberg-Marquardt."
---

# oracle-differentiable-deflectometry Oracle Skill

This skill reconstructs the standard solution for a single-case refractive deflectometry inverse problem. The forward model projects camera pixels into 3D, refracts each ray through two aspheric lens surfaces via Snell's law (with implicit-differentiation Newton ray-surface intersection), and intersects the exit ray with a flat display screen. The measured target is produced by four-step phase-shifting fringe analysis, temporal unwrapping, DC removal, pixel-to-mm conversion, and reference-trace subtraction. A Levenberg-Marquardt solver with autograd column-wise Jacobians minimizes the per-pixel screen-intersection residual over 8 unknowns (two curvatures, thickness, 3D origin, two tilt angles). The final scalar outputs are surface_0_roc_mm=1/c0, surface_1_roc_mm=1/c1, and thickness_mm. Each operation below is independently removable so its contribution can be ablated; keep constants, axis orders, and conventions exactly as stated.

<!-- ORACLE_OP_START op_010_contract -->
## Public task contract and submission schema

## Public contract and submission

Recover refractive lens parameters for each case listed in the run-local cases manifest and write one output file per case.

**Cases.** Read the run-local cases manifest (`public/cases.json` or the manifest under `public/`). It lists one case (`case_000`) with an `input_dir` and a `params` file. Each case exposes `params_data.json` and `input_data/raw_data.npz` plus a calibration folder, lens prescription, and display sinusoid images.

**Output.** For each `case_id`, write `outputs/{case_id}.npz` (base = run root). It must contain exactly these three scalar (0-d) arrays, dtype float32 or float64, all finite:
- `surface_0_roc_mm` — radius of curvature of the first lens surface in mm.
- `surface_1_roc_mm` — radius of curvature of the second lens surface in mm.
- `thickness_mm` — lens center thickness in mm.

**Invariants.** `finite_only: true` (no NaN/Inf) and `allow_cast: false`, so store real float scalars, not object arrays or Python floats needing cast. The radius of curvature relates to the fitted surface curvature `c` by `roc = 1.0 / c`; thickness is the fitted axial gap of the second surface. Do not emit tilt/origin in the submission even though they are estimated internally.

Use the run-local runtime interpreter `public/envs/runtime/.venv-posix/bin/python` for all execution; refer to it as `python` on the run PATH. Never read outside `public/`, `workspace/`, `outputs/`, `logs/agent/`, or `${CLAUDE_SKILL_DIR}`.
<!-- ORACLE_OP_END op_010_contract -->

<!-- ORACLE_OP_START op_020_data_loading -->
## Load case inputs: images, calibration, lens file, sinusoids

## Loading raw inputs for a case

Resolve paths relative to the case `input_dir` under `public/` (e.g. `public/cases/case_000/input_data/`). Read imaging config from the case `params_data.json`.

**params_data.json fields (case_000):**
- `imaging.filmsize = [768, 768]`, `full_filmsize = [2048, 2048]`.
- `imaging.sinusoid_periods = [70, 100, 110]` (px), `sinusoid_period_index = 0` (use periods from this index onward), `sinusoid_xs = [0]`.
- `imaging.display_pixel_size_mm = 0.115`, `max_image_val = 65535.0`.
- `calibration.texture_shift = [0.0, 1.1106231]`, `calibration.origin_shift = [0.0, 0.0, 0.0]`.
- `lens.file = 'ThorLabs/LE1234-A.txt'`.

**raw_data.npz** holds `imgs` and `refs`, each shape `(1, 3, 8, 2, 2048, 2048)` float32; axes are `(batch, periods, phase_shifts, cameras, H, W)`. Take batch index 0 and cast to float64. `imgs` = with lens, `refs` = without lens. 3 periods, 8 shifts (first 4 = x-direction, last 4 = y-direction), 2 cameras.

**calibration/** MATLAB `.mat` files: `cams.mat` (camera intrinsics/extrinsics and `display_pixel_size`), `rotation.mat` (mount rotation calibration points `p1`,`p2`), `gammas.mat` (gamma LUTs, optional), plus `checkerboard.png`. Load with `scipy.io.loadmat` and `matplotlib.image.imread`.

**lenses/ThorLabs/LE1234-A.txt** Zemax-style prescription (see the lens prescription operation).

**sinusoids/{x}.png** display fringe pattern images for period T=70; load with `imread`, average over the color channel, normalize each by its max.

Load lazily/free large arrays after use (the 2048x2048x48 stacks are large).
<!-- ORACLE_OP_END op_020_data_loading -->

<!-- ORACLE_OP_START op_030_camera_screen_geometry -->
## Camera pinhole projection and display screen plane conventions

## Camera and screen geometry

**Camera extrinsics from MATLAB.** For each camera i, intrinsics `f` (2), `c` (2), distortion `k` (3, zero-padded), `p` (2), rotation `R[...,i]`, translation `t[...,i]`. Convert MATLAB pose to world convention: `R_ours = R`, `t_ours = -R @ t`. Build a rigid `Transformation(R, t)` whose `transform_point(o)=R@o + t`, `transform_vector(d)=R@d`, and inverse `(R^T, -R^T t)`.

**Ray sampling (pinhole).** For pixel grid coordinates `uv` (add 0.5 pixel centering and a `crop_offset`, see cropping op), compute normalized image coords `xy = (uv + crop_offset - c) / f` (distortion coefficients are loaded but not applied in the forward path). Direction in camera space `d = normalize([xy_x, xy_y, 1])`, origin `o = 0`; transform both to world via the camera's `to_world`. Ray wavelength is fixed at 562.0 nm.

**Screen.** The display is a flat plane. `pixelsize = 1e-3 * display_pixel_size` (mm), `size = pixelsize * [1600, 2560]`. Screen transform is identity rotation, zero translation (plane at z=0 in world). Intersection of a ray with the screen (in screen-object frame): `t = -o_z / d_z`, local hit `p = o + t d`; valid when `mint<=t<=maxt` and `|p_xy - texture_shift| <= halfsize`. Screen-plane uv used only for rendering.

**Filmsize / crop_offset.** After cropping to `filmsize=[768,768]` from `full_filmsize=2048`, set `crop_offset = ((2048 - filmsize)/2) = [640, 640]` (int) and assign it to each camera so pinhole projection maps cropped pixels back to full-sensor coordinates.

Keep the `matlab2ours` sign convention, the 562 nm wavelength, the 0.5-pixel centering, and the crop_offset exactly; they define the world geometry.
<!-- ORACLE_OP_END op_030_camera_screen_geometry -->

<!-- ORACLE_OP_START op_040_material_ior -->
## Optical material index of refraction model

## Refractive index (dispersion) model

Each medium between surfaces has an index of refraction computed from a simplified Cauchy/Sellmeier two-term form:

`ior(wavelength_nm) = A + B / wavelength_nm^2`

where A,B are derived from catalog (n_d, Abbe V) pairs via:
```
lambdas = [656.3, 589.3, 486.1]   # nm (C, d, F lines)
ivs(a) = 1/a^2
B = (n - 1) / V / (ivs(486.1) - ivs(656.3))
A = n - B * ivs(589.3)
```
A material named `"n1/n2"` (slash) is parsed as literal `n=float(n1)`, `V=float(n2)`.

**Key table entries** used here: `air = [1.000293, inf]` (V=inf gives B=0, constant index), `n-bk7 = [1.51680, 64.17]`, `vacuum = [1.0, inf]`. The full catalog includes many glasses (f2, sf-series, baf10, pmma, etc.); N-BK7 is the lens glass for this case.

Evaluate all indices at the ray wavelength 562.0 nm. Refraction uses the ratio `eta = n_current / n_next` per surface in trace order (see the refraction operation).
<!-- ORACLE_OP_END op_040_material_ior -->

<!-- ORACLE_OP_START op_050_lens_prescription -->
## Lens prescription parsing and surface/material stack

## Lens prescription -> surfaces and materials

The prescription file has a title line, a header line, then whitespace-separated surface rows: `type distance roc diameter material`.

Parsing rules:
- Aperture radius `r = diameter / 2`. Curvature from `roc`: `c = 1/roc` if `roc != 0` else `0`.
- Maintain a running axial position `d_total` accumulating `distance`.
- Row types: `O` (object) resets `d_total=0` and drops that gap; `S`/`A` append an `Aspheric(r, d_total, c[, conic, ai])` surface (extra columns after material are conic then higher-order `ai`); `I` (image/stop) subtracts the last distance, pops the last material, and records `r_last`, `d_last` (sensor distance). `X` -> XYPolynomial, `B` -> BSpline (not needed for this biconvex case).
- Each row also appends a `Material(name)`.
- After parsing, `d_sensor = d_last + surfaces[-1].d`.

**For LE1234-A (this case):** two `S` surfaces, glass `N-BK7` between them, air outside. Surface[0] at `d_total=0`, surface[1] at `d_total=3.0` (initial thickness). The `roc=1000.0` values in the file are placeholders; the true curvatures are unknown and are the fit targets, so they get reinitialized (see solver initialization). Aperture radius `r = 25.4/2 = 12.7 mm`.

The lensgroup exposes the differentiable parameters at these attribute paths: `surfaces[0].c`, `surfaces[1].c`, `surfaces[1].d`, `origin`, `theta_x`, `theta_y`.
<!-- ORACLE_OP_END op_050_lens_prescription -->

<!-- ORACLE_OP_START op_060_aspheric_sag -->
## Aspheric sag function and surface derivatives

## Aspheric surface sag and derivatives

Each lens surface is an implicit surface `g(x,y) + h(z) = 0` with `h(z) = -z`, i.e. `z = sag(r^2)`. The sag as a function of `r2 = x^2 + y^2`:

```
tmp = r2 * c
sag = tmp / (1 + sqrt(1 - (1 + k) * tmp * c)) + higher_order(ai)
```
with curvature `c = 1/roc`, conic `k` (0 for a sphere), optional higher-order polynomial in `r2`. For the biconvex lens k=0 and no `ai`, giving `sag = c r2 / (1 + sqrt(1 - c^2 r2))`.

Radial derivative helper (`d sag / d r2`):
```
alpha_r2 = (1 + k) * c^2 * r2
t = sqrt(1 - alpha_r2)
d_sag_dr2 = c * (1 + t - 0.5*alpha_r2) / (t * (1 + t)^2)   (+ higher-order terms)
```
Cartesian surface derivatives: `dg/dx = 2*d_sag_dr2 * x`, `dg/dy = 2*d_sag_dr2 * y`, and `dh/dz = -1`. The surface normal is `normalize([dg/dx, dg/dy, -1])`.

Implement in PyTorch so `c` (and thickness) stay in the autograd graph. See the bundled helper for a reference implementation.

Provided helper: `${CLAUDE_SKILL_DIR}/scripts/op_060_aspheric_sag.py`.
<!-- ORACLE_OP_END op_060_aspheric_sag -->

<!-- ORACLE_OP_START op_070_newton_intersection -->
## Ray-surface intersection via implicit-differentiation Newton

## Ray-surface intersection (Newton, implicit differentiation)

For a ray `p(t) = o + t d` and implicit surface `f(x,y,z) = g(x,y) + h(z) + d_offset` (with `h(z)=-z`, `d_offset` the surface axial position), find the intersection parameter `t`.

**Newton iteration** (run under `torch.no_grad()`): initialize `t0 = (d_offset - o_z) / d_z`, `t_delta = 0`. Precompute `A = d_x^2 + d_y^2`, `B = 2(d_x o_x + d_y o_y)`, `C = o_x^2 + o_y^2`, so `r2(t) = A t^2 + B t + C`. Each step evaluates residual `s = sag(r2) - z` and its derivative-dot-direction `ds·D = dsag_dr2*(2A t + B) - d_z`, then `t_delta -= residual / (ds·D)`. Iterate up to `NEWTONS_MAXITER = 10`; tight tolerance `50e-6`, loose validity tolerance `300e-6`.

**Implicit differentiation.** Do NOT backpropagate through the Newton loop. After convergence (no_grad), reattach a single differentiable correction so gradients flow into surface/pose params:
```
t = t0 + t_delta                      # detached
t = t - (g(o_x+t d_x, o_y+t d_y) + h(o_z+t d_z) + d_offset) / (ds·D)
p = o + t * d
```
This treats the converged root as an implicit function of the parameters (correct gradients, lower cost than unrolling).

**Aperture validity.** After intersection, `r2 = p_x^2 + p_y^2`, `g_aperture = r^2 - r2`; the hit is valid where the Newton residual was within loose tolerance AND `g_aperture > 0` (inside the clear aperture). Combine with the incoming validity mask.

Keep MAXITER=10 and both tolerances; they balance convergence against the differentiable correction.
<!-- ORACLE_OP_END op_070_newton_intersection -->

<!-- ORACLE_OP_START op_080_snell_refraction -->
## Vector Snell's law refraction

## Refraction (vector Snell's law)

At each surface refract the incident unit direction `wi` using the surface normal `n` and index ratio `eta = n_from / n_to`.

Exact form (used):
```
cosi = sum(wi * n, axis=-1)
cost2 = 1 - (1 - cosi^2) * eta^2
valid = cost2 > 0                     # else total internal reflection
cost2 = clamp(cost2, min=1e-8)
wt = sqrt(cost2)[...,None] * n + eta * (wi - cosi[...,None] * n)
```
The returned `wt` is the refracted direction; `g = cost2` feeds the soft validity indicator. Renormalization is not required because inputs are unit vectors and the formula preserves norm up to the clamp.

**Normal orientation.** In forward tracing (rays travelling +z) pass `-n` (the outward normal negated) into refraction; ordering of `eta` follows trace direction: at surface i, `eta = materials[i].ior / materials[i+1].ior`. The `clamp(min=1e-8)` keeps the sqrt differentiable near grazing incidence. Keep eta ordering and the clamp exactly.
<!-- ORACLE_OP_END op_080_snell_refraction -->

<!-- ORACLE_OP_START op_090_lensgroup_trace -->
## Lensgroup pose transform and full camera-to-screen trace

## Lensgroup pose and forward trace

**Pose.** The lensgroup pose is `origin` (3D), `shift` (3D, from calibration `origin_shift`), and tilt angles `theta_x, theta_y, theta_z` (degrees). Rotation is the product of Rodrigues rotations about x, y, z applied to `deg2rad(theta_*)`:
`R = Rx(theta_x) @ Ry(theta_y) @ Rz(theta_z)`, world translation `t = origin + R @ shift`. `to_world = Transformation(R, t)`, `to_object = inverse`. Rebuild this whenever pose params require grad (call update()).

**Trace path (with element).** For each camera: sample rays (pinhole op), transform ray into lens-object frame via `to_object`; then for each surface i in order: intersect (Newton op) -> compute normal -> refract (Snell op, using `-n` forward, `eta = ior[i]/ior[i+1]`); accumulate soft mask via the differentiable Step indicator `ind(g)=Step` (forward `(g>0).float()`, backward `grad*exp(-(0.5*g)^2)`) and hard validity `valid &= valid_o & valid_d`. After the last surface, transform the exit ray back to world via `to_world` and intersect the screen plane. Predicted screen point `ps` uses the first two components `p[...,0:2]`.

**Scene.trace vs render.** `trace` returns per-camera `(points[H,W,3], valid[H,W], mask_g[H,W])` permuted so axis order is `(H,W,...)`; the pipeline stacks cameras -> tensor `[n_cam,H,W,3]`. When a validity mask is supplied, only masked pixels are sampled and scattered back into a full `[H,W]` grid. `render` instead shades the screen texture at the uv hit (bilinear, replicate boundary) for visualization only.

**Display texture.** Build the screen texture from the sinusoid images: average each over color channel, normalize by max, sum the selected `xs`, then `set_texture` flips the texture and rebuilds the Screen with `pixelsize` and identity pose at z=0. Apply the calibrated `texture_shift = [0.0, 1.1106231]`.

The forward callable returns predicted `ps` for valid pixels; this is the map differentiated in the solver. Keep the Rx@Ry@Rz order, degree units, and the +0:2 screen projection.
<!-- ORACLE_OP_END op_090_lensgroup_trace -->

<!-- ORACLE_OP_START op_100_fringe_phase -->
## Four-step phase-shifting fringe analysis

## Four-step phase-shifting

Each camera captures, per fringe period, 8 phase-shifted images: indices 0..3 encode the x-direction fringe, 4..7 encode the y-direction fringe. For a 4-image set `f0..f3` modeled as `f = a + b*cos(phi + psi)`:
```
a   = mean(f0..f3)                 # DC / background amplitude
b   = sum_i (fi - a)^2 / 2         # modulation (squared)
psi = arctan2(f3 - f1, f0 - f2)    # wrapped phase
```
Apply this separately to the x-set and y-set, producing per-direction `a`, `b`, `psi` each stacked as `[2 (x,y), ...]`. The stacking reorders the phase-shift axis (size 8) to the front before splitting into two 4-image sets.

Compute this for both the lens images (`imgs` -> a_cap,b_cap,psi_cap) and the reference images (`refs` -> a_ref,b_ref,psi_ref). The differential wrapped phase is `psi_cap - psi_ref`. `a` and `b` feed the validity mask; the differential phase feeds unwrapping.

Keep the exact `arctan2(f3-f1, f0-f2)` convention and the x-then-y ordering.
<!-- ORACLE_OP_END op_100_fringe_phase -->

<!-- ORACLE_OP_START op_110_validity_mask -->
## Per-camera validity mask from amplitude, modulation, connectivity

## Validity mask V

Build a binary per-camera mask selecting reliably measured lens pixels.

1. Aggregate over batch and period: `A = mean(a_cap, axis=(0,1))`, `B = mean(b_cap, axis=(0,1))`, and `I = |mean(refs - imgs, axis=(0,1))|` (all per camera).
2. Threshold `thres = 0.07`: `valid_ab = (A > thres) & (B > thres) & (I < 2.0*thres)`.
3. Connected components (`scipy.ndimage.label`). If <2 features, take label 1. Otherwise keep components whose pixel count is in `(200^2, 500^2)`; if several qualify, pick the one whose centroid is closest (L1) to the image center `filmsize/2`.
4. The mask `V` is the chosen connected component per camera; stack cameras -> `valid_map`.

This mask restricts every downstream operation (unwrapping, DC removal, residual, loss). Removing it degrades robustness by admitting low-modulation/background pixels. Keep `thres=0.07`, the `2*thres` intensity gate, and the size window.
<!-- ORACLE_OP_END op_110_validity_mask -->

<!-- ORACLE_OP_START op_120_measured_points -->
## Unwrap, DC removal, mm conversion, reference-trace displacement, mount origin

## Measured screen-intersection target ps_cap

Convert the differential wrapped phase into measured screen points in mm.

1. **Temporal unwrap.** For each direction (x,y), period T, and camera: `unwrapped = unwrap_phase(V * (psi_cap - psi_ref)) * T / (2*pi)` using `skimage.restoration.unwrap_phase`. Result is displacement in display pixels.
2. **DC (integer-period) removal.** For each period T, search integer multiples `k in [-10, 10]`: choose `DC = k*T` minimizing `sum(( unwrapped + DC )^2)` over valid pixels, then add the selected DC back (masked). This removes the absolute-phase offset ambiguity.
3. **Pixel -> mm.** Multiply by `screen.pixelsize` (mm/px). Then average over periods: `psi_x = mean_T`, `psi_y = mean_T`.
4. **Reference trace.** Trace the scene WITHOUT the lens element (`with_element=False`) to get reference screen points `ps_ref[...,0:2]`, masked by V.
5. **Displacement -> measured points.** `ps_cap = ps_ref - stack((psi_y, psi_x), axis=-1)` (note the y,x ordering when combining). This is the measured target `hat{p}(u)`.

**Mount origin estimate.** Estimate the lens mount 3D origin by triangulating the two cameras' rays through calibration points: solve least squares `min_{t1,t2} || L1(t1) - L2(t2) ||^2` with `A = stack(d0, -d1, axis=-1)`, `b = -(o0 - o1)`, `np.linalg.lstsq`; take the mean of the two ray points at the solved parameters. Use rotation-calibration points first, then recompute from the measured valid-region centers `C` and assign to `lensgroup.origin` before optimization. Keep the k in [-10,10] DC search, the period averaging, and the (psi_y, psi_x) stacking order.
<!-- ORACLE_OP_END op_120_measured_points -->

<!-- ORACLE_OP_START op_130_lm_solver -->
## Levenberg-Marquardt fit with autograd column-wise Jacobian

## Levenberg-Marquardt optimization

**Unknowns (8).** Parameter paths: `surfaces[0].c`, `surfaces[1].c`, `surfaces[1].d`, `origin` (3), `theta_x`, `theta_y`. Enable `requires_grad` on each.

**Initialization (critical).** `c0 = 0.001`, `c1 = 0.001` (NOT zero — zero gives a degenerate flat-surface Jacobian), `d1 = 3.0`, `theta_x = theta_y = 0.0`; `origin` from the mount estimate. Rebuild the pose (update) after setting.

**Residual and loss.** `residual = V[...,None] * (ps_cap - ps_pred)` with NaNs set to 0; loss `L = mean(residual^2)`.

**LM step (params: lambda0=0.1, mu=2.0, regularization='diag', max_iter=100):**
1. Predict `y = forward()`; record `L`. Stop if `L < 1e-16`.
2. Build the tall Jacobian `J` (M rows = valid residual entries, N=8 cols) column-wise via autograd JVP (perturb one parameter at a time using double-grad trick); reshape to `[-1, N]`. Form `JtJ = J^T J`.
3. Regularization matrix `R`: `'diag'` -> `diag(|diag(JtJ)|)` (Marquardt scaling for the very different parameter magnitudes, e.g. curvature ~1e-2 vs origin ~1e1); `'I'` -> identity.
4. Gradient term `b = J^T r` computed as `torch.autograd.grad(y, x, grad_outputs=residual)` per parameter, concatenated.
5. **Damping loop:** solve `(JtJ + lambda*R) delta = b` (`torch.linalg.solve`, fall back to `lstsq` on singular). Apply `+delta`. If new `L_current < L`: accept, `lambda /= mu`, break. Else revert (`-delta`), `lambda *= 2*mu`; abort inner loop after 20 tries or if `lambda > 1e16` or delta is NaN.
6. **Stopping:** after each outer iter, stop if `|L_prev - L| < 1e-8` or mean `|delta| < 1e-8`, or at max_iter. Typically converges in ~21 iterations.

Expected recovered values for this case: surface_0_roc ≈ -76.5 mm, surface_1_roc ≈ -31.7 mm, thickness ≈ 3.25 mm; mean displacement error ≈ 43 µm. Keep the diag regularization, the 0.001 curvature init, and the damping schedule.

Helper: `${CLAUDE_SKILL_DIR}/scripts/op_130_lm_solver.py`.
<!-- ORACLE_OP_END op_130_lm_solver -->

<!-- ORACLE_OP_START op_140_output_writing -->
## Assemble recovered parameters and write submission npz

## Write the submission

After LM converges, read the fitted scene parameters and convert curvatures to radii of curvature:
```
surface_0_roc_mm = 1.0 / surfaces[0].c
surface_1_roc_mm = 1.0 / surfaces[1].c
thickness_mm     = surfaces[1].d
```
Write `outputs/{case_id}.npz` (create `outputs/` if missing) containing exactly `surface_0_roc_mm`, `surface_1_roc_mm`, `thickness_mm` as finite 0-d float arrays (float32 or float64). Detach from autograd and cast to plain numpy floats; do not store Python objects or non-finite values.

Validate before finishing: all three keys present, each 0-d, finite. If any is NaN/Inf the submission fails the `finite_only` gate.

Helper: `${CLAUDE_SKILL_DIR}/scripts/op_140_write_output.py`.
<!-- ORACLE_OP_END op_140_output_writing -->
