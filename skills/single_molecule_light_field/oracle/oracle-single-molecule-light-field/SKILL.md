---
name: oracle-single-molecule-light-field
description: "Reconstruct 3D single-molecule positions from multi-view 2D localisations of a hexagonal Fourier light field microscope, then report retained-count and median lateral/axial precision metrics. Use when a case provides PeakFit-style 2D localisations plus microscope parameters and expects an outputs/{case_id}.npz with n_locs_3d_filtered, median_lateral_err_nm, median_axial_err_nm."
---

# oracle-single-molecule-light-field Oracle Skill

This skill localizes fluorescent molecules in 3D by exploiting parallax between micro-lens views. Each operation below is independently removable. Together they cover: reading the public contract and case inputs; converting raw PeakFit columns to a standardized micron-unit array; centring the field of view; deriving the optical geometry (back-focal-plane radius, magnification, pupil scaling); building and rotating the hexagonal micro-lens lattice; assigning each localisation to its nearest lens in normalised pupil coordinates; filtering to in-aperture lenses; computing the phase-averaged spherical angular-sensitivity model; solving the per-molecule ordinary-least-squares parallax inversion; grouping companion views by angle and disparity consistency; running the frame-by-frame fitting engine with z calibration; a two-pass per-view aberration correction; and writing the final scalar metrics. Individual formulas, constants, axis orders, and column layouts live inside their own operation sections so each can be ablated in isolation.

<!-- ORACLE_OP_START op_001_public_contract -->
## Public task contract and submission schema

Establish what the solver must produce before doing any computation.

Case discovery: read the run-local case manifest `cases.json` (search under `public/`; it may sit at `public/cases.json` or `public/visible_data/cases.json`). It has `version` and a `cases` list. Each case entry provides `id` (e.g. `case_000`), an `input_dir` (e.g. `cases/case_000/input_data`), and a `params` path (e.g. `cases/case_000/params_data.json`). Resolve those relative to the manifest's own directory. Under each case: `input_data/raw_data.npz` holds the raw localisations and the case params JSON holds all microscope settings.

Submission: for each case write exactly `outputs/{case_id}.npz` at the run root. The NPZ must contain three scalar (0-d) arrays:
- `n_locs_3d_filtered` — int32 or int64 — number of retained 3D localisations after the quality filter.
- `median_lateral_err_nm` — float32/float64 — median lateral localisation error in nanometres.
- `median_axial_err_nm` — float32/float64 — median axial localisation error in nanometres.

Hard invariants: every value must be finite (no NaN/Inf); the grader does not cast dtypes, so store `n_locs_3d_filtered` as a true integer scalar and the two errors as float scalars. Write with `numpy.savez` using these exact keys. Do not add extra required keys.

After writing, verify the file with the bundled checker: `python ${CLAUDE_SKILL_DIR}/scripts/op_001_public_contract.py outputs/case_000.npz`.
<!-- ORACLE_OP_END op_001_public_contract -->

<!-- ORACLE_OP_START op_002_load_localizations -->
## Load raw NPZ and convert PeakFit columns to standardized (N,8)

Load the case params JSON (call it `meta`) and `input_data/raw_data.npz`. The NPZ has a single key `localizations_2d`; cast it to float64. For this task it is a raw PeakFit export of shape (N, 35) (N=150996, 5000 frames).

Produce the standardized internal array `locs_2d_csv` of shape (N, 8) in physical units:
- If the loaded array already has 8 columns, use it as-is.
- If `meta['csv_format'].upper() == 'PEAKFIT'` and there are >= 14 columns, extract and convert using `pixel_size_sample` (see the optics operation; equivalently `pixel_size_camera / magnification`, and `magnification = (f_tube/f_obj)*(f_mla/f_fourier)`):

  - col 0: `[0] frame`  = raw[:,0]
  - col 1: `[1] X (um)`  = raw[:,9]  * pixel_size_sample
  - col 2: `[2] Y (um)`  = raw[:,10] * pixel_size_sample
  - col 3: `[3] sigma_X (um)` = raw[:,12] * pixel_size_sample
  - col 4: `[4] sigma_Y (um)` = raw[:,12] * pixel_size_sample   (same source column as sigma_X)
  - col 5: `[5] intensity (photons)` = raw[:,8]
  - col 6: `[6] background (photons)` = raw[:,7]
  - col 7: `[7] precision (um)` = raw[:,13] / 1000.0   (nm -> um)

Exact source column indices matter: X=9, Y=10, sigma=12, intensity=8, background=7, precision=13, frame=0. Raise an error for any other unsupported shape. Do not drop, sort, or deduplicate rows here.
<!-- ORACLE_OP_END op_002_load_localizations -->

<!-- ORACLE_OP_START op_003_center_localizations -->
## Centre X,Y at the optical axis

After building the standardized (N,8) array, subtract the mean of column 1 (X) from column 1 and the mean of column 2 (Y) from column 2, on a copy. This places the field-of-view centre at the origin (0,0), which is required because the micro-lens lattice is defined relative to its own centre. Do not touch any other column. Use the full-dataset mean (all N rows) computed once, not a per-frame mean.
<!-- ORACLE_OP_END op_003_center_localizations -->

<!-- ORACLE_OP_START op_004_optics_model -->
## Derived Fourier-microscope optical quantities

Compute all derived optical scalars from `meta`. Focal lengths are in millimetres; all resulting lengths are microns. Formulas (verified against the provided case: NA=1.27, f_obj=200/60=3.3333 mm, f_tube=200, f_fourier=175, f_mla=175, pitch=2390 um, pixel=16 um):

- `bfp_radius = 1000.0 * NA * f_obj * (f_fourier / f_tube)`  => 3704.1667 um
- `magnification = (f_tube / f_obj) * (f_mla / f_fourier)`   => 60.0
- `bfp_lens_count = 2.0 * bfp_radius / mla_lens_pitch`       => 3.09972
- `pixels_per_lens = mla_lens_pitch / pixel_size_camera`
- `pixel_size_sample = pixel_size_camera / magnification`    => 0.266667 um
- `rho_scaling = magnification / bfp_radius`                 => 0.01619798  (converts image-plane microns to normalised pupil coords)
- `xy_to_uv_scale = rho_scaling`
- `mla_to_uv_scale = 2.0 / bfp_lens_count`                   => 0.645219  (MLA lattice-spacing units -> pupil coords)
- `mla_to_xy_scale = mla_to_uv_scale / rho_scaling`          => 39.8333 um

Keep these values; downstream lens assignment, alpha, and OLS all depend on `rho_scaling`, `mla_to_uv_scale`, and `mla_to_xy_scale`. See `${CLAUDE_SKILL_DIR}/resources/op_004_optics_model.md` for the reference-value table.
<!-- ORACLE_OP_END op_004_optics_model -->

<!-- ORACLE_OP_START op_005_mla_lattice -->
## Hexagonal micro-lens lattice generation and rotation

Build the micro-lens lattice in lattice-spacing units, then rotate it. Parameters from `meta`: `mla_type='HEXAGONAL'`, `mla_lens_pitch=2390`, `mla_optic_size=10000`, `mla_centre=[0,0]`, `mla_rotation=30.8` (degrees), `mla_offset=[0,0]`.

Generation (hexagonal), exact convention:
```
width = optic_size / lens_pitch
marks = arange(-floor(width/2), ceil(width/2) + 1)
xx, yy = meshgrid(marks, marks)
length = max(xx.shape)
dx = tile([0.5, 0], [length, ceil(length/2)])[:length, :length]
x = yy * sqrt(3)/2
y = xx + dx.T + 0.5
lens_centres = column_stack((x.flatten('F'), y.flatten('F')))   # (K,2)
```
This yields 36 raw centres for the given optic. Use Fortran ('F') flattening order.

Rotation about `centre` by theta = deg2rad(mla_rotation):
```
x' = (x-cx)*cos(theta) - (y-cy)*sin(theta) + cx
y' = (x-cx)*sin(theta) + (y-cy)*cos(theta) + cy
```
Apply offset only if `mla_offset` is nonzero: shift lens_centres by `mla_offset / mla_to_xy_scale` (lattice units); for this case the offset is [0,0], so skip. Keep `centre` as float array [0,0]. See `${CLAUDE_SKILL_DIR}/resources/op_005_mla_lattice.md`.
<!-- ORACLE_OP_END op_005_mla_lattice -->

<!-- ORACLE_OP_START op_006_lens_assignment -->
## Assign localisations to nearest lens (kNN in pupil coords) and build the 13-column array

Create the working 13-column localisation array and assign each row to its nearest micro-lens view.

Build `locs_2d` (N,13), zero-initialised, from the centred (N,8) array:
- col 0 = frame
- cols 3:10 = the (N,8) columns 1:8, i.e. col3=X(um), col4=Y(um), col5=sigma_X, col6=sigma_Y, col7=intensity(photons), col8=background, col9=precision(um)
- cols 1,2 (U,V), 10,11 (alpha_U,alpha_V), 12 (lens index) are filled later.

Assignment:
```
xy_uv = locs_2d[:, 3:5] * xy_to_uv_scale            # (rho_scaling)
lens_uv = (mla.lens_centres - mla.centre) * mla_to_uv_scale
idx = NearestNeighbors(n_neighbors=1).fit(lens_uv).kneighbors(xy_uv, return_distance=False)
locs_2d[:, 1:3] = lens_uv[idx][:, 0, :]             # snap U,V to the assigned lens centre
locs_2d[:, 12]  = idx[:, 0]                          # lens index
```
Use scikit-learn `sklearn.neighbors.NearestNeighbors`. Critically, columns 1,2 are set to the assigned lens-centre (u,v), NOT the raw per-point pupil coordinate; the whole parallax model treats each view as a single discrete (u,v). Record `min_frame`, `max_frame` from int min/max of column 0. Initialise `filtered_locs_2d = locs_2d.copy()`.
<!-- ORACLE_OP_END op_006_lens_assignment -->

<!-- ORACLE_OP_START op_007_filter_lenses -->
## Aperture (BFP) lens filter and optional quality filters

Restrict `filtered_locs_2d` to valid views before computing alpha and fitting.

BFP lens filter (always on when `meta['filter_lenses']` is true, which it is here):
```
radius_sq = (bfp_radius / mla_lens_pitch) ** 2
centres   = mla.lens_centres - mla.centre
valid_idx = nonzero(sum(centres**2, axis=1) < radius_sq)[0]
keep = isin(filtered_locs_2d[:,12], valid_idx)
filtered_locs_2d = filtered_locs_2d[keep]
```
This removes localisations assigned to lenses outside the back-focal-plane circle (only ~0.16% removed here: 150996 -> 150758; 7 views remain).

Optional filters, applied in this order only if the corresponding `meta` key is not null (all null for this case, so they are no-ops):
- `filter_rhos=(lo,hi)`: keep rows with pupil radius sqrt(u^2+v^2) in [lo,hi] using cols 1,2.
- `filter_spot_sizes=(lo,hi)`: keep rows with col 5 (sigma_X) in [lo,hi].
- `filter_photons=(lo,hi)`: keep rows with col 7 (intensity) in [lo,hi].

Do not drop rows for any other reason.
<!-- ORACLE_OP_END op_007_filter_lenses -->

<!-- ORACLE_OP_START op_008_alpha_model -->
## Phase-averaged spherical angular-sensitivity model (INTEGRATE_SPHERE)

Compute the angular sensitivity alpha(u,v) for each retained localisation and store it in `filtered_locs_2d[:, 10:12]` (alpha_U, alpha_V). alpha encodes how a unit depth change shifts the apparent lateral position in a given view. Use `meta['alpha_model']` = `INTEGRATE_SPHERE`.

Since columns 1,2 were snapped to discrete lens centres, all rows of a given view share one alpha; you may compute per unique (u,v) and broadcast.

Models (na = num_aperture = 1.27, n = ref_idx_medium = 1.33):
- LINEAR: alpha = (u, v)
- SPHERE: rho=sqrt(u^2+v^2); alpha = -(na/n)/sqrt(1 - rho^2*(na/n)^2) * (u,v); set the sqrt argument to NaN where negative.
- INTEGRATE_SPHERE (use this): phase-average the SPHERE expression over each micro-lens aperture. Aperture width in pupil units = `mla_to_uv_scale`. For each (u,v), build an (m+1)x(m+1) sub-sample grid (m=10) spanning [u-ds2, u+ds2] x [v-ds2, v+ds2] with ds2 = mla_to_uv_scale/2; compute phi = -(na/n)/sqrt(1 - (u'^2+v'^2)(na/n)^2) with NaN where the radicand < 0; then alpha_U = nanmean(u'*phi), alpha_V = nanmean(v'*phi).

Reference alpha for the 7 views (m=10) is in `${CLAUDE_SKILL_DIR}/resources/op_008_alpha_model.md`. Use the bundled helper `${CLAUDE_SKILL_DIR}/scripts/op_008_alpha_model.py` (function `compute_alpha`). The central view (0,0) yields alpha ~ (0,0); outer views yield |alpha| ~ 1.06.
<!-- ORACLE_OP_END op_008_alpha_model -->

<!-- ORACLE_OP_START op_009_backward_ols -->
## Per-molecule OLS parallax inversion (backward model)

Given the multi-view rows for one candidate molecule (each row is a 13-col localisation with columns 1,2=u,v; 3,4=x,y; 10,11=alpha_u,alpha_v), solve for (x0,y0,z) by ordinary least squares.

Forward model per view i: x_i = x0 + u_i/rho + z*alpha_u_i ; y_i = y0 + v_i/rho + z*alpha_v_i.
Rearranged linear system A [x0,y0,z]^T = b:
```
x = locs[:,3] - locs[:,1]/rho_scaling
y = locs[:,4] - locs[:,2]/rho_scaling
alpha_u = locs[:,10]; alpha_v = locs[:,11]
A = vstack( column_stack(ones, zeros, alpha_u),
            column_stack(zeros, ones, alpha_v) )   # rows: x-eqs then y-eqs
b = concatenate((x, y))
model, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
mse = sum(residuals)/(A.shape[0]-rank) if residuals.size>0 else 0.0
cov = mse * inv(A.T @ A)
std_err = sqrt(diag(cov))     # [se_x0, se_y0, se_z]
```
Return `model` [x0,y0,z], `std_err` [se_x,se_y,se_z], and `mse`. The x-equation rows come first, then all y-equation rows (b is x-block then y-block). Use `rcond=None`. This routine is called both to score candidate fits and, inside grouping, to disambiguate duplicate-view points. Helper: `${CLAUDE_SKILL_DIR}/scripts/op_009_backward_ols.py`.
<!-- ORACLE_OP_END op_009_backward_ols -->

<!-- ORACLE_OP_START op_010_group_localisations -->
## Group companion views by parallax angle and disparity consistency

For a seed localisation, gather companion localisations (within the same frame) that are geometrically consistent with a single molecule at one depth z. seed[1],seed[2]=su,sv (u,v); seed[3],seed[4]=sx,sy (x,y).

Steps:
1. Drop points in the same view as the seed: remove rows where (u==su)&(v==sv).
2. Angle test: du=u-su, dv=v-sv, dx=x-sx, dy=y-sy; angles_uv=arctan2(dv,du), angles_xy=arctan2(dy,dx); diff=abs(angles_xy-angles_uv); keep where (diff<angle_tol) OR (diff>pi-angle_tol), where angle_tol=deg2rad(angle_tolerance). (Molecule shifts along +/- the view direction.)
3. Disparity test on survivors: recompute du,dv,dx,dy; dxy=sqrt(dx^2+dy^2), duv=sqrt(du^2+dv^2); disparity = where(duv>0, (dxy - duv/rho_scaling)/duv, 0). Build range_z = arange(-disparity_max, disparity_max + step/2, step) (step=disparity_step). For each z_i count points with disparity in (z_i - dist_search, z_i + dist_search]; best_z = range_z[argmax(count)]. Keep points with disparity in (best_z - dist_search, best_z + dist_search].
4. Duplicate-view resolution: start candidates = seed (as a single row). For unique (u,v) views appearing exactly once, add them directly. For views with multiple points, add the one point that minimises sqrt(sum(std_err^2)) from the OLS backward model of (current candidates + that point). `view_count` = number of unique companion views (seed not counted here).
5. Degeneracy guard: if view_count==1 and either var(candidates[:,1],ddof=1)<0.1 or var(candidates[:,2],ddof=1)<0.1, reject (return empty, view_count=0).

Return (candidates rows, view_count, duplicate_count). Parameter values differ per pass (see the fitting-engine operation): aberration pass uses disparity_max=5.0, angle_tolerance=2.0, dist_search=0.5, step=0.1; full pass uses disparity_max=8.0, angle_tolerance=1.0, dist_search=0.5, step=0.1. See `${CLAUDE_SKILL_DIR}/resources/op_010_group_localisations.md`.
<!-- ORACLE_OP_END op_010_group_localisations -->

<!-- ORACLE_OP_START op_011_light_field_fit_engine -->
## Frame-by-frame fitting engine, full-pass parameters, and z calibration

Core engine that turns grouped views into 3D localisations, run once per frame. Produces the (M,8) output table.

For each frame in [frame_min..frame_max]:
1. candidates = rows with col0==frame (copy). Order them central-views-first then by intensity: central = |u|<0.1 & |v|<0.1, sorted by descending col7 (intensity); outer = rows not at exact origin, sorted by descending col7; stack (central, outer).
2. Loop while candidates.shape[0] > min_views and reps < 100:
   - seed = candidates[0]; group companions (previous operation).
   - if view_count < min_views: drop seed (candidates=candidates[1:]); continue.
   - solve OLS backward model on grouped rows; dist = sqrt(sum(std_err^2)); if dist > threshold: drop seed; continue.
   - remove all grouped (used) rows from the candidate pool.
   - append output row: [model[0], model[1], model[2], mean(std_err[0:2]), std_err[2], view_count+1, sum(grouped[:,7]), frame]. Note n_views stored = view_count+1 (seed included) and photons = sum of intensity col7 over grouped rows.
3. After all frames, if z_calib is not None: multiply output column 2 (Z) by z_calib.

Output `locs_3d` (M,8): [0]X um, [1]Y um, [2]Z um (calibrated), [3]lateral_err um = mean(se_x,se_y), [4]axial_err um = se_z, [5]n_views, [6]photons, [7]frame.

Full-dataset pass parameters (fit on the aberration-corrected array; fall back to filtered array if no correction was applied): frame_min = data min frame, frame_max = data max frame, disparity_max=8.0, disparity_step=0.1, dist_search=0.5, angle_tolerance=1.0, threshold=0.3, min_views=2, z_calib=1.534. The reference full pass yields ~24,543 3D localisations. See `${CLAUDE_SKILL_DIR}/resources/op_011_light_field_fit_engine.md`.
<!-- ORACLE_OP_END op_011_light_field_fit_engine -->

<!-- ORACLE_OP_START op_012_aberration_correction -->
## Two-pass per-view aberration correction

Before the full fit, estimate and remove small systematic per-view XY offsets (optical aberrations) using a loose first pass on the early frames.

Pass 1 (run the fitting engine on `filtered_locs_2d`) with aberration parameters: frame_min = data min, frame_max = min(1000, data_max), disparity_max=5.0, disparity_step=0.1, dist_search=0.5, angle_tolerance=2.0, threshold=1.0, min_views=3, z_calib=None. Keep the per-molecule fit records (frame, model [x0,y0,z], grouped points, photon_count, std_err).

Build the correction table over the unique views `views = unique(filtered_locs_2d[:,1:3])` (V rows, one per view). Selection criteria (aberration_params): axial_window=1.0, photon_threshold=1, min_views=3. For each accepted fit with points.shape[0] > min_views AND abs(z) < axial_window AND photon_count > photon_threshold: compute per-view forward residual `err = xy - uv/rho - xy0 - z*alpha` (shape (K,2)); for each view row present, accumulate its residual and increment its count. Finally divide accumulated dx,dy by the count (0 where count==0). Correction table is (V,5): [0]u [1]v [2]mean_dx [3]mean_dy [4]n_points.

Apply correction to produce `corrected_locs_2d` = copy of `filtered_locs_2d` with, for each view row, X(col3) -= dx and Y(col4) -= dy where (u==row[0])&(v==row[1]). The full pass then fits this corrected array. Reference: 7 views, mean correction magnitude ~0.9 um. This step is quality scaffolding; the pipeline still produces output if skipped (the full pass then fits the uncorrected filtered array), but metrics differ. See `${CLAUDE_SKILL_DIR}/resources/op_012_aberration_correction.md`.
<!-- ORACLE_OP_END op_012_aberration_correction -->

<!-- ORACLE_OP_START op_013_output_metrics -->
## Quality filter, summary metrics, and NPZ writing

From the (M,8) `locs_3d` table, compute the three required scalar metrics and write the submission.

Convert errors to nm: lat_err_nm = 1000*locs_3d[:,3]; ax_err_nm = 1000*locs_3d[:,4]; n_views = locs_3d[:,5].

Quality mask (visualization/reporting filter): keep = (lat_err_nm < show_max_lateral_err) AND (n_views > show_min_view_count), where thresholds come from `meta` with defaults show_max_lateral_err=200.0 and show_min_view_count=3. Note the view test is strict greater-than (> 3), so only molecules with 4+ views pass.

Metrics:
- `n_locs_3d_filtered` = int(sum(keep))
- `median_lateral_err_nm` = float(median(lat_err_nm[keep]))
- `median_axial_err_nm` = float(median(ax_err_nm[keep]))

Write `outputs/{case_id}.npz` via `numpy.savez` with these exact keys, storing `n_locs_3d_filtered` as an integer scalar (e.g. `np.int64(...)`) and the two medians as float scalars. Ensure all three are finite. Reference values on the public case: n_locs_3d_filtered=21727, median_lateral_err_nm~51.79, median_axial_err_nm~55.33 (out of ~24,543 total). Create the `outputs/` directory if absent.
<!-- ORACLE_OP_END op_013_output_metrics -->

<!-- ORACLE_OP_START op_014_runtime_probe -->
## Cheap optics/assignment smoke probe

Optional pre-run sanity check before launching the full ~30-60 s fit. It loads a case, derives the optical scalars, builds and rotates the lattice, assigns localisations, and prints counts, without doing the frame loop. Run: `python ${CLAUDE_SKILL_DIR}/scripts/op_014_runtime_probe.py <path-to-case-dir>` where the case dir contains `input_data/raw_data.npz` and a params json. Expected on the public case: bfp_radius~3704.17, magnification=60, rho_scaling~0.016198, 7 lenses inside the BFP, ~150996 localisations across 5000 frames, ~150758 kept after the aperture filter. Non-mutating; useful to catch column-index, rotation, or scaling mistakes early.
<!-- ORACLE_OP_END op_014_runtime_probe -->
