---
name: oracle-single-molecule-light-field
description: "Reconstruct 3D single-molecule positions from multi-view 2D localisations of a hexagonal Fourier light field microscope, then report retained-count and median lateral/axial precision metrics. Use when a case provides PeakFit-style 2D localisations plus microscope parameters and expects an outputs/{case_id}.npz with n_locs_3d_filtered, median_lateral_err_nm, median_axial_err_nm."
---

# oracle-single-molecule-light-field Oracle Skill

This skill localizes fluorescent molecules in 3D by exploiting parallax between micro-lens views. Each operation below is independently removable. Together they cover: reading the public contract and case inputs; converting raw PeakFit columns to a standardized micron-unit array; centring the field of view; deriving the optical geometry (back-focal-plane radius, magnification, pupil scaling); building and rotating the hexagonal micro-lens lattice; assigning each localisation to its nearest lens in normalised pupil coordinates; filtering to in-aperture lenses; computing the phase-averaged spherical angular-sensitivity model; solving the per-molecule ordinary-least-squares parallax inversion; grouping companion views by angle and disparity consistency; running the frame-by-frame fitting engine with z calibration; a two-pass per-view aberration correction; and writing the final scalar metrics. Individual formulas, constants, axis orders, and column layouts live inside their own operation sections so each can be ablated in isolation.

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

Keep these values; downstream lens assignment, alpha, and OLS all depend on `rho_scaling`, `mla_to_uv_scale`, and `mla_to_xy_scale`. See `${CLAUDE_SKILL_DIR}/resources/resource_001.md` for the reference-value table.

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

Full-dataset pass parameters (fit on the aberration-corrected array; fall back to filtered array if no correction was applied): frame_min = data min frame, frame_max = data max frame, disparity_max=8.0, disparity_step=0.1, dist_search=0.5, angle_tolerance=1.0, threshold=0.3, min_views=2, z_calib=1.534. The reference full pass yields ~24,543 3D localisations. See `${CLAUDE_SKILL_DIR}/resources/resource_002.md`.

## Quality filter, summary metrics, and NPZ writing

From the (M,8) `locs_3d` table, compute the three required scalar metrics and write the submission.

Convert errors to nm: lat_err_nm = 1000*locs_3d[:,3]; ax_err_nm = 1000*locs_3d[:,4]; n_views = locs_3d[:,5].

Quality mask (visualization/reporting filter): keep = (lat_err_nm < show_max_lateral_err) AND (n_views > show_min_view_count), where thresholds come from `meta` with defaults show_max_lateral_err=200.0 and show_min_view_count=3. Note the view test is strict greater-than (> 3), so only molecules with 4+ views pass.

Metrics:
- `n_locs_3d_filtered` = int(sum(keep))
- `median_lateral_err_nm` = float(median(lat_err_nm[keep]))
- `median_axial_err_nm` = float(median(ax_err_nm[keep]))

Write `outputs/{case_id}.npz` via `numpy.savez` with these exact keys, storing `n_locs_3d_filtered` as an integer scalar (e.g. `np.int64(...)`) and the two medians as float scalars. Ensure all three are finite. Reference values on the public case: n_locs_3d_filtered=21727, median_lateral_err_nm~51.79, median_axial_err_nm~55.33 (out of ~24,543 total). Create the `outputs/` directory if absent.
