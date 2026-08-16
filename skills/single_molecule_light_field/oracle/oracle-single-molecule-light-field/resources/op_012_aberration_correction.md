# Aberration correction (two-pass)

Pass-1 fit params (fit_params_aberration):
  frames 1..min(1000, max_frame)
  disparity_max=5.0, disparity_step=0.1, dist_search=0.5
  angle_tolerance=2.0 deg, threshold=1.0 um, min_views=3, z_calib=None

View-error accumulation (aberration_params): axial_window=1.0, photon_threshold=1, min_views=3.
Select fits with points>min_views and |z|<axial_window and photon_count>photon_threshold.
forward residual per view: err = xy - uv/rho - xy0 - z*alpha  (see backward-OLS helper forward_model_error).
correction[view] = mean residual (dx,dy) over selected fits; count 0 -> 0.

Apply: corrected X -= dx, Y -= dy per matching (u,v) view.

Reference correction table (u, v, dx, dy, n) approx:
 (-0.6452, -0.0090, -0.0250,  1.0288, 832)
 (-0.3304,  0.5542,  0.9407,  0.6159, 845)
 (-0.3148, -0.5632, -0.8700,  0.4866, 956)
 ( 0.0000,  0.0000,  0.0713, -0.0304, 980)
 ( 0.3148,  0.5632,  0.9847, -0.5123, 884)
 ( 0.3304, -0.5542, -0.9295, -0.4738, 958)
 ( 0.6452,  0.0090,  0.0087, -1.0089, 897)
Mean correction magnitude ~0.913 um.
