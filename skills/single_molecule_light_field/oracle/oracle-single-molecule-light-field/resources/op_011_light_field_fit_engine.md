# Fitting engine details

Parameter containers per pass (frame_min/frame_max = -1 means resolve to data range; aberration frame_max resolves to min(1000, data_max)).

Full-dataset pass (fit_params_full):
  frame_min -> data min (1)
  frame_max -> data max (5000)
  disparity_max = 8.0
  disparity_step = 0.1
  dist_search = 0.5
  angle_tolerance = 1.0 deg
  threshold = 0.3 um   (max accepted sqrt(sum(std_err^2)))
  min_views = 2
  z_calib = 1.534      (optical Z -> physical Z, applied as locs[:,2] *= z_calib at the very end)

Frame loop invariants:
- central-first ordering: |u|<0.1 & |v|<0.1 then sort by -intensity(col7); outer sorted by -intensity; stack.
- while candidates > min_views and reps < 100.
- accept row = [x0, y0, z, mean(se_x,se_y), se_z, view_count+1, sum(grouped photons col7), frame].
- remove grouped rows from pool by exact row match; drop seed on failure.

Reference outcome on the public case:
- total 3D locs ~24,543; 2D locs used ~131,588.
- z range roughly -3.3 to +5.7 um after calibration.
- n_views histogram concentrated at 5-7 (median 6).

z_calib is applied once, after the frame loop, to column 2 only.
