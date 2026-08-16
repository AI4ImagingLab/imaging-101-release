# Grouping pseudocode

Inputs: seed (13,), in_points (M,13), rho_scaling, params {disparity_max, disparity_step, dist_search, angle_tolerance}.

angle_tol = deg2rad(angle_tolerance)
points = in_points without rows sharing seed's (u,v)
# angle test
du=u-su; dv=v-sv; dx=x-sx; dy=y-sy
keep = (abs(arctan2(dy,dx)-arctan2(dv,du)) < angle_tol) | (abs(...) > pi-angle_tol)
# disparity test (recompute du.. on survivors)
dxy=hypot(dx,dy); duv=hypot(du,dv)
disparity = where(duv>0, (dxy - duv/rho)/duv, 0)
range_z = arange(-disparity_max, disparity_max + step/2, step)
count[z] = sum( disparity in (z-dist_search, z+dist_search] )
best_z = range_z[argmax(count)]
keep = disparity in (best_z-dist_search, best_z+dist_search]
# duplicate views
unique (u,v): single-count views added directly; multi-count views -> pick point j
  minimising sqrt(sum(get_backward_model(candidates+point_j)[1]**2)).
view_count = number of unique companion views.
# degeneracy: if view_count==1 and (var(u,ddof=1)<0.1 or var(v,ddof=1)<0.1) -> reject.

Interpretation: (dxy - duv/rho) removes the fixed inter-view pupil offset, leaving the depth-proportional term; dividing by duv normalises to a per-view z estimate that should cluster for a true multi-view molecule.
