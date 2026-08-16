# Probe size / calibration reference

Threshold-sweep probe sizing (vacuum probe, shape (Qx,Qy)):

```
thresh_vals = linspace(0.01, 0.70, 100)
dp_max = probe.max()
r_vals[i] = sqrt( sum(probe > dp_max*thresh_vals[i]) / pi )
dr = gradient(r_vals)
stable = (dr <= 0) & (dr >= 2*median(dr))
probe_radius = mean(r_vals[stable])           # ~49.2 px
thresh_mean  = mean(thresh_vals[stable])
mask = probe > dp_max*thresh_mean
# center of mass over integer index grids (row=x, col=y)
qx0 = sum(row_idx * probe*mask)/sum(probe*mask)   # ~96.9
qy0 = sum(col_idx * probe*mask)/sum(probe*mask)   # ~88.3
```

Reciprocal + object sampling:

```
dk_mrad  = convergence_semiangle_mrad / probe_radius     # ~0.459 mrad/px
dk_invA  = dk_mrad*1e-3 / wavelength_A                    # ~0.02333 1/A
dx_obj_A = 1.0 / (Q * dk_invA)   with Q=192               # ~0.223 A/px
```

Interpretation: the ptychographic object is sampled at the fine detector-defined pixel size dx_obj_A, far finer than the 2.39 A scan step. Scan positions are therefore separated by R_pixel_size_A / dx_obj_A object pixels (~10.7 px).