# Sampling grid and PSF FOV sizing

## Sampling
- ns = newSpacingPx = 15
- sensor_res = lensPitch/ns = 10 um/px (both axes)
- nnum = ns + (1 - ns%2) -> odd; tex_nnum = nnum = 15
- tex_res = (sensor_res/(scale*M), sensor_res/(scale*M), depthStep) with scale=1 -> (0.5, 0.5, 10) um
- depths = arange(z_lo, z_hi + 0.5*depthStep, depthStep)  # singleton for benchmark cases

## PSF field-of-view (lenslets)
max_depth = depths[argmax(|depths + offset_fobj|)] + offset_fobj
zobj = fobj - (max_depth - offset_fobj); if isclose(zobj,fobj) or isclose(zobj,dof): zobj += 1e-5*fobj
z1 = zobj*fobj/(zobj - fobj)
tube_r = objective_radius*delta_ot*abs(1/z1 - 1/delta_ot)
z2 = ftl*(delta_ot - z1)/(delta_ot - z1 - ftl)
blur_r = tube_r*tube_to_mla*abs(1/z2 - 1/tube_to_mla)
psf_size_lenslets = ceil(blur_r/lensPitch) + 2

Worked psf_size_lenslets by depth (this config): z=0 ->3, z=-30 ->4, z=-50 ->6, z=-70 ->7, z=-100 ->9.
img_half = max(nnum*psf_size_lenslets, 2*nnum).

## Axes
yspace = xspace = sensor_res * arange(-img_half, img_half+1)
y_ml_space = x_ml_space = sensor_res * arange(-(nnum//2), nnum//2 + 1)   # length nnum

## Masks
sens_mask: disk over the nnum-local grid, radius = ulens_radius (microlens aperture).
tex_mask: disk over the nnum-local grid, radius = tex_nnum*tex_res/2, then a regularization
pass that fills cells whose neighborhood count is 0 and clears cells whose count is 2, so the
on-grid lenslet-center stamp pattern is exactly single-covered.