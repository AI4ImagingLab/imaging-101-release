# Derived optical quantities (reference values)

Inputs (case params, focal lengths in mm, lengths in um):
- num_aperture NA = 1.27
- focal_length_obj_lens f_obj = 200/60 = 3.3333333 mm
- focal_length_tube_lens f_tube = 200 mm
- focal_length_fourier_lens f_fourier = 175 mm
- focal_length_mla f_mla = 175 mm
- mla_lens_pitch = 2390 um
- pixel_size_camera = 16 um
- ref_idx_medium n = 1.33

Derived:
| quantity | formula | value |
|---|---|---|
| bfp_radius | 1000*NA*f_obj*(f_fourier/f_tube) | 3704.16667 um |
| magnification | (f_tube/f_obj)*(f_mla/f_fourier) | 60.0 |
| bfp_lens_count | 2*bfp_radius/pitch | 3.099721 |
| pixel_size_sample | pixel/mag | 0.2666667 um |
| rho_scaling | mag/bfp_radius | 0.016197975 |
| mla_to_uv_scale | 2/bfp_lens_count | 0.6452193 |
| mla_to_xy_scale | mla_to_uv_scale/rho_scaling | 39.833333 um |

Note: rho_scaling maps image-plane microns -> normalised pupil (u,v). mla_to_uv_scale maps MLA lattice-spacing units -> (u,v).
