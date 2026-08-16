# Camera geometry constants

Inputs (um / dimensionless): M, NA, ftl, WaveLength, n, lensPitch, fm, pixelPitch, newSpacingPx.

```
spacing_px      = lensPitch / pixelPitch
fobj            = ftl / M
delta_ot        = ftl + fobj
wave_number k   = 2*pi*n / WaveLength
objective_radius= fobj * NA
tube_radius     = objective_radius
dof             = fobj
offset_fobj     = dof - fobj            # = 0 for this config
tube_to_mla     = ftl                   # fall back to ftl when configured value is 0
mla_to_sensor   = fm                    # fall back to fm  when configured value is 0
ulens_radius    = tube_radius * mla_to_sensor / tube_to_mla
new_pixel_pitch = lensPitch / newSpacingPx
mla_magnification = M
```

Worked values for the benchmark config (M=20, NA=0.5, ftl=165000, n=1.0, WaveLength=0.525, lensPitch=150, fm=3000, pixelPitch=6.5, newSpacingPx=15):

```
spacing_px = 23.0769..., fobj = 8250, delta_ot = 173250, k = 11.96797,
objective_radius = 4125, ulens_radius = 75, offset_fobj = 0, new_pixel_pitch = 10
```

Validity guard: this reduced model only supports regular-grid, single-focus, plenoptic-1 arrays.