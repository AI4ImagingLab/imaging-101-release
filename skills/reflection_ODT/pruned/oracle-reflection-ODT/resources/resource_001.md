Normalized units

The physical voxel sizes in micrometers are not used directly in phase and propagation formulas. First normalize each component by background index over wavelength:

`res[j] = res_um[j] * n0 / wavelength_um`

For the public case:

- `dx_norm = 0.274 * 1.54 / 0.632 = 0.6677848101265823`
- `dy_norm = 0.274 * 1.54 / 0.632 = 0.6677848101265823`
- `dz_norm = 1.0 * 1.54 / 0.632 = 2.4367088607594938`

Use these normalized spacings in `fftfreq(...)/res[x or y]`, in `kz = c_gamma * 2*pi*res[z]`, and in the scatter phase factor `2*pi*res[z]/n0*dz`.