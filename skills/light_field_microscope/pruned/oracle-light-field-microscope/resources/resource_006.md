# Shift-variant PSF kernel bank

Constants (fixed, not in params): theta_samples = 128, kernel_tol = 0.005.

## Native PSF stack
for depth in depths + offset_fobj:  psf = Debye_field(depth, ...)  # over yspace,xspace

## integer_shift(image, sr, sc): zero-filled edge-clipped shift
sr, sc = round(sr), round(sc); out = zeros_like(image)
src_r0=max(0,-sr); src_r1=H-max(0,sr); dst_r0=max(0,sr); dst_r1=dst_r0+(src_r1-src_r0)
(analogous for cols); out[dst...] = image[src...]; return out

## Kernel loop
row_c = tex_nnum[0]//2; col_c = tex_nnum[1]//2
for lr in range(tex_nnum[0]):
  for lc in range(tex_nnum[1]):
    sr = lr-row_c; sc = lc-col_c
    for psf_ref in psf_stack:
      f = integer_shift(psf_ref, sr, sc)
      f = f * mla_array
      s = prop_to_sensor(f, sensor_res, mla_to_sensor, WaveLength)
      s = integer_shift(s, -sr, -sc)
      kern = abs(s)**2
      if kern.max() > 0:
        kern[kern < kern.max()*kernel_tol] = 0
        ksum = kern.sum()
        if ksum > 0: kern = kern/ksum
      store kern.astype(float64)

Adjoint kernel = kern[::-1, ::-1] (used by the backward projector).