Fourier-domain formulas

Use PyTorch or NumPy FFT conventions without manual normalization changes.

```
fx = fftfreq(nx) / res_x
fy = fftfreq(ny) / res_y
FY, FX = meshgrid(fy, fx, indexing='ij')
c_gamma = sqrt(max(1 - FX^2 - FY^2, 1e-8))
kz = c_gamma * 2*pi*res_z
eva_mask = exp(min((c_gamma - 0.2)*5, 0))
cutoff = sqrt(1 - (NA_obj/n0)^2)
t = exp(min(c_gamma - cutoff, 0.01) * 10000)
pupil = t / (1 + t)
```

The pupil is a steep smooth cutoff rather than a simple binary radius threshold. Keeping this exact smooth cutoff helps match the dark-field direct-beam rejection behavior.