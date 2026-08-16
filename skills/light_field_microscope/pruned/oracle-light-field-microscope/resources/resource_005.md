# Rayleigh-Sommerfeld (angular spectrum) propagation

Inputs: field_at_mla (ny,nx complex), sensor_res (dy,dx) um, z = mla_to_sensor (=fm), WaveLength.

```
k  = 2*pi/WaveLength
fy = fftfreq(ny, d=sensor_res[0]); fx = fftfreq(nx, d=sensor_res[1])
fy_grid, fx_grid = meshgrid(fy, fx, indexing='ij')
argument = (1 - WaveLength**2 * (fy_grid**2 + fx_grid**2)).astype(complex128)
transfer = exp(1j * sqrt(argument) * z * k)
U_sens   = exp(1j*k*z) * ifft2( fft2(field_at_mla) * transfer )
```

Evanescent handling: casting `argument` to complex before sqrt yields imaginary sqrt where
|k_perp| > k, so `transfer` becomes a real decaying exponential that suppresses those components
(equivalent to zeroing evanescent waves). Use scipy.fft functions.