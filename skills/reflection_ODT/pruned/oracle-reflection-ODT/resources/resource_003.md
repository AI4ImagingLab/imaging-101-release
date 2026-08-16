One-angle forward model pseudocode

```
u = incident_tilt(NA, m, n_ring)
for iz in range(Nz):
    u = propagate(u, dz_layer)
    u = scatter(u, dn[iz], dz_layer)
    u = propagate(u, dz_gap)
u = -u
for iz in range(Nz-1, -1, -1):
    u = propagate(u, dz_gap)
    u = propagate(u, dz_layer)
    u = scatter(u, dn[iz], dz_layer)
u = propagate(u, -Nz*(dz_gap + dz_layer))
a = fft2(u) * pupil
u = ifft2(a)
I = abs(u)**2
```

The upward pass scatters after the layer-thickness propagation. Do not replace the reflected geometry with a single-pass transmission model.