BPM operator details

Propagation:

```
a = fft2(u)
prop = exp(1j * kz * dz) * eva_mask
u_new = ifft2(a * prop)
```

Scattering:

```
phase_factor = 2*pi*res_z/n0*dz
u_new = u * exp(1j * dn_slice * phase_factor)
```

The scattering phase uses RI contrast directly. A negative RI contrast gives a negative phase delay in this convention. Do not rescale `delta_n` into absolute RI and do not clip it to nonnegative values.