# Debye objective PSF (pre-MLA field)

Inputs: depth_um, camera constants, yspace/xspace grids, theta_samples (=128).

```
alpha             = asin(NA/n)
sin_alpha_half_sq = sin(alpha/2)**2
demag             = 1/M
d1                = dof - depth_um
u                 = 4*k*depth_um*sin_alpha_half_sq          # Lommel defocus
koi               = demag/((d1*WaveLength)**2) * exp(-1j*u/(4*sin_alpha_half_sq))

yy, xx  = meshgrid(yspace, xspace, indexing='ij')
radial  = sqrt(yy**2 + xx**2) / M
v       = k * radial * sin(alpha)

theta   = linspace(0, alpha, theta_samples)          # shape (T,), broadcast over (T,H,W)
phase   = exp(1j*(u/2)*sin(theta/2)**2 / sin_alpha_half_sq)
pref    = sqrt(cos(theta)) * (1+cos(theta)) * sin(theta)
barg    = (sin(theta)/sin(alpha)) * v                # broadcast
integrand = pref * phase * j0(barg)                  # scipy.special.j0
integral  = trapz(integrand, x=theta, axis=0)        # integrate over theta
U_mla     = koi * integral                           # complex field
```

Notes: sqrt(cos theta)*(1+cos theta) is the aplanatic apodization; the extra sin(theta) is the solid-angle weight. Field is complex; downstream squares its magnitude after MLA + propagation.