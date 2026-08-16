# SSNP forward model — end-to-end pseudocode

Precompute (once, float64 / complex128):
```
fx = fftfreq(nx)/res[0]; fy = fftfreq(ny)/res[1]
FY, FX = meshgrid(fy, fx, indexing='ij')
c_gamma = sqrt(clamp(1 - FX**2 - FY**2, min=1e-8))
kz      = c_gamma * (2*pi*res[2])
eva     = exp(clamp((c_gamma - 0.2)*5.0, max=0.0))
cutoff  = sqrt(1 - (NA/n0)**2)
mask    = exp(clamp(c_gamma - cutoff, max=0.01)*10000.0)
pupil   = (mask/(1+mask)).astype(complex128)
```

Incident field for angle m:
```
na=NA/n0; th=2*pi*m/n_angles; ca=na*cos(th); cb=na*sin(th)
ca = int(ca*nx*res[0])/(nx*res[0]); cb = int(cb*ny*res[1])/(ny*res[1])
x=arange(nx); y=arange(ny)
tilt = exp(2j*pi*cb*res[1]*y)[:,None] * exp(2j*pi*ca*res[0]*x)[None,:]
kz_in = sqrt(max(1-ca**2-cb**2,1e-8)) * (2*pi*res[2])
u=tilt; ud=tilt*(1j*kz_in)
```

P operator (dz in slice units):
```
ck=cos(kz*dz)*eva; sk=sin(kz*dz)*eva
a=fft2(u); ad=fft2(ud)
an = ck*a + (sk/kz)*ad
adn = (-kz*sk)*a + ck*ad
u=ifft2(an); ud=ifft2(adn)
```

Q operator (slice dn2d, dz=1):
```
pf = (2*pi*res[2]/n0)**2 * dz
ud = ud - pf*dn2d*(2*n0+dn2d)*u
```

Main loop + camera:
```
for iz in range(nz):
    u,ud = P(u,ud,dz=1.0)
    u,ud = Q(u,ud,dn[iz],dz=1.0)
u,ud = P(u,ud,dz=-nz/2.0)          # back to focal plane
af = fft2(u); adf = fft2(ud)
uf = ifft2((af - 1j*adf/kz)*0.5)    # forward component
phi = ifft2(fft2(uf)*pupil)
I = abs(phi)**2
```
