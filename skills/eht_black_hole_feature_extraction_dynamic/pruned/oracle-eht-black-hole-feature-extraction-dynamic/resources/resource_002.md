# NUFFT forward model reference

Complex multiply of real-valued [real; imag] channel tensors:
```
xy_real = x_real*y_real - x_imag*y_imag
xy_imag = x_real*y_imag + x_imag*y_real
```

Forward (per image batch, eps = 1e-16):
```
x = images.float().unsqueeze(-1)
x = cat([x, zeros_like(x)], dim=-1)    # add imaginary channel
x = x.unsqueeze(0)
kdata = nufft_ob(x, ktraj_vis)          # KbNufft(im_size=(npix,npix), numpoints=3)
kdata = kdata.transpose(-1, -2)
vis   = complex_mul(kdata, pulsefac_vis).squeeze(0)   # (B, 2, M)
visamp = sqrt(vis[:,0]**2 + vis[:,1]**2 + eps)
```

Closure phase (degrees):
```
for leg in (0,1,2): angL = atan2(vis_sel[:,1], vis_sel[:,0])
cphase = (sign0*ang0 + sign1*ang1 + sign2*ang2) * 180/pi
```

Log closure amplitude:
```
amp_ab = sqrt(vis_ab[:,0]**2 + vis_ab[:,1]**2 + eps)
logcamp = log(amp_ij) + log(amp_kl) - log(amp_ik) - log(amp_jl)
```
Index selection uses the cphase_ind_list (3 legs) and camp_ind_list (4 legs)
from the closure-extraction step.
