# Euler-Rodrigues rotation and warp

Unit axis `[ax, ay, az]`, angle array `phi` (here `phi = -theta_rot`):
```
a = cos(phi/2)
b = -ax*sin(phi/2)
c = -ay*sin(phi/2)
d = -az*sin(phi/2)
aa,bb,cc,dd = a*a,b*b,c*c,d*d
bc,ad,ac,ab,bd,cd = b*c,a*d,a*c,a*b,b*d,c*d
R = [[aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)],
     [2*(bc-ad),   aa+cc-bb-dd, 2*(cd+ab)],
     [2*(bd+ac),   2*(cd-ab),   aa+dd-bb-cc]]
```
R has shape `(3, 3, *batch)` when `phi` is batched.

Warp:
```
t_M       = (t_frame - t_start_obs) + t_geo - t_injection   # GM_c3 = 1
theta_rot = t_M * Omega
theta_rot = where(t_M < 0, NaN, theta_rot)
R         = rodrigues(axis=[0,0,1], phi=-theta_rot)
warped    = einsum('ij...,j...->...i', R, coords)           # (*spatial, 3)
```
