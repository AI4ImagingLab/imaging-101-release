# SimpleCrescent renderer reference

Input: params tensor shape (B, 4), each column in [0, 1].

Grid (constructed once for a given npix):
```
gap = 1.0 / npix
xs = torch.arange(-1 + gap, 1, 2 * gap)          # length npix
grid_y, grid_x = torch.meshgrid(-xs, xs, indexing='ij')
grid_r = torch.sqrt(grid_x**2 + grid_y**2)
grid_theta = torch.atan2(grid_y, grid_x)
```

Feature mapping (half_fov = 0.5 * fov_uas):
```
r     = r_range[0]/half_fov + p[:,0]*(r_range[1]-r_range[0])/half_fov
sigma = width_range[0]/half_fov + p[:,1]*(width_range[1]-width_range[0])/half_fov
s     = p[:,2]
eta   = 181/180 * pi * (2*p[:,3] - 1)      # radians
```
(r, sigma, s, eta broadcast as (B,1,1).)

Image:
```
ring = exp(-0.5 * (grid_r - r)**2 / sigma**2)
S    = 1 + s * cos(grid_theta - eta)
crescent = S * ring
crescent = crescent / (crescent.sum(dim=(-1,-2), keepdim=True) + 1e-4)
```

Physical position angle in degrees for a unit param p3:
```
pa_deg = 181/180 * 180 * (2*p3 - 1) = 181 * (2*p3 - 1)
```
Convention: pa measured as atan2(grid_y, grid_x); 0 deg -> +x (increasing column),
+90 deg -> +y (decreasing row). Periodic modulo 360.

Diameter and width (for completeness of the 4-param physical vector):
```
diameter_uas = 2 * (r_range[0] + p0 * (r_range[1] - r_range[0]))
width_uas    = width_range[0] + p1 * (width_range[1] - width_range[0])
asymmetry    = p2
```
