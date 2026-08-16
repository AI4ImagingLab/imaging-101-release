Lens deflection and ray-shooting (numpy only).

```python
import numpy as np

def ellipticity2phi_q(e1, e2):
    phi = np.arctan2(e2, e1) / 2.0
    c = min(np.sqrt(e1**2 + e2**2), 0.9999)
    q = (1.0 - c) / (1.0 + c)
    return phi, q

def spep_deflection(x, y, theta_E, gamma, e1, e2, center_x=0.0, center_y=0.0):
    phi_G, q = ellipticity2phi_q(e1, e2)
    gamma = max(1.4, min(2.9, float(gamma)))
    q = max(0.01, q)
    phi_E = theta_E * q
    E = phi_E / (((3.0 - gamma) / 2.0) ** (1.0 / (1.0 - gamma)) * np.sqrt(q))
    eta = 3.0 - gamma
    cos_phi, sin_phi = np.cos(phi_G), np.sin(phi_G)
    dx = x - center_x; dy = y - center_y
    xt1 = cos_phi * dx + sin_phi * dy
    xt2 = -sin_phi * dx + cos_phi * dy
    xt2dq2 = xt2 / (q * q)
    P2 = xt1 * xt1 + xt2 * xt2dq2
    a = np.where(P2 > 0, P2, 1e-6) if hasattr(P2, '__len__') else max(P2, 1e-6)
    fac = (1.0 / eta) * (a / (E * E)) ** (eta / 2.0 - 1.0) * 2.0
    fx_p = fac * xt1; fy_p = fac * xt2dq2
    return cos_phi * fx_p - sin_phi * fy_p, sin_phi * fx_p + cos_phi * fy_p

def shear_deflection(x, y, g1, g2):
    return g1 * x + g2 * y, g2 * x - g1 * y

def ray_shoot(x, y, kwargs_spemd, kwargs_shear):
    ax1, ay1 = spep_deflection(x, y, **kwargs_spemd)
    ax2, ay2 = shear_deflection(x, y, kwargs_shear['gamma1'], kwargs_shear['gamma2'])
    return x - ax1 - ax2, y - ay1 - ay2
```
