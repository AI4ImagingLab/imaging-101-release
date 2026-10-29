"""Reproduce author-defined new noiseless Taylor-vortex observations.
Adapted from Sample_TaylorVortex2D.m, GPL-3.0-or-later, Triad2024.
This is maintainer-only synthetic generation, not an existing measured dataset.
Native grid1001, domain[-3L0,3L0], x-fast/y-second, t=.3s, H=nu=1e-6,
rho1000, L0=.001m, P0=.001Pa. Finite-difference convective acceleration
is author preprocessing, not an exact analytic pressure derivative; truncation
and VTKfloat32 quantization are retained. No random noise or hidden seed.
"""
import numpy as np

def generate(n=1001):
    """Return normalized float32 gradient(2,n,n), coords(n), true Pa field(n,n)."""
    coordinate=np.linspace(-3.,3.,n)
    x,y=np.meshgrid(coordinate*.001,coordinate*.001)
    radius=np.hypot(x,y);angle=np.arctan2(y,x)
    t=.3;nu=1e-6;H=1e-6;rho=1000.;p0=.001
    tangential=H/(8*np.pi)*radius/(nu*t*t)*np.exp(-radius*radius/(4*nu*t))
    u=-tangential*np.sin(angle);v=tangential*np.cos(angle)
    delta=coordinate[1]*.001-coordinate[0]*.001
    uy,ux=np.gradient(u,delta,delta);vy,vx=np.gradient(v,delta,delta)
    gradient=np.stack([-rho*(u*ux+v*uy)/p0*.001,-rho*(u*vx+v*vy)/p0*.001]).astype(np.float32)
    pressure=-rho*H**2/(64*np.pi**2*nu*t**3)*np.exp(-radius*radius/(2*nu*t))
    pressure-=pressure[1,1]
    return gradient,coordinate.astype(np.float32),pressure

if __name__=="__main__":
    from pathlib import Path
    g,q,p=generate();Path("data").mkdir(exist_ok=True);Path("evaluation").mkdir(exist_ok=True)
    np.savez_compressed("data/raw_data.npz",pressure_gradient=g[None],coordinate=q[None])
    np.savez_compressed("evaluation/ground_truth.npz",reconstruction=p[None])
