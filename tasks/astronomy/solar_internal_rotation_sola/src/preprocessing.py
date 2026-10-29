"""Known solar calibration and complete measured rotational splittings."""
from pathlib import Path
import numpy as np

def load_data(path='data/raw_data.npz'):
    """Return all public arrays with the singleton batch dimension removed."""
    with np.load(Path(path),allow_pickle=False) as z:
        data={k:z[k][0].copy() for k in z.files}
    if data['splittings'].shape!=(29617,) or data['mesh'].shape!=(5151,3):raise ValueError('Expected complete native observation and mesh')
    if any(not np.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite input')
    return data

def quadrature(data):
    """Native r-weighted meridional quadrature, sqrt and inverse sqrt."""
    w=data['mesh'][:,0]*data['mesh'][:,2]
    sqrt=np.sqrt(w);inverse=np.ones_like(w);np.divide(1.,sqrt,out=inverse,where=w!=0)
    return w,sqrt,inverse

def cubic_interpolate(x,y,value):
    """Native DLIR four-point Lagrange interpolation including endpoint rule."""
    x=np.asarray(x);y=np.asarray(y)
    exact=np.flatnonzero(x==value)
    if len(exact):return float(y[exact[0]])
    n=int(np.searchsorted(x,value))
    m=max(2,min(n,len(x)-2));start=m-2
    nodes=x[start:start+4];weights=np.empty(4)
    for j in range(4):
        other=np.arange(4)!=j
        weights[j]=np.prod(value-nodes[other])/np.prod(nodes[j]-nodes[other])
    pivot=n if n<=1 else 3 if n>=len(x)-1 else 2
    weights[pivot]=(1+weights[pivot])-np.sum(weights)
    return float(np.dot(weights,y[start:start+4]))

def radial_target_widths(data,width=.06,reference_radius=.7):
    """Known-model sound-speed scaling; dimensionless radius/width."""
    x=data['model_radius'];a1=data['model_A1'];a2=data['model_A2'];gamma=data['model_Gamma1']
    mass,radius,pressure,density=data['model_constants']
    cs=np.empty_like(x);cs[1:]=x[1:]*np.sqrt(a1[1:]/a2[1:])
    cs[0]=np.sqrt(radius*pressure*gamma[0]/(6.67232e-8*mass*density))
    reference=cubic_interpolate(x,cs,reference_radius)
    radii=data['mesh'][::51,0]
    return np.array([width*cubic_interpolate(x,cs,r)/reference for r in radii])
