"""Instrument polar readings to normalized Coe thermal Arai observations.

Adapted mathematical conventions from PmagPy SPD (BSD-3-Clause), Lori
Jonestrask / Ron Shaar. Source revision and full notices: test/validation.md.
"""
from pathlib import Path
import numpy as np

NAMES=('A-3-3','BR06-4F','HEL2-2d','KF-3-1','LV6C3A')

def load_observations(path):
    """Load public NPZ arrays without pickle; return independent array dictionary."""
    with np.load(Path(path),allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def polar_vectors(magnitude,declination,inclination):
    """Return (...,3) Cartesian readings, x north, y east, z down."""
    d=np.deg2rad(declination);i=np.deg2rad(inclination)
    return np.asarray(magnitude)[...,None]*np.stack([np.cos(d)*np.cos(i),np.sin(d)*np.cos(i),np.sin(i)],axis=-1)

def arai_observations(raw,index):
    """Pair the full specimen's native Coe zero/in-field steps, preserving order.

    Returns temperature Celsius, normalized gained TRM x, residual NRM y,
    normalized zero-field Cartesian vectors, and all native vectors/codes.
    Check steps stay in the diagnostic acquisition, outside reconstruction pairs.
    """
    n=int(raw['count'][0,index]);temp=raw['temperature'][0,index,:n];code=raw['treatment'][0,index,:n]
    vec=polar_vectors(raw['magnitude'][0,index,:n],raw['declination'][0,index,:n],raw['inclination'][0,index,:n])
    scale=np.linalg.norm(vec[0]);zero=np.flatnonzero(code==0)
    t=[];x=[];y=[];z=[]
    for row in zero:
        temperature=temp[row]
        matches=np.flatnonzero((temp==temperature)&(code==1))
        if temperature==0:
            trm=0.
        elif len(matches)==1:
            trm=np.linalg.norm(vec[matches[0]]-vec[row])/scale
        else:
            raise ValueError('A thermal zero-field step must have exactly one in-field partner')
        t.append(temperature);x.append(trm);y.append(np.linalg.norm(vec[row])/scale);z.append(vec[row]/scale)
    return dict(temperature=np.asarray(t),x=np.asarray(x),y=np.asarray(y),vectors=np.asarray(z),native_vectors=vec/scale,native_temperature=temp,native_code=code)

def check_observations(arai):
    """Measured repeat pTRM and tail points, including their prior heating temperature."""
    t=arai['temperature'];x=arai['x'];y=arai['y'];vec=arai['native_vectors'];temp=arai['native_temperature'];code=arai['native_code']
    result={}
    for kind,number in [('ptrm',2),('tail',3)]:
        points=[];temperatures=[];starting=[]
        for row in np.flatnonzero(code==number):
            match=np.flatnonzero(t==temp[row])
            if not len(match) or row==0 or temp[row-1] not in t:continue
            j=match[0]
            if kind=='ptrm':point=(np.linalg.norm(vec[row]-vec[row-1]),y[j])
            else:point=(x[j],np.linalg.norm(vec[row]))
            points.append(point);temperatures.append(temp[row]);starting.append(temp[row-1])
        result[kind]=dict(points=np.asarray(points,dtype=np.float64).reshape(-1,2),temperature=np.asarray(temperatures),starting=np.asarray(starting))
    return result
