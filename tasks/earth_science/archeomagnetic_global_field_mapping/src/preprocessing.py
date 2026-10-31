"""Prepare observed archeomagnetic D/I/F and geodetic coordinates."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def geodetic_to_geocentric(latitude, height=0.):
    """Latitude radians, height m -> colatitude radians, radius km, cd, sd."""
    a,b=6378137.,6356752.
    f=(a-b)/a; e2=f*(2-f)
    rn=a/np.sqrt(1-e2*np.sin(latitude)**2)
    eta=e2*rn/(rn+height)
    glat=np.arctan((1-eta)*np.tan(latitude))
    radius=(rn+height)*np.sqrt(1-eta*(2-eta)*np.sin(latitude)**2)/1000
    return np.pi/2-glat,radius,np.cos(latitude-glat),np.sin(latitude-glat)

def load_observations(path=None):
    """Return arrays time, coords(N,5), value, sigma, kind (4=F,5=I,6=D)."""
    groups=[]
    with np.load(path or ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        for key,kind in [('intensity',4),('inclination',5),('declination',6)]:
            a=z[key][0].copy(); value=a[:,3]; sigma=a[:,4]
            if kind>4: value=np.deg2rad(value); sigma=np.deg2rad(sigma)
            theta,r,cd,sd=geodetic_to_geocentric(np.deg2rad(a[:,1]))
            coords=np.column_stack((theta,np.deg2rad(np.where(a[:,2]>180,a[:,2]-360,a[:,2])),r,cd,sd))
            groups.append((a[:,0],coords,value,sigma,np.full(len(a),kind,dtype=np.int64)))
    return tuple(np.concatenate([g[i] for g in groups]) for i in range(5))
