"""Acquisition conventions extracted from Hamm et al., MIT license (NOTICE)."""
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates

NZ, NR = 120, 41
DZ, DR = 1.5/NZ, (1.1376-.624)/NR
SAMPLING = (DZ+DR)/2
Z = (np.linspace(-.75,.75,NZ+1)[:-1]+DZ/2)[::-1]
R = np.linspace(.624,1.1376,NR+1)[:-1]+DR/2

def load_data(path='data/raw_data.npz'):
    """Load batch-first arrays and remove only their single batch dimension."""
    with np.load(Path(path),allow_pickle=False) as f:
        return {k:f[k][0] for k in f.files}

def interpolate_equilibria(psi, radial, vertical):
    """Native (R,Z,time) magnetic flux -> (time,120,41), quadratic spline."""
    rr=(R-radial.min())/np.mean(np.diff(radial))
    zz=-(Z-vertical.max())/np.mean(np.diff(vertical))
    coords=np.stack(np.meshgrid(zz,rr,indexing='ij')).reshape(2,-1)
    return np.array([map_coordinates(psi[:,:,i].T[::-1],coords,order=2,mode='nearest',prefilter=True).reshape(NZ,NR) for i in range(psi.shape[2])])

def prepare(data, equilibrium='measured'):
    """Return normalized geometry, power observations, matched flux and vessel mask."""
    channels=data['good_channels'].astype(int)-1
    scale=1/data['geometry'].max()
    matrix=(data['geometry']*scale)[channels].reshape(len(channels),-1)
    values=(data['bolometer']*(data['etendue'][None,:]*4*np.pi))*scale
    flux=interpolate_equilibria(data[equilibrium+'_flux'],data[equilibrium+'_r'],data[equilibrium+'_z'])
    idx=np.abs(data[equilibrium+'_time'][None,:]-data['time'][:,None]).argmin(axis=1)
    return matrix,values[:,channels],flux[idx],data['vessel_mask'].astype(bool)
