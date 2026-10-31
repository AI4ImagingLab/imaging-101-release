"""Count/exposure annular measurements, GPLv3 adaptation of PyProffit."""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
ROOT=Path(__file__).resolve().parents[1]
PIXEL_ARCMIN=.2491218

def load_data(path=None):
    with np.load(path or ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        return {key:z[key][0].copy() for key in z.files}

def image_center(counts,exposure):
    work=counts.copy();work[exposure<=0]=0
    smoothed=gaussian_filter(work,5)
    y,x=np.where(smoothed==smoothed.max())
    return float(x.mean()),float(y.mean())

def annular_profile(data,edges,center=None,min_exposure=.05):
    """Return rates/errors in count/s/arcmin2 using native pixel exposure."""
    counts=data['counts'];exposure=data['exposure'];bkg=data['instrument_background']
    center=image_center(counts,exposure) if center is None else center
    y,x=np.indices(counts.shape);radius=np.hypot(x-center[0],y-center[1])*PIXEL_ARCMIN
    valid=exposure>min_exposure*exposure.max()
    n=len(edges)-1;profile=np.zeros(n);error=np.zeros(n);area=np.zeros(n);effexp=np.zeros(n);ncounts=np.zeros(n)
    for i in range(n):
        mask=valid&(radius<np.round(edges[i+1],5)+.5e-5)
        if i:mask &= radius>=np.round(edges[i],5)+.5e-5
        else:mask &= radius>=0
        number=np.count_nonzero(mask)
        if not number:continue
        area[i]=number*PIXEL_ARCMIN**2
        profile[i]=np.sum((counts[mask]-bkg[mask])/exposure[mask])/area[i]
        error[i]=np.sqrt(np.sum(counts[mask]/exposure[mask]**2))/area[i]
        effexp[i]=np.mean(exposure[mask]);ncounts[i]=np.sum(counts[mask])
    return {'profile':profile,'error':error,'area':area,'exposure':effexp,'counts':ncounts,'center':np.asarray(center)}

def background_subtraction(data):
    """Fit the author's outer constant objective by its closed-form minimizer."""
    center=image_center(data['counts'],data['exposure'])
    edges=np.arange(136,dtype=float)/3
    outer=annular_profile(data,edges,center)
    centers=(edges[1:]+edges[:-1])/2
    mask=(centers>=30)&(centers<=40)&(outer['error']>0)
    weights=1/outer['error'][mask]**2
    background=float(np.sum(outer['profile'][mask]*weights)/np.sum(weights))
    background_error=float(1/np.sqrt(np.sum(weights)))
    source=annular_profile(data,data['radial_edges'],center)
    source['profile_raw']=source['profile'].copy();source['error_raw']=source['error'].copy()
    source['profile']-=background
    source['error']=np.sqrt(source['error']**2+background_error**2)
    source.update(background=background,background_error=background_error,outer_profile=outer['profile'],outer_error=outer['error'],outer_centers=centers)
    return source
