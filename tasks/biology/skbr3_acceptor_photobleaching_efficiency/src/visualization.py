"""Actual numeric comparisons and traceable science figure exports."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt

PIXEL_UM=.0558035713935203

def comparison_metrics(estimate,reference,mask):
    x=np.asarray(estimate,dtype=np.float64)[mask];y=np.asarray(reference,dtype=np.float64)[mask]
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(y)==0 or np.linalg.norm(y)==0:raise ValueError('invalid metric arrays')
    nx=np.linalg.norm(x)
    return {'ncc':float(x@y/nx/np.linalg.norm(y)) if nx else 0.,'dynamic_range_nrmse':float(np.sqrt(np.mean((x-y)**2))/np.ptp(y)),'max_abs':float(np.max(abs(x-y)))}

def image_panel(ax,array,title,unit='ADU',signed=False,mask=None,limits=None):
    a=np.asarray(array)
    shown=np.ma.array(a,mask=~mask) if mask is not None else a
    if limits is None:
        if signed:
            bound=float(np.max(abs(a[mask] if mask is not None else a))) or 1.
            limits=(-bound,bound)
        else:limits=(0,float(np.max(a)) or 1.)
    im=ax.imshow(shown,cmap='RdBu_r' if signed else 'viridis',vmin=limits[0],vmax=limits[1],origin='upper',extent=(0,a.shape[1]*PIXEL_UM,a.shape[0]*PIXEL_UM,0))
    ax.set(title=title,xlabel='x (µm)',ylabel='y (µm)');plt.colorbar(im,ax=ax,fraction=.046,pad=.04,label=unit)
    return im

def save_figure(fig,name,numeric,metadata,export_dir):
    """Export PNG/SVG/PDF plus exact plotted arrays and hashed provenance JSON."""
    folder=Path(export_dir);folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    for suffix in ['png','svg','pdf']:
        path=folder/(name+'.'+suffix);fig.savefig(path,dpi=150,bbox_inches='tight');paths.append(path)
    npz=folder/(name+'.npz');np.savez_compressed(npz,**{k:np.asarray(v) for k,v in numeric.items()});paths.append(npz)
    record=dict(metadata);record.update(name=name,numeric_keys={k:list(np.shape(v)) for k,v in numeric.items()},files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (folder/(name+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf8')
    return record
