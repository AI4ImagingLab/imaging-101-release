"""Numerical summaries and traceable actual-result figure export."""
from pathlib import Path
import hashlib
import json
import numpy as np


def metrics(estimate, reference):
    x=np.asarray(estimate,dtype=np.float64).ravel();y=np.asarray(reference,dtype=np.float64).ravel()
    dr=np.ptp(y)
    if dr<=0 or np.linalg.norm(y)==0:raise ValueError('Degenerate reference')
    return {'ncc':float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))) if np.linalg.norm(x) else 0.,
            'nrmse':float(np.sqrt(np.mean((x-y)**2))/dr),
            'max_error_over_reference_range':float(np.max(abs(x-y))/dr)}


def projection_metrics(prediction, measured):
    p=np.asarray(prediction,dtype=np.float64);g=np.asarray(measured,dtype=np.float64)
    if p.shape!=g.shape or not np.isfinite(p).all() or np.any(p<0):raise ValueError('Invalid prediction')
    return {'relative_L2':float(np.linalg.norm(p-g)/np.linalg.norm(g)),
            'poisson_nll_without_factorial':float(np.sum(p-g*np.log(p+1e-11)))}


def image(ax, array, title, unit, signed=False, vmax=None):
    import matplotlib.pyplot as plt
    a=np.asarray(array)
    if signed:
        limit=max(float(np.max(abs(a))) if vmax is None else vmax,1e-12)
        im=ax.imshow(a.T,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit)
    else:
        im=ax.imshow(a.T,origin='lower',cmap='viridis',vmin=0,vmax=vmax)
    ax.set_title(title,fontsize=10);ax.set_xlabel('First displayed index');ax.set_ylabel('Second displayed index')
    cb=ax.figure.colorbar(im,ax=ax,shrink=.78,pad=.03);cb.set_label(unit,fontsize=8)
    return im


def export_figure(fig, directory, stem, arrays, caption):
    """Save PNG/SVG/PDF plus paired numeric NPZ and JSON with full hashes."""
    folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
    for value in arrays.values():
        if not np.issubdtype(np.asarray(value).dtype,np.number):raise ValueError('Numeric figure sources required')
    np.savez_compressed(folder/(stem+'.npz'),**arrays)
    for extension in ['png','svg','pdf']:
        fig.savefig(folder/(stem+'.'+extension),dpi=150,bbox_inches='tight')
    files={}
    for extension in ['png','svg','pdf','npz']:
        path=folder/(stem+'.'+extension);files[extension]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    record={'caption':caption,'files':files,'numeric_keys':{k:{'shape':list(np.asarray(a).shape),'dtype':str(np.asarray(a).dtype)} for k,a in arrays.items()}}
    (folder/(stem+'.json')).write_text(json.dumps(record,indent=2),encoding='utf8')
    return record
