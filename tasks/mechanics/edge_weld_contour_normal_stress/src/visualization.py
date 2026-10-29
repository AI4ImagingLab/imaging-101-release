"""Numerically identified scientific plots; backend selection belongs to callers."""
from pathlib import Path
import hashlib,json
import numpy as np

def comparison_metrics(estimate,reference):
    a=np.asarray(estimate,dtype=np.float64).ravel();b=np.asarray(reference,dtype=np.float64).ravel()
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.ptp(b)==0 or np.linalg.norm(b)==0:raise ValueError('Invalid metric arrays')
    return dict(ncc=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)>0 else 0.,nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)),max_abs_mpa=float(np.max(abs(a-b))))

def stress_panel(ax,xy,values,title,limit=None):
    v=np.asarray(values).ravel()
    scale=float(np.max(abs(v))) if limit is None else float(limit)
    artist=ax.scatter(xy[:,0],xy[:,1],c=v,s=7,cmap='RdBu_r',vmin=-scale,vmax=scale)
    ax.set_title(title);ax.set_aspect('equal');ax.set_xlabel('author x [mm]');ax.set_ylabel('author y [mm]')
    return artist

def export_figure(fig,directory,name,numeric,caption):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if not numeric or any(np.asarray(v).dtype.kind not in 'biufc' for v in numeric.values()):raise ValueError('Numeric figure sources required')
    np.savez_compressed(directory/(name+'.npz'),**numeric)
    for extension in ('png','svg','pdf'):fig.savefig(directory/(name+'.'+extension),dpi=160,bbox_inches='tight')
    record=dict(name=name,caption=caption,files={})
    for extension in ('png','svg','pdf','npz'):
        path=directory/(name+'.'+extension)
        record['files'][extension]=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (directory/(name+'.json')).write_text(json.dumps(record,indent=2),encoding='utf8')
    return record
