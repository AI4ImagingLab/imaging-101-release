"""Metrics and plotting helpers; importing this module never selects a backend."""
from pathlib import Path
import numpy as np

def map_metrics(estimate,reference):
    """Per-hour cosine NCC and own-reference-range NRMSE, no alignment."""
    x=np.asarray(estimate).reshape(25,-1);r=np.asarray(reference).reshape(25,-1)
    norms=np.linalg.norm(x,axis=1)*np.linalg.norm(r,axis=1)
    ncc=np.divide(np.sum(x*r,axis=1),norms,out=np.zeros(25),where=norms>0)
    ranges=np.ptp(r,axis=1)
    if np.any(ranges<=0):raise ValueError('Reference map range is zero')
    return ncc,np.sqrt(np.mean((x-r)**2,axis=1))/ranges

def save_figure(fig,directory,name):
    """Save actual plotted figure as PNG/SVG/PDF; return paths."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    for extension in ['png','svg','pdf']:
        path=directory/f'{name}.{extension}';fig.savefig(path,dpi=170,bbox_inches='tight');paths.append(str(path))
    return paths
