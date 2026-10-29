"""Actual IUE curves; preserve original ordering, units and sample centering."""
from pathlib import Path
import numpy as np

def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as z:a=z['observations'][0].copy()
    if a.shape!=(3,60,3) or not np.isfinite(a).all() or np.any(a[:,:,2]<=0):raise ValueError('Expected three complete60epoch light curves with positive errors')
    return a

def prepare(observations):
    rows=[a[np.argsort(a[:,0])] for a in observations];means=np.array([np.mean(a[:,1]) for a in rows])
    t=np.concatenate([a[:,0] for a in rows]);y=np.concatenate([a[:,1]-m for a,m in zip(rows,means)]);e=np.concatenate([a[:,2] for a in rows]);ids=np.repeat(np.arange(1,4),60)
    order=np.argsort(t);t,y,e,ids=t[order],y[order],e[order],ids[order]
    return dict(time=t,value=y,error=e,channel=ids,means=means,design=np.eye(3)[ids-1],baseline=float(np.ptp(t)))
