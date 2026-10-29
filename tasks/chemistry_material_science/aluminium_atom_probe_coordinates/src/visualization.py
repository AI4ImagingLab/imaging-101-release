"""Local atomic geometry metrics and spatial plots; no backend selection."""
import numpy as np
import matplotlib.pyplot as plt
def metrics(a,r):
    """Cosine NCC and dynamic-range NRMSE, all coordinates in fixed event order."""
    a=np.asarray(a).ravel();r=np.asarray(r).ravel();norm=np.linalg.norm(r);span=np.ptp(r)
    if norm==0 or span==0:raise ValueError('Degenerate reference')
    na=np.linalg.norm(a);return {'ncc':float(np.dot(a,r)/(na*norm)) if na else 0.,'nrmse':float(np.sqrt(np.mean((a-r)**2))/span)}
def depth_histogram(xyz,bins=80):
    """All-event depth histogram: return integer counts and nm bin edges."""
    return np.histogram(xyz[:,2],bins=bins)
def point_view(xyz,title,ax=None):
    """3D scatter, deterministic display decimation only; returns Matplotlib axis."""
    if ax is None:ax=plt.figure(figsize=(6,5)).add_subplot(projection='3d')
    view=xyz[::max(1,len(xyz)//12000)];ax.scatter(*view.T,s=.25,c=view[:,2],cmap='viridis',rasterized=True)
    ax.set(xlabel='x / nm',ylabel='y / nm',zlabel='z / nm',title=title);return ax
