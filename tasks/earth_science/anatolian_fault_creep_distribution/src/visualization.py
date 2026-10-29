"""Scientific notebook plotting and independently defined comparison helpers."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm


def reference_metrics(estimate,reference):
    """Cosine and dynamic-range NRMSE without mean removal or scale fitting."""
    a=np.asarray(estimate).ravel();b=np.asarray(reference).ravel()
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected matching finite comparison vectors')
    norm=np.linalg.norm(a)*np.linalg.norm(b);span=np.ptp(b)
    if span<=0 or np.linalg.norm(b)<=0:raise ValueError('Degenerate reference')
    return {'ncc':float(a@b/norm) if norm>0 else 0.,
            'nrmse':float(np.sqrt(np.mean((a-b)**2))/span)}


def signed_norm(values,fallback=1e-14):
    """Symmetric signed normalization, including the all-zero case."""
    extent=max(float(np.max(abs(values))),fallback)
    return TwoSlopeNorm(vcenter=0.,vmin=-extent,vmax=extent)


def fault_panel(ax,triangles,values,title,norm=None,cmap='viridis'):
    """All1292native patches, east-depth projection; no interpolation/crop."""
    vertices=np.asarray(triangles)[:,:,[0,2]]
    collection=PolyCollection(vertices,array=np.asarray(values),cmap=cmap,norm=norm,
                              edgecolors='none',rasterized=True)
    ax.add_collection(collection);ax.autoscale_view();ax.invert_yaxis()
    ax.set(xlabel='East (km)',ylabel='Depth (km)',title=title)
    return collection


def scatter_panel(ax,xy,values,title,norm=None,cmap='viridis',size=8):
    """Native irregular observation positions, with physical coordinate axes."""
    artist=ax.scatter(xy[:,0],xy[:,1],c=values,s=size,norm=norm,cmap=cmap,
                      linewidths=0,rasterized=True)
    ax.set(xlabel='East (km)',ylabel='North (km)',title=title)
    ax.set_aspect('equal',adjustable='box')
    return artist


def save_figure(figure,directory,stem):
    """Export same actually computed figure to paired PNG/SVG/PDF privately."""
    if directory is None:return
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for extension in ['png','svg','pdf']:
        figure.savefig(directory/(stem+'.'+extension),dpi=170,bbox_inches='tight')


def vector_extent(ax,xy,vectors,scale,margin=.06):
    """Include every native tail and displayed tip; vectors share xy units/scale."""
    points=np.concatenate([np.asarray(xy),np.asarray(xy)+np.asarray(vectors)/scale])
    low=points.min(axis=0);high=points.max(axis=0)
    padding=np.maximum(high-low,1.)*margin
    ax.set_xlim(low[0]-padding[0],high[0]+padding[0])
    ax.set_ylim(low[1]-padding[1],high[1]+padding[1])
