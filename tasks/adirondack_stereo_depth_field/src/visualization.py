"""Scientific review helpers; metrics never call the independent Judge."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import FOCAL_LENGTH, LEFT_CENTRE

def agreement(estimate, reference, mask, bad1_threshold=1.):
    """Return cosine NCC and range NRMSE; optional exceedance in input units.

    Default one-pixel threshold is appropriate for disparity; pass None for
    depth so a one-millimetre exceedance is not mislabeled as disparity bad1.
    """
    x=np.asarray(estimate,dtype=float)[mask];y=np.asarray(reference,dtype=float)[mask]
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(y)<=0 or np.linalg.norm(y)==0:
        raise ValueError('Nonfinite or degenerate metric input')
    nx=np.linalg.norm(x)
    metrics={'ncc':float(np.dot(x,y)/nx/np.linalg.norm(y)) if nx else 0.,
             'nrmse':float(np.sqrt(np.mean((x-y)**2))/np.ptp(y))}
    if bad1_threshold is not None:
        metrics['bad1']=float(np.mean(abs(x-y)>bad1_threshold))
    return metrics

def save_review_figure(fig, name, arrays, export_dir=None):
    """Export PNG/SVG/PDF and numeric sources when a private path is supplied."""
    if export_dir:
        path=Path(export_dir);path.mkdir(parents=True,exist_ok=True)
        for extension in ('png','svg','pdf'):
            fig.savefig(path/(name+'.'+extension),dpi=160,bbox_inches='tight')
        np.savez_compressed(path/(name+'.npz'),**arrays)
        (path/(name+'.json')).write_text(json.dumps({'figure':name,'numeric_source':name+'.npz',
          'provenance':'Actual actively executed cleaned reconstruction; original method and independent measurement explicitly labeled in panels.',
          'arrays':{k:{'shape':list(np.asarray(v).shape),'dtype':str(np.asarray(v).dtype)} for k,v in arrays.items()}},indent=2),encoding='utf8')
    return fig

def image_panel(ax, image, title, *, unit='pixel', limits=None, cmap='viridis'):
    """Plot a native-grid scalar or RGB array; no quality crop or interpolation."""
    kwargs=dict(interpolation='nearest',rasterized=True)
    if np.asarray(image).ndim==2:
        kwargs['cmap']=cmap
        if limits:kwargs.update(vmin=limits[0],vmax=limits[1])
    artist=ax.imshow(image,**kwargs)
    ax.set(title=title,xlabel='Column (pixel)',ylabel='Row (pixel)')
    if np.asarray(image).ndim==2:plt.colorbar(artist,ax=ax,label=unit,fraction=.046,pad=.03)
    return artist

def point_cloud(ax, depth, rgb, title, stride=5):
    """Display calibrated XYZ in mm; stride affects rendering only."""
    v,u=np.indices(depth.shape);z=depth[::stride,::stride].ravel().astype(float)
    u=u[::stride,::stride].ravel();v=v[::stride,::stride].ravel()
    finite=np.isfinite(z)
    x=(u-LEFT_CENTRE[0])*z/FOCAL_LENGTH;y=(v-LEFT_CENTRE[1])*z/FOCAL_LENGTH
    colors=rgb[::stride,::stride].reshape(-1,3)/255.
    ax.scatter(x[finite],z[finite],-y[finite],c=colors[finite],s=1,rasterized=True,depthshade=False)
    ax.set(title=title,xlabel='X (mm)',ylabel='Z (mm)',zlabel='-Y (mm)')
    ax.view_init(elev=18,azim=-62)
    return ax
