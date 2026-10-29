"""Actual complex-modulus science figures; no backend mutation."""
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
def save(fig,name):
    if os.environ.get('IMAGING_FIGURE_DIR'):
        root=Path(os.environ['IMAGING_FIGURE_DIR']);root.mkdir(parents=True,exist_ok=True)
        for ext in ['png','svg','pdf']:fig.savefig(root/(name+'.'+ext),dpi=170,bbox_inches='tight')
    return fig
def scalar_metrics(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    if np.ptp(b)<=0 or np.linalg.norm(b)<=0:raise ValueError('Degenerate reference')
    return dict(ncc=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)>0 else 0.,nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
def panel(ax,a,title,label,cmap='viridis',limits=None):
    if limits is None:limits=(float(np.nanmin(a)),float(np.nanmax(a)))
    im=ax.imshow(a.T,origin='lower',cmap=cmap,vmin=limits[0],vmax=limits[1],extent=[-.75,191.25,-.75,119.25],rasterized=True)
    ax.set(title=title,xlabel='Native x (mm)',ylabel='Native y (mm)')
    ax.figure.colorbar(im,ax=ax,label=label,shrink=.8,extend='both' if limits[0]>np.nanmin(a) or limits[1]<np.nanmax(a) else 'neither')
    return im
