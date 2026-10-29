"""Actual multispectral science plots and independent presentation metrics."""
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
def metrics(a,b):
    a=np.asarray(a,np.float64).ravel();b=np.asarray(b,np.float64).ravel()
    if np.ptp(b)<=0 or np.linalg.norm(b)<=0:raise ValueError('Degenerate reference')
    return dict(ncc=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)>0 else 0.,nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
def save(fig,name):
    if os.environ.get('IMAGING_FIGURE_DIR'):
        root=Path(os.environ['IMAGING_FIGURE_DIR']);root.mkdir(parents=True,exist_ok=True)
        for ext in ['png','svg','pdf']:fig.savefig(root/(name+'.'+ext),dpi=170,bbox_inches='tight')
    return fig
def panel(ax,a,title,label,cmap='viridis',limits=None):
    if limits is None:limits=(float(np.nanmin(a)),float(np.nanmax(a)))
    edge=(.024975/2+.024975/110/2)*1000
    im=ax.imshow(a,origin='lower',extent=[-edge,edge,-edge,edge],cmap=cmap,vmin=limits[0],vmax=limits[1],rasterized=True)
    ax.set(title=title,xlabel='Nominal x (mm)',ylabel='Nominal y (mm)')
    ax.figure.colorbar(im,ax=ax,label=label,shrink=.8,extend='both' if limits[0]>np.nanmin(a) or limits[1]<np.nanmax(a) else 'neither')
    return im
