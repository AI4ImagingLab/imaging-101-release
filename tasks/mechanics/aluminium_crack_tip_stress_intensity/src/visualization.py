"""Measured-data and fracture-field visualizations with tested metrics."""
import numpy as np
import matplotlib.pyplot as plt


def metrics(a,b):
    """Cosine NCC and dynamic-range NRMSE without alignment or rescaling."""
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    denominator=np.linalg.norm(a)*np.linalg.norm(b);span=np.ptp(b)
    if denominator==0 or span==0:raise ValueError('Degenerate comparison')
    return dict(ncc=float(a@b/denominator),nrmse=float(np.sqrt(np.mean((a-b)**2))/span))


def field_panel(fields,titles,r,phi,cmap='coolwarm'):
    """Show fields on the declared radius/azimuth image grid."""
    fig,axs=plt.subplots(1,len(fields),figsize=(5*len(fields),4.5),squeeze=False)
    for ax,value,title in zip(axs[0],fields,titles):
        bound=max(float(np.max(np.abs(value))),np.finfo(float).tiny)
        art=ax.pcolormesh(np.rad2deg(phi),r,value,shading='nearest',cmap=cmap,vmin=-bound,vmax=bound)
        fig.colorbar(art,ax=ax,label='MPa');ax.set_title(title)
        ax.set_xlabel('Crack-frame azimuth [degree]');ax.set_ylabel('Radius [mm]')
    fig.tight_layout();return fig
