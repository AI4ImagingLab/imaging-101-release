"""Quantitative comparisons and actual full-grid diagnostics; no backend mutation."""
import numpy as np
def metrics(estimate,reference):
    e=np.asarray(estimate,dtype=float).ravel();r=np.asarray(reference,dtype=float).ravel()
    width=np.ptp(r)
    if width<=0:raise ValueError('NRMSE requires nonconstant reference')
    norm=np.linalg.norm(e)*np.linalg.norm(r)
    return dict(ncc=float(e@r/norm) if norm else 0.,nrmse=float(np.sqrt(np.mean((e-r)**2))/width))
def panel(images,titles,units,columns=3,cmap='viridis',shared=False):
    import matplotlib.pyplot as plt
    rows=(len(images)+columns-1)//columns;fig,axes=plt.subplots(rows,columns,figsize=(4.2*columns,3.5*rows),squeeze=False,constrained_layout=True)
    vmin=min(np.min(x) for x in images) if shared else None;vmax=max(np.max(x) for x in images) if shared else None
    if shared and cmap in {'RdBu_r','coolwarm','seismic'}:
        vmax=max(abs(vmin),abs(vmax));vmin=-vmax
    for ax,a,title,unit in zip(axes.ravel(),images,titles,units):
        im=ax.imshow(a,cmap=cmap,origin='upper',vmin=vmin,vmax=vmax);ax.set_title(title,fontsize=10);ax.set_xlabel('Column (native pixel)');ax.set_ylabel('Row (native pixel)');fig.colorbar(im,ax=ax,label=unit,shrink=.8)
    for ax in axes.ravel()[len(images):]:ax.set_visible(False)
    return fig
