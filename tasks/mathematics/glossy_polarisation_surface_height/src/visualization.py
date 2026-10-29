"""Review figures and independently tested real-valued comparison metrics."""
import numpy as np
import matplotlib.pyplot as plt


def metrics(estimate, reference, mask):
    """Cosine NCC and reference-dynamic-range NRMSE on the declared mask."""
    a, b = np.asarray(estimate)[mask], np.asarray(reference)[mask]
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    span = np.ptp(b)
    if norm == 0 or span == 0:
        raise ValueError('Undefined cosine or dynamic-range metric')
    return dict(ncc=float(a@b/norm), nrmse=float(np.sqrt(np.mean((a-b)**2))/span))


def image_panel(arrays, titles, mask=None, cmap='viridis', signed=None, shared_groups=()):
    """Plot images; signed panels use symmetric limits and a diverging map.

    shared_groups lists panel-index groups that share physical color limits.
    """
    signed=[False]*len(arrays) if signed is None else signed
    limits={}
    for group in shared_groups:
        values=np.concatenate([np.asarray(arrays[i])[mask] if mask is not None else np.asarray(arrays[i]).ravel() for i in group])
        for i in group:limits[i]=(float(values.min()),float(values.max()))
    fig, axs = plt.subplots(1,len(arrays),figsize=(5*len(arrays),5),squeeze=False)
    for i,(ax,a,title) in enumerate(zip(axs[0],arrays,titles)):
        shown=np.ma.array(a,mask=~mask) if mask is not None else a
        if signed[i]:
            bound=max(float(np.max(np.abs(shown))),np.finfo(float).tiny)
            vmin,vmax=-bound,bound
        else:vmin,vmax=limits.get(i,(None,None))
        artist=ax.imshow(shown,cmap='coolwarm' if signed[i] else cmap,vmin=vmin,vmax=vmax)
        fig.colorbar(artist,ax=ax,shrink=.8)
        ax.set_title(title);ax.set_xlabel('column (pixel)');ax.set_ylabel('row (pixel)')
    fig.tight_layout()
    return fig
