"""Maintainer plotting and independently checked presentation metrics."""
import numpy as np
import matplotlib.pyplot as plt

def metrics(estimate,reference):
    x=np.asarray(estimate,float).ravel();y=np.asarray(reference,float).ravel()
    norm=np.linalg.norm(x)*np.linalg.norm(y);span=np.ptp(y)
    if np.linalg.norm(y)==0 or span==0:raise ValueError('Degenerate comparison reference')
    return {'ncc':float(x@y/norm) if norm else 0.,'nrmse':float(np.sqrt(np.mean((x-y)**2))/span)}

def source_comparison(fresh,reference,source=0):
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    x=fresh['reconstruction'][0,source];y=reference['reconstruction'][0,source]
    maximum=max(x.max(),y.max())
    for ax,image,title in zip(axes,[y,x,x-y],['Original upstream','Fresh local','Fresh minus upstream']):
        obj=ax.imshow(image,cmap='viridis' if 'minus' not in title else 'coolwarm',vmin=0 if 'minus' not in title else -np.max(abs(x-y)),vmax=maximum if 'minus' not in title else np.max(abs(x-y)))
        ax.set_title(title);ax.set_xlabel('Column [pixel]');ax.set_ylabel('Row [pixel]');fig.colorbar(obj,ax=ax)
    return fig
