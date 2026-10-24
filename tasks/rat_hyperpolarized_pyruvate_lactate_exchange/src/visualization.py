"""Plotting and independently named scientific summary metrics."""
import numpy as np
import matplotlib.pyplot as plt
def relative_l2(pred,observed):
    return float(np.linalg.norm(np.asarray(pred)-observed)/np.linalg.norm(observed))
def compare(pred,reference):
    a=np.asarray(pred,dtype=float).ravel();b=np.asarray(reference,dtype=float).ravel();dr=np.ptp(b)
    return {'ncc':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),'dr_nrmse':float(np.sqrt(np.mean((a-b)**2))/dr),'max_abs_over_range':float(abs(a-b).max()/dr)}
def image(ax,array,title,unit='',signed=False):
    a=np.asarray(array);limit=float(abs(a).max())
    kw={'cmap':'RdBu_r','vmin':-limit,'vmax':limit} if signed else {'cmap':'viridis'}
    im=ax.imshow(a.T,origin='lower',extent=(-1.5,94.5,-1.5,94.5),**kw);ax.set_title(title);ax.set_xlabel('native x (mm)');ax.set_ylabel('native y (mm)');plt.colorbar(im,ax=ax,label=unit,shrink=.82)
    return im
