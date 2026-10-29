"""Native spatial plots and independently defined imagewise agreement."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import PIXEL_WIDTH_UM


def agreement(estimate,reference,mask):
    """Cosine NCC and dynamic-range NRMSE for one fixed input-defined image."""
    x=np.asarray(estimate,dtype=float)[mask];y=np.asarray(reference,dtype=float)[mask]
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.linalg.norm(x)==0 or np.linalg.norm(y)==0 or np.ptp(y)==0:raise ValueError('Undefined image agreement')
    return {'ncc':float(np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)),
            'nrmse':float(np.sqrt(np.mean((x-y)**2))/np.ptp(y))}


def ph_image(ax,image,mask,title,limits=(5.,7.),cmap='viridis',unit='pH'):
    """Plot full native FOV, preserving measured support and micrometre axes."""
    h,w=image.shape
    im=ax.imshow(np.ma.array(image,mask=~mask),extent=(0,w*PIXEL_WIDTH_UM,h*PIXEL_WIDTH_UM,0),cmap=cmap,vmin=limits[0],vmax=limits[1],rasterized=True)
    ax.set(title=title,xlabel='x (micrometre)',ylabel='y (micrometre)')
    plt.colorbar(im,ax=ax,label=unit,shrink=.8,extend='both')
    return im
