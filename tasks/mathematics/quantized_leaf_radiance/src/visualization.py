"""Comparable plots and explicitly defined real-image accuracy measures."""
import numpy as np
import matplotlib.pyplot as plt


def metrics(image, reference, lower, upper):
    """Uncentered NCC, dynamic-range NRMSE, unit-range PSNR, interval error."""
    a=np.asarray(image,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    rmse=float(np.sqrt(np.mean((a-b)**2)))
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return {'ncc':float(np.sum(a*b)/denominator) if denominator else 0.,
            'nrmse':rmse/float(np.ptp(b)),
            'psnr_db':float(-20*np.log10(rmse)) if rmse else None,
            'interval_violation':float(max(0.,np.max(lower-a),np.max(a-upper)))}


def image_panel(images, titles, ncols=3, error=False, vmax=None):
    """Equal sized image panels; common absolute-error scale when requested."""
    nrows=(len(images)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(4*ncols,4*nrows),squeeze=False,layout='constrained')
    artist=None
    for ax,image,title in zip(axes.flat,images,titles):
        artist=ax.imshow(image,cmap='magma' if error else None,vmin=0 if error else None,vmax=vmax if error else None)
        ax.set_title(title,fontsize=10);ax.set_axis_off()
    for ax in list(axes.flat)[len(images):]:ax.set_visible(False)
    if error:fig.colorbar(artist,ax=list(axes.flat)[:len(images)],shrink=.72,label='Absolute normalized RGB error (channel RMS)')
    return fig
