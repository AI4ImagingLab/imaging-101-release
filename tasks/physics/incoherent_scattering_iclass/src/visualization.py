"""Review figures and separately identified experimental comparisons."""
import numpy as np
import matplotlib.pyplot as plt


def align_circular(image, reference):
    """Maximal circular cross-correlation; return shifted image and signed shift."""
    q = np.fft.ifft2(np.fft.fft2(reference)*np.fft.fft2(image).conj()).real
    shift = np.unravel_index(np.argmax(q),q.shape)
    aligned = np.roll(image,shift,axis=(0,1))
    signed = tuple(int(a if a <= n//2 else a-n) for a,n in zip(shift,image.shape))
    return aligned,signed


def comparison_metrics(image, reference):
    """NCC/range-NRMSE with declared shift/peak normalization and contrast.

    Widefield scores describe independent measurement agreement, not truth
    accuracy: illumination, support, windowing and exposure differ.
    """
    aligned,shift = align_circular(np.asarray(image),np.asarray(reference))
    a,b = aligned.astype(np.float64)/aligned.max(),reference.astype(np.float64)/reference.max()
    ncc = np.vdot(a.ravel(),b.ravel()).real/(np.linalg.norm(a)*np.linalg.norm(b))
    nrmse = np.sqrt(np.mean((a-b)**2))/(b.max()-b.min())
    bright,background = np.percentile(a,[99.5,50])
    return {'ncc':float(ncc),'nrmse':float(nrmse),'shift':shift,
            'contrast':float((bright-background)/(bright+background))}


def review_figure(images, titles):
    """Actual images, each divided by its explicitly stated peak, range [0,1]."""
    fig,axes = plt.subplots(2,3,figsize=(14,9))
    for axis,image,title in zip(axes.flat,images,titles):
        axis.imshow(image/image.max(),cmap='gray',vmin=0,vmax=1)
        axis.set_title(title)
        axis.axis('off')
    fig.tight_layout()
    return fig


def transfer_figure(phase, mtf):
    """Correction phase, empirical MTF, and inferred detection |PSF|."""
    otf = mtf*np.conj(phase)
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(otf)))
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    for axis,image,title,cmap in zip(axes,[np.angle(phase),mtf,np.abs(psf)],
                                   ['Correction phase [rad]','Relative MTF','Inferred |PSF|'],
                                   ['twilight','magma','gray']):
        im=axis.imshow(image,cmap=cmap)
        axis.set_title(title)
        fig.colorbar(im,ax=axis,fraction=.046)
    fig.tight_layout()
    return fig
