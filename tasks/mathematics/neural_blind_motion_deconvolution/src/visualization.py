"""Maintainer comparison metrics and scientific figures, without backend changes."""
import numpy as np
from scipy.signal import fftconvolve

MARGIN=50
MAX_SHIFT=25


def align_translation(image,reference):
    """Choose one integer shift by demeaned correlation in a fixed +/-25 box.

    Return aligned and reference 653x568 interiors plus (dy,dx). No rescaling,
    rotation, interpolation or optimization of photometric parameters is used.
    """
    a=np.asarray(reference,dtype=np.float64);b=np.asarray(image,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or min(a.shape)<=2*MARGIN:
        raise ValueError('alignment requires equal 2-D images larger than 100 pixels')
    correlation=fftconvolve(a-a.mean(),(b-b.mean())[::-1,::-1],mode='full')
    row,col=np.array(correlation.shape)//2
    box=correlation[row-MAX_SHIFT:row+MAX_SHIFT+1,col-MAX_SHIFT:col+MAX_SHIFT+1]
    iy,ix=np.unravel_index(np.argmax(box),box.shape)
    dy,dx=int(iy-MAX_SHIFT),int(ix-MAX_SHIFT)
    aligned=b[MARGIN-dy:b.shape[0]-MARGIN-dy,MARGIN-dx:b.shape[1]-MARGIN-dx]
    return aligned,a[MARGIN:-MARGIN,MARGIN:-MARGIN],(dy,dx)


def image_metrics(image,reference):
    """Cosine NCC and reference-range NRMSE under the declared integer gauge."""
    a,b,shift=align_translation(image,reference)
    norm=np.linalg.norm(a)*np.linalg.norm(b);dynamic=np.ptp(b)
    if norm==0 or dynamic==0:
        raise ValueError('undefined NCC or NRMSE for degenerate image/reference')
    return {'ncc':float(np.sum(a*b)/norm),'nrmse':float(np.sqrt(np.mean((a-b)**2))/dynamic),
            'shift_row':shift[0],'shift_column':shift[1]}


def forward_nrmse(image,kernel,observation):
    """Unaligned valid correlation consistency on the 703x618 observation ROI."""
    prediction=fftconvolve(np.asarray(image,dtype=float),np.asarray(kernel,dtype=float)[::-1,::-1],mode='valid')
    target=np.asarray(observation,dtype=float)[25:-25,25:-25]
    return float(np.sqrt(np.mean((prediction-target)**2))/np.ptp(target))


def comparison_figure(observation,reference,upstream,fresh):
    """Paired reference, blur, reproduced upstream, fresh result and residuals."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,11),constrained_layout=True)
    for j,(label,image) in enumerate([('Independent short exposure',reference),('Real blurred observation',observation),('Reproduced upstream',upstream),('Fresh cleaned reconstruction',fresh)]):
        aligned,truth,shift=align_translation(image,reference)
        axes[0,j].imshow(image,cmap='gray',vmin=0,vmax=1);axes[0,j].set_title(label)
        err=aligned-truth
        axes[1,j].imshow(err,cmap='coolwarm',vmin=-.12,vmax=.12)
        axes[1,j].set_title(f'Interior error; shift={shift}\nNRMSE={image_metrics(image,reference)["nrmse"]:.5f}')
    for ax in axes.ravel():ax.axis('off')
    return fig


def diagnostic_figure(result,upstream_kernel,observation):
    """Both kernels, loss stages, and physical observation discrepancy."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(18,4),constrained_layout=True)
    for ax,k,title in zip(axes[:2],[upstream_kernel,result.kernel],['Upstream kernel','Fresh kernel']):
        ax.imshow(k,cmap='inferno');ax.set_title(title);ax.set_xlabel('column offset [pixel]')
    axes[2].semilogy(result.history[:500,0],result.history[:500,1],label='MSE')
    axes[2].semilogy(result.history[500:,0],result.history[500:,1],label='1-SSIM')
    axes[2].axvline(500,color='gray',linestyle=':');axes[2].legend();axes[2].set_xlabel('iteration (two distinct losses)')
    pred=fftconvolve(result.reconstruction,result.kernel[::-1,::-1],mode='valid')
    axes[3].imshow(pred-observation[25:-25,25:-25],cmap='coolwarm',vmin=-.03,vmax=.03)
    axes[3].set_title('Unaligned forward residual\nvalid interior; no padding')
    return fig
