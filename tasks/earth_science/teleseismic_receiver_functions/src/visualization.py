"""Science figures from fresh arrays; imports do not select a backend."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import reconvolution
def metrics(estimate,reference):
    a=np.asarray(estimate).ravel(); b=np.asarray(reference).ravel()
    return dict(ncc=float(a@b/np.linalg.norm(a)/np.linalg.norm(b)),nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
def make_figures(raw,zrt,methods,sweep,norm):
    time=np.arange(raw.shape[-1])/20.; lag=time-30
    figs={}
    fig,ax=plt.subplots(2,1,figsize=(10,6),layout='constrained')
    for c,name in enumerate(['Vertical','North','East']):ax[0].plot(time,raw[c]*1e6,label=name,lw=.85)
    ax[0].set(xlabel='Seconds after acquisition start',ylabel='Calibrated velocity (μm/s)',title='Tohoku earthquake recorded at Walferdange');ax[0].legend(ncol=3)
    for c,name in enumerate(['Z source','R response','T response']):ax[1].plot(time,zrt[c]*1e6,label=name,lw=.9)
    ax[1].set(xlabel='Seconds after acquisition start',ylabel='Filtered velocity (μm/s)',title='Detrended, tapered, 0.05–1 Hz observations');ax[1].legend(ncol=3)
    figs['observations']=fig
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharex='col')
    for c,label in enumerate(['Radial','Transverse']):
        for name,out in methods.items():
            style={'Fresh water level':dict(color='#0072B2',lw=1.5),'Original water level':dict(color='black',lw=.9,ls='--'),'Fresh time domain':dict(color='#D55E00',lw=1.),'Fresh iterative (cap 400)':dict(color='#009E73',lw=1.,ls=':')}[name]
            axs[c,0].plot(lag,out[c],label=name,**style)
        axs[c,0].set(xlim=(-15,90),ylabel='Normalized receiver response',title=label+' conversion response');axs[c,0].axvline(0,color='.5',ls=':',lw=.8)
        for alpha,out in sweep.items():axs[c,1].plot(lag,out[c],label=f'Water level {alpha:g}',lw=1)
        axs[c,1].set(xlim=(-15,90),title=label+' regularization sensitivity')
    axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
    for ax in axs[-1]:ax.set_xlabel('Lag relative to direct P response (s)')
    figs['receiver_comparisons']=fig
    pred,obs=reconvolution(zrt,methods['Fresh water level'],norm)
    fig,axs=plt.subplots(2,2,figsize=(11,6),layout='constrained')
    for c,label in enumerate(['Radial','Transverse']):
        axs[c,0].plot(time,obs[c]*1e6,label='Observed + Gaussian low pass',lw=1)
        axs[c,0].plot(time,pred[c]*1e6,label='Reconvolved fresh response',lw=.8,alpha=.8)
        axs[c,0].set(ylabel='Velocity (μm/s)',title=label+' forward consistency')
        axs[c,1].plot(time,(pred[c]-obs[c])*1e6,color='#a43820',lw=.8)
        axs[c,1].axhline(0,color='.5',lw=.5);axs[c,1].set(ylabel='Signed residual (μm/s)',title=label+' residual; full acquisition')
    axs[0,0].legend(fontsize=8)
    for ax in axs[-1]:ax.set_xlabel('Seconds after acquisition start')
    figs['forward_consistency']=fig
    return figs
