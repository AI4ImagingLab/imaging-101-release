"""Review figures from measured input and actually computed reconstructions."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import CAMERAS, observations

def metrics(estimate, reference):
    a=np.asarray(estimate).ravel(); b=np.asarray(reference).ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))

def observations_figure(data):
    fig,axs=plt.subplots(4,2,figsize=(12,10),layout='constrained')
    for row,camera in enumerate(CAMERAS):
        _,led,flu=observations(data,camera)
        for ax,signal,label in zip(axs[row],(led,flu),('LED diffraction','Fluorescent diffraction')):
            for c,color in enumerate(('r','g','b')):ax.plot(signal[:,c],color=color,lw=1)
            ax.set(title=camera+' | '+label,xlabel='Sensor column (pixel)',ylabel='Relative linear signal')
    return fig

def response_figure(result,truth,upstream):
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained'); wave=np.arange(400,701,10)
    for i,ax in enumerate(axs.flat):
        for c,color in enumerate(('r','g','b')):
            ax.plot(wave,truth[0,i,:,c],color=color,ls='--',label='Independent measurement' if c==0 else None)
            ax.plot(wave,result[0,i,:,c],color=color,label='Fresh reconstruction' if c==0 else None)
            ax.scatter(wave[::3],upstream[0,i,::3,c],color=color,s=15,marker='x',label='Original run' if c==0 else None)
        ncc,error=metrics(result[0,i],truth[0,i]); ax.set(title=f'{CAMERAS[i]} | NCC {ncc:.4f}, NRMSE {error:.4f}',xlabel='Wavelength (nm)',ylabel='Relative sensitivity');ax.legend(fontsize=8);ax.axhline(0,color='gray',lw=.5)
    return fig

def diagnostic_figure(details):
    fig,axs=plt.subplots(4,3,figsize=(14,11),layout='constrained')
    for i,d in enumerate(details):
        axs[i,0].plot(d['mapping']);axs[i,0].set(title=CAMERAS[i]+' mapping',xlabel='Pixel',ylabel='Spectral bin')
        # Signed attenuation is retained: the unconstrained basis is not a physical calibration.
        axs[i,1].plot(np.arange(400,721),d['attenuation']);axs[i,1].axhline(0,color='gray',lw=.5);axs[i,1].set(title='Unconstrained nuisance efficiency',xlabel='Wavelength (nm)',ylabel='Relative coefficient')
        residual=d['pred_dispersed']-d['dispersed']
        for c,color in enumerate(('r','g','b')):axs[i,2].plot(residual[:,c],color=color,lw=.7)
        axs[i,2].set(title='Forward residual (signed)',xlabel='Pixel',ylabel='Normalized signal difference')
    return fig

def comparison_figure(results,truth):
    names=list(results); errors=np.array([[metrics(results[n][0,i],truth[0,i])[1] for i in range(4)] for n in names])
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained');x=np.arange(4);width=.8/len(names)
    for j,n in enumerate(names):ax.bar(x-.4+(j+.5)*width,errors[j],width,label=n)
    ax.set(xticks=x,xticklabels=CAMERAS,ylabel='Dynamic-range NRMSE',title='Fresh reconstructions: method and basis comparisons');ax.legend(fontsize=8,ncol=3)
    return fig

def parity_figure(result,upstream):
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    delta=result[0]-upstream[0]
    for i,camera in enumerate(CAMERAS):axs[0].plot(np.arange(400,701,10),np.max(np.abs(delta[i]),axis=1),label=camera)
    axs[0].set(xlabel='Wavelength (nm)',ylabel='Maximum RGB absolute difference',title='Fresh versus independently captured original');axs[0].legend(fontsize=8)
    axs[1].bar(CAMERAS,np.max(np.abs(delta),axis=(1,2)));axs[1].set(ylabel='Maximum absolute difference',title='Floating-point parity, not truth accuracy');axs[1].tick_params(axis='x',rotation=20)
    return fig
