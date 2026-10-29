"""All-region diagnostics for relative HRF/neural input estimation."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import convolve_periodic
def roi_residual(result):
    predicted=convolve_periodic(result['neural_drive'][0].T,result['hrf'][0].T)
    observed=result['filtered_bold']
    return np.sqrt(np.mean((predicted-observed)**2,axis=0))/np.sqrt(np.mean(observed**2,axis=0))
def make_figures(data,fresh,original,ablation):
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,array,title,unit in [(axes[0],fresh['filtered_bold'].T,'Measured BOLD, actual source filtering','Standardized BOLD'),(axes[1],fresh['neural_drive'][0],'Fresh model-conditioned relative input','Relative discrete input')]:
        im=ax.imshow(array,aspect='auto',extent=[0,302,400,0],cmap='RdBu_r');fig.colorbar(im,ax=ax,label=unit);ax.set(title=title,xlabel='Retained-record time (s)',ylabel='Anatomical parcel index')
        ax.axhline(200,color='black',lw=.6)
    fig2,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained');hrf=fresh['hrf'][0];ref=original['hrf'][0];limit=np.max(np.abs(ref))
    for ax,a,title in [(axes[0],ref,'Original author canonical+derivative HRF'),(axes[1],hrf,'Fresh local canonical+derivative HRF')]:
        im=ax.imshow(a,aspect='auto',extent=[0,24,400,0],cmap='RdBu_r',vmin=-limit,vmax=limit);fig2.colorbar(im,ax=ax,label='BOLD per unit discrete input');ax.set(title=title,xlabel='HRF lag (s)',ylabel='Parcel index')
    im=axes[2].imshow(np.abs(hrf-ref),aspect='auto',extent=[0,24,400,0]);fig2.colorbar(im,ax=axes[2],label='Absolute amplitude difference');axes[2].set(title='Fresh/original difference, all400parcels',xlabel='HRF lag (s)',ylabel='Parcel index')
    fig3,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');modeled=convolve_periodic(fresh['neural_drive'][0].T,hrf.T)
    for ax,i in zip(axes.flat,[0,100,200,300]):
        t=np.arange(152)*2;ax.plot(t,fresh['filtered_bold'][:,i],label='Observed, filtered');ax.plot(t,modeled[:,i],'--',label='Fresh forward prediction')
        ax.set(title=f'Fixed anatomical example parcel{i+1}',xlabel='Retained-record time (s)',ylabel='Standardized BOLD');ax.legend(fontsize=8)
    fig4,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');full=roi_residual(fresh);reduced=roi_residual(ablation)
    axes[0].plot(full,label='Fresh canonical+two derivatives');axes[0].plot(reduced,alpha=.65,label='Fresh canonical only');axes[0].set(title='All400ROI observation consistency',xlabel='Parcel index',ylabel='RMS residual / RMS observed BOLD');axes[0].legend(fontsize=7)
    t=np.arange(37)*2/3
    for i in [0,100,200,300]:axes[1].plot(t,hrf[i],label=str(i+1))
    axes[1].set(title='Fixed anatomical HRF examples',xlabel='HRF lag (s)',ylabel='BOLD per unit discrete input');axes[1].legend(title='Parcel',fontsize=7)
    peak_full=np.argmax(hrf,axis=1)*2/3;peak_reduced=np.argmax(ablation['hrf'][0],axis=1)*2/3
    axes[2].hist(peak_full,bins=np.arange(0,25,2/3),alpha=.6,label='Fresh full basis');axes[2].hist(peak_reduced,bins=np.arange(0,25,2/3),alpha=.6,label='Fresh canonical only');axes[2].set(title='Model dependence, no neural ground truth',xlabel='HRF sampled peak lag (s)',ylabel='Parcel count');axes[2].legend(fontsize=7)
    return [fig,fig2,fig3,fig4]
