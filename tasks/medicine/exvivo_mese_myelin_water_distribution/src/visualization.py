"""Figures use only actually computed curves, masses and observed calibration."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import T2_MS

def make_figures(data,results):
    """Return five figures; each result contains the four batch-first outputs."""
    mask=data['mask'];display=data['display_mask'];figs=[]
    fig,axs=plt.subplots(1,4,figsize=(14,3.5),constrained_layout=True)
    for ax,a,title in zip(axs,[data['observed'][...,0],data['observed'][...,-1],data['t1_ms'],data['flip_scale']],['First echo / AU','Last echo / AU','Independent T1 / ms','Independent flip scale']):
        im=ax.imshow(a,origin='lower');ax.contour(mask,levels=[.5],colors='white',linewidths=.5);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.75)
    figs.append(fig)
    fig,axs=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for ax,(name,r) in zip(axs.flat,results.items()):
        im=ax.imshow(np.where(mask,r['myelin_fraction'][0],np.nan),origin='lower',vmin=0,vmax=1,cmap='viridis');ax.contour(display,levels=[.5],colors='white',linewidths=.5);ax.set_title(name);ax.set_xlabel('column / pixel');ax.set_ylabel('row / pixel')
    fig.colorbar(im,ax=axs,shrink=.8,label='Conditional myelin mass fraction');figs.append(fig)
    coords=np.argwhere(mask);selected=coords[[len(coords)//4,len(coords)//2,3*len(coords)//4]]
    fig,axs=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    colors=['#b6b6b6','#777777','#fa8c16','#1354a5','#e24a33','#23a18b']
    for row,(i,j) in enumerate(selected):
        axs[row,0].plot(np.arange(1,33)*9.2,data['observed'][i,j],'ko',ms=3,label='Observed mean of 4')
        for (name,r),color in zip(results.items(),colors):
            axs[row,0].plot(np.arange(1,33)*9.2,r['predicted_echoes'][0,i,j],color=color,label=name,lw=1.3)
            axs[row,1].semilogx(T2_MS,r['reconstruction'][0,i,j],color=color,label=name,lw=1.3)
        axs[row,0].set_title(f'Pixel ({i},{j}): complete 32-echo curve');axs[row,0].set_xlabel('Echo time / ms');axs[row,0].set_ylabel('AU')
        axs[row,1].set_title('T2 bin masses, not density');axs[row,1].set_xlabel('T2 / ms');axs[row,1].set_ylabel('AU per bin');axs[row,1].axvspan(15,40,color='#23a18b',alpha=.1)
    axs[0,0].legend(fontsize=7);figs.append(fig)
    centers=np.zeros_like(data['flip_scale'],dtype=float);centers[mask]=data['cluster_centroids'][data['labels'],1];delta=centers-data['flip_scale'];fresh=next(reversed(results.values()));error=np.sqrt(np.mean((fresh['predicted_echoes'][0]-data['observed'])**2,axis=-1))
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,a,title in [(axs[0,0],np.where(mask,delta,np.nan),'Cluster kappa minus independent kappa'),(axs[0,1],np.where(mask,error,np.nan),'Fresh full-curve residual RMS / AU')]:
        im=ax.imshow(a,origin='lower');ax.contour(display,levels=[.5],colors='white',linewidths=.5);ax.set_title(title);fig.colorbar(im,ax=ax)
    axs[1,0].hist(abs(delta[mask]),bins=40,alpha=.6,label='All 2629 fit pixels');axs[1,0].hist(abs(delta[display]),bins=40,alpha=.6,label='1752 display pixels');axs[1,0].set_xlabel('Absolute kappa clustering error');axs[1,0].set_ylabel('Pixels');axs[1,0].legend()
    axs[1,1].scatter(abs(delta[mask]),error[mask],s=5,alpha=.4);axs[1,1].set_xlabel('Absolute kappa clustering error');axs[1,1].set_ylabel('Curve RMS residual / AU');axs[1,1].set_title('Approximation and fit diagnostics; no exclusions')
    figs.append(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,region,title in [(axs[0],display,'Display region'),(axs[1],mask&~display,'Fitting halo')]:
        for (name,r),color in zip(results.items(),colors):ax.semilogx(T2_MS,r['reconstruction'][0][region].mean(axis=0),label=name,color=color)
        ax.axvspan(15,40,color='#23a18b',alpha=.1);ax.set_title(title+' mean mass spectrum');ax.set_xlabel('T2 / ms');ax.set_ylabel('Mean AU per bin')
    axs[1].legend(fontsize=7);figs.append(fig)
    return figs
