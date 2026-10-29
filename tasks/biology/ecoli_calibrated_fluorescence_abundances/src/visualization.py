"""Maintainer plots and independently checked numerical summaries."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import FLUOROPHORES,PIXEL_PITCH_M
from .physics_model import forward,relative_observation_error

def agreement(estimate,reference):
    a,b=np.ravel(estimate),np.ravel(reference)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))

def acquisition_figure(image,m):
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    extent=[0,30*PIXEL_PITCH_M*1e6,30*PIXEL_PITCH_M*1e6,0]
    im=axes[0].imshow(image.sum(axis=-1),extent=extent,cmap='magma');fig.colorbar(im,ax=axes[0],label='Summed digital intensity')
    axes[0].set(xlabel='x (µm)',ylabel='y (µm)',title='Measured author ROI')
    axes[1].plot(np.arange(1,25),m);axes[1].set(xlabel='Detector channel',ylabel='Normalized control spectrum',title='13 independently stained controls')
    corr=m.T@m/np.outer(np.linalg.norm(m,axis=0),np.linalg.norm(m,axis=0))
    im=axes[2].imshow(corr,vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=axes[2],label='Spectral cosine similarity')
    axes[2].set(title='Spectral overlap',xticks=np.arange(13),yticks=np.arange(13));axes[2].set_xticklabels(FLUOROPHORES,rotation=90);axes[2].set_yticklabels(FLUOROPHORES)
    return fig

def abundance_figure(methods):
    fig,axes=plt.subplots(len(methods),13,figsize=(22,2.3*len(methods)),layout='constrained',squeeze=False)
    for row,(name,a) in enumerate(methods.items()):
        for c in range(13):
            im=axes[row,c].imshow(a[...,c],vmin=0,vmax=1,cmap='magma');axes[row,c].set_xticks([]);axes[row,c].set_yticks([])
            if row==0:axes[row,c].set_title(FLUOROPHORES[c],fontsize=9)
            if c==0:axes[row,c].set_ylabel(name)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='Relative abundance (shared 0–1 scale)')
    return fig

def residual_figure(image,m,methods):
    fig,axes=plt.subplots(2,len(methods),figsize=(4*len(methods),7),layout='constrained',squeeze=False)
    for j,(name,a) in enumerate(methods.items()):
        error,gain=relative_observation_error(a,image,m);p=gain*forward(a,m)
        r=np.sqrt(np.mean((image-p)**2,axis=-1));r[[0,-1]]=np.nan;r[:,[0,-1]]=np.nan
        im=axes[0,j].imshow(r,vmin=0,vmax=35,cmap='inferno');fig.colorbar(im,ax=axes[0,j],label='Spectral RMSE (digital units)');axes[0,j].set_title(f'{name}: relative error {error:.3f}')
        axes[1,j].plot(image[14,14],label='Measured');axes[1,j].plot(p[14,14],label='Reprojected');axes[1,j].set(xlabel='Channel index',ylabel='Digital intensity',title='Same central pixel in every method');axes[1,j].legend()
    return fig
