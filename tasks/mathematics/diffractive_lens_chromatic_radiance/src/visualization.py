"""Actual reconstruction figures; display clipping never changes evaluated arrays."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

def metrics(x,r):
    a=x.ravel();b=r.ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))

def image_comparison(images,title):
    fig,axs=plt.subplots(1,len(images),figsize=(4*len(images),7),layout='constrained')
    for ax,(name,a) in zip(np.atleast_1d(axs),images.items()):ax.imshow(np.clip(a,0,1));ax.set_title(name);ax.axis('off')
    fig.suptitle(title+' | display only clipped to [0,1]')
    return fig

def zoom_comparison(images):
    fig,axs=plt.subplots(2,len(images),figsize=(4*len(images),7),layout='constrained')
    # Fixed image regions, identical across every comparison, not metric crops.
    regions=[(slice(200,500),slice(100,400)),(slice(950,1250),slice(400,700))]
    for row,sl in enumerate(regions):
        for col,(name,a) in enumerate(images.items()):axs[row,col].imshow(np.clip(a[sl],0,1));axs[row,col].set(title=name+f' | region {row+1}');axs[row,col].axis('off')
    return fig

def psf_figure(details):
    fig,axs=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for row,c in enumerate((0,2)):
        p=details[c]['psf']
        for col,k in enumerate((0,19,39,59)):
            v=p[:,:,k];lim=max(abs(v.min()),abs(v.max()));im=axs[row,col].imshow(v,cmap='coolwarm',norm=SymLogNorm(linthresh=1e-4,vmin=-lim,vmax=lim))
            axs[row,col].set(title=f'{"Red" if c==0 else "Blue"} | patch {k}',xlabel='Kernel x',ylabel='Kernel y');fig.colorbar(im,ax=axs[row,col],shrink=.75)
    fig.suptitle('Signed diagnostic kernels | symmetric logarithmic color, linear threshold 1e-4')
    return fig

def diagnostics(result,original,image,details):
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    error=np.max(abs(result-original),axis=-1);im=axs[0].imshow(error/1e-11,cmap='magma');fig.colorbar(im,ax=axs[0],label='Absolute difference / 1e-11');axs[0].set_title('Fresh-original max RGB difference',fontsize=10)
    names=['Observed','Fresh'];vals=[image,result]
    for i,a in enumerate(vals):axs[1].hist(a.ravel(),bins=120,histtype='step',density=True,label=names[i])
    axs[1].set(xlabel='Unclipped relative signal',ylabel='Density',title='Signed numerical range');axs[1].legend()
    for c,label in [(0,'Red'),(2,'Blue')]:
        histories=[details[c]['initial']]+[r['deblur'] for r in details[c]['iteration']]
        scores=[metrics(a,original[:,:,c])[1] for a in histories]
        axs[2].plot(range(4),scores,'o-',label=label)
    axs[2].set(xlabel='Iteration (0 = initialization)',ylabel='NRMSE versus original final',title='Iteration comparison, not truth error');axs[2].legend()
    return fig
