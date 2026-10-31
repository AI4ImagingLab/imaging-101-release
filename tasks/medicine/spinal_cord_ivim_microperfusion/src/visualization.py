import numpy as np
import matplotlib.pyplot as plt
from .physics_model import signal

def map_figure(parameters,mask):
    fig,axes=plt.subplots(6,4,figsize=(13,16),layout='constrained')
    names=['S0 (intensity)','D (10^-3 mm²/s)','f (%)','D* (10^-3 mm²/s)'];scales=[1,1000,100,1000]
    x,y,z=np.where(mask);xmin,xmax=x.min()-2,x.max()+3;ymin,ymax=y.min()-2,y.max()+3
    for j in range(4):
        hi=np.percentile(parameters[...,j][mask]*scales[j],99)
        for k in range(6):
            a=np.where(mask[:,:,k],parameters[:,:,k,j]*scales[j],np.nan)
            im=axes[k,j].imshow(a[xmin:xmax,ymin:ymax].T,origin='lower',vmin=0,vmax=hi,cmap='viridis',extent=(xmin,xmax,ymin,ymax));axes[k,j].set_title(f'{names[j]}, slice {k}');fig.colorbar(im,ax=axes[k,j],shrink=.7)
    return fig

def acquisition_figure(image,mask):
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for k,ax in enumerate(axes.flat):
        im=ax.imshow(image[:,:,k,0].T,origin='lower',cmap='gray');ax.contour(mask[:,:,k].T,levels=[.5],colors=['tab:red'],linewidths=.5);ax.set_title(f'Measured b=5, native slice {k}');fig.colorbar(im,ax=ax)
    return fig

def residual_values(image,mask,b,parameters):
    p=parameters[mask];pred=signal(b[None],*(p[:,j,None] for j in range(4)));return pred-image[mask]

def residual_figure(image,mask,b,one,two,mono):
    res=[residual_values(image,mask,b,p) for p in [one,two,mono]]
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for r,name in zip(res,['Fresh joint','Fresh fixed D','Fresh monoexponential']):
        axes[0].plot(b,np.sqrt(np.mean(r*r,axis=0)),'o-',label=name)
        axes[1].hist(np.sqrt(np.mean(r*r,axis=1)),bins=50,histtype='step',label=name)
    axes[0].set(xlabel='b (s/mm²)',ylabel='Signal RMSE (intensity)');axes[0].legend();axes[1].set(xlabel='Voxel RMS residual',ylabel='Count');axes[1].legend()
    p=one[mask];axes[2].scatter(p[:,2],p[:,3]*1000,s=2,alpha=.25);axes[2].set(xlabel='f (fraction)',ylabel='D* (10^-3 mm²/s)',title='Vascular coupling / zero-f ambiguity')
    return fig

def spectral_figure(image,mask,b,one,two,mono):
    coords=np.argwhere(mask);fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');dense=np.linspace(0,800,200)
    for ax,index in zip(axes,[len(coords)//4,len(coords)//2,3*len(coords)//4]):
        loc=tuple(coords[index]);ax.plot(b,image[loc],'o',label='Measured')
        for p,name in zip([one,two,mono],['Fresh joint','Fresh fixed D','Fresh mono']):ax.plot(dense,signal(dense,*p[loc]),label=name)
        ax.set(title=str(loc),xlabel='b (s/mm²)',ylabel='Magnitude intensity');ax.legend()
    return fig

def reference_figure(fresh,reference,mask):
    fig,axes=plt.subplots(1,4,figsize=(14,3.5),layout='constrained')
    for j,ax in enumerate(axes):
        x=reference[...,j][mask];y=fresh[...,j][mask];ax.scatter(x,y,s=2,alpha=.3);ax.plot([x.min(),x.max()],[x.min(),x.max()],'k--');ax.set(xlabel='Original',ylabel='Fresh',title=['S0','D','f','D*'][j])
    return fig
