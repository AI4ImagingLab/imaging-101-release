import numpy as np
import matplotlib.pyplot as plt
from .physics_model import POOL_NAMES,lorentzian,predict

def bound_activity(parameters,mask,lower,upper):
    """Fraction within one-millionth of each component's parameter-box range."""
    scaled=(parameters[mask]-lower)/(upper-lower)
    return np.mean((scaled<=1e-6)|(scaled>=1-1e-6),axis=0)

def acquisition_figure(image,mask,normalization):
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    values=[normalization['m0'],mask,normalization['m0_final']/np.maximum(normalization['m0'],np.finfo(float).tiny)]
    for ax,v,title in zip(axes,values,['First measured reference','Input-only support','Last / first reference']):
        im=ax.imshow(v,cmap='viridis',vmin=.5 if 'Last' in title else None,vmax=1.5 if 'Last' in title else None)
        ax.set_title(title);fig.colorbar(im,ax=ax)
    return fig

def amplitude_figure(parameters,mask):
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for k,ax in enumerate(axes.flat):
        im=ax.imshow(np.where(mask,100*parameters[...,k,0],np.nan),vmin=0,vmax=[100,50,30,50,25,25][k],cmap='magma')
        ax.set_title(POOL_NAMES[k]);fig.colorbar(im,ax=ax,label='Apparent peak attenuation (%)')
    return fig

def spectral_parameter_figure(parameters,field,mask):
    fig,axes=plt.subplots(2,6,figsize=(19,7),layout='constrained')
    for k in range(6):
        for row,index,label in [(0,1,'FWHM (ppm)'),(1,2,'Center (ppm)')]:
            im=axes[row,k].imshow(np.where(mask,parameters[...,k,index],np.nan),cmap='viridis');axes[row,k].set_title(POOL_NAMES[k]+' '+label);fig.colorbar(im,ax=axes[row,k])
    return fig

def spectrum_figure(spectra,offsets,parameters,field,positions,joint):
    fig,axes=plt.subplots(2,len(positions),figsize=(15,8),layout='constrained')
    order=np.argsort(offsets);x=offsets[order]
    for col,(row,px) in enumerate(positions):
        p=parameters[row,px];b=field[row,px];ax=axes[0,col]
        ax.plot(x,spectra[row,px,order],'o',ms=3,label='Measured')
        ax.plot(x,predict(x,p,b),label='Fresh two-step')
        ax.plot(x,predict(x,joint[col],b),'--',label='Fresh joint iterate (see status)')
        ax.set(title=f'Native pixel ({row},{px})',xlabel='RF offset (ppm)',ylabel='Normalized water signal');ax.legend(fontsize=8)
        for k in range(6):axes[1,col].plot(x,lorentzian(x-b,*p[k]),label=POOL_NAMES[k])
        axes[1,col].set(xlabel='RF offset (ppm)',ylabel='Fresh component attenuation');axes[1,col].legend(fontsize=7,ncol=2)
    return fig

def reference_figure(parameters,upstream,mask):
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for k,ax in enumerate(axes.flat):
        x=upstream[...,k,0][mask];y=parameters[...,k,0][mask]
        ax.scatter(x,y,s=1,alpha=.2);ax.plot([0,x.max()],[0,x.max()],'k--',lw=.7)
        ax.set(title=POOL_NAMES[k],xlabel='Original amplitude (fraction)',ylabel='Fresh amplitude (fraction)')
    return fig

def residual_figure(spectra,offsets,parameters,field,mask):
    p=predict(offsets,parameters[mask],field[mask]);q=predict(offsets,parameters[mask],np.zeros(mask.sum()))
    errors=np.sqrt(np.mean((p-spectra[mask])**2,axis=1));ablation=np.sqrt(np.mean((q-spectra[mask])**2,axis=1))
    maps=[]
    for v in [errors,ablation]:
        out=np.full(mask.shape,np.nan);out[mask]=v;maps.append(out)
    fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
    for ax,v,label in zip(axes[:2],maps,['Fresh full model','Field=0 forward ablation']):
        im=ax.imshow(v,vmin=0,vmax=.25,cmap='magma');ax.set_title(label);fig.colorbar(im,ax=ax,label='Normalized spectral RMSE')
    im=axes[2].imshow(np.where(mask,field,np.nan),cmap='coolwarm');axes[2].set_title('Fresh field shift');fig.colorbar(im,ax=axes[2],label='ppm')
    axes[3].hist(errors,bins=40,alpha=.6,label='Full model');axes[3].hist(ablation,bins=40,alpha=.5,label='Field=0');axes[3].set(xlabel='Pixel spectral RMSE',ylabel='Supported pixels');axes[3].legend()
    return fig,errors,ablation
