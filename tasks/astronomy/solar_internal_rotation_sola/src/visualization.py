"""Computed observation, rotation, resolution and uncertainty review plots."""
import numpy as np
import matplotlib.pyplot as plt

def observations(data):
    y=data['splittings'];error=np.abs(data['splitting_error']);fig,ax=plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(np.arange(len(y)),y,'.',ms=.8);ax[0].set(xlabel='Native coefficient index',ylabel='Frequency coefficient (nHz)',title=f'All {len(y):,} observations')
    ax[1].hist(np.log10(error[error>0]),bins=60);ax[1].set(xlabel='log10 reported error (nHz)',ylabel='Coefficient count',title='Uneven observational precision')
    ax[2].plot(data['mesh'][::51,0],data['radial_kernel_1'][:,[100,500,1000]]);ax[2].set(xlabel='Radius / solar radius',ylabel='Calibrated radial factor',title='Example known-model sensitivity factors')
    fig.tight_layout();return fig

def field_comparison(data,fresh,upstream,error):
    radius=data['mesh'][::51,0];theta=np.rad2deg(data['mesh'][:51,1]);x=fresh['reconstruction'][0];r=upstream['reconstruction'][0]
    fig,ax=plt.subplots(1,4,figsize=(17,4),sharey=True)
    for a,z,title,cmap in zip(ax,[r,x,x-r,error],['Reproduced original method','Fresh full local SOLA','Fresh - original','Conditional formal error'],['viridis','viridis','coolwarm','magma']):
        kwargs=dict(vmin=float(min(r.min(),x.min())),vmax=float(max(r.max(),x.max()))) if title in ['Reproduced original method','Fresh full local SOLA'] else {}
        if title=='Fresh - original':
            limit=max(float(np.max(np.abs(z))),np.finfo(float).eps)
            kwargs=dict(vmin=-limit,vmax=limit)
        m=a.pcolormesh(theta,radius,z,shading='auto',cmap=cmap,**kwargs);fig.colorbar(m,ax=a,label='nHz');a.set(xlabel='Colatitude (deg)',title=title)
    ax[0].set_ylabel('Radius / solar radius');fig.tight_layout();return fig

def resolution(data,diagnostics):
    radius=data['mesh'][::51,0];theta=np.rad2deg(data['mesh'][:51,1]);idx=diagnostics['kernel_indices'];kernel=diagnostics['averaging_kernels'];fig,ax=plt.subplots(1,4,figsize=(17,4))
    for a,j in zip(ax[:3],[3,4,6]):
        limit=max(float(np.max(np.abs(kernel[j]))),np.finfo(float).eps)
        m=a.pcolormesh(theta,radius,kernel[j],shading='auto',cmap='coolwarm',vmin=-limit,vmax=limit);center=data['mesh'][idx[j]]
        a.plot(np.rad2deg(center[1]),center[0],'kx',ms=7);a.set(xlabel='Colatitude (deg)',ylabel='Radius / solar radius',title=f'Averaging kernel at r={center[0]:.2f}');fig.colorbar(m,ax=a,label='Density / meridional area')
    cov=diagnostics['covariance'];diag=np.sqrt(np.diag(cov));corr=cov/diag[:,None]/diag[None,:]
    m=ax[3].imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax[3].set(xlabel='Selected target',ylabel='Selected target',title='Conditional error correlation');fig.colorbar(m,ax=ax[3])
    fig.tight_layout();return fig

def residuals(observed,predicted,sigma):
    z=(predicted-observed)/sigma;fig,ax=plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(np.arange(len(z)),z,'.',ms=.8);ax[0].set(xlabel='Native coefficient index',ylabel='Residual / reported error',title='All-coefficient forward consistency')
    ax[1].hist(z,bins=80,density=True);xx=np.linspace(-5,5,200);ax[1].plot(xx,np.exp(-xx**2/2)/np.sqrt(2*np.pi),'--',label='Unit Gaussian comparison');ax[1].set(xlabel='Residual / reported error',ylabel='Density',title=f'Weighted RMS {np.sqrt(np.mean(z*z)):.5f}');ax[1].legend(fontsize=8)
    ax[2].plot(observed,predicted,'.',ms=.8);lo=min(observed.min(),predicted.min());hi=max(observed.max(),predicted.max());ax[2].plot([lo,hi],[lo,hi],'k--');ax[2].set(xlabel='Observed (nHz)',ylabel='Predicted (nHz)',title='Physical calibrated data prediction')
    fig.tight_layout();return fig
