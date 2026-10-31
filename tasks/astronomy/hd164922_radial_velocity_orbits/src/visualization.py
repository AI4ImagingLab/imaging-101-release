"""Computed review diagnostics; backend is owned by the caller."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import forward,components,unpack_orbits

def residual_diagnostics(data,vector):
    """Return residual m/s, total sigma m/s and dimensionless weighted RMS."""
    residual=data['velocity']-forward(data['time'],data['instrument'],vector)
    sigma=np.sqrt(data['uncertainty']**2+vector[13:16][data['instrument']]**2)
    return residual,sigma,float(np.sqrt(np.mean((residual/sigma)**2)))

def parameter_errors(vector,reference):
    """Return 16 absolute physical errors (orbital rows then offsets and jitter)."""
    a=unpack_orbits(vector);b=unpack_orbits(reference);d=np.abs(a-b)
    d[:,1]=np.abs((a[:,1]-b[:,1]+b[:,0]/2)%b[:,0]-b[:,0]/2)
    d[:,3]=np.abs(np.angle(np.exp(1j*(a[:,3]-b[:,3]))))
    return np.r_[d.ravel(),np.abs(vector[10:]-reference[10:])]

def plot_observations(data):
    """Return Figure for full irregularly sampled observations and instrumental errors."""
    fig,ax=plt.subplots(figsize=(11,3.5),layout='constrained')
    for k,name in enumerate(['HIRES pre','HIRES post','APF']):
        m=data['instrument']==k
        ax.errorbar(data['time'][m]-2450000,data['velocity'][m],data['uncertainty'][m],fmt='.',ms=4,alpha=.7,label=f'{name} (n={m.sum()})')
    ax.set(xlabel='BJD - 2450000 [days]',ylabel='Radial velocity [m/s]',title='All 401 original observations; reported errors exclude fitted jitter');ax.legend(ncol=3)
    return fig

def plot_fit(data,vector,upstream,native=None):
    """Return Figure with fresh/upstream curves and instrument-specific residuals."""
    fig,ax=plt.subplots(2,1,figsize=(11,5.5),sharex=True,layout='constrained',height_ratios=[2,1]);t=data['time'];grid=np.linspace(t.min(),t.max(),2500)
    for k,name in enumerate(['HIRES pre','HIRES post','APF']):
        m=data['instrument']==k;ax[0].errorbar(t[m]-2450000,data['velocity'][m]-vector[10+k],np.sqrt(data['uncertainty'][m]**2+vector[13+k]**2),fmt='.',alpha=.5,ms=3,label=name)
    if native is not None:ax[0].plot(grid-2450000,components(grid,native).sum(0),lw=1,color='C7',alpha=.6,label='Original default stop')
    ax[0].plot(grid-2450000,components(grid,upstream).sum(0),lw=2,color='k',alpha=.4,label='Original API refined MAP')
    ax[0].plot(grid-2450000,components(grid,vector).sum(0),lw=1,color='C3',label='Fresh local MAP')
    r,s,w=residual_diagnostics(data,vector)
    for k in range(3):
        m=data['instrument']==k;ax[1].errorbar(t[m]-2450000,r[m],s[m],fmt='.',ms=3,alpha=.55)
    ax[0].set(ylabel='Offset-corrected RV [m/s]',title='Independent original and active local Keplerian reconstructions');ax[0].legend(ncol=3,fontsize=8)
    ax[1].axhline(0,color='k',lw=.8);ax[1].set(xlabel='BJD - 2450000 [days]',ylabel='Residual [m/s]',title=f'Fresh residual / total uncertainty RMS = {w:.6f}')
    return fig

def phase_bins(phase,velocity):
    """Native 25-bin, two-cycle phase summary; return native centred-cycle coordinates."""
    x=np.r_[phase,phase+1];y=np.r_[velocity,velocity]
    count,edges=np.histogram(x,bins=25);sy,_=np.histogram(x,bins=edges,weights=y);sy2,_=np.histogram(x,bins=edges,weights=y*y)
    valid=count>=3;centre=(edges[:-1]+edges[1:])/2-1;valid &= (centre>=-.5)&(centre<=.5)
    mean=sy[valid]/count[valid]
    error=np.sqrt(np.maximum(sy2[valid]/count[valid]-mean*mean,0))/np.sqrt(count[valid])
    return centre[valid],mean,error

def plot_phases(data,vector):
    """Return per-planet folded observations after removing the other fresh planet."""
    fig,axs=plt.subplots(1,2,figsize=(11,3.8),layout='constrained');orbits=unpack_orbits(vector);c=components(data['time'],vector)
    for j,ax in enumerate(axs):
        P,tc,e,w,K=orbits[j];phase=np.remainder((data['time']-tc)/P,1)
        y=data['velocity']-vector[10:13][data['instrument']]-c[1-j]
        for k,name in enumerate(['HIRES pre','HIRES post','APF']):
            m=data['instrument']==k;ax.errorbar(np.where(phase[m]>.5,phase[m]-1,phase[m]),y[m],np.sqrt(data['uncertainty'][m]**2+vector[13+k]**2),fmt='.',ms=3,alpha=.5,label=name)
        bc,by,be=phase_bins(phase,y);ax.errorbar(bc,by,be,fmt='ro',ms=4,mec='white',label='Native phase bins (~0.08 cycles)')
        ph=np.linspace(-.5,.5,500);ax.plot(ph,components(tc+ph*P,vector)[j],color='k',lw=1.5)
        ax.set(xlabel='Phase from inferior conjunction [cycles]',ylabel='Planet-isolated RV [m/s]',title=f'Planet {j+1}: P={P:.4f} d, e={e:.4f}')
    axs[0].legend(fontsize=8)
    return fig
