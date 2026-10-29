"""Maintainer review of computed spectra, physical structure and full ensemble."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import R_SUN,M_JUP,R_JUP
LABELS=['Stellar radius (solar)','Planet mass (Jupiter)','Reference radius (Jupiter)','Temperature (K)','log10 scattering','log10 metallicity','log10 cloud pressure (Pa)','Error multiplier']
UNITS=np.array([R_SUN,M_JUP,R_JUP,1.,1.,1.,1.,1.])

def spectra(raw,observed_model,original,high_wave,high_depth,predictive=None):
    fig,axes=plt.subplots(2,1,figsize=(13,8),height_ratios=[3,1],sharex=True,layout='constrained')
    x=raw['wavelength_bins'].mean(1)*1e6
    for start,end,label,color in [(0,10,'STIS','tab:blue'),(10,38,'WFC3','tab:orange'),(38,42,'Spitzer','tab:green')]:
        axes[0].errorbar(x[start:end],raw['transit_depth'][start:end]*1e6,yerr=raw['uncertainty'][start:end]*1e6,xerr=np.diff(raw['wavelength_bins'][start:end],axis=1).ravel()*.5e6,fmt='o',color=color,label=label,ms=4)
    axes[0].plot(high_wave*1e6,high_depth*1e6,color='black',lw=.7,alpha=.6,label='Fresh full physical spectrum')
    axes[0].plot(x,observed_model*1e6,'rx',label='Fresh band mean');axes[0].plot(x,original*1e6,'+',color='purple',label='Original full-run band mean')
    if predictive is not None:axes[0].fill_between(x,np.percentile(predictive,16,axis=0)*1e6,np.percentile(predictive,84,axis=0)*1e6,color='red',alpha=.15,label='Fresh sampled model 16-84%')
    axes[0].set(ylabel='Blocked stellar flux (ppm)',title='Measured HD209458b transmission: all 42 bands');axes[0].legend(ncol=2,fontsize=9)
    axes[1].axhline(0,color='gray');axes[1].plot(x,(observed_model-raw['transit_depth'])/raw['uncertainty'],'o');axes[1].set(xscale='log',xlabel='Wavelength (micrometers)',ylabel='Residual / nominal sigma')
    return fig

def traces(chain,logp,original_logp):
    fig,axes=plt.subplots(4,2,figsize=(14,11),layout='constrained')
    for j,a in enumerate(axes.flat):
        a.plot((chain[:,:,j]/UNITS[j]).T,lw=.25,alpha=.25);a.set(xlabel='Transition (all 1000 retained)',ylabel=LABELS[j])
    fig.suptitle('Fresh 50 walker trajectories: finite protocol, no convergence claim')
    fig2,a=plt.subplots(figsize=(11,4),layout='constrained');a.plot(np.max(logp,axis=0),label='Fresh per-step maximum');a.plot(np.max(original_logp,axis=0),label='Original per-step maximum',alpha=.7);a.set(xlabel='Transition',ylabel='Normalized log posterior',title='Likelihood exploration including initialization');a.legend()
    return fig,fig2

def corner(chain,original_chain):
    x=chain.reshape(-1,8)/UNITS;y=original_chain.reshape(-1,8)/UNITS
    fig,axes=plt.subplots(8,8,figsize=(19,18),layout='constrained')
    short=['Rs','Mp','Rp','T','log S','log Z','log Pc','error']
    for i in range(8):
        for j in range(8):
            a=axes[i,j]
            if i<j:a.set_axis_off();continue
            if i==j:
                a.hist(y[:,i],bins=40,density=True,histtype='step',color='tab:orange',label='Original')
                a.hist(x[:,i],bins=40,density=True,histtype='step',color='tab:blue',label='Fresh')
                if i==0:a.legend(fontsize=7)
            else:a.hist2d(x[:,j],x[:,i],bins=40,cmap='Blues')
            if i==7:a.set_xlabel(short[j])
            else:a.tick_params(labelbottom=False)
            if j==0:a.set_ylabel(short[i])
            else:a.tick_params(labelleft=False)
            a.tick_params(labelsize=7)
    fig.suptitle('All 8 parameter marginals and fresh pair distributions; full finite chains retained',fontsize=14)
    return fig
