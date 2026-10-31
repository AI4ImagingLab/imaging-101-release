"""Figures computed from fresh arrays; importing this module never sets a backend."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

def style():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':110,'axes.titleweight':'semibold','savefig.bbox':'tight'})

def acquisition(intensity,variance,q,energy):
    fig,axs=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
    for ax,value,title in zip(axs,[intensity,np.sqrt(variance),np.isfinite(intensity)],
                              ['Original intensity','Measured standard deviation','Acquisition validity']):
        image=ax.pcolormesh(energy,q,value,shading='nearest',cmap='viridis')
        fig.colorbar(image,ax=ax,label='dimensionless' if title.endswith('validity') else 'original intensity units')
        ax.set(xlabel='Energy transfer (meV)',ylabel=r'$Q$ ($\mathrm{\AA}^{-1}$)',title=title)
    return fig

def decomposition(result,q):
    stage=result['stages'][-1];e=np.arange(-15,80)
    fields=[stage['scaled_observation'],stage['sp'],stage['mp'],stage['ms'],stage['total'],stage['scaled_observation']-stage['total']]
    names=['Scaled observation','Single phonon','Multiphonon (orders 2–4)','Multiple scattering','Total coupled model','Observation − total']
    fig,axs=plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
    limit=np.nanpercentile(abs(fields[0]),99)
    for ax,value,title in zip(axs.flat,fields,names):
        norm=SymLogNorm(linthresh=limit/100,vmin=-limit if title==names[-1] else 0,vmax=limit)
        im=ax.pcolormesh(e,q,value,shading='nearest',cmap='RdBu_r' if title==names[-1] else 'magma',norm=norm)
        fig.colorbar(im,ax=ax,label=r'Modeled intensity ($\mathrm{meV}^{-1}$)')
        ax.set(xlabel='Energy transfer (meV)',ylabel=r'$Q$ ($\mathrm{\AA}^{-1}$)',title=title)
    return fig

def density_comparison(fresh,tighter,source,tight_source,initial):
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained');e=np.arange(80.)
    ax=axs[0]
    for value,label,color,ls in [(initial,'Initial prior','#8c8c8c','--'),(fresh['history'][0],'First physical update','#bc7a20',':'),(source['final_g'],'Reproduced source: default 6','#222222','-'),(fresh['reconstruction'],'Fresh: default 6','#007f85','--'),(tight_source['reconstruction'],'Reproduced source: tighter 14','#8d3c82','-'),(tighter['reconstruction'],'Fresh: tighter 14','#d05b45',':')]:
        ax.plot(e,value,label=label,color=color,ls=ls,lw=1.8)
    ax.set(xlabel='Phonon energy (meV)',ylabel=r'Density ($\mathrm{meV}^{-1}$)',title='Full comparison groups');ax.legend(fontsize=7)
    g=fresh['reconstruction'];sd=np.sqrt(fresh['variance'])
    axs[1].plot(e,g,color='#007f85',label='Fresh default')
    axs[1].fill_between(e,np.maximum(0,g-sd),g+sd,color='#007f85',alpha=.2,label='Original approximate ±1σ protocol')
    axs[1].axvspan(0,7,color='gray',alpha=.15);axs[1].axvline(55,color='gray',ls=':')
    axs[1].set(xlabel='Phonon energy (meV)',ylabel=r'Density ($\mathrm{meV}^{-1}$)',title='Support and uncertainty diagnostic');axs[1].legend(fontsize=7)
    axs[2].plot(e,fresh['reconstruction']-source['final_g'],label='Default parity')
    axs[2].plot(e,tighter['reconstruction']-tight_source['reconstruction'],label='Tighter parity')
    axs[2].set(xlabel='Phonon energy (meV)',ylabel=r'Fresh − source ($\mathrm{meV}^{-1}$)',title='Numerical agreement, not physical truth');axs[2].legend(fontsize=8)
    return fig

def residuals(result,q,history_errors):
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained');e=np.arange(-15,80);s=result['stages'][-1]
    observed=np.nansum(s['scaled_observation'],axis=0);model=np.nansum(s['total'],axis=0)
    axs[0].plot(e,observed,label='Scaled measured Q marginal',color='#222222')
    axs[0].plot(e,model,label='Fresh complete model',color='#007f85',ls='--')
    axs[0].plot(e,np.nansum(s['sp'],axis=0),label='Single phonon',alpha=.7)
    axs[0].set(xlabel='Energy transfer (meV)',ylabel=r'Summed intensity ($\mathrm{meV}^{-1}$)',title='All acquired momentum bins');axs[0].legend(fontsize=7)
    zoom=axs[0].inset_axes([.47,.35,.49,.39])
    zoom.plot(e[22:],observed[22:],color='#222222',lw=1)
    zoom.plot(e[22:],model[22:],color='#007f85',ls='--',lw=1)
    zoom.set(title='Inelastic region',xlim=(7,79));zoom.tick_params(labelsize=7)
    axs[1].plot(e,observed-model,color='#b14b42');axs[1].axhline(0,color='gray',lw=.7)
    axs[1].axvspan(7,79,color='#007f85',alpha=.06)
    axs[1].set(xlabel='Energy transfer (meV)',ylabel=r'Marginal residual ($\mathrm{meV}^{-1}$)',title='Conditional target region ≥7 meV')
    zoom=axs[1].inset_axes([.35,.35,.6,.42])
    zoom.plot(e[22:],(observed-model)[22:],color='#b14b42',lw=1)
    zoom.axhline(0,color='gray',lw=.6)
    zoom.set(title='Target residual, enlarged',xlim=(7,79));zoom.tick_params(labelsize=7)
    axs[2].semilogy(np.arange(len(history_errors)),history_errors,'o-',color='#007f85')
    axs[2].axhline(.00795285686754709,color='#b14b42',ls='--',label='Frozen target boundary')
    axs[2].set(xlabel='Physical update (0 = prior)',ylabel='Relative marginal L2',title='Full default convergence');axs[2].legend(fontsize=7)
    return fig

def balance_bias(e,a,hs,masses,source_masses):
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    positive=e>0;negative=e<0;beta=1/(300*.0862)
    for n in range(4):
        axs[0].semilogy(e[positive],a[n,positive],label=f'Order {n+1}: loss')
        axs[0].semilogy(e[positive],a[n,negative][::-1]*np.exp(beta*e[positive]),ls=':',color=axs[0].lines[-1].get_color())
    axs[0].set(xlabel='Absolute energy (meV)',ylabel=r'Spectral density ($\mathrm{meV}^{-1}$)',title='Detailed balance: gain × Boltzmann factor');axs[0].legend(fontsize=7)
    axs[1].plot(np.arange(1,5),a.sum(axis=1),'o-',label='Fresh finite-grid spectrum')
    axs[1].axhline(1,color='gray',ls='--');axs[1].set(xticks=[1,2,3,4],xlabel='Phonon order',ylabel='Discrete mass',title='A1 endpoint bias is retained')
    defect=1-np.asarray(masses)
    axs[2].loglog(hs,defect,'o-',label='Fresh Debye density')
    axs[2].loglog(hs,1-np.asarray(source_masses),'x',ms=9,label='Independent source check')
    axs[2].loglog(hs,defect[0]*np.asarray(hs)/hs[0],ls='--',color='gray',label='First-order reference')
    axs[2].set(xlabel='Energy spacing h (meV)',ylabel='1 − A1 discrete mass',title='Bias vanishes under refinement');axs[2].legend(fontsize=7)
    axs[2].set_xticks(hs,labels=[str(h) for h in hs])
    axs[2].minorticks_off()
    return fig
