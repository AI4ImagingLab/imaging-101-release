"""Scientific comparisons from supplied freshly computed arrays; no reconstruction cache."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import calzetti,forward
def residual_statistic(prediction,data):return float(np.mean(((prediction-data['flux'])/data['uncertainty'])**2))
def marginal(weights,coordinate):
    grid=np.unique(coordinate);return grid,np.array([weights[coordinate==x].sum() for x in grid])
def acquisition_figure(data,result,original,rejected):
    fig,ax=plt.subplots(2,2,figsize=(13,8),constrained_layout=True);a=ax.ravel();wave=data['wavelength']
    a[0].plot(data['observed_wavelength'],data['observed_flux'],lw=.5,color='.35');a[0].set(title='All 3835 original SDSS detector bins',xlabel='Observed vacuum wavelength [Angstrom]',ylabel='Flux [1e-17 erg/s/cm2/Angstrom]')
    a[1].plot(wave,data['flux'],color='.6',lw=.6,label='Measured');a[1].plot(wave,result['prediction'],lw=.8,label='Fresh full fit');a[1].plot(wave,original['prediction'][0],ls=':',lw=.6,label='Native fit');a[1].legend();a[1].set(title='Complete supported-band reconstruction',xlabel='Rest vacuum wavelength [Angstrom]',ylabel='Flux [same units]')
    a[2].plot(wave,(result['prediction']-data['flux'])/data['uncertainty'],lw=.6);a[2].axhline(0,color='k',lw=.7);a[2].set(title='All supported residuals; no clipping in evaluation',xlabel='Rest vacuum wavelength [Angstrom]',ylabel='Residual / measured standard error')
    context=data['context_wavelength'];edge=context>7350
    a[3].plot(context[edge],data['context_flux'][edge],'.-',label='Measured red edge');a[3].plot(context[edge],rejected['prediction'][edge],label='Rejected old padded model');a[3].axvline(7400,color='k',ls='--',label='Independent support limit');a[3].legend(fontsize=8);a[3].set(title='Disclosed template-edge failure',xlabel='Rest vacuum wavelength [Angstrom]',ylabel='Flux [same units]')
    right=a[3].twinx();right.plot(context[edge],(rejected['prediction'][edge]-data['context_flux'][edge])/data['context_uncertainty'][edge],color='crimson',ls=':',lw=1.2);right.set_ylabel('Rejected residual / standard error',color='crimson');right.tick_params(axis='y',labelcolor='crimson')
    return fig
def population_figure(data,out,original):
    fig,ax=plt.subplots(2,2,figsize=(13,8),constrained_layout=True);a=ax.ravel();w=out['reconstruction'][0];age=data['age'];metal=data['metallicity']
    scatter=a[0].scatter(age,metal,c=w,s=35+400*w,cmap='viridis',vmin=0,vmax=w.max());fig.colorbar(scatter,ax=a[0],label='Light weight');a[0].set(xscale='log',yscale='log',xlabel='Age [Gyr]',ylabel='Metallicity [solar]',title='Recovered distribution on all 198 template nodes')
    x,y=marginal(w,age);a[1].stem(x,y,basefmt=' ');a[1].set(xscale='log',xlabel='Age [Gyr]',ylabel='Total light fraction',title='Age marginal; no invented intermediate ages')
    x,y=marginal(w,metal);a[2].stem(x,y,basefmt=' ');a[2].set(xscale='log',xlabel='Metallicity [solar]',ylabel='Total light fraction',title='Metallicity marginal')
    a[3].plot(w,label='Fresh');a[3].plot(original['reconstruction'][0],'--',label='Native');a[3].legend();a[3].set(xlabel='Public template index',ylabel='Light weight',title='All 198 population coefficients: native comparison')
    return fig
def dust_figure(data,out,result,no_dust,calzetti_fit):
    fig,ax=plt.subplots(2,2,figsize=(13,8),constrained_layout=True);a=ax.ravel();wave=data['wavelength'];e=out['ebv'][0,0];law=calzetti(e,wave);law/=np.median(law)
    a[0].plot(wave,result['highpass']['fine'],color='.75',lw=.5,label='Fine attenuation');a[0].plot(wave,out['attenuation'][0],label='Empirical running median');a[0].plot(wave,law,'--',label='Closest Calzetti curve');a[0].legend(fontsize=8);a[0].set(xlabel='Rest wavelength [Angstrom]',ylabel='Relative attenuation',title='Dust / continuum separation')
    a[1].plot(result['highpass']['dust_grid'],result['highpass']['dust_loss']);a[1].axvline(e,color='k',ls=':');a[1].set(xlabel='E(B-V) [mag]',ylabel='Clipped fourth-power discrepancy',title='Original discrete extinction-law fit',yscale='log')
    results=[result['prediction'],no_dust['prediction'],calzetti_fit['prediction']];labels=['Empirical curve','No attenuation','Calzetti-only']
    bars=a[2].bar(labels,[residual_statistic(p,data) for p in results]);a[2].bar_label(bars,fmt='%.3f');a[2].set(ylabel='Mean standardized squared residual',title='Three fresh complete population fits')
    x,y=marginal(out['reconstruction'][0],data['age']);a[3].plot(x,y,'o-',label='Empirical curve')
    for label,r in [('No attenuation',no_dust),('Calzetti-only',calzetti_fit)]:
        x,y=marginal(r['best_weights'],data['age']);a[3].plot(x,y,'.--',label=label)
    a[3].legend(fontsize=8);a[3].set(xscale='log',xlabel='Age [Gyr]',ylabel='Light fraction',title='Population sensitivity to continuum assumptions')
    return fig
def exploration_figure(data,result):
    fig,ax=plt.subplots(2,2,figsize=(13,8),constrained_layout=True);a=ax.ravel();fit=result['final'];delta=fit['chis']-fit['chis'].min();keep=delta<10
    ages=fit['weights'][keep]@np.log10(data['age']);metals=fit['weights'][keep]@np.log10(data['metallicity'])
    sc=a[0].scatter(ages,metals,c=delta[keep],s=10,cmap='plasma');fig.colorbar(sc,ax=a[0],label='Clipped objective above minimum');a[0].set(xlabel='Light-weighted log10(age/Gyr)',ylabel='Light-weighted log10(Z/solar)',title='Near-optimal candidate degeneracy; not a posterior')
    a[1].hist(np.maximum(delta,1e-6),bins=np.geomspace(1e-6,max(delta)+1,45));a[1].set(xscale='log',yscale='log',xlabel='Clipped objective above minimum (display floor 1e-6)',ylabel='Candidate count',title='All 19602 complete final candidates')
    for name,f in [('Highpass',result['highpass']['fit']),('Final',fit)]:
        a[2].plot(f['history'][:,0],f['history'][:,2],'.-',label=name);a[3].plot(f['history'][:,0],f['history'][:,1],'.-',label=name)
    a[2].legend();a[2].set(xlabel='BIC branching iteration',ylabel='Minimum clipped objective',title='Actual inverse progression')
    a[3].legend();a[3].set(xlabel='BIC branching iteration',ylabel='Retained pre-mixing candidates',title='Explicit convergence before full mixture expansion')
    return fig
