"""Scientific figure groups from computed arrays; no backend changes or solver calls."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm

def setup():
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.dpi':120,'savefig.dpi':170,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})

def _map(fig,ax,a,title,extent,unit,cmap='viridis',norm=None):
    im=ax.imshow(a,origin='lower',extent=extent,cmap=cmap,norm=norm,interpolation='nearest',aspect='equal')
    ax.set(title=title,xlabel='x / mm',ylabel='y / mm');fig.colorbar(im,ax=ax,shrink=.8,label=unit)
    return im

def _signed(a):
    bound=max(float(np.max(np.abs(a))),np.finfo(float).eps)
    return TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)

def _finish(fig,title,subtitle):
    fig.suptitle(title,fontsize=16,fontweight='bold');fig.text(.5,.01,subtitle,ha='center',fontsize=9);fig.tight_layout(rect=(0,.045,1,.95));return fig

def observations(pos,disp,sigma,extent):
    setup();fig,axs=plt.subplots(2,2,figsize=(12,9))
    for ax,a,title,unit in zip(axs.flat,[disp[...,0],disp[...,1],sigma[...,0],sigma[...,1]],['Signed horizontal displacement','Signed vertical displacement','MC random uncertainty: horizontal','MC random uncertainty: vertical'],['pixel']*4):
        _map(fig,ax,a,title,extent,unit,cmap='RdBu_r' if title.startswith('Signed') else 'magma',norm=_signed(a) if title.startswith('Signed') else None)
    return _finish(fig,'1 | Native observation and uncertainty','All32×32 author-existing simulated measurements; MC uncertainty is estimated from correlation, not true error.')

def calibration(cal,extent):
    x,y,gx,gy,sgx,sgy=cal;fig,axs=plt.subplots(2,2,figsize=(12,9))
    for ax,a,title in zip(axs.flat,[gx,gy,sgx,sgy],['Horizontal density gradient','Vertical density gradient','Qualified horizontal gradient sigma','Qualified vertical gradient sigma']):
        signed='sigma' not in title;_map(fig,ax,a,title,extent,'kg m$^{-4}$',cmap='RdBu_r' if signed else 'magma',norm=_signed(a) if signed else None)
    return _finish(fig,'2 | The optical conversion and its approximation','Sigma uses exact derivatives of the mean law; spatially independent calibration-error contributions are a conditional assumption.')

def results(wls,poisson,extent):
    fig,axs=plt.subplots(2,2,figsize=(12,9));r=[wls['reconstruction'][0],poisson['reconstruction'][0]];s=[wls['density_sigma'][0],poisson['density_sigma'][0]]
    rn=Normalize(min(a.min() for a in r),max(a.max() for a in r));sn=Normalize(0,max(a.max() for a in s))
    for i in range(2):
        _map(fig,axs[0,i],r[i],['Fresh WLS density','Fresh Poisson density'][i],extent,'kg m$^{-3}$',norm=rn)
        _map(fig,axs[1,i],s[i],['Fresh WLS standard uncertainty','Fresh Poisson standard uncertainty'][i],extent,'kg m$^{-3}$',cmap='magma',norm=sn)
    return _finish(fig,'3 | Two active integrations of the same full observations','Common density and uncertainty scales; neither reconstruction is an independent density truth.')

def comparisons(wls,poisson,upstream,extent,cal):
    fig,axs=plt.subplots(2,2,figsize=(12,9));a=wls['reconstruction'][0]-upstream['reconstruction'][0];b=poisson['reconstruction'][0]-wls['reconstruction'][0]
    for ax,v,title in [(axs[0,0],a,'Fresh WLS − qualified original'),(axs[0,1],b,'Poisson − WLS')]:_map(fig,ax,v,title,extent,'kg m$^{-3}$',cmap='RdBu_r',norm=_signed(v))
    d=wls['diagnostics'];n=32*31;residual=d['D']@wls['reconstruction'][0].ravel()-d['rhs'];standard=residual*np.sqrt(d['weights'])
    for ax,v,title in [(axs[1,0],standard[:n].reshape(32,31),'Horizontal standardized edge residual'),(axs[1,1],standard[n:2*n].reshape(31,32),'Vertical standardized edge residual')]:_map(fig,ax,v,title,extent,'standard deviations',cmap='RdBu_r',norm=_signed(v))
    return _finish(fig,'4 | Numerical reproduction and observation residuals','Symmetric signed scales include every value; different colorbar ranges distinguish roundoff from physical data mismatch.')

def uncertainty(wls,poisson,mc_sigma,uncorrected,extent):
    fig,axs=plt.subplots(2,2,figsize=(12,9));analytic=wls['density_sigma'][0]
    norm=Normalize(0,max(analytic.max(),mc_sigma.max()))
    _map(fig,axs[0,0],analytic,'Analytical conditional WLS sigma',extent,'kg m$^{-3}$',cmap='magma',norm=norm)
    _map(fig,axs[0,1],mc_sigma,'2048 independent linear-noise draws',extent,'kg m$^{-3}$',cmap='magma',norm=norm)
    axs[1,0].plot(analytic[16],label='Analytic');axs[1,0].plot(mc_sigma[16],'--',label='Empirical draws');axs[1,0].set(title='Central-row propagation check',xlabel='x sample',ylabel='Standard uncertainty / kg m$^{-3}$');axs[1,0].legend()
    for a,label,color in [(analytic,'Qualified WLS','#187f85'),(poisson['density_sigma'][0],'Poisson','#b06526'),(uncorrected['density_sigma'][0],'Original unit-defective WLS','#8b4b9f')]:
        values=np.sort(a.ravel());axs[1,1].plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=color)
    axs[1,1].set(xscale='log',title='Complete sigma distributions',xlabel='Standard uncertainty / kg m$^{-3}$',ylabel='Empirical cumulative fraction');axs[1,1].legend(fontsize=8)
    return _finish(fig,'5 | Covariance propagation and source qualification','Monte Carlo verifies the fixed conditional linear model, not physical coverage. Original sigma conversion has a proven unit defect.')

def sensitivity(wls,uniform,mc_only,noise_results,noise_levels,extent):
    fig,axs=plt.subplots(2,2,figsize=(12,9));a=uniform['reconstruction'][0]-wls['reconstruction'][0];b=mc_only[0]-wls['reconstruction'][0]
    _map(fig,axs[0,0],a,'Uniform-gradient weighting − selected WLS',extent,'kg m$^{-3}$',cmap='RdBu_r',norm=_signed(a))
    _map(fig,axs[0,1],b,'MC-only variance − full conditional model',extent,'kg m$^{-3}$',cmap='RdBu_r',norm=_signed(b))
    ref=wls['reconstruction'][0];errors=[np.sqrt(np.mean((r['reconstruction'][0]-ref)**2))/np.ptp(ref) for r in noise_results]
    axs[1,0].plot(noise_levels,errors,'o-',color='#187f85');axs[1,0].set(title='Fresh sensor-noise sensitivity',xlabel='Added MC-noise multiplier',ylabel='Range-normalized density RMS difference')
    labels=['Selected WLS','Uniform edges','MC-only'];fields=[wls['density_sigma'][0],uniform['density_sigma'][0],mc_only[1]]
    rms=[np.sqrt(np.mean(a*a)) for a in fields];axs[1,1].bar(labels,rms,color=['#187f85','#b06526','#8b4b9f']);axs[1,1].set(title='Conditional uncertainty depends on assumptions',ylabel='Whole-field RMS sigma / kg m$^{-3}$');axs[1,1].tick_params(axis='x',rotation=15)
    return _finish(fig,'6 | Weighting ablation and reproducible perturbations','Ablations are fresh full-grid computations; noise uses explicit seeds and leaves the bound benchmark unchanged.')

def save_group(fig,folder,name):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:fig.savefig(folder/(name+'.'+suffix),bbox_inches='tight')
