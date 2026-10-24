"""Real time-height lidar fields, forward residuals and prior sensitivity."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,TwoSlopeNorm
from .physics_model import OUTPUT,forward_signal

def make_figures(data,output,stages,source,experiments,export_dir=None):
    figures=[];r=data['range'][OUTPUT]/1000;t=(data['time_start']-data['time_start'][0])/60
    beta=output['reconstruction'][0];alpha=output['extinction'][0];signal=data['range_corrected_signal'][:,OUTPUT]
    bm=data['molecular_backscatter'][OUTPUT];am=data['molecular_extinction'][OUTPUT]
    observed=signal/signal[:,-1,None];prediction=forward_signal(beta,bm,am,reference=len(r)-1)
    extent=[t[0],t[-1],r[0],r[-1]]
    def finish(fig,title):fig.suptitle(title,fontsize=14);fig.tight_layout(rect=(0,0,1,.95));figures.append(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4.5));im=axs[0].imshow(signal.T,origin='lower',aspect='auto',extent=extent,cmap='magma',norm=LogNorm());fig.colorbar(im,ax=axs[0],label='Range-corrected return [ADC count m$^2$]');axs[0].set(xlabel='Minutes after first record',ylabel='Height above lidar [km]',title='All66real observation profiles')
    for i in [0,32,65]:axs[1].plot(observed[i],r,label=f'Profile{i+1}')
    axs[1].set(xlabel='Return / fitted reference return',ylabel='Height above lidar [km]');axs[1].legend()
    finish(fig,'1. Complete real2024Cordoba time-height acquisition; calibrated observations')
    fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    for ax,field,label in [(axs[0],beta*1e6,'Aerosol backscatter [Mm$^{-1}$ sr$^{-1}$]'),(axs[1],alpha*1e6,'Aerosol extinction [Mm$^{-1}$]')]:
        limit=np.max(abs(field));im=ax.imshow(field.T,origin='lower',aspect='auto',extent=extent,cmap='RdBu_r',norm=TwoSlopeNorm(0,-limit,limit));fig.colorbar(im,ax=ax,label=label);ax.set(xlabel='Minutes after first record',ylabel='Height above lidar [km]')
    finish(fig,'2. Fresh coupled aerosol fields; negative weak-signal estimates are retained')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));delta=beta-source['backscatter'];im=axs[0].imshow(delta.T,origin='lower',aspect='auto',extent=extent,cmap='coolwarm');fig.colorbar(im,ax=axs[0],label='Fresh minus C++original [m$^{-1}$ sr$^{-1}$]');axs[0].set(xlabel='Minutes after first record',ylabel='Height above lidar [km]')
    axs[1].scatter(source['backscatter'].ravel()[::50]*1e6,beta.ravel()[::50]*1e6,s=3);limits=np.array([beta.min(),beta.max()])*1e6;axs[1].plot(limits,limits,'k--');axs[1].set(xlabel='Original [Mm$^{-1}$ sr$^{-1}$]',ylabel='Fresh [Mm$^{-1}$ sr$^{-1}$]',title='Full-field numerical parity')
    finish(fig,'3. Independent original C++and fresh local field comparison')
    fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    for i,color in zip([0,32,65],['C0','C1','C2']):
        axs[0].plot(observed[i],r,color=color,alpha=.45,label=f'Observed{i+1}');axs[0].plot(prediction[i],r,color=color,ls='--',label=f'Forward{i+1}')
    axs[0].set(xlabel='Relative elastic return',ylabel='Height above lidar [km]');axs[0].legend(fontsize=7)
    relative=(prediction-observed)/np.sqrt(np.mean(observed**2));im=axs[1].imshow(relative.T,origin='lower',aspect='auto',extent=extent,cmap='coolwarm');fig.colorbar(im,ax=axs[1],label='Return residual / global observed RMS');axs[1].set(xlabel='Minutes after first record',ylabel='Height above lidar [km]')
    finish(fig,'4. Full nonlinear elastic-lidar forward consistency; smoothing/noise visible')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for lr,result in experiments.items():
        a=result['extinction'][0];axs[0].plot(np.mean(a,axis=0)*1e6,r,label=f'LR{lr}sr');column=np.trapezoid(a,data['range'][OUTPUT],axis=1);axs[1].plot(t,column,label=f'LR{lr}sr')
    axs[0].set(xlabel='Time-mean extinction [Mm$^{-1}$]',ylabel='Height above lidar [km]');axs[0].legend();axs[1].set(xlabel='Minutes after first record',ylabel='Extinction integral over evaluated heights [1]');axs[1].legend()
    finish(fig,'5. Actually inverted lidar-ratio sensitivity; measured-height integral only')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));raw=stages['beta_unsmoothed'][:,OUTPUT]
    for i in [0,32,65]:
        axs[0].plot(raw[i]*1e6,r,alpha=.3,label=f'Unsmooth{i+1}');axs[0].plot(beta[i]*1e6,r,label=f'Smooth{i+1}')
    axs[0].set(xlabel='Backscatter [Mm$^{-1}$ sr$^{-1}$]',ylabel='Height above lidar [km]');axs[0].legend(fontsize=7)
    axs[1].plot(t,np.mean(beta<0,axis=1)*100,label='Negative aerosol estimates');axs[1].set(xlabel='Minutes after first record',ylabel='Fraction of evaluated height bins [%]');axs[1].legend()
    finish(fig,'6. Preserved in-place smoothing and uncertainty of weak aerosol retrievals')
    if export_dir is not None:
        dest=Path(export_dir);dest.mkdir(parents=True,exist_ok=True)
        numeric={**{'input_'+k:v for k,v in data.items()},**{'fresh_'+k:v for k,v in output.items()},**{'stage_'+k:v for k,v in stages.items()},**{'source_'+k:v for k,v in source.items()},'forward_prediction':prediction,'normalized_observation':observed}
        for lr,result in experiments.items():
            for k,v in result.items():numeric[f'LR{lr}_{k}']=v
        np.savez_compressed(dest/'numeric_sources.npz',**numeric)
        for i,fig in enumerate(figures,1):
            for ext in ('png','svg','pdf'):fig.savefig(dest/f'figure_{i:02}.{ext}',dpi=150,bbox_inches='tight')
        (dest/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.suffix in ('.npz','.png','.svg','.pdf')},indent=2),encoding='utf8')
    return figures
