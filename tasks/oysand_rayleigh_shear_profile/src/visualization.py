"""Real dispersion, stratified fields, actual CPSO history and sensitivity."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from .solvers import physical_model
from .physics_model import phase_velocity

def layer_curve(parameters):
    h=parameters[:3]*1000;vs=parameters[3:7]*1000
    edges=np.r_[0,np.cumsum(h),np.sum(h)+5.]
    return np.repeat(vs,2),np.column_stack([edges[:-1],edges[1:]]).ravel()

def make_figures(data,output,stages,source,export_dir=None):
    figures=[];history=stages['parameters'];loss=stages['objectives'];best=np.r_[output['thickness'][0]/1000,output['reconstruction'][0]/1000,[.3,.3,.49,.49]]
    model=physical_model(best);pred=phase_velocity(data['period'],*model.T);period=data['period'];observed=data['phase_velocity'];sigma=data['uncertainty'];wavelength=period*observed*1000
    def finish(fig,title):fig.suptitle(title,fontsize=14);fig.tight_layout(rect=(0,0,1,.95));figures.append(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));axs[0].errorbar(1/period,observed*1000,sigma*1000,fmt='o',ms=4);axs[0].set(xlabel='Frequency [Hz]',ylabel='Phase velocity [m/s]',title='Every real composite dispersion point')
    axs[1].errorbar(wavelength,observed*1000,sigma*1000,fmt='o',ms=4);axs[1].set(xlabel='Wavelength [m]',ylabel='Phase velocity [m/s]',title='Author uncertainty half-widths')
    finish(fig,'1. Oysand real surface-wave observations and known uncertainty weights')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));axs[0].errorbar(period,observed*1000,sigma*1000,fmt='k.',label='Measured');axs[0].plot(period,pred*1000,label='Fresh explicit Dunkin/CPSO');axs[0].plot(period,source['prediction']*1000,'--',label='Reproduced original');axs[0].set(xlabel='Period [s]',ylabel='Phase velocity [m/s]');axs[0].legend(fontsize=8)
    axs[1].bar(np.arange(len(period)),(pred-observed)/sigma);axs[1].axhline(0,color='k',lw=.7);axs[1].axhspan(-1,1,color='gray',alpha=.12);axs[1].set(xlabel='Observed period index',ylabel='Velocity residual / supplied uncertainty')
    finish(fig,'2. Full dispersion fit and every standardized residual')
    fig,axs=plt.subplots(1,2,figsize=(11,5));x,z=layer_curve(best);axs[0].plot(x,z,'C0',lw=2,label='Fresh');source_params=np.r_[source['model'][:-1,0],source['model'][:,2],[.3,.3,.49,.49]];xs,zs=layer_curve(source_params);axs[0].plot(xs,zs,'k--',label='Reproduced original');axs[0].invert_yaxis();axs[0].set(xlabel='Shear velocity Vs [m/s]',ylabel='Depth below surface [m]');axs[0].legend()
    labels=['h1','h2','h3','Vs1','Vs2','Vs3','Vs4'];error=np.r_[output['thickness'][0]-source['model'][:-1,0]*1000,output['reconstruction'][0]-source['model'][:,2]*1000];axs[1].bar(labels,error);axs[1].set(ylabel='Fresh minus original [m or m/s]',title='All seven independent unknowns')
    finish(fig,'3. Reconstructed piecewise spatial field and every layer thickness')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));global_loss=stages['global_objectives'];axs[0].plot(np.arange(1,len(global_loss)+1),global_loss,label='Fresh');axs[0].plot(np.arange(1,len(source['global_misfits'])+1),source['global_misfits'],'--',label='Original');axs[0].set(xlabel='CPSO generation',ylabel='Best uncertainty-weighted RMS',yscale='log');axs[0].legend()
    finite=np.where(np.isfinite(loss),loss,np.nan);axs[1].plot(np.arange(1,len(loss)+1),np.nanmedian(finite,axis=1),label='Finite population median');axs[1].plot(np.arange(1,len(loss)+1),np.nanmin(finite,axis=1),label='Generation minimum');axs[1].set(xlabel='Generation',ylabel='Weighted RMS',yscale='log');axs[1].legend()
    finish(fig,'4. Actually executed 40-particle,300-generation search; no cached optimization')
    fig,axs=plt.subplots(1,2,figsize=(12,5));flat=history.reshape(-1,11);lf=loss.ravel();selected=np.flatnonzero(np.isfinite(lf)&(lf<np.nanmin(lf)+.2));selected=selected[np.linspace(0,len(selected)-1,min(80,len(selected))).astype(int)]
    for i in selected:
        xx,zz=layer_curve(flat[i]);axs[0].plot(xx,zz,color='C0',alpha=.08)
    axs[0].plot(x,z,'k',lw=2);axs[0].invert_yaxis();axs[0].set(xlabel='Vs [m/s]',ylabel='Depth [m]',title='Actual low-misfit explored models')
    im=axs[1].scatter(flat[selected,2]*1000,flat[selected,5]*1000,c=lf[selected],s=15);fig.colorbar(im,ax=axs[1],label='Weighted RMS');axs[1].set(xlabel='Third layer thickness [m]',ylabel='Third layer Vs [m/s]',title='Thickness/velocity tradeoff; not a posterior')
    finish(fig,'5. Nonuniqueness of a seven-parameter depth field under fixed elastic priors')
    sensitivity=[]
    for k in range(7):
        delta=best[k]*.01;plus=best.copy();minus=best.copy();plus[k]+=delta;minus[k]-=delta
        p1=phase_velocity(period,*physical_model(plus).T);p0=phase_velocity(period,*physical_model(minus).T)
        sensitivity.append((p1-p0)/(.02*observed))
    sensitivity=np.asarray(sensitivity)
    fig,axs=plt.subplots(1,2,figsize=(13,4.5));im=axs[0].imshow(sensitivity,aspect='auto',cmap='coolwarm',origin='lower');axs[0].set(yticks=np.arange(7),yticklabels=labels,xlabel='Observed period index',ylabel='Perturbed unknown');fig.colorbar(im,ax=axs[0],label='Delta c / (0.02 c_observed), +/-1% parameter')
    for k in range(3,7):axs[1].plot(period,sensitivity[k],label=labels[k])
    axs[1].set(xlabel='Period [s]',ylabel='Observed-speed-normalized sensitivity');axs[1].legend()
    finish(fig,'6. Actually evaluated local physical sensitivity of all seven unknowns')
    if export_dir is not None:
        dest=Path(export_dir);dest.mkdir(parents=True,exist_ok=True)
        numeric={**{'input_'+k:v for k,v in data.items()},**{'fresh_'+k:v for k,v in output.items()},**{'stages_'+k:np.asarray(v) for k,v in stages.items()},**{'source_'+k:v for k,v in source.items()},'prediction':pred,'sensitivity':sensitivity,'selected_ensemble_indices':selected}
        np.savez_compressed(dest/'numeric_sources.npz',**numeric)
        for i,fig in enumerate(figures,1):
            for ext in ('png','svg','pdf'):fig.savefig(dest/f'figure_{i:02}.{ext}',dpi=150,bbox_inches='tight')
        (dest/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.suffix in ('.npz','.png','.svg','.pdf')},indent=2),encoding='utf8')
    return figures
