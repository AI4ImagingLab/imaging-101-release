"""Scientific plots and numerical exports; importing does not change backend."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import response,jacobian

def similarity(estimate,reference):
    """Return cosine NCC and dynamic-range NRMSE for finite nonconstant profiles."""
    a=np.asarray(estimate).ravel();b=np.asarray(reference).ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))

def make_figures(data,model,history,upstream,sensitivity,unconstrained):
    """Return four figure groups; every method result must be provided explicitly."""
    figures={};period=data['period'];z=data['layer_top'];pred=response(1/period,model,z)
    initial=response(1/period,history['models'][0],z)
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    names=['log10 apparent resistivity [ohm m]','Impedance phase [rad]']
    for k,(key,std) in enumerate([('log_apparent_resistivity','log_resistivity_std'),('impedance_phase','phase_std')]):
        ax[0,k].errorbar(period,data[key],yerr=data[std],fmt='o',ms=4,color='#36454f',label='Observed ±1 SD')
        ax[0,k].plot(period,initial[k],ls='--',color='#a9a9a9',label='Uniform initial profile')
        ax[0,k].plot(period,pred[k],color='#b83b32',label='Fresh corrected Occam')
        ax[0,k].set(xscale='log',xlabel='Period [s]',ylabel=names[k]);ax[0,k].legend(fontsize=8)
        ax[1,k].axhspan(-1,1,color='#e2ebde');ax[1,k].axhline(0,color='black',lw=.7)
        ax[1,k].plot(period,(pred[k]-data[key])/data[std],'o-',color='#326789')
        ax[1,k].set(xscale='log',xlabel='Period [s]',ylabel='Standardized residual [1]')
    fig.suptitle('Australian MT sounding: two measured response channels')
    figures['01_observations_and_residuals']=fig
    fig,ax=plt.subplots(1,3,figsize=(12,5),layout='constrained')
    # A step curve assigns each value to its actual depth interval; the final
    # half-space is shown to 200 km only for plotting and explicitly labeled.
    bottom=max(200000,z[-1]*1.2);edges=np.r_[z,bottom]/1000
    for values,label,color,style in [(history['models'][0],'Uniform initial','#999999','--'),(upstream,'Corrected original oracle','#326789','-'),(model,'Fresh local standard','#b83b32',':')]:
        ax[0].stairs(10**values,edges,orientation='horizontal',label=label,color=color,ls=style,lw=2)
    ax[0].set(xscale='log',ylim=(200,0),xlabel='Resistivity [ohm m]',ylabel='Depth [km]');ax[0].legend(fontsize=8)
    ax[0].set_title('Profile; half-space displayed to 200 km')
    ax[1].plot(np.arange(len(unconstrained['rms'])),unconstrained['rms'],'o-',color='#326789',label='Fresh target-zero path')
    ax[1].axhline(1,color='#b83b32',ls='--',label='Standard discrepancy target')
    ax[1].set(yscale='log',xlabel='Nonlinear update',ylabel='Weighted observation RMS [1]');ax[1].legend(fontsize=8)
    for k,values in enumerate(unconstrained['models'][1:]):
        ax[2].plot(values,np.maximum(z/1000,.0001),label=str(k+1),color=plt.cm.viridis(k/5))
    ax[2].set(yscale='log',ylim=(200,.0001),xlabel='log10 resistivity [ohm m]',ylabel='Layer-top depth [km]');ax[2].legend(title='Update',fontsize=7)
    fig.suptitle('Fresh nonlinear progression and independently reproduced reference')
    figures['02_profiles_and_convergence']=fig
    j1,j2,_=jacobian(1/period,model,z)
    weighted=np.vstack([j1/data['log_resistivity_std'][:,None],j2/data['phase_std'][:,None]])
    singular=np.linalg.svd(weighted,compute_uv=False)
    fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for k,j in enumerate([j1/data['log_resistivity_std'][:,None],j2/data['phase_std'][:,None]]):
        im=ax[k].imshow(np.log10(np.maximum(abs(j),1e-12)),aspect='auto',origin='lower',cmap='magma',extent=(-.5,24.5,-.5,22.5))
        ax[k].set(xlabel='Layer index (surface = 0)',ylabel='Observed period index',title=['Resistivity sensitivity','Phase sensitivity'][k]);fig.colorbar(im,ax=ax[k],label='log10 |standardized sensitivity|')
    ax[2].semilogy(np.arange(1,26),singular,'o-',color='#326789');ax[2].set(xlabel='Singular component',ylabel='Singular value [1]',title='Limited independent depth information')
    figures['03_jacobian_and_identifiability']=fig
    fig,ax=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    targets=[];residuals=[];roughness=[]
    for target,(m,h) in sensitivity.items():
        ax[0].stairs(10**m,edges,orientation='horizontal',label=f'Target {target:g}')
        targets.append(target);residuals.append(h['final_rms']);roughness.append(float(np.sum(np.diff(m)**2)))
    ax[0].set(xscale='log',ylim=(100,0),xlabel='Resistivity [ohm m]',ylabel='Depth [km]',title='Fresh discrepancy-target sensitivity');ax[0].legend(fontsize=8)
    ax[0].set_xticks([2,5,10,20,50],labels=['2','5','10','20','50']);ax[0].minorticks_off()
    ax[1].plot(residuals,roughness,'o-',color='#326789')
    for target,r,q in zip(targets,residuals,roughness):ax[1].annotate(f'{target:g}',(r,q),xytext=(4,4),textcoords='offset points')
    ax[1].set(xlabel='Achieved standardized RMS [1]',ylabel='Squared first-difference roughness [1]',title='Local search: fit / roughness tradeoff')
    for target,(m,h) in sensitivity.items():ax[2].plot(period,(response(1/period,m,z)[1]-data['impedance_phase'])/data['phase_std'],label=f'{target:g}')
    ax[2].axhspan(-1,1,color='#e2ebde',zorder=-1);ax[2].set(xscale='log',xlabel='Period [s]',ylabel='Phase residual / SD [1]',title='Frequency-resolved consequences')
    figures['04_regularization_sensitivity']=fig
    return figures

def export_figures(figures,sources,destination):
    """Write actual PNG/SVG/PDF figures and exact numeric sources with hashes."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    files=[]
    for name,fig in figures.items():
        for extension in ('png','svg','pdf'):
            path=destination/f'{name}.{extension}';fig.savefig(path,dpi=160);files.append(path)
    numeric=destination/'figure_sources.npz';np.savez_compressed(numeric,**sources);files.append(numeric)
    manifest={'task':'australia_magnetotelluric_occam_profile','reference_kind':'corrected original method; not actual truth','files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
