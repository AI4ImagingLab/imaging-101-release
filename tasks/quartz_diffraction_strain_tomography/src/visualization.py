"""Actual six-component truth, oracle and forward-model review figures."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
LABELS=('xx','yy','zz','xy','xz','yz')
def component_metrics(field,truth):
    return np.array([[np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),np.sqrt(np.mean((a-b)**2))/np.ptp(b)] for a,b in zip(field,truth)])

def make_figures(data,field,diagnostics,truth,original_gpu,original_cpu,cpu_rounding,cpu_projection,source_projection,short,short_diagnostics):
    model=diagnostics['model'];prediction=model.field_prediction(field);residual=prediction-data['strain_measurements'];sigma=data['noise_std'];valid=model.mask
    normalized=np.divide(residual,sigma,out=np.zeros_like(residual),where=sigma!=0)
    groups={};fig,axes=plt.subplots(3,6,figsize=(21,11),layout='constrained')
    for i,label in enumerate(LABELS):
        limit=max(abs(truth[i].min()),abs(truth[i].max()))*1e6
        for row,a,title in [(0,truth[i],'Author simulation truth'),(1,field[i],'Fresh corrected local'),(2,field[i]-truth[i],'Fresh minus truth')]:
            clim=limit if row<2 else max(abs((field[i]-truth[i]).min()),abs((field[i]-truth[i]).max()))*1e6
            im=axes[row,i].imshow(a*1e6,cmap='coolwarm',vmin=-clim,vmax=clim);axes[row,i].set_title(label+': '+title,fontsize=10);fig.colorbar(im,ax=axes[row,i],label='microstrain',shrink=.8)
            axes[row,i].set_xlabel('original column');axes[row,i].set_ylabel('original row')
    groups['01_six_tensor_components']=fig
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');methods=[original_gpu,field,original_cpu,cpu_rounding];names=['Original GPU (unmatched BP)','Fresh corrected GPU','Original CPU strip branch','CPU Joseph rounding diagnostic'];metrics=np.array([component_metrics(x,truth) for x in methods]);x=np.arange(6)
    for j,name in enumerate(names):
        axes[0,0].bar(x+(j-1.5)*.18,metrics[j,:,1],width=.18,label=name);axes[0,1].plot(x,metrics[j,:,0],marker='o',label=name)
    for ax in axes[0]:ax.set_xticks(x,LABELS);ax.set_xlabel('Tensor component');ax.legend(fontsize=7)
    axes[0,0].set_ylabel('Full-array dynamic-range NRMSE');axes[0,1].set_ylabel('Full-array cosine NCC')
    axes[0,0].set_title('Truth accuracy: separately identified methods');axes[0,1].set_title('All six components evaluated')
    axes[1,0].semilogy(diagnostics['history'][:,0],label='Fresh default: maxiter390');axes[1,0].semilogy(short_diagnostics['history'][:,0],label='Fresh diagnostic: maxiter20');axes[1,0].set_xlabel('Actual objective evaluation');axes[1,0].set_ylabel('Sum weighted squared residuals');axes[1,0].legend();axes[1,0].set_title('Original optimizer and tolerance retained')
    axes[1,1].bar(x-.15,component_metrics(short,truth)[:,1],width=.3,label='Fresh maxiter20');axes[1,1].bar(x+.15,component_metrics(field,truth)[:,1],width=.3,label='Fresh default108 iterations');axes[1,1].set_xticks(x,LABELS);axes[1,1].set_ylabel('Truth dynamic-range NRMSE');axes[1,1].legend();axes[1,1].set_title('Actual convergence comparison')
    groups['02_accuracy_convergence']=fig
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');arrays=[data['strain_measurements']*1e6,prediction*1e6,normalized,np.where(valid,data['noise_std']*1e6,np.nan),model.weights,np.sqrt(np.sum(model.sampler[3:]**2,axis=0))];titles=['Existing simulated measured strain','Fresh forward-predicted strain','Residual / supplied standard deviation','Propagated noise: material rays only','Normalized inverse-noise weight','Off-diagonal directional sensitivity'];units=['microstrain','microstrain','dimensionless','microstrain','dimensionless','dimensionless']
    for ax,a,title,unit in zip(axes.ravel(),arrays,titles,units):
        im=ax.imshow(a,aspect='auto',cmap='coolwarm' if 'strain' in title.lower() or 'Residual' in title else 'viridis');ax.set_title(title,fontsize=10);ax.set_xlabel('Detector bin');ax.set_ylabel('Acquisition event');fig.colorbar(im,ax=ax,label=unit)
    groups['03_diffraction_forward_residual']=fig
    probe=np.sin(np.arange(390)).reshape(6,65)*.001
    probe_field=np.array([model.basis.dot(c).reshape(106,96) for c in probe])
    fresh_projection=model.mean_project(probe_field)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');cpu_error=np.sqrt(np.mean((cpu_projection-source_projection)**2,axis=0));fresh_error=np.sqrt(np.mean((fresh_projection-source_projection)**2,axis=0))
    im=axes[0,0].imshow(cpu_error,aspect='auto');fig.colorbar(im,ax=axes[0,0],label='Probe strain RMS difference');axes[0,0].set_title('CPU arithmetic: texture-bin differences')
    im=axes[0,1].imshow(fresh_error,aspect='auto');fig.colorbar(im,ax=axes[0,1],label='Probe strain RMS difference');axes[0,1].set_title('Visible CUDA template vs source: exact')
    for ax in axes[0]:ax.set_xlabel('Detector bin');ax.set_ylabel('Acquisition event')
    diff=cpu_rounding-field;im=axes[1,0].imshow(diff[0]*1e6,cmap='coolwarm');fig.colorbar(im,ax=axes[1,0],label='microstrain');axes[1,0].set_title('CPU-rounding inverse minus exact GPU: xx');axes[1,0].set_xlabel('original column');axes[1,0].set_ylabel('original row')
    axes[1,1].bar(x,component_metrics(cpu_rounding,field)[:,1]);axes[1,1].set_xticks(x,LABELS);axes[1,1].set_ylabel('CPU vs corrected GPU dynamic-range NRMSE');axes[1,1].set_title('Inverse conditioning amplifies small perturbations')
    groups['04_numerical_fidelity']=fig
    numeric=dict(reconstruction=field,truth=truth,original_gpu=original_gpu,original_cpu=original_cpu,cpu_rounding=cpu_rounding,short_reconstruction=short,history=diagnostics['history'],short_history=short_diagnostics['history'],prediction=prediction,measurement=data['strain_measurements'],normalized_residual=normalized,noise_std=sigma,weights=model.weights,source_projection=source_projection,fresh_projection=fresh_projection,cpu_projection=cpu_projection,method_metrics=metrics)
    summary={'iterations':diagnostics['iterations'],'objective':diagnostics['objective'],'weighted_residual_norm':float(np.linalg.norm(model.weights*residual)),'standardized_residual_rms_inside_rays':float(np.sqrt(np.mean(normalized[valid]**2))),'truth_components':component_metrics(field,truth).tolist(),'cpu_projection_relative_error':float(np.linalg.norm(cpu_projection-source_projection)/np.linalg.norm(source_projection)),'fresh_projection_max_error':float(np.max(abs(fresh_projection-source_projection)))}
    return groups,numeric,summary

def export_figures(groups,numeric,summary,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);files=[]
    for name,fig in groups.items():
        for extension in ['png','svg','pdf']:
            path=directory/(name+'.'+extension);fig.savefig(path,dpi=150);files.append(path)
    path=directory/'numeric_sources.npz';np.savez_compressed(path,**numeric);files.append(path)
    path=directory/'measurements.json';path.write_text(json.dumps(summary,indent=2),encoding='utf8');files.append(path)
    manifest={'description':'Existing author simulation; actual fresh corrected reconstruction and shortened convergence run; separately identified unmodified GPU/CPU and CPU rounding captures.','files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files]};(directory/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
