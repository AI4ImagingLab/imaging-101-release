"""Actual heat-transfer comparisons and private scientific exports."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt

def similarity(estimate,reference):
    """Cosine NCC and reference-dynamic-range NRMSE."""
    a=np.asarray(estimate).ravel();b=np.asarray(reference).ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))

def make_figures(data,q,terms,upstream,historical,sensitivity):
    """Return three groups; supplied method arrays are never replaced by caches."""
    half_pixel=.00635
    figures={};extent=(data['x'][1]-half_pixel,data['x'][-2]+half_pixel,data['y'][1]-half_pixel,data['y'][-2]+half_pixel);indexes=[0,12,24]
    fig,ax=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for col,index in enumerate(indexes):
        im=ax[0,col].imshow(data['temperature'][index,1:-1,1:-1],origin='lower',extent=extent,cmap='inferno',aspect='auto',vmin=data['temperature'].min(),vmax=data['temperature'].max())
        ax[0,col].set(title=f'Measured temperature, {data["time"][index]:g} s',xlabel='x [m]',ylabel='y [m]');fig.colorbar(im,ax=ax[0,col],label='K')
        im=ax[1,col].imshow(q[index],origin='lower',extent=extent,cmap='magma',aspect='auto',vmin=q.min(),vmax=q.max())
        ax[1,col].set(title=f'Fresh incident flux, {data["time"][index]:g} s',xlabel='x [m]',ylabel='y [m]');fig.colorbar(im,ax=ax[1,col],label='kW/m²')
    fig.suptitle('Real steel-panel thermography and full-window inverse heat balance')
    figures['01_measurements_and_reconstruction']=fig
    fig,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for axis,key,sign in zip(ax.ravel()[:4],['storage','convection','radiation','conduction'],[1,1,1,-1]):
        image=sign*terms[key][12]/940
        im=axis.imshow(image,origin='lower',extent=extent,aspect='auto',cmap='coolwarm' if key=='conduction' else 'viridis')
        axis.set(title=f'{key.capitalize()} contribution at 112 s',xlabel='x [m]',ylabel='y [m]');fig.colorbar(im,ax=axis,label='Incident-equivalent kW/m²')
    for key,sign in [('storage',1),('convection',1),('radiation',1),('conduction',-1)]:ax[1,1].plot(data['time'],sign*terms[key].mean(axis=(1,2))/940,label=key)
    ax[1,1].plot(data['time'],q.mean(axis=(1,2)),'k--',label='sum');ax[1,1].set(xlabel='Time after ignition [s]',ylabel='Spatial mean [kW/m²]',title='Actual four-term energy budget');ax[1,1].legend(fontsize=7)
    no_conduction=q+terms['conduction']/940;no_storage=q-terms['storage']/940
    ax[1,2].plot(data['time'],np.sqrt(np.mean((no_conduction-q)**2,axis=(1,2))),label='Omit lateral conduction')
    ax[1,2].plot(data['time'],np.sqrt(np.mean((no_storage-q)**2,axis=(1,2))),label='Omit thermal storage')
    ax[1,2].set(xlabel='Time after ignition [s]',ylabel='RMS field change [kW/m²]',yscale='log',title='Fresh omission diagnostics');ax[1,2].legend(fontsize=8)
    figures['02_energy_terms_and_ablations']=fig
    fig,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for axis,value,title in [(ax[0,0],upstream[12],'Reproduced original HFITS'),(ax[0,1],historical[12],'Historical author irradiance')]:
        im=axis.imshow(value,origin='lower',extent=extent,aspect='auto',cmap='magma');axis.set(title=title,xlabel='x [m]',ylabel='y [m]');fig.colorbar(im,ax=axis,label='kW/m²')
    difference=q[12]-upstream[12]
    im=ax[0,2].imshow(difference,origin='lower',extent=extent,aspect='auto',cmap='coolwarm');ax[0,2].set(title='Fresh local minus original HFITS',xlabel='x [m]',ylabel='y [m]');fig.colorbar(im,ax=ax[0,2],label='kW/m²')
    for h,field in sensitivity.items():ax[1,0].plot(data['time'],field.mean(axis=(1,2)),label=f'hf=hb={h:g}')
    ax[1,0].set(xlabel='Time after ignition [s]',ylabel='Mean incident flux [kW/m²]',title='Fresh convection sensitivity');ax[1,0].legend(fontsize=8)
    for h,field in sensitivity.items():ax[1,1].plot(np.mean(data['temperature'][:,1:-1,1:-1],axis=(1,2)),field.mean(axis=(1,2)),label=f'{h:g}')
    ax[1,1].set(xlabel='Mean observed temperature [K]',ylabel='Mean incident flux [kW/m²]',title='Exposure versus temperature')
    pairs=[('Fresh vs original',q-upstream),('Fresh vs historical',q-historical)]
    for name,delta in pairs:
        rms=np.sqrt(np.nanmean(delta**2,axis=(1,2)))
        ax[1,2].plot(data['time'],np.maximum(rms,1e-16),label=name)
    ax[1,2].set(xlabel='Time after ignition [s]',ylabel='RMS field difference [kW/m²]',yscale='log',title='Reproduction / historical differences');ax[1,2].legend(fontsize=8)
    fig.suptitle('Original reference, historical estimate and boundary assumptions')
    figures['03_reference_and_model_sensitivity']=fig
    return figures

def export_figures(figures,sources,destination):
    """Write PNG/SVG/PDF, exact NPZ plot sources and SHA256 manifest privately."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);files=[]
    for name,fig in figures.items():
        for extension in ('png','svg','pdf'):
            path=destination/f'{name}.{extension}';fig.savefig(path,dpi=160);files.append(path)
    numeric=destination/'figure_sources.npz';np.savez_compressed(numeric,**sources);files.append(numeric)
    manifest={'task':'fire_wall_inverse_heatflux','reference_kind':'unmodified original method; historical author estimate is separate, neither is actual truth','files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
