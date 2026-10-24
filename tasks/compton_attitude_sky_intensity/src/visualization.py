"""Scientific review plots; numerical inputs are supplied by active callers."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from .physics_model import ring_pixel_angles,PIXEL_AREA

COLORS=['#306c9c','#d27b38','#438a72','#975c8e']

def image_metrics(estimate,reference):
    a=np.asarray(estimate,dtype=np.float64).ravel();b=np.asarray(reference,dtype=np.float64).ravel()
    return dict(ncc=float(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)) if np.linalg.norm(a)>0 else 0.,nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))

def _sky(values):
    theta,phi=ring_pixel_angles()
    points=np.column_stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    longitude,latitude=np.meshgrid(np.deg2rad(np.linspace(180,-180,361)),np.deg2rad(np.linspace(-90,90,181)))
    locations=np.stack([np.cos(latitude)*np.cos(longitude),np.cos(latitude)*np.sin(longitude),np.sin(latitude)],axis=-1)
    nearest=cKDTree(points).query(locations)[1]
    return np.asarray(values).ravel()[nearest]

def _map(ax,values,title,vmin=0,vmax=None,cmap='magma'):
    artist=ax.imshow(_sky(values),extent=[180,-180,-90,90],origin='lower',aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set(title=title,xlabel='Galactic longitude [deg]',ylabel='Latitude [deg]');ax.set_xticks([180,90,0,-90,-180]);ax.set_yticks([-90,-45,0,45,90])
    return artist

def export_figure(fig,destination,arrays,metadata):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:fig.savefig(destination.with_suffix('.'+suffix),dpi=180,bbox_inches='tight',facecolor='white')
    np.savez_compressed(destination.with_suffix('.npz'),**arrays)
    destination.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')

def acquisition_figure(data):
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    events=data.events.sum(axis=1,dtype=np.uint64);background=data.background.sum(axis=1,dtype=np.uint64)
    axes[0].plot(events,label='Observed total',color=COLORS[0]);axes[0].plot(background,label='Known injected background',color=COLORS[1]);axes[0].plot(events-background,label='Source counts',color=COLORS[2]);axes[0].set(xlabel='Attitude bin',ylabel='Photon counts',title='Complete observed acquisition');axes[0].legend(fontsize=8)
    histogram=np.zeros(256,dtype=np.uint64)
    for row in data.events:histogram+=np.bincount(row,minlength=256).astype(np.uint64)
    valid=histogram>0;axes[1].bar(np.flatnonzero(valid),histogram[valid],color=COLORS[0]);axes[1].set(xlabel='Photons per bin',ylabel='Number of measurement bins',yscale='log',title='Sparse counting regime')
    profiles=[]
    for index,color in zip([0,768,1536],COLORS):
        profile=data.operator.response[index].reshape(60,3072).sum(axis=1,dtype=np.float64)*data.operator.response_area
        profiles.append(profile);axes[2].plot(np.arange(60)*3+1.5,profile,label=f'Incoming pixel {index}',color=color)
    axes[2].set(xlabel='Compton scatter angle [deg]',ylabel='Direction-summed response [cm²]',title='Independent detector calibration');axes[2].legend(fontsize=8)
    return fig,dict(attitude_events=events,attitude_background=background,count_histogram=histogram,response_profiles=np.asarray(profiles))

def iteration_figure(result,original_maps):
    fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained');maximum=float(np.max(original_maps))
    for column,iteration in enumerate([1,7,30]):
        for row,(label,maps) in enumerate([('Original source',original_maps),('Fresh local',result.maps)]):
            artist=_map(axes[row,column],maps[iteration-1],f'{label}: update {iteration}',vmax=maximum)
    fig.colorbar(artist,ax=axes.ravel().tolist(),label='Band intensity [photons cm⁻² s⁻¹ sr⁻¹]',shrink=.85)
    return fig,dict(original_maps=original_maps[[0,6,29]],fresh_maps=result.maps[[0,6,29]])

def parity_figure(result,reference):
    estimate=result.reconstruction.ravel();reference=np.asarray(reference).ravel();error=estimate.astype(np.float64)-reference
    bound=max(float(np.max(abs(error))),np.finfo(float).eps)
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    artist=_map(axes[0],estimate,'Fresh complete 30-update field',vmax=float(reference.max()));fig.colorbar(artist,ax=axes[0],label='photons cm⁻² s⁻¹ sr⁻¹')
    artist=_map(axes[1],error,'Fresh minus original: signed error',vmin=-bound,vmax=bound,cmap='RdBu_r');fig.colorbar(artist,ax=axes[1],label='photons cm⁻² s⁻¹ sr⁻¹')
    axes[2].scatter(reference,estimate,s=8,alpha=.5,color=COLORS[0]);axes[2].plot([0,reference.max()],[0,reference.max()],color='black',lw=1)
    values=image_metrics(estimate,reference);axes[2].set(xlabel='Original intensity',ylabel='Fresh intensity',title=f"All3072pixels: NCC={values['ncc']:.10f}\nNRMSE={values['nrmse']:.3g}")
    return fig,dict(reference=reference,fresh=estimate,signed_error=error)

def forward_figure(data,source_expectation):
    observed=(data.events.astype(np.int16)-data.background.astype(np.int16)).reshape(278,60,3072)
    expected=np.asarray(source_expectation).reshape(278,60,3072)
    fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained');arrays={};statistics={}
    for col,(name,reduce_axes) in enumerate([('Attitude',(1,2)),('Scatter angle',(0,2)),('Scattered direction',(0,1))]):
        obs=observed.sum(axis=reduce_axes,dtype=np.float64);pred=expected.sum(axis=reduce_axes,dtype=np.float64)
        good=pred>=20;residual=np.divide(obs-pred,np.sqrt(pred),out=np.zeros_like(obs),where=pred>0)
        pearson=float(np.mean(residual[good]**2));statistics[name]=dict(pearson_per_bin=pearson,counts_covered=float(obs[good].sum()/obs.sum()))
        axes[0,col].plot(obs,label='Observed source counts',color=COLORS[0],lw=1);axes[0,col].plot(pred,label='Fresh folded expectation',color=COLORS[1],lw=1);axes[0,col].set(title=name,xlabel='Physical bin index',ylabel='Photon counts')
        axes[1,col].plot(np.flatnonzero(good),residual[good],'.',ms=2,color=COLORS[0]);axes[1,col].axhline(0,color='black',lw=.7);axes[1,col].set(xlabel='Bins with expected count ≥20',ylabel='(observed − expected) / sqrt(expected)',title=f'Descriptive Pearson/bin={pearson:.3f}')
        arrays[name+'_observed']=obs;arrays[name+'_expected']=pred;arrays[name+'_residual']=residual;arrays[name+'_normal_approx_display_mask']=good
    axes[0,0].legend(fontsize=8);fig.suptitle('All counts enter the forward totals; bottom panels restrict only the normal-approximation display',fontsize=11)
    return fig,arrays,statistics

def convergence_figure(result,original_maps,original_likelihood):
    iterations=np.arange(1,len(result.maps)+1);fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    axes[0].plot(iterations,original_likelihood,label='Original',lw=2,color=COLORS[1]);axes[0].plot(iterations,result.log_likelihood,'--',label='Fresh',color=COLORS[0]);axes[0].set(xlabel='Update',ylabel='Poisson log likelihood',title='Full-array objective');axes[0].legend()
    flux=result.maps.sum(axis=1,dtype=np.float64)*PIXEL_AREA
    axes[1].plot(iterations,flux,color=COLORS[2]);axes[1].set(xlabel='Update',ylabel='Integrated source flux [photons cm⁻² s⁻¹]',title='Absolute flux convergence')
    error=np.asarray([image_metrics(a,original_maps[-1])['nrmse'] for a in result.maps]);axes[2].semilogy(iterations,error,color=COLORS[0]);axes[2].set(xlabel='Update',ylabel='Dynamic-range NRMSE vs original update30',title='Finite-iteration sensitivity')
    return fig,dict(iteration=iterations,original_likelihood=original_likelihood,fresh_likelihood=result.log_likelihood,flux=flux,error_vs_original30=error)

def negative_controls_figure(result,reference,exposure,source_count):
    reference=np.asarray(reference).ravel();fresh=result.reconstruction.ravel()
    fields=[fresh,np.zeros_like(fresh),np.full_like(fresh,np.mean(fresh)),fresh*.95,np.roll(fresh,307)]
    labels=['Fresh','Zero','Flat','0.95 scale','Permuted'];values=[image_metrics(a,reference) for a in fields]
    closure=np.asarray([abs(np.dot(np.asarray(exposure,dtype=np.float64),a)-source_count)/source_count for a in fields])
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for axis,name,bound in zip(axes,['ncc','nrmse','closure'],[.999999,1e-5,.001]):
        numbers=np.asarray([v[name] for v in values]) if name!='closure' else closure
        axis.bar(labels,numbers,color=[COLORS[0],*[COLORS[1]]*4]);axis.axhline(bound,color='black',ls='--',label='Frozen boundary');axis.tick_params(axis='x',rotation=30);axis.set(title=name.upper(),ylabel='Measured metric');axis.legend(fontsize=8)
        if name!='ncc':axis.set_yscale('symlog',linthresh=1e-8)
    return fig,dict(fields=np.asarray(fields),ncc=np.asarray([v['ncc'] for v in values]),nrmse=np.asarray([v['nrmse'] for v in values]),source_count_closure=closure)
