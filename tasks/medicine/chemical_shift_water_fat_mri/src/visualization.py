"""Scientific plots from fresh maps and independently identified author outputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt

def map_panel(ax,values,title,cmap='gray',vmin=None,vmax=None,unit='signal a.u.'):
    im=ax.imshow(values.T,origin='lower',extent=(-.75,150.75,-.75,150.75),cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set(title=title,xlabel='x (mm)',ylabel='y (mm)')
    plt.colorbar(im,ax=ax,fraction=.046,pad=.03,label=unit)
    return im

def metrics_for_display(a,b):
    """Dynamic-range RMS and fat dominance disagreement; no hidden true label."""
    result={}
    for key in a:
        ref=b[key].astype(float);estimate=a[key].astype(float)
        result[key+'_nrmse']=float(np.sqrt(np.mean((estimate-ref)**2))/np.ptp(ref))
    mask=(b['water']+b['fat'])>.05*np.max(b['water']+b['fat'])
    result['fat_dominance_disagreement_percent']=float(100*np.mean(((a['fat_fraction']>50)!=(b['fat_fraction']>50))[mask]))
    return result

def make_figures(data,fresh,original,variants,noise_results,diagnostics,unmodified):
    raw=data['echoes'][0];figures=[];numeric={'observed_echoes':raw}
    for name,result in {'canonical':fresh,**variants}.items():
        for key,value in result.items():numeric[name+'_'+key]=value
    for key,value in diagnostics.items():numeric['diagnostic_'+key]=value
    z=1
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    fig.suptitle('Bound challenge case17 | three complex echoes, complete acquisition retained')
    magnitude_max=float(np.abs(raw).max())
    for n in range(3):
        map_panel(axes[0,n],np.abs(raw[n,:,:,z]),f'Echo{n+1}, TE={1000*[.00287,.00607,.00927][n]:.2f} ms',vmin=0,vmax=magnitude_max)
        map_panel(axes[1,n],np.angle(raw[n,:,:,z]),f'Observed phase, echo{n+1}',cmap='twilight',vmin=-np.pi,vmax=np.pi,unit='radian')
    figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True);fig.suptitle('Fresh complete-volume chemical separation | first-echo coefficient convention')
    for ax,key,cmap,unit in zip(axes.flat,['water','fat','fat_fraction','fieldmap','r2star'],['gray','gray','magma','twilight','viridis'],['signal a.u.','signal a.u.','percent','ppm','s^-1']):
        map_panel(ax,fresh[key][0,:,:,z],key,cmap=cmap,vmin=0,unit=unit)
    map_panel(axes[1,2],np.sqrt(np.mean(np.abs(diagnostics['predicted_echoes']-raw)**2,axis=0))[:,:,z],'Fresh complex model RMS residual',cmap='inferno')
    figures.append(fig)
    names=['canonical','single_scale','ICM_only','weak_smoothness','strong_smoothness'];all_results={'canonical':fresh,**variants}
    fig,axes=plt.subplots(3,5,figsize=(20,11),constrained_layout=True);fig.suptitle('Actual reconstructed method/regularization comparisons | same complete observations')
    for col,name in enumerate(names):
        result=all_results[name]
        map_panel(axes[0,col],result['water'][0,:,:,z],name,unit='water a.u.')
        map_panel(axes[1,col],result['fat_fraction'][0,:,:,z],'Fat signal fraction',cmap='magma',vmin=0,vmax=100,unit='percent')
        map_panel(axes[2,col],result['fieldmap'][0,:,:,z],'Alias-interval field',cmap='twilight',vmin=0,vmax=4.92,unit='ppm')
    figures.append(fig)
    fig,axes=plt.subplots(2,4,figsize=(18,9),constrained_layout=True);fig.suptitle('Fresh added-noise experiments | unperturbed standard is a comparison, not true tissue')
    noise_metrics={};levels=[];disagreement={'multi':[],'single':[]}
    for index,(level,pair) in enumerate(noise_results.items()):
        levels.append(float(level))
        for row,mode in enumerate(['multi','single']):
            result=pair[mode];map_panel(axes[row,index],result['fat_fraction'][0,:,:,z],f'{mode}, complex noise={level}',cmap='magma',vmin=0,vmax=100,unit='percent')
            score=metrics_for_display(result,fresh);noise_metrics[str(level)+'_'+mode]=score;disagreement[mode].append(score['fat_dominance_disagreement_percent'])
            for key,value in result.items():numeric[f'noise_{level}_{mode}_{key}']=value
    figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True);fig.suptitle('Measured-signal consistency and multiscale noise behavior | no anatomical truth inferred')
    pred=diagnostics['predicted_echoes'];residual=pred-raw
    axes[0,0].scatter(raw.real.ravel()[::13],pred.real.ravel()[::13],s=2,alpha=.3);axes[0,0].set(xlabel='observed real signal (a.u.)',ylabel='predicted real signal (a.u.)',title='All slices; sampled for display')
    axes[0,1].hist(residual.real.ravel(),bins=100,histtype='step',label='real');axes[0,1].hist(residual.imag.ravel(),bins=100,histtype='step',label='imaginary');axes[0,1].set(xlabel='complex model residual (a.u.)',ylabel='voxel-echo count',title='Noise and model mismatch');axes[0,1].legend()
    for mode in disagreement:axes[0,2].plot(levels,disagreement[mode],'o-',label=mode)
    axes[0,2].set(xlabel='added noise / foreground signal',ylabel='fat-dominance disagreement (%)',title='Relative to unperturbed standard');axes[0,2].legend()
    axes[1,0].plot(diagnostics['profile_labels_ppm'],diagnostics['profile_costs']);axes[1,0].set(xlabel='field candidate (ppm)',ylabel='complex residual (scaled)',title='Actual voxel variable-projection landscape')
    for index,axis in enumerate(axes[1,1:]):
        key=['water','fat_fraction'][index]
        for name in names:axis.plot(all_results[name][key][0,:,50,z],label=name)
        axis.set(xlabel='stored x index',ylabel='signal a.u.' if key=='water' else 'percent',title=key+' cross-section')
        axis.legend(fontsize=7)
    figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True);fig.suptitle('Author-source parity, volume coverage and explicit scientific limits')
    keys=['water','fat','fat_fraction','fieldmap','r2star'];error=[]
    for key in keys:error.append(metrics_for_display(fresh,original)[key+'_nrmse'])
    axes[0,0].plot(keys,error,'o');axes[0,0].tick_params(axis='x',rotation=25);axes[0,0].set(ylabel='dynamic-range NRMSE',title='Fresh versus captured author result',ylim=(-1e-7,1e-6))
    for index,value in enumerate(error):axes[0,0].annotate(f'{value:.2g}',(index,value),xytext=(0,8),textcoords='offset points',ha='center')
    axes[0,1].hist(fresh['r2star'].ravel(),bins=np.arange(-.5,145.5),histtype='step');axes[0,1].set(xlabel='R2* (s^-1)',ylabel='voxel count',title='Three echoes: decay identifiability is limited')
    for plane in range(4):axes[0,2].plot(np.mean(fresh['fat_fraction'][0,:,:,plane],axis=0),label=f'z={5*plane:.0f} mm')
    axes[0,2].set(xlabel='y index',ylabel='mean fat signal fraction (%)',title='All four slices retained');axes[0,2].legend(fontsize=7)
    map_panel(axes[1,0],fresh['fat_fraction'][0,:,:,0],'First slice',cmap='magma',vmin=0,vmax=100,unit='percent')
    map_panel(axes[1,1],fresh['fat_fraction'][0,:,:,3],'Fourth slice',cmap='magma',vmin=0,vmax=100,unit='percent')
    axes[1,2].axis('off');axes[1,2].text(0,.95,'No independent spatial truth is supplied.\nReference agreement is numerical reproduction.\nThe field is reported in one echo-alias interval.\nFat signal fraction is not calibrated tissue fat.\nLow-signal voxels and three-echo decay are fragile.\nAdded-noise curves measure comparison stability,\nnot historical multi-case clinical accuracy.',va='top',fontsize=12)
    figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True);fig.suptitle('Qualified pair-energy correction | captured unmodified source remains explicitly identified')
    map_panel(axes[0,0],unmodified['fat_fraction'][0,:,:,z],'Captured unmodified source',cmap='magma',vmin=0,vmax=100,unit='percent')
    map_panel(axes[0,1],fresh['fat_fraction'][0,:,:,z],'Fresh qualified field energy',cmap='magma',vmin=0,vmax=100,unit='percent')
    map_panel(axes[0,2],fresh['fat_fraction'][0,:,:,z]-unmodified['fat_fraction'][0,:,:,z],'Qualified minus unmodified',cmap='coolwarm',vmin=-100,vmax=100,unit='percentage point')
    source_defect=metrics_for_display(unmodified,fresh)
    axes[1,0].bar(keys,[source_defect[key+'_nrmse'] for key in keys]);axes[1,0].tick_params(axis='x',rotation=25);axes[1,0].set(ylabel='dynamic-range NRMSE',title='Effect on all five complete maps')
    axes[1,1].bar(['unmodified chosen labels','qualified chosen labels'],[16,3]);axes[1,1].set(ylabel='declared spatial energy',title='Independent asymmetric two-voxel proof')
    axes[1,1].tick_params(axis='x',rotation=15)
    axes[1,2].axis('off');axes[1,2].text(0,.95,'Candidates at q,p: A=(0,1), B=(2,4).\nThe q-first squared-distance table is(1,16,1,4).\nPassing it as(p,q) interchanges mixed states.\nCorrecting endpoints to(q,p) restores that energy.\nOnly this primitive endpoint order is changed.\nNo threshold is relaxed and no result is truth.\nMuch of the change is in fragile low signal.',va='top',fontsize=11)
    figures.append(fig)
    for key,value in unmodified.items():numeric['unmodified_original_'+key]=value
    numeric['endpoint_toy_table']=np.array([1,16,1,4]);numeric['endpoint_toy_selected_physical_energies']=np.array([16,3])
    metrics={'variant_comparisons':{name:metrics_for_display(result,fresh) for name,result in variants.items()},'source_comparison':metrics_for_display(fresh,original),'unmodified_source_comparison':source_defect,'noise_comparisons':noise_metrics,'forward_rms':float(np.sqrt(np.mean(np.abs(residual)**2))),'interpretation':'All fresh comparisons computed; captured unmodified source is labeled. Endpoint correction independently justified. No spatial truth. Added-noise agreement is relative to unperturbed standard only.'}
    return figures,numeric,metrics

def export_figures(figures,numeric,metrics,directory):
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    for index,figure in enumerate(figures,1):
        for suffix in ['png','svg','pdf']:figure.savefig(path/f'figure_{index}.{suffix}',dpi=150)
    np.savez_compressed(path/'numeric_sources.npz',**numeric)
    (path/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n',encoding='utf8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file() and p.name!='hash_manifest.json'}
    (path/'hash_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf8')
