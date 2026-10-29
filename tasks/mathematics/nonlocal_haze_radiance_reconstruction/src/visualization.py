"""Computed scientific comparisons; no backend mutation and no cached-as-fresh."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import recover_radiance,image_formation,rgb_to_gray
from .solvers import reconstruct,refine_transmission,contrast_stretch

def compare(estimate,reference):
    x=np.asarray(estimate,dtype=np.float64).ravel();r=np.asarray(reference,dtype=np.float64).ravel()
    return {'ncc':float(x@r/(np.linalg.norm(x)*np.linalg.norm(r))),'nrmse':float(np.sqrt(np.mean((x-r)**2))/np.ptp(r))}

def style():
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':10,'figure.dpi':110,'savefig.dpi':160,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})

def image_panel(ax,image,title,cmap=None,vmin=None,vmax=None,colorbar=False):
    artist=ax.imshow(image,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
    ax.set(title=title,xlabel='Column (pixel)',ylabel='Row (pixel)')
    if colorbar:ax.figure.colorbar(artist,ax=ax,shrink=.8,pad=.02)
    return artist

def observations_and_recovery(image,restored,t):
    style();fig,ax=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    image_panel(ax[0],image,'A | Real hazy observation')
    image_panel(ax[1],restored,'B | Fresh restored display (counts)')
    image_panel(ax[2],t,'C | Fresh transmission (dimensionless)','viridis',.1,1,True)
    return fig

def haze_line_mechanism(image,airlight,stages):
    style();fig,ax=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    image_panel(ax[0],stages['weight'],'A | Haze-line reliability','magma',0,1,True)
    centered=(image-airlight).reshape(-1,3);labels=stages['labels'].ravel()
    bins=np.argsort(stages['count'])[-5:];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    for b,c in zip(bins,colors):
        ids=np.flatnonzero(labels==b);ids=ids[::max(1,len(ids)//600)]
        ax[1].scatter(centered[ids,0],centered[ids,1],s=5,alpha=.4,color=c,label=f'Line {b}, n={stages["count"][b]}')
    ax[1].scatter([0],[0],marker='*',s=130,c='black',label='Airlight origin')
    ax[1].set(title='B | Non-local color groups',xlabel='R − airlight R',ylabel='G − airlight G');ax[1].legend(fontsize=8)
    counts=stages['count'][stages['count']>0]
    ax[2].hist(counts,bins=np.geomspace(1,counts.max()+1,25),color='#0072B2',alpha=.8)
    ax[2].axvline(50,color='#D55E00',linestyle='--',label='Source reliability count scale')
    ax[2].set(xscale='log',title='C | Uneven prior support',xlabel='Pixels per occupied haze-line',ylabel='Number of haze-lines');ax[2].legend(fontsize=8)
    return fig

def source_comparison(restored,t,original_image,original_t):
    style();fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    image_panel(ax[0,0],original_image,'A | Independently executed author display')
    image_panel(ax[0,1],restored,'B | Fresh local display')
    image_panel(ax[1,0],t-original_t,'C | Local − author transmission','RdBu_r',-1e-11,1e-11,True)
    ids=np.linspace(0,t.size-1,2500,dtype=int)
    ax[1,1].scatter(original_t.ravel()[ids],t.ravel()[ids],s=3,color='#0072B2',alpha=.4,label='Actual sampled pixels')
    ax[1,1].plot([.1,1],[.1,1],color='#999999',linestyle='--',label='Identity')
    ax[1,1].set(title='D | Method agreement, not physical truth',xlabel='Author transmission',ylabel='Fresh local transmission');ax[1,1].legend(fontsize=9)
    return fig

def refinement_ablation(image,stages):
    style();fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    initial=stages['initial'];t=stages['transmission']
    image_panel(ax[0,0],initial,'A | No WLS refinement','viridis',.1,1,True)
    image_panel(ax[0,1],t,'B | Full weighted graph solve','viridis',.1,1,True)
    delta=t-initial;limit=float(np.max(np.abs(delta)))
    image_panel(ax[1,0],delta,'C | Refined − initial transmission','RdBu_r',-limit,limit,True)
    row=220;ax[1,1].plot(initial[row],label='Initial',color='#999999',alpha=.8);ax[1,1].plot(t[row],label='Refined',color='#0072B2')
    ax[1,1].set(title=f'D | Native row {row}',xlabel='Column (pixel)',ylabel='Transmission');ax[1,1].legend()
    return fig

def sensitivity(stages,image,settings=(.03,.1,.3)):
    style();fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained',sharey='row');arrays={};records=[]
    for column,lam in enumerate(settings):
        t,system=refine_transmission(stages['initial'],stages['weight'],image,lam)
        arrays[f't_lambda_{str(lam).replace(".","p")}']=t
        image_panel(ax[0,column],t,f'{chr(65+column)} | λ = {lam:g}','viridis',.1,1,True)
        ax[1,column].plot(stages['transmission'][220],label='Standard λ=0.1',color='#999999',linestyle='--')
        ax[1,column].plot(t[220],label=f'λ={lam:g}',color='#0072B2')
        ax[1,column].set(title=f'{chr(68+column)} | Row 220 sensitivity',xlabel='Column (pixel)',ylabel='Transmission');ax[1,column].legend(fontsize=9)
        vector=t.ravel(order='F');records.append({'lambda':lam,'data_term':float(np.sum(system['reliability']*(t-system['anchored'])**2)),'smoothness_term':float(vector@system['smooth']@vector),'nrmse_vs_standard':compare(t,stages['transmission'])['nrmse']})
    return fig,arrays,records

def physical_and_display_conventions(image,airlight,t,restored,stages):
    style();fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    physical=recover_radiance(image,t,airlight,leave_haze=1.)
    biased=stages['radiance']
    image_panel(ax[0,0],physical,'A | Physical inverse, clipped (normalized)')
    image_panel(ax[0,1],biased,'B | Source residual-haze convention')
    image_panel(ax[0,2],restored,'C | Source contrast display (counts)')
    residual=image_formation(physical,t,airlight)-image
    image_panel(ax[1,0],np.sqrt(np.mean(residual**2,axis=-1)),'D | Clipping consistency residual (RMS)','magma',0,float(np.max(np.sqrt(np.mean(residual**2,axis=-1)))),True)
    ax[1,1].hist(t.ravel(),bins=60,color='#0072B2');ax[1,1].set(title='E | Transmission distribution',xlabel='Transmission (dimensionless)',ylabel='Pixels')
    values=[physical.mean(),biased.mean(),restored.mean()/255]
    ax[1,2].bar(['Physical','Residual haze','Display'],values,color=['#999999','#E69F00','#0072B2']);ax[1,2].set(title='F | Display changes the intensity scale',ylabel='Mean normalized intensity')
    return fig,{'physical_clipped':physical,'physical_forward_residual':residual}

def export_figures(figures,directory,numeric_sources,metrics):
    """Optional private atlas export; standard CLI writes no figures."""
    import json,hashlib
    path=Path(directory);path.mkdir(parents=True,exist_ok=True);inventory=[]
    for name,fig in figures:
        for ext in ['png','svg','pdf']:
            file=path/(name+'.'+ext);fig.savefig(file,bbox_inches='tight')
            inventory.append({'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
    np.savez_compressed(path/'numeric_sources.npz',**numeric_sources)
    (path/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf8')
    (path/'figure_manifest.json').write_text(json.dumps(inventory,indent=2),encoding='utf8')
