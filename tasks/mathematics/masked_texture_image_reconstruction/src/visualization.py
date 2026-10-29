"""Actual scientific comparisons, finite-budget sensitivities and numeric exports."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import gradient,symgrad
from .solvers import reconstruct

def metric_pair(x,truth,mask):
    scale=float(np.ptp(truth));den=np.linalg.norm(x)*np.linalg.norm(truth)
    return dict(ncc=float(np.vdot(x,truth)/den) if den else 0.,nrmse=float(np.sqrt(np.mean((x-truth)**2))/scale),missing_nrmse=float(np.sqrt(np.mean((x[~mask]-truth[~mask])**2))/scale),known_error=float(np.max(np.abs(x[mask]-truth[mask]))))

def make_figures(y,mask,truth,fresh,upstream):
    """Fresh maps/objectives are supplied by caller; extra sensitivity actively runs."""
    figures=[];numeric=dict(observations=y,mask=mask,truth=truth);table={}
    def panel(ax,a,title,cmap='gray',vmin=0,vmax=1,extent=None):
        im=ax.imshow(a,cmap=cmap,vmin=vmin,vmax=vmax,extent=extent,interpolation='nearest');ax.set_title(title,fontsize=10);ax.set_xlabel('column (pixel)');ax.set_ylabel('row (pixel)');plt.colorbar(im,ax=ax,fraction=.045,pad=.03)
    for order,result in fresh.items():
        x=result['u'];numeric[f'fresh_order{order}']=x;numeric[f'objective{order}']=result['objective'];numeric[f'original_order{order}']=upstream[f'order{order}'];table[f'order{order}']=metric_pair(x,truth,mask)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,a,title in zip(axes.flat,[truth,y,mask,fresh[1]['u'],fresh[2]['u'],fresh[3]['u']],['Actual complete author photograph','Bound observations: zero means missing','Independent author sampling mask','Fresh TV: 8000 updates','Fresh TGV2: 8000 updates','Fresh TGV3: 8000 updates']):panel(ax,a,title)
    fig.suptitle('Full native photographic inverse | identical observations and finite iteration budget');figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True);r0,r1,c0,c1=70,150,80,200;extent=(c0-.5,c1-.5,r1-.5,r0-.5)
    for j,order in enumerate([1,2,3]):
        panel(axes[0,j],fresh[order]['u'][r0:r1,c0:c1],f'Fresh order{order}: same native detail',extent=extent)
        panel(axes[1,j],fresh[order]['u']-truth,f'Order{order} minus actual photograph','RdBu_r',-.4,.4)
    fig.suptitle('Recovered stripes and shared-scale signed errors | texture loss remains visible');figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True);names=['TV','TGV2','TGV3']
    for j,key in enumerate(['ncc','nrmse','missing_nrmse']):
        vals=[table[f'order{k}'][key] for k in [1,2,3]];axes[0,j].bar(names,vals,color=['#E69F00','#0072B2','#009E73']);axes[0,j].set_title(key+' | actual truth');axes[0,j].set_ylabel('dimensionless')
        if key=='ncc':axes[0,j].set_ylim(0,1)
        for k,val in enumerate(vals):axes[0,j].text(k,val,f'{val:.5f}',ha='center',va='bottom',fontsize=9)
    for order in [1,2,3]:
        axes[1,0].plot(fresh[order]['objective'][1:],label=names[order-1]);diff=fresh[order]['objective']-upstream[f'objective{order}'];numeric[f'objective_difference{order}']=diff
        axes[1,1].plot(np.arange(320),fresh[order]['u'][110],label=names[order-1])
        error=fresh[order]['u'][~mask]-truth[~mask];axes[1,2].hist(error,bins=np.linspace(-.4,.4,81),density=True,histtype='step',label=names[order-1])
    axes[1,0].set_title('Actual feasible objective histories');axes[1,0].set_xlabel('update');axes[1,0].set_ylabel('dimensionless regularizer')
    axes[1,1].plot(truth[110],color='black',lw=1,label='actual photograph');axes[1,1].set_title('Paired native row110 profile');axes[1,1].set_xlabel('column (pixel)');axes[1,1].set_ylabel('encoded luminance')
    axes[1,2].set_title('Missing-only errors');axes[1,2].set_xlabel('encoded-luminance error');axes[1,2].set_ylabel('density')
    for ax in axes[1]:ax.legend(fontsize=8)
    fig.suptitle('Independent truth metrics, complete objective histories and acquired/missing separation');figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True);differences=[fresh[k]['u']-upstream[f'order{k}'] for k in [1,2,3]];lim=max(max(float(np.max(np.abs(d))) for d in differences),1e-14)
    for j,order in enumerate([1,2,3]):
        panel(axes[0,j],differences[j],f'Fresh minus independent original order{order}','RdBu_r',-lim,lim)
    g=gradient(fresh[2]['u']);strain=symgrad(fresh[2]['v']);numeric['fresh_gradient']=g;numeric['fresh_tgv_aux_vector']=fresh[2]['v'];numeric['fresh_symgrad_aux']=strain
    panel(axes[1,0],np.sqrt(np.sum(g*g,axis=2)),'Fresh gradient magnitude','magma',0,.4)
    panel(axes[1,1],np.sqrt(np.sum((g-fresh[2]['v'])**2,axis=2)),'Fresh first TGV decomposition','magma',0,.4)
    panel(axes[1,2],np.sqrt(np.sum(strain*strain*np.array([1,1,2]),axis=2)),'Fresh symmetric-tensor magnitude','magma',0,.1)
    fig.suptitle('Original-code parity and explicit local first/second derivative decomposition');figures.append(fig)
    sensitivity={}
    for weight in [.5,np.sqrt(2.),3.]:
        result=reconstruct(y,mask,order=2,niter=2000,alpha1=weight);key=f'{weight:.8g}';sensitivity[key]=result;numeric['sensitivity_'+key]=result['u'];table['sensitivity_'+key]=metric_pair(result['u'],truth,mask)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for j,(key,result) in enumerate(sensitivity.items()):
        panel(axes[0,j],result['u'],f'Fresh alpha1={key}: fixed2000 updates')
        panel(axes[1,j],result['u']-truth,f'Error alpha1={key}','RdBu_r',-.4,.4)
    fig.suptitle('Genuine regularizer sensitivity | separate2000-update budget, not a convergence claim');figures.append(fig)
    return figures,numeric,table

def export_figures(figures,numeric,metrics,directory):
    target=Path(directory);target.mkdir(exist_ok=True,parents=True);manifest={}
    for i,fig in enumerate(figures,1):
        for extension in ['png','svg','pdf']:
            p=target/f'figure_{i}.{extension}';fig.savefig(p,dpi=160,bbox_inches='tight');manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    np.savez_compressed(target/'numeric_sources.npz',**numeric);(target/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf8')
    for name in ['numeric_sources.npz','metrics.json']:manifest[name]=hashlib.sha256((target/name).read_bytes()).hexdigest()
    (target/'hash_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')

