"""Active real-pattern orientation review; never selects a Matplotlib backend."""
from pathlib import Path
import time,json,hashlib,importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt
from .preprocessing import load_data
from .solvers import reconstruct,index_patterns,index_direct,refine
from .orientations import sample_cubic_grid,qu_apply
from .physics_model import interpolate_master

def pole_color(q):
    """Cubic-invariant sample-normal pole triangle RGB, defined explicitly here.

    Sort absolute crystal normal components a<=b<=c, take(c-b,sqrt2(b-a),sqrt3*a),
    divide by its maximum and square-root each channel. Red001,green101,blue111.
    This is an explanatory pole color, not a claimed external IPF implementation.
    """
    q=torch.from_numpy(np.asarray(q).copy());q=q/q.norm(dim=-1,keepdim=True);v=qu_apply(q,torch.tensor([0.,0,1.],dtype=q.dtype)).abs().numpy();a,b,c=np.moveaxis(np.sort(v,axis=-1),-1,0);rgb=np.stack([c-b,np.sqrt(2)*(b-a),np.sqrt(3)*a],-1);return np.sqrt(np.clip(rgb/np.maximum(rgb.max(-1,keepdims=True),1e-15),0,1))
def review(task_dir,figure_dir):
    """Reconstruct and compare original/local/PC/component paths; return figs,summary."""
    task=Path(task_dir);dest=Path(figure_dir);dest.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);raw,master=load_data(task/'data/raw_data.npz')
    q,diag=reconstruct(raw,master,return_diagnostics=True);np.savez(task/'output/reconstruction.npz',reconstruction=q[None]);up=dict(np.load(task/'notebooks/reference_outputs/upstream_methods.npz',allow_pickle=False))
    spec=importlib.util.spec_from_file_location('ebsd_review_judge',task/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge);report,code=judge.evaluate(task/'output/reconstruction.npz')
    if code:raise RuntimeError(report)
    # Actual reduced-dimensional matching/refinement; PCA basis is legitimately reused.
    p=torch.from_numpy(diag['cleaned_patterns']);mp=torch.from_numpy(diag['master']);rays=torch.from_numpy(diag['rays']);mask=torch.from_numpy(diag['mask']);V=torch.from_numpy(diag['components'])
    torch.manual_seed(20261021);start=time.perf_counter();coarse_di,_=index_direct(mp,p,rays,mask);direct_index_seconds=time.perf_counter()-start;start=time.perf_counter();qdi,dotsdi=refine(mp,p,rays,coarse_di);direct_refine_seconds=time.perf_counter()-start;qdi=qdi.numpy().reshape(40,50,4)
    torch.manual_seed(20261021);sample_cubic_grid();start=time.perf_counter();coarse100,_=index_patterns(mp,p,rays,mask,V[:,-100:]);q100,dots100=refine(mp,p,rays,coarse100);seconds100=time.perf_counter()-start;q100=q100.numpy().reshape(40,50,4)
    torch.manual_seed(20261021);sample_cubic_grid();start=time.perf_counter();coarse25,_=index_patterns(mp,p,rays,mask,V[:,-25:]);q25,dots25=refine(mp,p,rays,coarse25);seconds25=time.perf_counter()-start;q25=q25.numpy().reshape(40,50,4)
    # Independent published PC conversion; no orientation-based fitting or selection.
    paper_pc=(.4210,1-.7794,.5049);qpaper,dpaper=reconstruct(raw,master,projection_center=paper_pc,return_diagnostics=True)
    angle100=judge.disorientation_degrees(q,q100).reshape(40,50);angle_original=judge.disorientation_degrees(q,up['corrected_original512']).reshape(40,50);angle_endpoint=judge.disorientation_degrees(q,up['original_uncorrected512']).reshape(40,50);angle_refinement=judge.disorientation_degrees(q,diag['coarse']).reshape(40,50);angle25=judge.disorientation_degrees(q,q25).reshape(40,50);anglepc=judge.disorientation_degrees(q,qpaper).reshape(40,50)
    cosines=judge.forward_cosines(q);summary={'judge':report,'fresh_baseline_seconds':diag['seconds'],'dictionary_count':diag['dictionary_count'],'comparison25_reused_basis_seconds':seconds25,'paper_pc_full_seconds':dpaper['seconds'],'paper_pc_bruker':list(paper_pc),'mean_endpoint_change_deg':float(angle_endpoint.mean()),'max_local_original_difference_deg':float(angle_original.max()),'mean_refinement_change_deg':float(angle_refinement.mean()),'mean25pc_change_deg':float(angle25.mean()),'mean_paperpc_change_deg':float(anglepc.mean()),'independent_forward_cosine_quantiles':np.quantile(cosines,[0,.05,.5,.95,1]).tolist(),'reference_is_not_truth':True};nums={'fresh_q':q,'original_q':up['corrected_original512'],'original_uncorrected_q':up['original_uncorrected512'],'coarse_q':diag['coarse'].reshape(40,50,4),'q25':q25,'paper_pc_q':qpaper,'local_original_deg':angle_original,'endpoint_change_deg':angle_endpoint,'refinement_change_deg':angle_refinement,'components25_change_deg':angle25,'paper_pc_change_deg':anglepc,'independent_forward_cosines':cosines,'components':diag['components'],'eigenvalues':diag['eigenvalues'],'mask':diag['mask'],'baseline_refinement_dots':diag['refinement_dots'],'components25_refinement_dots':dots25.numpy(),'paper_pc_refinement_dots':dpaper['refinement_dots']};figs=[]
    summary.update(comparison100_reused_basis_seconds=seconds100,mean100pc_change_deg=float(angle100.mean()),fresh100_vs_original_max_deg=float(judge.disorientation_degrees(q100,up['corrected_original']).max()))
    nums.update(q100=q100,components100_change_deg=angle100,components100_refinement_dots=dots100.numpy(),original100_q=up['corrected_original'])
    summary['max_coarse_local_original_difference_deg']=float(judge.disorientation_degrees(diag['coarse'],up['corrected_coarse512']).max())
    nums['original_refinement_mean_dots']=up['original_refinement512_mean_dots'];nums['actual_runtime_seconds']=np.array([diag['seconds'],seconds25,dpaper['seconds']])
    # Pixel centers: x90..163.5um,y75..133.5um; origin upper follows native raster.
    extent=(89.25,164.25,134.25,74.25)
    def image(ax,a,title,unit=None,limits=None,cmap='viridis',spatial=True):
        palette=plt.get_cmap(cmap).copy();palette.set_bad('.8')
        im=ax.imshow(a,origin='upper',extent=extent if spatial else None,cmap=palette,**({} if limits is None else {'vmin':limits[0],'vmax':limits[1]}));ax.set(title=title,xlabel='Original scan x (µm)' if spatial else 'Detector column',ylabel='Original scan y (µm)' if spatial else 'Detector row')
        if unit is not None:plt.colorbar(im,ax=ax,label=unit,shrink=.8)
    points=[(0,0),(20,25),(39,49)];fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for j,(r,c) in enumerate(points):
        image(ax[0,j],raw[r,c],f'Native detector60² at ROI({r},{c})','ADU',(0,255),spatial=False);image(ax[1,j],diag['cleaned_patterns'][r*50+c],f'Actual processed detector30² at({r},{c})','Normalized intensity',(0,1),spatial=False);nums[f'raw_pattern_{r}_{c}']=raw[r,c];nums[f'processed_pattern_{r}_{c}']=diag['cleaned_patterns'][r*50+c]
    fig.suptitle('1. Real native patterns and public-ROI-only preprocessing');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for a,v,title in zip(ax[0],[up['original_uncorrected512'],up['corrected_original512'],q],['Unmodified original endpoint convention','Qualified corrected original','Fresh local corrected reconstruction']):image(a,pole_color(v),title)
    image(ax[1,0],angle_endpoint,'Actual effect of endpoint correction','Disorientation (degree)');image(ax[1,1],angle_original,'Fresh versus corrected original','Disorientation (degree)',(0,1e-4));image(ax[1,2],cosines.reshape(40,50),'Independent public-pattern consistency','Best24-representative cosine',(0,1));fig.suptitle('2. Cubic pole triangle color: red001,green101,blue111; no measured orientation truth');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for a,n in zip(ax.ravel()[:4],[1,2,100,512]):
        component=np.full(900,np.nan);component[diag['mask']]=diag['components'][:,-n];lim=float(np.nanmax(abs(component)));image(a,component.reshape(30,30),f'Fresh PCA component{n}(descending variance)','Coefficient',(-lim,lim),'RdBu_r',False)
    values=np.maximum(diag['eigenvalues'][::-1],0);cumulative=np.cumsum(values)/values.sum();ax[1,1].semilogy(np.arange(1,len(values)+1),np.maximum(values,1e-12));ax[1,1].set(xlabel='Component rank',ylabel='Covariance eigenvalue',title='Actual dictionary spectrum');ax[1,2].plot(np.arange(1,len(values)+1),cumulative);ax[1,2].axvline(512,color='k',ls='--');ax[1,2].set(xlabel='Retained components',ylabel='Cumulative eigenvalue fraction',title='Subspace capacity on this selected dictionary');nums['eigenvalue_cumulative_fraction']=cumulative;fig.suptitle('3. Active covariance-PCA basis; gray outside public detector mask');figs.append(fig)
    fig,ax=plt.subplots(3,3,figsize=(14,11),layout='constrained')
    qt=torch.from_numpy(q.reshape(-1,4));rn=rays/rays.norm(dim=-1,keepdim=True);predicted=interpolate_master(mp,qu_apply(qt[[r*50+c for r,c in points]][:,None],rn[None]),align_corners=True,normalize_coords=False).numpy()
    for j,(r,c) in enumerate(points):
        obs=diag['cleaned_patterns'][r*50+c].ravel().copy();pred=predicted[j].ravel().copy();obs-=obs.mean();pred-=pred.mean();obs/=max(np.linalg.norm(obs),1e-15);pred/=max(np.linalg.norm(pred),1e-15);res=pred-obs;limit=max(abs(obs).max(),abs(pred).max());rlim=abs(res).max();image(ax[0,j],obs.reshape(30,30),f'Observed normalized pattern({r},{c})','L2-normalized intensity',(-limit,limit),'RdBu_r',False);image(ax[1,j],pred.reshape(30,30),'Fresh calibrated forward pattern','L2-normalized intensity',(-limit,limit),'RdBu_r',False);image(ax[2,j],res.reshape(30,30),'Normalized intensity residual','Difference',(-rlim,rlim),'RdBu_r',False);nums[f'normalized_observed_{r}_{c}']=obs;nums[f'normalized_forward_{r}_{c}']=pred;nums[f'normalized_residual_{r}_{c}']=res
    fig.suptitle('4. Actual zero-mean unit-L2 observations/forward predictions; no fitted gain or truth');figs.append(fig)
    fig,ax=plt.subplots(3,3,figsize=(16,12),layout='constrained')
    image(ax[0,0],angle_refinement,'Coarse to four-round refinement','Disorientation(degree)');image(ax[0,1],angle100,'100PC versus512PC','Disorientation(degree)');image(ax[0,2],angle25,'25PC versus512PC','Disorientation(degree)');image(ax[1,0],anglepc,'Published PC versus source PC','Disorientation(degree)')
    for dots,label,style in [(diag['refinement_dots'],'Fresh512PC','o-'),(dots100.numpy(),'Fresh100PC','s-'),(dots25.numpy(),'Fresh25PC','^-')]:ax[1,1].plot(np.arange(1,5),dots.mean(0),style,label=label)
    ax[1,1].plot(np.arange(1,5),up['original_refinement512_mean_dots'][1:],'x--',label='Original512PC');ax[1,1].set(xlabel='Refinement round',ylabel='Mean unnormalized dot',title='Actual source/local trajectories');ax[1,1].legend(fontsize=8)
    ax[1,2].hist(cosines,bins=35);ax[1,2].set(xlabel='Best24 public-pattern cosine',ylabel='Pixels',title=f'All2000pixels; mean{cosines.mean():.4f}')
    times=[diag['seconds'],seconds100,seconds25,dpaper['seconds']];ax[2,0].bar(['512PC full','100PC reused','25PC reused','PaperPC full'],times);ax[2,0].tick_params(axis='x',labelrotation=20);ax[2,0].set(ylabel='Actual one-thread CPU seconds',title='Reused paths exclude basis learning')
    ax[2,1].hist(angle100.ravel(),bins=35,alpha=.6,label='100PC vs512PC');ax[2,1].hist(angle25.ravel(),bins=35,alpha=.6,label='25PC vs512PC');ax[2,1].set(xlabel='Disorientation(degree)',ylabel='Pixels',title='All-point compression sensitivity');ax[2,1].legend(fontsize=8)
    stages=diag['stage_seconds'];ax[2,2].bar(['PCA learning','PCA matching','PCA refinement'],[stages['pca_learning'],stages['indexing'],stages['refinement']]);ax[2,2].tick_params(axis='x',labelrotation=15);ax[2,2].set(ylabel='Actual one-thread CPU seconds',title='Full PCA costs are retained')
    nums['actual_runtime_seconds']=np.array(times);nums['pca_stage_seconds']=np.array([stages[k] for k in ['preprocessing','pca_learning','indexing','refinement']]);fig.suptitle('5. Actual512PC /100PC /25PC / independent-PC sensitivity; no calibration fitting');figs.append(fig)
    direct_parity=judge.disorientation_degrees(qdi,up['original_direct']).reshape(40,50);direct_change=judge.disorientation_degrees(qdi,q).reshape(40,50)
    summary.update(direct_index_seconds=direct_index_seconds,direct_refinement_seconds=direct_refine_seconds,direct_max_original_difference_deg=float(direct_parity.max()),direct_vs_pca_mean_deg=float(direct_change.mean()),pca_stage_seconds=diag['stage_seconds'])
    nums.update(original_direct_q=up['original_direct'],fresh_direct_q=qdi,direct_coarse_q=coarse_di.numpy(),direct_original_deg=direct_parity,direct_vs_pca_deg=direct_change,direct_refinement_dots=dotsdi.numpy(),matching_seconds=np.array([diag['stage_seconds']['indexing'],direct_index_seconds]))
    fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for a,v,title in zip(ax[0],[up['original_direct'],qdi,q],['Original direct DI + refinement','Fresh direct DI + refinement','Fresh512PC DI + refinement']):image(a,pole_color(v),title)
    image(ax[1,0],direct_change,'Direct DI versus512PC DI','Disorientation(degree)');image(ax[1,1],direct_parity,'Fresh direct DI versus original','Disorientation(degree)',(0,1e-4))
    ax[1,2].bar(['512PC projected dot','716-pixel direct dot'],[diag['stage_seconds']['indexing'],direct_index_seconds]);ax[1,2].tick_params(axis='x',labelrotation=12);ax[1,2].set(ylabel='Actual one-thread CPU seconds',title='Matching only\nLearning/refinement excluded')
    fig.suptitle('6. Actual uncompressed direct DI comparison: same2000patterns,102448dictionary,4refinement rounds');figs.append(fig)
    np.savez_compressed(dest/'numeric_sources.npz',**nums);(dest/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8');manifest={}
    for i,fig in enumerate(figs,1):
        for ext in ['png','svg','pdf']:
            path=dest/f'figure_{i}.{ext}';fig.savefig(path,dpi=150);manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        plt.close(fig)
    for name in ['numeric_sources.npz','metrics.json']:manifest[name]=hashlib.sha256((dest/name).read_bytes()).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');return figs,summary
