"""Scientific views from actual fresh normal fields; no backend selection."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from .physics_model import angular_errors
METHODS=['l2','l1','sbl','rpca']
LABELS=['L2','L1 IRLS','Source-SBL heuristic','RPCA']
def metrics(image,truth,support):
    a=image[support].ravel();r=truth[support].ravel()
    unit_error=float(abs(np.linalg.norm(image[support],axis=-1)-1).max())
    if (~support).any():unit_error=max(unit_error,float(abs(image[~support]).max()))
    return {'ncc':float(a@r/(np.linalg.norm(a)*np.linalg.norm(r))),'nrmse':float(np.sqrt(np.mean((a-r)**2))/(r.max()-r.min())),'mean_angular_error_deg':float(angular_errors(image,truth,support).mean()),'unit_norm_max_error':unit_error}
def normal_rgb(n,mask):return np.where(mask[...,None],np.clip((n+1)/2,0,1),0)
def render_figures(observations,lights,support,truth,results,original,records,subset_results,traces):
    figures=[];numeric={'observations':observations,'light_vectors':lights,'support':support,'truth':truth}
    for name,a in results.items():numeric['fresh_'+name]=a;numeric['angle_'+name]=angular_errors(a,truth,support)
    chosen=[0,9,19,29,39,49];vmax=float(np.quantile(observations,.995))
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for ax,i in zip(axes.flat,chosen):
        im=ax.imshow(observations[i],cmap='gray',norm=PowerNorm(.5,vmin=0,vmax=vmax));ax.set_title(f'Light {i}: ({lights[i,0]:.2f}, {lights[i,1]:.2f}, {lights[i,2]:.2f})');ax.set_xlabel('Column [pixel]');ax.set_ylabel('Row [pixel]')
    fig.colorbar(im,ax=axes,label='Author-relative intensity [1]',shrink=.8);fig.suptitle('Existing author-rendered observations; common square-root display scale');figures.append(fig)
    fig,axes=plt.subplots(2,5,figsize=(17,7),layout='constrained')
    axes[0,0].imshow(normal_rgb(truth,support));axes[0,0].set_title('Actual renderer normals')
    im=axes[1,0].imshow(np.where(support,truth[...,2],np.nan),cmap='coolwarm',vmin=-1,vmax=1);axes[1,0].set_title('Actual normal z [1]');fig.colorbar(im,ax=axes[1,0],shrink=.7)
    for j,(name,label) in enumerate(zip(METHODS,LABELS),1):
        axes[0,j].imshow(normal_rgb(results[name],support));axes[0,j].set_title('Fresh '+label)
        err=np.full(support.shape,np.nan);err[support]=numeric['angle_'+name];im=axes[1,j].imshow(err,cmap='magma',vmin=0,vmax=30);axes[1,j].set_title(f'Mean angle {records[name]["mean_angular_error_deg"]:.3f} deg')
    for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=axes[1,1:],label='Angular error [degree]',shrink=.8);fig.suptitle('All four original comparison methods actively reconstructed; RGB=(normal+1)/2');figures.append(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');xx=np.arange(4);w=.35
    original_angle=[metrics(original[m],truth,support)['mean_angular_error_deg'] for m in METHODS];fresh_angle=[records[m]['mean_angular_error_deg'] for m in METHODS]
    axes[0,0].bar(xx-w/2,original_angle,w,label='Independent original run');axes[0,0].bar(xx+w/2,fresh_angle,w,label='Fresh local');axes[0,0].set_ylabel('Foreground mean angular error [degree]');axes[0,0].legend()
    axes[0,1].bar(xx,[records[m]['seconds'] for m in METHODS]);axes[0,1].set_yscale('log');axes[0,1].set_ylabel('Fresh solver time [s]')
    for ax in axes[0]:ax.set_xticks(xx,LABELS,rotation=12);ax.grid(axis='y',alpha=.2)
    for m,label in zip(METHODS,LABELS):
        angles=np.sort(numeric['angle_'+m]);axes[1,0].plot(angles,np.arange(1,len(angles)+1)/len(angles),label=label)
        diff=np.max(abs(results[m]-original[m]),axis=-1);numeric['original_difference_'+m]=diff;axes[1,1].plot(np.quantile(diff[support],[0,.5,.9,.99,1]),marker='o',label=label)
    axes[1,0].set(xlim=(0,40),xlabel='Foreground angular error [degree]',ylabel='Empirical cumulative fraction [1]');axes[1,0].legend()
    axes[1,1].set(xlabel='Quantile index (0, .5, .9, .99, 1)',ylabel='Fresh minus original max component [1]');axes[1,1].legend();fig.suptitle('Matched foreground metrics; background never dilutes accuracy');figures.append(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(pixel,trace) in zip(axes.flat[:3],traces.items()):
        b=observations.reshape(50,-1)[:,pixel];ax.plot(b,'k.',label='Observed')
        for method,color in [('l2','tab:blue'),('l1','tab:orange'),('sbl','tab:green')]:ax.plot(lights@trace[method].ravel(),color=color,label=method.upper()+' diffuse fit')
        ax.set(xlabel='Illumination index',ylabel='Relative intensity [1]',title=f'Actual pixel row {pixel//256}, col {pixel%256}');ax.legend(fontsize=8)
        numeric[f'pixel_{pixel}_observed']=b
        for method in ['l2','l1','sbl']:numeric[f'pixel_{pixel}_{method}_coefficient']=trace[method]
    for pixel,trace in traces.items():
        loss=np.array([abs(v['r']).sum() for v in trace['iterations']]);numeric[f'pixel_{pixel}_l1_loss']=loss;axes[1,1].plot(loss,label=str(pixel))
    axes[1,1].set(xlabel='IRLS iteration',ylabel='Sum of absolute intensity residuals [1]',title='Actual convex-objective trajectories');axes[1,1].legend(title='Flattened pixel');fig.suptitle('Independent physical diagnostic: robust fits separate non-diffuse observations');figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');counts=[10,25,50]
    for j,count in enumerate(counts):
        a=subset_results[count];err=np.full(support.shape,np.nan);err[support]=angular_errors(a,truth,support)
        axes[0,j].imshow(normal_rgb(a,support));axes[0,j].set_title(f'Fresh L1 with {count} actual lights')
        im=axes[1,j].imshow(err,cmap='magma',vmin=0,vmax=30);axes[1,j].set_title(f'Foreground angle {np.nanmean(err):.3f} degrees');numeric[f'light_subset_{count}']=a
        for row in axes:row[j].set_xlabel('Column [pixel]');row[j].set_ylabel('Row [pixel]')
    fig.colorbar(im,ax=axes[1],label='Angular error [degree]',shrink=.8);fig.suptitle('Executed acquisition-subset sensitivity; same scene, unaltered observations');figures.append(fig)
    return figures,numeric
