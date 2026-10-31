"""Active full-observation experiments and figures; no backend selection."""
from pathlib import Path
import json,time
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct
from .preprocessing import as_matrix


def summary(result,info):
    l=result['reconstruction'][0];s=result['sparse_component'][0]
    singular=np.linalg.svd(l.reshape(len(l),-1),compute_uv=False)
    return {'iterations':info['iterations'],'converged':info['converged'],'relative_primal':info['relative_primal'],'lambda':info['lambda'],'rank_1e_6':int(np.count_nonzero(singular>1e-6*singular[0])),'sparse_fraction_1e_3':float(np.mean(np.abs(s)>1e-3)),'low_energy':float(np.sum(l*l)),'sparse_energy':float(np.sum(s*s))}


def experiments(frames,canonical,diag):
    """Reconstruct full native movies at two lambda values and temporal subsets.

    The canonical reconstruction is freshly supplied by the notebook. Variant
    arrays are released after selected previews and scalar summaries are saved.
    A maximum-iteration stop is reported and never relabeled as convergence.
    """
    ids=np.array([0,20,40,60,83]);records=[];previews=[];traces=[]
    for factor in [.75,1.,1.25]:
        start=time.perf_counter()
        if factor==1:result,info=canonical,diag
        else:result,info=reconstruct(frames,lambda_factor=factor)
        row=summary(result,info);seconds=float(diag['runtime_seconds']) if factor==1 and 'runtime_seconds' in diag else time.perf_counter()-start
        row.update(condition='lambda',factor=factor,seconds=seconds)
        records.append(row);previews.append(result['sparse_component'][0,ids].copy());traces.append(info['history'].copy())
        if factor!=1:del result,info
    temporal=[]
    for stride in [2,4]:
        start=time.perf_counter();result,info=reconstruct(frames[::stride]);row=summary(result,info)
        target=canonical['reconstruction'][0,::stride]
        row.update(condition='temporal_stride',factor=stride,seconds=time.perf_counter()-start,low_relative_change=float(np.linalg.norm(result['reconstruction'][0]-target)/np.linalg.norm(target)))
        records.append(row);temporal.append(result['reconstruction'][0,0].copy());del result,info
    d=as_matrix(frames);u,s,v=np.linalg.svd(d,full_matrices=False)
    pca=[]
    for rank in [1,3,5]:
        l=(u[:,:rank]*s[:rank])@v[:rank];res=d-l
        records.append({'condition':'pca_rank','factor':rank,'relative_residual':float(np.linalg.norm(res)/np.linalg.norm(d)),'sparse_fraction_1e_3':float(np.mean(np.abs(res)>1e-3))})
        pca.append(res[:,ids].T.reshape(len(ids),*frames.shape[1:]).copy())
    return {'records':records,'ids':ids,'lambda_sparse':np.asarray(previews),'lambda_traces':traces,'temporal_low':np.asarray(temporal),'pca_sparse':np.asarray(pca),'raw_singular':s}


def figures(frames,result,diag,upstream,exp):
    """Four scientific figures; all intensities use fixed physical conventions."""
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110})
    ids=exp['ids'];l=result['reconstruction'][0];s=result['sparse_component'][0];figs=[]
    limit=float(np.max(np.abs(s[ids])))
    fig,ax=plt.subplots(3,5,figsize=(14,7),constrained_layout=True)
    for j,t in enumerate(ids):
        ax[0,j].imshow(frames[t],cmap='gray',vmin=0,vmax=255);ax[0,j].set_title('Digital frame '+str(t))
        a=ax[1,j].imshow(l[t],cmap='gray',vmin=0,vmax=1)
        b=ax[2,j].imshow(s[t],cmap='RdBu_r',vmin=-limit,vmax=limit)
        for i in range(3):ax[i,j].set_xticks([]);ax[i,j].set_yticks([])
    for i,label in enumerate(['Observed (ADU)','Fresh low rank (ADU/255)','Fresh signed sparse (ADU/255)']):ax[i,0].set_ylabel(label)
    fig.colorbar(a,ax=ax[1,:],shrink=.6);fig.colorbar(b,ax=ax[2,:],shrink=.6)
    fig.suptitle('1888 real film: full84-frame statistical decomposition, no component truth');figs.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True);h=diag['history'];o=upstream
    ax[0,0].semilogy(o[:,0],o[:,1],color='.65',lw=4,label='Independent original');ax[0,0].semilogy(h[:,0],h[:,1],label='Fresh local')
    ax[0,0].set(xlabel='ADMM iteration',ylabel='Additive residual Frobenius norm');ax[0,0].legend()
    ax[0,1].plot(np.mean(l*l,axis=(1,2)),label='Low rank');ax[0,1].plot(np.mean(s*s,axis=(1,2)),label='Signed sparse');ax[0,1].set(xlabel='Digital frame',ylabel='Mean squared normalized intensity');ax[0,1].legend()
    ls=np.linalg.svd(l.reshape(len(l),-1),compute_uv=False);ax[1,0].semilogy(exp['raw_singular'],label='Observed');ax[1,0].semilogy(ls,label='Fresh low rank');ax[1,0].set(xlabel='Singular-value index',ylabel='Singular value (ADU/255)');ax[1,0].legend()
    residual=frames/255-l-s;ax[1,1].plot(np.sqrt(np.mean(residual**2,axis=(1,2))))
    ax[1,1].set(xlabel='Digital frame',ylabel='Additive residual RMS (ADU/255)');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(12,7),constrained_layout=True);v=float(np.max(np.abs(exp['lambda_sparse'][:,2])))
    for j,factor in enumerate([.75,1.,1.25]):
        ax[0,j].imshow(exp['lambda_sparse'][j,2],cmap='RdBu_r',vmin=-v,vmax=v);ax[0,j].set_title('Fresh lambda x'+str(factor));ax[0,j].set(xlabel='x (pixel)',ylabel='y (pixel)')
        row=exp['records'][j];ax[1,j].semilogy(exp['lambda_traces'][j][:,0],exp['lambda_traces'][j][:,1]);ax[1,j].set(xlabel='Iteration',ylabel='Additive residual norm',title=f"Rank {row['rank_1e_6']}; sparse {row['sparse_fraction_1e_3']:.1%}\nConverged: {row['converged']}")
    fig.suptitle('Full84-frame regularization sensitivity; same signed display scale');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(12,7),constrained_layout=True);v=float(np.max(np.abs(exp['pca_sparse'][:,2])))
    for j,rank in enumerate([1,3,5]):
        ax[0,j].imshow(exp['pca_sparse'][j,2],cmap='RdBu_r',vmin=-v,vmax=v);ax[0,j].set_title('Fresh PCA residual, rank '+str(rank))
        ax[0,j].set(xlabel='x (pixel)',ylabel='y (pixel)')
    for j,(a,title) in enumerate([(l[0],'All84frames'),(exp['temporal_low'][0],'Every2nd digital frame'),(exp['temporal_low'][1],'Every4th digital frame')]):
        ax[1,j].imshow(a,cmap='gray',vmin=0,vmax=1);ax[1,j].set_title(title);ax[1,j].set(xlabel='x (pixel)',ylabel='y (pixel)')
    fig.suptitle('Alternative low-rank baselines and temporal-sampling sensitivity');figs.append(fig)
    return figs


def export(figs,exp,directory):
    p=Path(directory);p.mkdir(parents=True,exist_ok=True)
    for i,f in enumerate(figs,1):
        for suffix in ['png','svg','pdf']:f.savefig(p/f'figure-{i}.{suffix}',dpi=160)
    arrays={k:v for k,v in exp.items() if k not in ['records','lambda_traces']}
    arrays.update({f'lambda_trace_{i}':a for i,a in enumerate(exp['lambda_traces'])})
    np.savez_compressed(p/'figure_source.npz',**arrays)
    (p/'experiment_metrics.json').write_text(json.dumps(exp['records'],indent=2,allow_nan=False),encoding='utf8')
