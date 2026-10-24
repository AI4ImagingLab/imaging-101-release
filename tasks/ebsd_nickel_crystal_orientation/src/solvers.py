"""MIT2025 ZacharyVarley; readable local cubic-Ni PCA dictionary inverse.

General torch linear algebra only. All task acquisition, normalization,
orientation sampling, covariance, indexing and custom refinement steps are local.
"""
import torch,time
import numpy as np
from .preprocessing import prepare_patterns,prepare_master
from .physics_model import detector_rays,get_radial_mask,interpolate_master
from .orientations import sample_cubic_grid,qu_apply,cu2qu,qu_prod,cubic_reduce,get_local_orientation_grid
def update_covariance(aggregate,count,mean,x):
    """Original Welford/Chan two-delta update; returns new aggregate,count,mean."""
    n=len(x);batch=x.mean(0,keepdim=True);new=(mean*count+batch*n)/(count+n)
    delta=x.float()-mean.float();delta_prime=x.float()-new.float()
    aggregate+=torch.einsum('ij,ik->jk',delta,delta_prime).to(aggregate.dtype)
    count+=n;mean.copy_(new);return aggregate,count,mean
def learn_components(master,rays,mask,n_components=100,resolution_deg=2.,chunk=256):
    """On-demand dictionary covariance, eigh and last/top eigenvector columns."""
    orientations=sample_cubic_grid(resolution_deg);n_pixels=int(mask.sum());aggregate=torch.zeros(n_pixels,n_pixels);mean=torch.zeros(1,n_pixels);count=torch.tensor([0],dtype=torch.int64)
    rays=rays[mask][None]
    for q in torch.split(orientations,chunk):
        sim=interpolate_master(master,qu_apply(q[:,None,:],rays)).reshape(len(q),-1);sim-=sim.mean(-1,keepdim=True)
        update_covariance(aggregate,count,mean,sim)
    covariance=aggregate/(count-1).float();values,vectors=torch.linalg.eigh(covariance)
    return vectors[:,-n_components:],values,{'covariance':covariance,'mean':mean,'dictionary_count':len(orientations)}
def index_patterns(master,patterns,rays,mask,components,resolution_deg=2.,dictionary_chunk=256,query_chunk=256):
    """Exact original float32 projected dot-product KNN, top1, no quantization."""
    dictionary=sample_cubic_grid(resolution_deg);N=len(patterns);best=torch.full((N,1),-torch.inf);indices=torch.zeros((N,1),dtype=torch.int64);queries=[]
    for start in range(0,N,query_chunk):
        p=patterns[start:start+query_chunk].flatten(1)[:,mask];p=p-p.mean(-1,keepdim=True);queries.append((start,p@components))
    masked=rays[mask][None];offset=0
    for q in torch.split(dictionary,dictionary_chunk):
        sim=interpolate_master(master,qu_apply(q[:,None,:],masked)).reshape(len(q),-1);sim=sim-sim.mean(-1,keepdim=True);data=sim@components
        for start,query in queries:
            values,idx=torch.topk(query@data.T,1,dim=1,largest=True,sorted=False);idx+=offset;end=start+len(query)
            merged=torch.cat([best[start:end],values],1);merged_idx=torch.cat([indices[start:end],idx],1);choose=torch.topk(merged,1,dim=1,largest=True,sorted=False)[1];best[start:end]=torch.gather(merged,1,choose);indices[start:end]=torch.gather(merged_idx,1,choose)
        offset+=len(q)
    return dictionary[indices[:,0]],best[:,0]
def refine(master,patterns,rays,initial,iterations=4,batch_size=64,semi_edge_deg=2.):
    """Original27-point cubochoric pattern search; shrink only on central winner."""
    rays=rays/rays.norm(dim=-1,keepdim=True);base=get_local_orientation_grid(semi_edge_deg,1,3).view(1,-1,3);qout=initial.clone();dots=torch.zeros(len(initial),iterations)
    for start in range(0,len(initial),batch_size):
        end=min(start+batch_size,len(initial));grid=base.clone().repeat(end-start,1,1);obs=patterns[start:end].flatten(1);obs=obs-obs.mean(-1,keepdim=True);current=initial[start:end]
        for step in range(iterations):
            proposal=cubic_reduce(qu_prod(cu2qu(grid),current[:,None,:]));rotated=qu_apply(proposal[:,:,None,:],rays[None,None]);sim=interpolate_master(master,rotated,align_corners=True,normalize_coords=False);sim=sim-sim.mean(-1,keepdim=True);scores=torch.einsum('ij,ikj->ik',obs,sim);values,index=scores.max(-1);dots[start:end,step]=values;current=proposal[torch.arange(end-start),index];grid[index==((grid.shape[1]-1)/2)]*=.5
        qout[start:end]=current
    return qout,dots
def index_direct(master,patterns,rays,mask,resolution_deg=2.,dictionary_chunk=256,query_chunk=256):
    """Original uncompressed DI:716-pixel direct dot, no PCA or surrogate basis.

    This separate source branch normalizes detector rays before rotating them,
    simulates all900pixels then masks, and does not renormalize rotated rays.
    """
    dictionary=sample_cubic_grid(resolution_deg);rays=rays/rays.norm(dim=-1,keepdim=True);N=len(patterns);best=torch.full((N,1),-torch.inf);indices=torch.zeros((N,1),dtype=torch.int64);queries=[]
    for start in range(0,N,query_chunk):
        p=patterns[start:start+query_chunk].flatten(1)[:,mask];queries.append((start,p-p.mean(-1,keepdim=True)))
    offset=0
    for q in torch.split(dictionary,dictionary_chunk):
        sim=interpolate_master(master,qu_apply(q[:,None],rays[None]),align_corners=False,normalize_coords=False).reshape(len(q),-1)[:,mask];sim=sim-sim.mean(-1,keepdim=True)
        for start,query in queries:
            values,idx=torch.topk(query@sim.T,1,dim=1,largest=True,sorted=False);idx+=offset;end=start+len(query);merged=torch.cat([best[start:end],values],1);merged_idx=torch.cat([indices[start:end],idx],1);choose=torch.topk(merged,1,dim=1,largest=True,sorted=False)[1];best[start:end]=torch.gather(merged,1,choose);indices[start:end]=torch.gather(merged_idx,1,choose)
        offset+=len(q)
    return dictionary[indices[:,0]],best[:,0]
def reconstruct(raw,master,n_components=512,resolution_deg=2.,refinement_iterations=4,return_diagnostics=False,projection_center=(.4221,.2179,.4954)):
    """Return float32(40,50,4), scalar-first sample-ray->crystal quaternion.

    Seed20261021 fixes dictionary permutation only; no target/truth is generated.
    """
    if not 1<=n_components<=716:raise ValueError('Invalid component count')
    if not isinstance(refinement_iterations,int) or refinement_iterations<0:raise ValueError('Invalid refinement iteration count')
    torch.manual_seed(20261021);start=time.perf_counter();patterns=prepare_patterns(raw);master=prepare_master(master);rays=detector_rays(projection_center);mask=get_radial_mask((30,30)).flatten();t1=time.perf_counter();components,values,stats=learn_components(master,rays,mask,n_components,resolution_deg);t2=time.perf_counter();coarse,scores=index_patterns(master,patterns,rays,mask,components,resolution_deg);t3=time.perf_counter();q,dots=refine(master,patterns,rays,coarse,refinement_iterations);t4=time.perf_counter();result=q.numpy().reshape(40,50,4)
    if not return_diagnostics:return result
    return result,{'components':components.numpy(),'eigenvalues':values.numpy(),'coarse':coarse.numpy(),'dot_scores':scores.numpy(),'refinement_dots':dots.numpy(),'cleaned_patterns':patterns.numpy(),'master':master.numpy(),'rays':rays.numpy(),'mask':mask.numpy(),'seconds':time.perf_counter()-start,'stage_seconds':{'preprocessing':t1-start,'pca_learning':t2-t1,'indexing':t3-t2,'refinement':t4-t3},'dictionary_count':stats['dictionary_count']}
