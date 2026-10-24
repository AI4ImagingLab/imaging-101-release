"""Local iterative structure-factor retrieval; no upstream library dependency."""
import numpy as np
from .physics_model import shell_grid,scattering_target,reciprocal_projection,real_space_update,profile_chi2
from .preprocessing import SIZE

def reconstruct(data: dict, seed: int=41, max_steps: int=6000, convergence_fraction: float=.01, return_history: bool=False) -> dict:
    grid=shell_grid();target,errors,selected=scattering_target(data,grid)
    rho=np.random.RandomState(seed).random_sample((SIZE,SIZE,SIZE));support=np.ones_like(rho,dtype=bool);sigma=3.
    chi=[];radius=[];support_volume=[]
    # The intensity is normalized to unit integrated density. Source monitoring
    # compares original arbitrary data units, so undo its I(0) scaling here.
    i0=data['smoothed_intensity'][0,0]
    for j in range(max_steps):
        projected,info=reciprocal_projection(rho,target,selected,grid)
        chi.append(profile_chi2(grid['q'],info['intensity']*i0,data))
        ratio=info['intensity'][1]/info['intensity'][0]
        radius.append(np.sqrt(-3*np.log(complex(ratio)))/grid['q'][1])
        newrho,support,sigma=real_space_update(projected,support,j,sigma)
        support_volume.append(np.sum(support)*grid['voxel']**3)
        # Original aliases rho/newrho after the first iteration. Its break state
        # therefore equals this last constrained update, not the previous one.
        rho=newrho
        if j>201 and np.std(chi[j-100:j])<convergence_fraction*np.mean(chi[j-100:j]):break
    rho,info=reciprocal_projection(rho,target,selected,grid)
    if np.sum(np.abs(rho[rho<0]))>np.sum(rho[rho>0]):rho=-rho
    rho*=1/np.sum(rho);rho/=grid['voxel']**3
    result={'reconstruction':rho[None]}
    if return_history:result['history']={'chi':np.array(chi),'rg':np.array(radius),'support_volume':np.array(support_volume),'iterations':j+1,'last_support':support,'target':target,'q':grid['q']}
    return result
