"""Complete local self-consistent SELCON1/2/3 inverse.
Adapted from SSCalcPy SelconsFunction.py, Hoffmann2021–2025, CC-BY-NC-SA4.0.
No source framework import. Original selection/tie/relaxation conventions preserved.
"""
import numpy as np
from .physics_model import spectral_svd,structure_estimate,spectral_refit,rmsd
from .preprocessing import prepare
def select_candidates(A,F,q):
    solutions=[];indices=[]
    for last in range(2,A.shape[1]):
        dec=spectral_svd(A[:,:last+1])
        for basis in range(min(7,last+1)):
            solutions.append(structure_estimate(F[:,:last+1],q,dec,basis+1));indices.append((last,basis))
    solutions=np.array(solutions).T
    sum_rule=.05;fraction_rule=-.025
    while True:
        selected=[];parameters=[];valid=0;number=0
        for last in range(2,A.shape[1]):
            best_sum=10000;best_basis=0;best=None
            for basis in range(min(7,last+1)):
                candidate=solutions[:,number];number+=1
                error=abs(1-candidate.sum())
                if error<=sum_rule and candidate.min()>=fraction_rule:
                    valid+=1
                    if error<=best_sum:best_sum=error;best_basis=basis;best=candidate
            # Preserve original zero-based convention: a rank-one winner is not retained.
            if best_basis>0:parameters.append(np.array([last,best_basis]));selected.append(best)
        if valid:
            if not selected:raise ValueError('Original selection has only rank-one candidates')
            return selected,parameters
        fraction_rule-=.005;sum_rule+=.01
def helix_selection(solutions,hj_helix):
    helix=np.array(solutions)[:,0]
    if hj_helix>.65:return np.flatnonzero(helix>.65)
    if hj_helix>=.25:center=(hj_helix+helix.max())/2
    elif hj_helix>=.15:center=(hj_helix+helix.mean())/2
    else:center=(hj_helix+helix.min())/2
    return np.flatnonzero((helix<=center+.03)&(helix>=center-.03))
def selcon(A,F,q,ir_only=False):
    hj5=structure_estimate(F,q,spectral_svd(A),5)
    distances=np.zeros(A.shape[1])
    for i in range(A.shape[1]):distances[i]=rmsd(q,A[:,i])
    order=np.argsort(distances)
    # The original sorting allocates float64, after the initial float32 HJ5 stage.
    sorted_A=np.zeros(A.shape);sorted_F=np.zeros(F.shape)
    for i,j in enumerate(order):sorted_A[:,i]=A[:,j];sorted_F[:,i]=F[:,j]
    augmented_A=np.c_[q,sorted_A];augmented_F=np.c_[sorted_F[:,0],sorted_F]
    for attempt in range(50):
        candidates,params=select_candidates(augmented_A,augmented_F,q)
        sel1=np.mean(np.array(candidates).T,axis=1)
        converged=rmsd(augmented_F[:,0],sel1)<.0025
        augmented_F[:,0]=sel1
        if converged:break
    errors=np.array([rmsd(spectral_refit(spectral_svd(augmented_A[:,:last+1]),basis+1),q) for last,basis in params])
    rule=.025 if ir_only else .25
    while not np.any(errors<=rule):rule+=.01
    accepted=np.flatnonzero(errors<=rule)
    solutions=[candidates[i] for i in accepted];params2=[params[i] for i in accepted]
    sel2=np.mean(np.array(solutions).T,axis=1)
    retained=helix_selection(solutions,hj5[0])
    fallback=not len(retained)
    if fallback:retained=np.arange(len(solutions))
    final_solutions=np.array([solutions[i] for i in retained]).T
    final_params=np.array([params2[i] for i in retained])
    fractions=np.mean(final_solutions,axis=1);spread=np.std(final_solutions,axis=1)
    refits=np.zeros((len(q),len(final_params)))
    for i,(last,basis) in enumerate(final_params):refits[:,i]=spectral_refit(spectral_svd(augmented_A[:,:last+1]),basis+1)
    fitted=np.mean(refits,axis=1)
    return {'fractions':fractions,'spectrum':fitted,'spread':spread,'ensemble_spectra':refits,'hj5':hj5,'ix':order,'sel1':sel1,'sel2':sel2,'sel3':fractions,'Mean_refit_prot':fitted,'refit_np_param':final_params,'refit_rule':rule,'attempts':attempt+1,'fallback':fallback}
def reconstruct(data,ir_scale=15.,mode='joint'):
    A,F,q=prepare(data,ir_scale,mode)
    details=[selcon(A,F,q[:,i],ir_only=(mode=='ir')) for i in range(q.shape[1])]
    spectra=np.stack([d['spectrum'] for d in details])
    if mode=='joint' and ir_scale!=0:spectra[:,66:]/=ir_scale
    return {'reconstruction':np.stack([d['fractions'] for d in details])[None],'spectra':spectra[None]},details
