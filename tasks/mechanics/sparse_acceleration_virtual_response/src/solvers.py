"""Full covariance matching, explicit Kalman filtering and RTS smoothing."""
import time
import numpy as np
from scipy.optimize import minimize
from .preprocessing import observation_indices
from .physics_model import modal_model,augmented_model,hyperparameter_cost,LOWER,SPAN
INITIAL_NORMALIZED=np.array([0.11347605884015177, 0.22762184346398384])

def fit_hyperparameters(observations,model):
    """Constrained Hellinger fit; every invocation relearns from observations."""
    history=[INITIAL_NORMALIZED.tolist()]
    result=minimize(hyperparameter_cost,INITIAL_NORMALIZED,args=(observations,model),method='SLSQP',bounds=[(0.,1.),(0.,1.)],callback=lambda x:history.append(x.tolist()),options={'maxiter':200,'ftol':1e-12,'eps':np.sqrt(np.finfo(float).eps)})
    if not result.success:raise RuntimeError('Hyperparameter optimization failed: '+result.message)
    return LOWER+SPAN*result.x,dict(normalized=result.x,objective=float(result.fun),iterations=int(result.nit),message=str(result.message),history=history)

def kalman_rts(transition,noise,measurement,measurement_noise,observations,indices,count,initial_covariance,maps=None,callback=None):
    """Explicit Gaussian recursions, preserving native skipped time-zero update.

    Store forward covariances for the backward recursion, then keep only
    mapped diagonal variances and three full smoothed-covariance samples.
    This avoids retaining unconsumed full smoothed covariance copies.
    """
    A,Q,H,R=transition,noise,measurement,measurement_noise;d=len(A);eye=np.eye(d)
    mean_f=np.zeros((count,d));mean_p=np.zeros_like(mean_f)
    covariance_f=np.empty((count,d,d));covariance_p=np.empty_like(covariance_f)
    covariance_f[0]=initial_covariance;covariance_p[0]=initial_covariance
    measured={int(k):i for i,k in enumerate(indices)}
    for k in range(1,count):
        mean_p[k]=A@mean_f[k-1];covariance_p[k]=A@covariance_f[k-1]@A.T+Q
        if k in measured:
            residual=observations[measured[k]]-H@mean_p[k]
            S=H@covariance_p[k]@H.T+R;gain=covariance_p[k]@H.T@np.linalg.inv(S)
            mean_f[k]=mean_p[k]+gain@residual
            J=eye-gain@H;covariance_f[k]=J@covariance_p[k]@J.T+gain@R@gain.T
        else:mean_f[k]=mean_p[k];covariance_f[k]=covariance_p[k]
        if callback is not None and k%20000==0:callback('filter',k,count)
    mean=mean_f.copy();Ps=covariance_f[-1].copy();maps={} if maps is None else maps
    variance={name:np.empty((count,len(matrix))) for name,matrix in maps.items()}
    samples={};chosen={0,count//2,count-1}
    def capture(k,P):
        for name,L in maps.items():variance[name][k]=np.diag(L@P@L.T)
        if k in chosen:samples[k]=P.copy()
    capture(count-1,Ps)
    for k in range(count-2,-1,-1):
        gain=covariance_f[k]@A.T@np.linalg.inv(covariance_p[k+1])
        mean[k]=mean_f[k]+gain@(mean[k+1]-mean_p[k+1])
        Ps=covariance_f[k]+gain@(Ps-covariance_p[k+1])@gain.T;capture(k,Ps)
        if callback is not None and k%20000==0:callback('smooth',k,count)
    info=dict(variance=variance,covariance_samples=np.stack([samples[k] for k in sorted(chosen)]),first_filtered_state=mean_f[:10].copy(),first_predicted_state=mean_p[:10].copy(),first_filtered_covariance=covariance_f[:10].copy(),first_predicted_covariance=covariance_p[:10].copy())
    return mean,info

def reconstruct(data,modes=3,hyperparameters=None,callback=None):
    """Infer all100001x10 physical fields; fixed HP only isolates test stages."""
    start=time.perf_counter();times=data['reconstruction_time'];dt=float(times[1]-times[0])
    model=modal_model(data['mass'],data['stiffness'],data['damping'],data['sensor_matrix'],dt,modes)
    if hyperparameters is None:hp,optimization=fit_hyperparameters(data['accelerations'],model)
    else:hp=np.asarray(hyperparameters);optimization={'role':'explicit fixed-HP stage comparison; not standard pipeline'}
    aug=augmented_model(model,hp);d=aug['na'];phi=model['Phi_r'];n=len(data['mass'])
    displacement=np.zeros((n,d));displacement[:,:modes]=phi
    velocity=np.zeros((n,d));velocity[:,modes:2*modes]=phi
    force=np.zeros((modes,d));force[np.arange(modes),2*modes+3*np.arange(modes)]=1
    maps=dict(reconstruction=displacement,velocity=velocity,acceleration=aug['H_ae'],modal_force=force)
    state,info=kalman_rts(aug['F_ad'],aug['Q_ad'],aug['H_ad'],np.eye(model['ny'])*1e-4,data['accelerations'],observation_indices(data['observation_time'],times),len(times),aug['P_0'],maps,callback)
    fields={k:state@L.T for k,L in maps.items()}
    info.update(model=model,augmented=aug,hyperparameters=hp,optimization=optimization,state=state,runtime_seconds=time.perf_counter()-start)
    return fields,info
