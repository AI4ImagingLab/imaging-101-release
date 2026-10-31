"""Complete sampled Gauss–Seidel MENT; five full24-profile sweeps."""
import copy
import numpy as np
from .preprocessing import COORDS,preprocess
from .physics_model import VOXEL_EDGES,fit_covariance,normalization,grid_points,projected_coordinates,probability
EDGES=np.r_[COORDS-(COORDS[1]-COORDS[0])/2,COORDS[-1]+COORDS[1]-COORDS[0]]

def sample_grid(p,size,rng):
    index=rng.choice(p.size,size,replace=True,p=p/p.sum())
    multi=np.unravel_index(index,(35,)*4)
    z=np.empty((size,4))
    for k in range(4):z[:,k]=rng.uniform(VOXEL_EDGES[multi[k]],VOXEL_EDGES[multi[k]+1])
    return z

def simulated_profile(z,V,M,direction):
    u=(z@V.T)@M.T
    positions=u[:,0] if direction==0 else u[:,2] if direction==1 else np.sum(u*(np.array([1.,0,-1.,0])/np.sqrt(2)),axis=1)
    values=np.histogram(positions,EDGES,density=True)[0]
    return values/values.sum()/(COORDS[1]-COORDS[0])

def reconstruct(raw,epochs=5,nsamp=1_000_000,seed=0,rng_state=None):
    measured,moments=preprocess(raw);matrices=raw['transfer_matrices']
    covariance=fit_covariance(moments,matrices);N=normalization(covariance);V=np.linalg.inv(N)
    z=grid_points();projected=projected_coordinates(z,V,matrices)
    prior=np.exp(-.5*np.sum(z*z,axis=1))/(2*np.pi)**2
    h=(measured.reshape(24,100)>0).astype(float);history=[h.copy()]
    rng=np.random.default_rng(seed)
    if rng_state is not None:rng.bit_generator.state=copy.deepcopy(rng_state)
    first={}
    for epoch in range(epochs):
        for index in range(24):
            p=probability(h,projected,prior)
            sampled=sample_grid(p,nsamp,rng)
            predicted=simulated_profile(sampled,V,matrices[index//3],index%3)
            target=measured.reshape(24,100)[index]
            ratio=np.ones(100);valid=(predicted>0)&(target>0)
            ratio[valid]=target[valid]/predicted[valid]
            h[index]*=1+.2*(ratio-1)
            if epoch==0 and index==0:first=dict(probability=p.reshape((35,)*4),sample_prefix=sampled[:1000],simulated=predicted,updated_function=h[0].copy())
        history.append(h.copy());print('LOCAL_FULL_EPOCH',epoch+1,flush=True)
    p=probability(h,projected,prior);mass=p/p.sum()
    return dict(reconstruction=mass.reshape((35,)*4),normalization=N,covariance=covariance,functions=h,history=np.asarray(history),first=first)

