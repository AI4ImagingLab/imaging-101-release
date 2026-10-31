import numpy as np
from .physics_model import forward,step

def reconstruct(data,steps=1000,repeats=20,seed=41):
 n=data['spectra'].shape[0];rng=np.random.RandomState(seed)
 sza=(data['solar_zenith']*data['scaler_scale'][0]+data['scaler_offset'][0]).astype(np.float32)
 q=np.c_[np.tile(sza,repeats),rng.uniform(size=(n*repeats,5))]
 target=np.tile(data['spectra'][:,5::10][:,:100],(repeats,1)).astype(np.float32)
 accumulator=np.full_like(q,.1)
 for i in range(steps):
  residual,gradient=forward(q,data,target);q,accumulator=step(q,gradient,accumulator)
 # Preserve published code's pre-final-update MAE selection, then reevaluate output.
 mae=np.mean(abs(residual),1).reshape(repeats,n);indices=np.argmin(mae,axis=0)*n+np.arange(n)
 chosen=q[indices];physical=(chosen-data['scaler_offset'])/data['scaler_scale']
 result=dict(reconstruction=physical[:,1:][None],modeled_spectra=forward(chosen,data)[None])
 return result,dict(selection_mae=mae.min(0),iterations=steps,starts=repeats)
