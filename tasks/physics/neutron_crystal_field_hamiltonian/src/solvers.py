"""Three original Nelder-Mead stages, with the prevalidated converged budget."""
import numpy as np
from scipy.optimize import minimize
from .physics_model import diagonalize,neutron_spectrum,magnetization,predict
from .preprocessing import estimate_resolution

def initial_hypotheses():
    """Published coarse point-charge hypotheses: guesses, not calibration/truth."""
    tail=[.00054856,.06884367,.07168863,.16511265,.14361942,.0001,.0001,.0002,.0001]
    return np.array([[-.0360740328,-.000382694983,-.013895981,3.15219896e-6,-4.36048862e-6,3.36282008e-5]+tail,
       [.023257538,-.000473413709,.0126946732,1.08102667e-6,4.47713362e-5,2.23404722e-5]+tail])

def objective(parameters,data,width):
    _,e,v=diagonalize(parameters[:6]);loss=0
    for k,T in enumerate(data['temperature']):
        m=data['fit_mask'][k];E=data['energy'][k,m]*(1+T*parameters[14])-parameters[10]
        residual=parameters[6]*neutron_spectrum(E,T,width,parameters[7+k],e,v)+parameters[11+k]-data['intensity'][k,m]
        loss+=np.nansum((residual/data['reported_error'][k,m])**2)
    return loss

def select_model(parameters,data):
    """Use independent observed directional contrast; fit no magnetic amplitude."""
    contrasts=[]
    for axis in ['ab','c']:
        m=(data['field_'+axis]>=.8)&(data['field_'+axis]<=1.2)
        contrasts.append(float(np.mean(data['moment_'+axis][m])))
    observed=np.sign(contrasts[0]-contrasts[1]);model_contrasts=[]
    for p in parameters:
        ab=-magnetization(p[:6],.5,[1.,0,0])[0];c=-magnetization(p[:6],.5,[0,0,1.])[2]
        model_contrasts.append(float(ab-c))
    matches=np.flatnonzero(np.sign(model_contrasts)==observed)
    if len(matches)!=1:raise ValueError('Independent magnetic anisotropy does not uniquely discriminate the two hypotheses')
    return int(matches[0]),dict(observed_moment_1T=contrasts,model_contrast_1T=model_contrasts)

def reconstruct(data,maxfev=30000):
    width,diagnostics=estimate_resolution(data);initial=initial_hypotheses();stages=[];traces=[]
    def run_stage(start,indices):
        trace=[]
        def fun(x):
            p=start.copy();p[indices]=x;loss=objective(p,data,width)
            if len(trace)<40:trace.append((x.copy(),loss))
            return loss
        r=minimize(fun,start[indices],method='Nelder-Mead',options={'maxfev':maxfev})
        p=start.copy();p[indices]=r.x;stages.append(dict(success=bool(r.success),status=int(r.status),nfev=int(r.nfev),loss=float(r.fun),message=str(r.message)));traces.append(trace)
        if not r.success:raise RuntimeError('Selected converged protocol failed: '+str(r.message))
        return p
    warm=run_stage(initial[0],np.arange(7));one=run_stage(warm,np.arange(15));two=run_stage(initial[1],np.arange(15))
    parameters=np.stack([one,two]);selection,selection_diag=select_model(parameters,data)
    predictions=dict(reconstruction=parameters[None],neutron_spectrum=np.stack([predict(p,data['energy'],data['temperature'],width) for p in parameters])[None],resolution_coefficients=width[None],selected_model=np.array([selection],dtype=np.int64))
    diagnostics.update(initial_parameters=initial,warmup_parameters=warm,stages=stages,selection=selection_diag,trace_x=[np.array([x for x,y in tr]) for tr in traces],trace_loss=[np.array([y for x,y in tr]) for tr in traces])
    return predictions,diagnostics
