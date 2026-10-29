"""Published bounded nonlinear least-squares parameter inference, locally assembled."""
import numpy as np
from scipy.optimize import curve_fit
from .preprocessing import case_data,NCASES,MAX_TIME
from .physics_model import arterial_concentration,steady_signal,predict
INITIAL=np.array([.02,.15,.1,120.])
LOWER=np.array([0.,0.,0.,30.]);UPPER=np.array([.05,.3,.3,np.inf])

def fit_case(time,signal,aorta,cal,variant='measured'):
    """Return parameters, prediction, diagnostics; variants preserve original comparisons."""
    cal=dict(cal)
    if variant not in ['measured','ss_literature','linear','linear_delay']:raise ValueError('Unknown variant')
    if variant!='measured':cal['tissue_t1']=1.017 if cal['field_strength']==1 else 1.542
    sequence='lin' if variant.startswith('linear') else 'SS'
    ca=arterial_concentration(aorta,cal,sequence)
    reference=1/cal['tissue_t1'] if sequence=='lin' else steady_signal(1.,1/cal['tissue_t1'],cal['tr'],cal['flip_angle'])
    scale=np.mean(signal[:cal['n0']])/reference
    delayed=variant=='linear_delay'
    initial=np.r_[0.,INITIAL] if delayed else INITIAL.copy()
    lower=np.r_[0.,LOWER] if delayed else LOWER
    upper=np.r_[3.,UPPER] if delayed else UPPER
    def function(_, *p):
        delay=p[0] if delayed else 0.
        pars=p[1:] if delayed else p
        return predict(pars,time,ca,cal,scale,sequence,delay)[0]
    parameters,covariance=curve_fit(function,None,signal,p0=initial,bounds=(lower,upper))
    pars=parameters[1:] if delayed else parameters;delay=parameters[0] if delayed else 0.
    fitted,C,R1=predict(pars,time,ca,cal,scale,sequence,delay)
    return parameters,fitted,{'arterial_concentration':ca,'compartments':C,'R1':R1,'scale':scale,'covariance':covariance}

def reconstruct(data,variant='measured'):
    """Actively reconstruct every renal curve; padding remains exactly zero."""
    parameters=[];fitted=np.zeros((1,NCASES,MAX_TIME));diagnostics=[]
    for i in range(NCASES):
        t,y,a,cal=case_data(data,i);p,s,d=fit_case(t,y,a,cal,variant)
        parameters.append(p);fitted[0,i,:len(t)]=s;diagnostics.append(d)
    return {'reconstruction':np.array([parameters]),'signal':fitted},diagnostics
