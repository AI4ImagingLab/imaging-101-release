"""Published similarity CPD EM with local E/M updates; MIT pycpd adaptation."""
import numpy as np
from .preprocessing import prepare
from .physics_model import transform,initialize_variance,expectation
MAX_ITERATIONS=400
TOLERANCE=1e-6
OUTLIER_FRACTION=.1

def one_step(target,source,transformed_source,variance,previous_objective,outlier_fraction=OUTLIER_FRACTION,tolerance=TOLERANCE):
    """Explicit Gaussian-posterior weighted Procrustes and variance update."""
    p,p1,pt1,px,mass=expectation(target,transformed_source,variance,outlier_fraction)
    mux=np.divide(np.sum(px,axis=0),mass)
    muy=np.divide(np.sum(np.dot(p.T,source),axis=0),mass)
    xhat=target-np.tile(mux,(len(target),1));yhat=source-np.tile(muy,(len(source),1))
    ypy=np.dot(p1.T,np.sum(yhat*yhat,axis=1))
    a=np.dot(np.dot(xhat.T,p.T),yhat)
    u,_,v=np.linalg.svd(a,full_matrices=True)
    sign=np.ones(target.shape[1]);sign[-1]=np.linalg.det(np.dot(u,v))
    rotation=np.dot(np.dot(u,np.diag(sign)),v).T
    scale=np.trace(np.dot(a.T,rotation.T))/ypy
    translation=mux.T-scale*np.dot(rotation.T,muy.T)
    transformed=transform(source,scale,rotation,translation)
    xpx=np.dot(pt1.T,np.sum(xhat*xhat,axis=1));trar=np.trace(np.dot(a,rotation))
    q=(xpx-2*scale*trar+scale*scale*ypy)/(2*variance)+target.shape[1]*mass/2*np.log(variance)
    difference=abs(q-previous_objective)
    updated_variance=(xpx-scale*trar)/(mass*target.shape[1])
    if updated_variance<=0:updated_variance=tolerance/10
    return {'R':rotation,'t':translation,'s':scale,'TY':transformed,'sigma2':updated_variance,'q':q,'diff':difference,'A':a,'YPY':ypy}

def reconstruct(data,stride=40,outlier_fraction=OUTLIER_FRACTION,max_iterations=MAX_ITERATIONS,tolerance=TOLERANCE):
    """Fit selected observations; return ALL registered source points and pose."""
    x,y,origin,unit=prepare(data,stride)
    variance=initialize_variance(x,y);transformed=y;q=np.inf;difference=np.inf;history=[]
    iteration=0
    while iteration<max_iterations and difference>tolerance:
        state=one_step(x,y,transformed,variance,q,outlier_fraction,tolerance)
        variance,q,difference,transformed=state['sigma2'],state['q'],state['diff'],state['TY']
        iteration+=1;history.append([iteration,q,variance,state['s']])
    if iteration==0:raise ValueError('At least one EM iteration is required')
    scale,rotation=state['s'],state['R']
    translation=state['t']*unit+origin-scale*origin@rotation
    aligned=transform(data['source_points'],scale,rotation,translation)
    output={'reconstruction':aligned[None],'rotation':rotation[None],'translation':translation[None],'scale':np.array([[scale]],dtype=np.float64)}
    return output,{'history':np.asarray(history),'iterations':iteration,'variance':variance,'objective_change':difference,'coordinate_unit_m':unit,'origin_m':origin}
