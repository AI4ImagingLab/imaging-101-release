"""Explicit low-rank geostatistical Gauss-Newton, damping and line search.

Mathematical extraction from pyPCGA, BSD3; full provenance in NOTICE/validation.
General linear systems use the equivalent explicit Schur/Woodbury formula,
independently checked against all24 original matrix-free GMRES candidates.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigsh
from .preprocessing import prepare,NOISE_VARIANCE
from .physics_model import HydraulicModel,covariance_operator

def prior_eigenpairs(n_modes=50,length_scale=200.,seed=1234):
    op=LinearOperator((7500,7500),matvec=lambda x:covariance_operator(x,length_scale),dtype=np.float64)
    values,vectors=eigsh(op,k=n_modes,which='LM',v0=np.random.default_rng(seed).normal(size=7500))
    order=np.argsort(values)[::-1]
    return values[order],vectors[:,order]

def jacobian_products(model,field,base,directions,precision=1e-8):
    field=np.asarray(field,dtype=np.float64).reshape(-1);directions=np.asarray(directions,dtype=np.float64)
    result=np.empty((len(base),directions.shape[1]),dtype=np.float32)
    for j in range(directions.shape[1]):
        direction=directions[:,j];mag=float(field@direction);absmag=float(abs(field)@abs(direction))
        delta=(1 if mag>=0 else -1)*np.sqrt(precision)*max(abs(mag),absmag)/(np.linalg.norm(direction)+np.finfo(float).eps)**2
        if delta==0:delta=np.sqrt(precision)
        perturbed=model.forward(field+delta*direction)
        result[:,j]=(perturbed-base)/delta
    return result

def inverse_data_covariance(HZ,variance):
    """Return explicit inverse of HZ HZ^T + variance I using50-D solves."""
    H=np.asarray(HZ,dtype=np.float64);small=H.T@H+variance*np.eye(H.shape[1])
    def apply(vector):
        v=np.asarray(vector,dtype=np.float64)
        return (v-H@np.linalg.solve(small,H.T@v))/variance
    return apply

def cokriging_candidates(HX,HZ,Z,X,rhs,alphas):
    H=np.asarray(HZ,dtype=np.float64);h=np.asarray(HX,dtype=np.float64);rhs=np.asarray(rhs,dtype=np.float64).reshape(-1,1)
    fields=[];betas=[]
    for alpha in alphas:
        inv=inverse_data_covariance(H,float(alpha)*NOISE_VARIANCE)
        beta=np.linalg.solve(h.T@inv(h),h.T@inv(rhs));xi=inv(rhs-h@beta)
        fields.append((X@beta+Z@(H.T@xi))[:,0]);betas.append(beta[:,0])
    return np.column_stack(fields),np.column_stack(betas)

def objective(field,predicted,observed,U,eigenvalues,X,beta=None):
    r=np.asarray(field).reshape(-1,1);residual=np.asarray(observed).reshape(-1,1)-np.asarray(predicted).reshape(-1,1)
    if beta is not None:
        coefficients=(U.T@(r-X@np.asarray(beta).reshape(1,1)))/np.sqrt(eigenvalues)[:,None]
        penalty=float(np.sum(coefficients**2))
    else:
        coefficients=(U.T@r)/np.sqrt(eigenvalues)[:,None];drift=(U.T@X)/np.sqrt(eigenvalues)[:,None]
        cross=drift.T@coefficients
        penalty=float(np.sum(coefficients**2)-(cross.T@np.linalg.solve(drift.T@drift,cross)).item())
    return float(.5*np.sum(residual**2)/NOISE_VARIANCE+.5*penalty)

def variance_diagnostics(HZ,HX,Z,X,alpha):
    """Return labeled source heuristic and complete conditional covariance diagonal.

    The source posterior helper omits the lower-input Schur terms. Its diagnostic
    is reproduced for comparison; the complete saddle inverse is also computed.
    Neither array is actual geological uncertainty or independent truth.
    """
    H=np.asarray(HZ,dtype=np.float64);h=np.asarray(HX,dtype=np.float64);inv=inverse_data_covariance(H,float(alpha)*NOISE_VARIANCE)
    ih=inv(h);S=float((h.T@ih).item());cross=h.T@inv(H)
    K=H.T@inv(H)-(H.T@ih)@cross/S
    diagonal=np.sum((Z@K)*Z,axis=1);t=(cross@Z.T).ravel()/S;drift=X.ravel()
    source=np.minimum(1.,1-diagonal-drift*t)
    # Unknown drift can legitimately increase variance above the prior marginal;
    # preserve it in this independent complete-covariance diagnostic.
    complete=1-diagonal-2*drift*t+drift*drift/S
    return source,complete

def reconstruct(raw,n_modes=50,max_iterations=10,relative_tolerance=.01,compute_variance=True):
    observed,wells=prepare(raw);model=HydraulicModel(wells)
    values,U=prior_eigenpairs(n_modes);Z=U*np.sqrt(values);X=np.ones((7500,1))/np.sqrt(7500)
    field=np.ones(7500);prediction=model.forward(field);initial=prediction.copy();best_value=np.inf;best=None;trace=[]
    alphas=10.**np.linspace(0.,3.,24);stage=None
    for iteration in range(max_iterations):
        J=jacobian_products(model,field,prediction,np.column_stack([X,Z,field]))
        HX=J[:,:1];HZ=J[:,1:1+n_modes];Hs=J[:,-1:];rhs=observed[:,None]-prediction[:,None]+Hs
        candidates,betas=cokriging_candidates(HX,HZ,Z,X,rhs,alphas)
        forecasts=[model.forward(candidates[:,j]) for j in range(24)]
        losses=[objective(candidates[:,j],forecasts[j],observed,U,values,X,betas[:,j]) for j in range(24)]
        selected=int(np.argmin(losses));new=candidates[:,selected];new_prediction=forecasts[selected];value=losses[selected];used_line_search=False
        stage=dict(HX=HX,HZ=HZ,Z=Z,X=X,alpha=float(alphas[selected]))
        if value<best_value:
            best_value=value;best=new.copy();best_prediction=new_prediction.copy();best_iteration=iteration+1
        else:
            # The native extrapolating source grid includes -.1 and1.1.
            deltas=np.linspace(-.1,1.1,24);line_candidates=[d*field+(1-d)*new for d in deltas]
            line_predictions=[model.forward(v) for v in line_candidates]
            line_losses=[objective(v,p,observed,U,values,X) for v,p in zip(line_candidates,line_predictions)]
            pick=int(np.argmin(line_losses));new=line_candidates[pick];new_prediction=line_predictions[pick];value=line_losses[pick];used_line_search=True
            if value<best_value:best_value=value;best=new.copy();best_prediction=new_prediction.copy();best_iteration=iteration+1
            elif iteration>1:
                trace.append(dict(iteration=iteration+1,objective=value,stop='no_objective_progress',line_search=True));break
        change=float(np.linalg.norm(field-new)/np.linalg.norm(field))
        trace.append(dict(iteration=iteration+1,objective=value,relative_change=change,head_rms=float(np.sqrt(np.mean((new_prediction-observed)**2))),alpha=float(alphas[selected]),line_search=used_line_search))
        field=new.copy();prediction=new_prediction.copy()
        if change<relative_tolerance:break
    result=dict(reconstruction=best.reshape(1,75,100),predicted_heads=best_prediction.reshape(1,30,29),initial_heads=initial.reshape(1,30,29),trace=trace,best_iteration=best_iteration,eigenvalues=values,eigenvectors=U,last_linearization=stage)
    if compute_variance:
        source,complete=variance_diagnostics(**stage);result.update(source_variance_diagnostic=source.reshape(1,75,100),conditional_variance=complete.reshape(1,75,100))
    return result
