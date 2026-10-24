"""Explicit accelerated Lucy maximum entropy with a floating Gaussian default."""
import numpy as np
from .preprocessing import prepare
from .physics_model import DopplerModel

def objective_derivatives(model,x,data,weights,ns=7):
    """Return H,S,raw gradients and current forward/default intermediates."""
    predicted=model.forward(x)
    if np.any(predicted<=0) or np.any(x<=0):raise FloatingPointError('Likelihood requires positive latent field and projection')
    H=float(np.sum(weights*(data*np.log(predicted)-predicted)))
    gradH=model.backward(weights*(data/predicted-1))
    q=model.smear(x,ns);logratio=np.log(x)-np.log(q);ex=np.exp(np.log(q))
    S=float(np.sum(-x*(logratio-1)-ex))
    gradS=-logratio+model.smear((x-ex)/q,ns)
    return H,S,gradH,gradS,predicted,q,logratio,ex

def newton_coefficients(model,x,old,new,data,weights,predicted,q,logratio,ex,alpha,ns=7):
    """Analytic2D Newton coefficients, preserving source near-singular fallback."""
    h1=model.forward(old);h2=model.forward(new);q1=model.smear(old,ns);q2=model.smear(new,ns)
    ratio=data/predicted-1;phw=data*weights;z=x-ex
    hl=np.sum(weights*h1*ratio);hm=np.sum(weights*h2*ratio)
    hll=-np.sum((h1/predicted)**2*phw);hmm=-np.sum((h2/predicted)**2*phw);hlm=-np.sum(h1/predicted*h2/predicted*phw)
    t1=q1/q;t2=q2/q
    sl=np.sum(-old*logratio+z*t1);sm=np.sum(-new*logratio+z*t2)
    a=old/x-t1;b=new/x-t2
    sll=-np.sum(x*a*a);smm=-np.sum(x*b*b);slm=-np.sum(x*a*b)
    ql=hl+alpha*sl;qm=hm+alpha*sm;qll=hll+alpha*sll;qmm=hmm+alpha*smm;qlm=hlm+alpha*slm
    determinant=qmm*qll-qlm**2;ratio_det=abs(determinant)/(qll**2+qmm**2)
    if ratio_det<1e-3:return 0.,float(-qm/qmm)
    return float((qlm*qm-qmm*ql)/determinant),float((qlm*ql-qll*qm)/determinant)

def solve(model,state,alpha=.002,ns=7,accuracy=8e-4,max_iterations=150,current_direction_only=False):
    """Compute full source-selected positive latent field and diagnostic trace.

    Q and gradients in each log row describe the pre-update field. Saved state
    after each iteration includes the positive update and flux renormalization.
    current_direction_only is an explicitly labeled notebook ablation.
    """
    if alpha<=0 or ns<1 or accuracy<=0 or max_iterations<1:raise ValueError('Invalid solver configuration')
    data=state['positive_data'];weights=state['weights'];total=state['total_flux'];x=state['initial_field'].copy();old=None;trace=[];first=[];small_steps=[]
    for iteration in range(1,max_iterations+1):
        H,S,gH,gS,predicted,q,logratio,ex=objective_derivatives(model,x,data,weights,ns)
        norm=float(np.sum(x))
        gH+=-np.sum(x*gH)/norm+total/norm-1
        gS-=np.sum(x*gS)/norm
        new=x*(gH+alpha*gS)
        if old is None:old=new.copy()
        c1,c2=newton_coefficients(model,x,old,new,data,weights,predicted,q,logratio,ex,alpha,ns)
        if current_direction_only:
            # Selected analytical Newton step in new direction only, not a named comparator.
            c1,c2=newton_coefficients(model,x,new,new,data,weights,predicted,q,logratio,ex,alpha,ns)
        amplitude=np.hypot(c1,c2)
        if not np.isfinite(amplitude) or amplitude==0:raise FloatingPointError('Invalid Newton multiplier')
        a,b=c1/amplitude,c2/amplitude;update=a*old+b*new
        negative=update<0;flm=float(np.max(-update[negative]/x[negative])) if np.any(negative) else 0.
        limit=min(1e5,float(np.float32(.9))/flm) if flm>0 else 1e5
        step=min(amplitude,limit);small_steps.append(step)
        if iteration>=20 and max(small_steps[-20:])<.01:
            trace.append(dict(iteration=iteration,objective=H+alpha*S,convergence=None,step=step,stop='vanishing_updates'));break
        x+=step*update;old=update
        denominator=np.abs(gH)+alpha*np.abs(gS)
        ratio=np.divide(np.abs(gH+alpha*gS),denominator,out=np.zeros_like(x),where=denominator!=0)
        convergence=float(np.sqrt(np.sum(ratio*ratio)/len(x)))
        x*=total/np.sum(x)
        trace.append(dict(iteration=iteration,objective=float(H+alpha*S),likelihood=H,entropy=S,convergence=convergence,step=float(step)))
        if iteration<=2:first.append(dict(field_after_update=x.copy(),forward_before_update=predicted.copy(),likelihood_gradient_projected=gH.copy(),entropy_gradient_projected=gS.copy(),floating_default_before_update=q.copy(),new_direction=new.copy(),saved_direction_after_update=old.copy()))
        if convergence<=accuracy:break
    return x,trace,first

def reconstruct(raw,alpha=.002,ns=7,accuracy=8e-4,max_iterations=150,model=None,current_direction_only=False):
    """Reconstruct every native144² signed emissivity sample from raw spectra."""
    prepared=prepare(raw)
    if model is None:model=DopplerModel(prepared)
    state=model.initialize_data(prepared['line_spectra'])
    x,trace,first=solve(model,state,alpha,ns,accuracy,max_iterations,current_direction_only)
    vector=(x-state['baseline_field'])/model.dv**2*model.denu
    return dict(reconstruction=model.scatter(vector)[None],iterations=len(trace),trace=trace,first_iterations=first,positive_field=x,model=model,state=state,prepared=prepared)
