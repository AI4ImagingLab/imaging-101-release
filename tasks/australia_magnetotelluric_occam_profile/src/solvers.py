"""Explicit Occam update/search adapted from Dong Hao (2011), AGPL-3.0."""
import numpy as np
from .physics_model import response,jacobian,weighted_rms,LEGACY_RHO_COEFFICIENT

INITIAL_LOG_RESISTIVITY=2.0
TARGET_RMS=1.0
MAX_ITERATIONS=20

def roughness_matrix(n):
    """Return first differences with zero first row, shape (n,n)."""
    d=np.eye(n)-np.eye(n,k=-1);d[0,0]=0
    return d

def occam_update(model,residual_rho,residual_phase,std_rho,std_phase,j1,j2,damping):
    """Solve the smooth-model linearized normal equations, returning (L,)."""
    d=roughness_matrix(model.size);a=j1/std_rho[:,None];b=j2/std_phase[:,None]
    rhs=a.T@((residual_rho+j1@model)/std_rho)+b.T@((residual_phase+j2@model)/std_phase)
    return np.linalg.solve(damping*d.T@d+a.T@a+b.T@b,rhs)

def reconstruct(data,target_rms=TARGET_RMS,max_iterations=MAX_ITERATIONS):
    """Return (log10 resistivity:(25,), history:dict) using all observations.

    Decade damping search and sqrt(10) target smoothing preserve upstream
    initialization, acceptance inequalities, Jacobian reuse, and stopping rules.
    """
    f=1/data['period'];z=data['layer_top']
    observed=np.r_[data['log_apparent_resistivity'],data['impedance_phase']]
    errors=np.r_[data['log_resistivity_std'],data['phase_std']]
    m=np.full(z.size,INITIAL_LOG_RESISTIVITY);damping=10.;factor=10.
    models=[m.copy()];rms=[];lambdas=[];searches=[]
    def measure(v):
        rr,pp=response(f,v,z)
        return rr,pp,weighted_rms(np.r_[rr,pp],observed,errors)
    rms.append(measure(m)[2])
    for iteration in range(max_iterations):
        j1,j2,c=jacobian(f,m,z)
        rho=np.log10(LEGACY_RHO_COEFFICIENT*np.abs(c[:,0])**2*f)
        phase=np.arctan2(c[:,0].real,-c[:,0].imag)
        previous=weighted_rms(np.r_[rho,phase],observed,errors)
        dr=data['log_apparent_resistivity']-rho;dp=data['impedance_phase']-phase
        def update(lam):
            return occam_update(m,dr,dp,data['log_resistivity_std'],data['phase_std'],j1,j2,lam)
        left=damping/factor;right=damping*factor
        ml=update(left);mm=update(damping);mr=update(right)
        for k in range(10):
            rl,pl,el=measure(ml);rm,pm,em=measure(mm);rr,pr,er=measure(mr)
            searches.append((iteration+1,left,damping,right,el,em,er))
            if em<=el and em<=er:break
            if em>el and er>el:
                if left/factor<=0.0001:break
                mr=mm;right=damping;mm=ml;damping=left;left=damping/factor;ml=update(left)
            elif er<em and er<el:
                if damping*factor>=100000:break
                ml=mm;left=damping;mm=mr;damping=right;right=damping*factor;mr=update(right)
            else:break
            rl,pl,el=measure(ml);rm,pm,em=measure(mm);rr,pr,er=measure(mr)
        m=mm.copy();models.append(m.copy());rms.append(em);lambdas.append(damping)
        if em>previous:
            damping/=factor**2
            continue
        if em<=target_rms:
            dr=data['log_apparent_resistivity']-rm;dp=data['impedance_phase']-pm
            while True:
                damping*=np.sqrt(factor)
                candidate=occam_update(m,dr,dp,data['log_resistivity_std'],data['phase_std'],j1,j2,damping)
                cr,cp,ce=measure(candidate)
                if ce<target_rms:mm=candidate;rm=cr;pm=cp
                if ce>target_rms:break
            m=mm.copy();models.append(m.copy());rms.append(measure(m)[2]);lambdas.append(damping/np.sqrt(factor))
            break
    rr,pp,err=measure(m)
    return m,{'models':np.asarray(models),'rms':np.asarray(rms),'damping':np.asarray(lambdas),'searches':np.asarray(searches),'predicted_log_resistivity':rr,'predicted_phase':pp,'final_rms':err}
