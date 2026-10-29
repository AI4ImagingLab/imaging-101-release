"""Explicit selected SYNCmoss custom LM and independent-start site inversion.

Adapted from ESRF SYNCmoss minimi_lib.py, MIT; see LICENSE. Only the bound,
unweighted, unlinked-parameter mathematical path used by this task is retained.
"""
import numpy as np
from .physics_model import transmission

def source_norm(values):
    return np.sqrt(np.sum((np.array([values])/len(values))**2))

def jacobian(function,parameters,velocity,fixed,base):
    rows=[];active=[]
    for k in range(len(parameters)):
        if k in fixed:continue
        step=np.maximum(1e-12,abs(parameters[k])/1e6)
        bumped=parameters.copy();bumped[k]+=step
        row=(function(velocity,bumped)-base)/step
        if not np.all(row==0):rows.append(row);active.append(k)
    return np.asarray(rows),np.asarray(active,dtype=np.int64)

def elimination(hessian,damping,gradient):
    """Preserve source Gaussian elimination and back-substitution ordering."""
    n=len(hessian);aug=np.empty((n,n+1));aug[:,:n]=hessian;aug[:,n]=gradient
    aug[np.arange(n),np.arange(n)]+=damping
    for i in range(n):
        pivot=i
        for j in range(i,n):
            if abs(aug[j,i])>0:pivot=j;break
        if pivot!=i:aug[[i,pivot]]=aug[[pivot,i]]
        for row in range(i+1,n):
            factor=aug[row,i]
            if abs(factor)>0:
                for col in range(i,n+1):aug[row,col]=aug[row,col]-factor/aug[i,i]*aug[i,col]
    x=np.zeros(n)
    for i in range(n):
        row=n-1-i;x[row]=aug[row,n]/aug[row,row]
        for j in range(i):x[row]+=-aug[row,n-1-j]*x[n-1-j]/aug[row,row]
    return x

def paired_lm(velocity,observed,initial,fixed,bounds,outer_limit=75,inner_limit=25,eps=1e-10,recheck=True):
    """Author diagonal/uniform LM damping state machine, no optimizer wrapper."""
    original_fixed=list(fixed);fixed=list(fixed);p=np.array(initial,dtype=float)
    tau=.001;nu0=2.618;nu=nu0;accepted=0;bound_hit=.5;gain=0.;strategy_count=0;stall=0;branch=0;flags=np.zeros(len(p),dtype=int);stop=0
    model=transmission(velocity,p);J,active=jacobian(transmission,p,velocity,fixed,model)
    H=J@J.T;chi=observed-model;gradient=J@chi
    if not len(gradient) or np.max(abs(gradient))<eps:stop=1
    damping=tau*np.diag(H).astype(float);mu_strategy=2;history=[]
    variance=abs(observed)+1
    for outer in range(outer_limit):
        if (stall in [2,3] and mu_strategy==1) or stop==1:break
        gain=-1.;inner=0;strategy_count+=1
        if strategy_count%2==0:
            J,active=jacobian(transmission,p,velocity,fixed,model)
            if not len(J):break
            H=J@J.T;damping=tau*np.diag(H).astype(float);gradient=J@chi
            if np.max(abs(gradient))<eps:stop=1
            stall+=1;mu_strategy=2;branch=0
        else:
            damping=np.zeros(len(J))
            if len(H):damping[:]=np.amax(H)*tau
            stall+=1;mu_strategy=1;branch=0
        nu=nu0;stop=0
        while gain<=0 and stop==0 and inner<inner_limit:
            step=elimination(H,damping,gradient) if len(J) else np.array([])
            free_values=np.array([p[i] for i in range(len(p)) if i not in fixed])
            if source_norm(step)<eps**2*source_norm(free_values):stop=2
            trial=p.copy();fraction=np.full(len(p),np.inf)
            trial[active]+=step
            for k in active:
                if trial[k]<bounds[0,k]:flags[k]=-1
                if trial[k]>bounds[1,k]:flags[k]=1
            if np.any(flags!=0):
                for k in range(len(p)):
                    if flags[k]==-1 and p[k]!=trial[k]:fraction[k]=(p[k]-bounds[0,k])/(p[k]-trial[k])
                    if flags[k]==1 and p[k]!=trial[k]:fraction[k]=(p[k]-bounds[1,k])/(p[k]-trial[k])
                clamped=step*np.min(fraction);bound_hit=int(np.flatnonzero(fraction==np.min(fraction))[0])
                trial[active]=p[active]+clamped
                trial[bound_hit]=bounds[int((abs(flags[bound_hit])+np.sign(flags[bound_hit]))/2),bound_hit]
            trial_model=transmission(velocity,trial);trial_chi=observed-trial_model
            predicted_drop=float(np.sum(step*(damping*step+gradient))) if len(step) else 0.
            if np.all(np.isfinite(trial_model)):
                with np.errstate(divide='ignore',invalid='ignore'):gain=float((source_norm(chi)**2-source_norm(trial_chi)**2)/abs(predicted_drop))
            else:gain=-1.
            if gain<=0:
                if stall in [2,3] and mu_strategy==1 and bound_hit!=.5 and stop==2:fixed.append(bound_hit);stall=0
                damping*=nu;nu=nu0*nu;flags[:]=0;bound_hit=.5;branch=1
            else:
                old_chi=chi;chi=trial_chi;gradient=J@chi
                if len(gradient) and np.max(abs(gradient))<eps:stop=3
                if accepted:
                    dof=len(observed)-len(active)
                    if np.sum(old_chi**2/variance)/dof-np.sum(chi**2/variance)/dof<eps:stop=4;tau=.001;nu=nu0
                p=trial;model=trial_model
                history.append({'outer':outer,'inner':inner,'sse':float(np.sum(chi**2)),'gain':gain})
                if mu_strategy==2:
                    if branch==0:damping=tau*np.diag(H).astype(float);tau/=nu
                    if branch==1:damping*=nu0;tau=.001
                else:
                    if branch==0:
                        factor=abs(float(1-(2*gain-1)**3));damping=damping*factor if factor<1/nu0 else damping/nu0
                    if branch==1:damping*=nu0
                    tau=.001
                nu=nu0;gain=-1.;accepted=1;inner+=1
                if stop not in [3,4]:stall=0
                if bound_hit!=.5:
                    fixed.append(bound_hit);fixed.sort();flags[:]=0;bound_hit=.5;inner=inner_limit;stall=0
                    if strategy_count%2==0:strategy_count+=1
    if len(fixed)!=len(original_fixed) and recheck:
        new=p.copy();failed=False
        for index in np.setdiff1d(fixed,original_fixed):
            nudge=max(1e-12,abs(new[index])/1e6)
            if p[index]==bounds[0,index]:new[index]+=nudge;failed|=new[index]>bounds[1,index]
            if p[index]==bounds[1,index]:new[index]-=nudge;failed|=new[index]<bounds[0,index]
        if not failed:
            p,extra=paired_lm(velocity,observed,new,original_fixed,bounds,max(int(outer_limit/2),1),inner_limit,eps,False);history+=extra
    return p,history

def configuration(start_index=0,count=4):
    first=[(.3,.25),(.4,.55),(1.1,1.05),(1.2,1.45)]
    second=[(.25,.3),(.55,.65),(.95,1.),(1.35,1.5)]
    p=[1.,0.,0.,0.,0.,0.,0.,0.]
    for delta,epsilon in (first if start_index==0 else second)[:count]:p.extend([.12 if start_index==0 else .15,delta,epsilon,.25 if start_index==0 else .3,0.,0.,0.,0.,1.])
    p=np.array(p);free=[0]+[8+9*i+j for i in range(count) for j in [0,1,2,3]];fixed=np.setdiff1d(np.arange(len(p)),free)
    lower=np.full(len(p),-np.inf);upper=np.full(len(p),np.inf);lower[0]=.95;upper[0]=1.05
    for i in range(count):lower[8+9*i:12+9*i]=[.0001,-.2,.02,.02];upper[8+9*i:12+9*i]=[5.,1.7,2.,1.5]
    return p,fixed,np.stack([lower,upper])

def outputs(parameters,velocities):
    sites=[];thickness=[];curves=[]
    for p,v in zip(parameters,velocities):
        rows=np.array([p[start:start+4] for start in range(8,len(p),9)])
        rows=rows[np.lexsort((rows[:,2],rows[:,1]))]
        sites.append(np.stack([rows[:,1],2*rows[:,2],rows[:,3]],axis=-1));thickness.append(rows[:,0]);curves.append(transmission(v,p))
    return {'reconstruction':np.array(sites)[None],'optical_thickness':np.array(thickness)[None],'baseline':np.asarray(parameters)[:,0][None],'transmission':np.array(curves)[None]}

def reconstruct(velocities,observations):
    parameters=[];records=[]
    for velocity,observed in zip(velocities,observations):
        candidates=[]
        for start in [0,1]:
            p,fixed,bounds=configuration(start);fit,history=paired_lm(velocity,observed,p,fixed,bounds)
            sse=float(np.sum((transmission(velocity,fit)-observed)**2));candidates.append((sse,fit,history))
        best=min(candidates,key=lambda r:r[0]);parameters.append(best[1]);records.append({'sse':best[0],'starts_sse':[r[0] for r in candidates],'history':best[2]})
    return outputs(parameters,velocities),records
