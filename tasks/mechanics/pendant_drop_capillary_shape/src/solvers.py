# Adapted from OpenDrop, copyright2020 Joseph Berry/Rico Tabor; GPL3. See LICENSE and test/validation.md.
"""Explicit native closest-profile Jacobian and custom damped normal updates."""
import math
import numpy as np
from .preprocessing import prepare,fit_circle,bond_number
from .physics_model import YoungLaplaceSolution
def rotation(angle):
    return np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
def jacobian_residual(points,parameters,profile_size=0.):
    x0,y0,radius,bond,angle=parameters
    solution=YoungLaplaceSolution(bond,radius);matrix=rotation(angle)
    rz=(matrix@(points-[x0,y0]).T).T
    guesses=[-.05*max(profile_size,4),.05*max(profile_size,4)]
    closest=np.empty((len(points),3));warnings=0
    for i,(r,z) in enumerate(rz):
        side=int(r>0) # numpy2.4 requires integer list index; identical branch convention.
        s,(er,ez),exceeded=solution.closest((r,z),guesses[side],10,1e-6)
        warnings+=int(exceeded);closest[i]=s,er,ez;guesses[side]=s
    s,er,ez=closest.T
    error=np.copysign(np.linalg.norm((er,ez),axis=0),er)
    rs,zs,phi,dr, dz,dphi=solution(s).T
    r=rs+er;z=zs+ez
    dx,dy=-(matrix.T@[er,ez])/error
    dR=-(er*rs+ez*zs)/(radius*error)
    dB=-(er*dr+ez*dz)/error
    da=(er*-z+ez*r)/error
    J=np.stack((dx,dy,dR,dB,da),axis=1)
    return J,np.stack((s,error),axis=1),warnings
def silhouette(parameters,profile_size,count=401):
    x,y,r,b,a=parameters
    s=(np.linspace(0,1,count)-.5)*2*profile_size
    rz=YoungLaplaceSolution(b,r)(s)[:,:2]
    return (rotation(a).T@rz.T).T+[x,y]
def solve(contour):
    points=prepare(contour);initial=fit_circle(points)
    parameters=np.array([*initial,bond_number(points,*initial),0.])
    history=[parameters.copy()];objectives=[];profile_size=0.;damping=cutoff=0.;ssr=math.inf;warnings=0;reason=0
    for step in range(11):
        J,res,nwarn=jacobian_residual(points,parameters,profile_size);warnings+=nwarn
        A=J.T@J;v=J.T@res[:,1]
        delta=-np.linalg.inv(A+damping*np.diag(np.diag(A)))@v
        next_damping=damping;next_cutoff=cutoff;next_ssr=np.sum(res[:,1]**2)
        if not math.isinf(ssr):
            ratio=(ssr-next_ssr)/(delta@(-2*v-A@delta))
            if ratio<.25:
                factor=np.clip(2-(next_ssr-ssr)/(delta@v),2,10)
                if next_damping==0:
                    next_cutoff=1/np.linalg.norm(np.linalg.inv(A),np.inf);next_damping=next_cutoff;factor/=2
                next_damping*=factor
            elif ratio>.75:
                next_damping/=2
                if 0<next_damping<cutoff:next_damping=0
        if next_ssr<ssr:
            profile_size=abs(res[:,0]).max();parameters=parameters+delta;history.append(parameters.copy())
            if abs(delta/parameters).max()<1e-6:reason|=1
            if abs(v).max()<1e-6:reason|=2
        else:next_ssr=ssr
        damping,cutoff,ssr=next_damping,next_cutoff,next_ssr
        objective=ssr/(len(points)-4);objectives.append(objective)
        if objective<1e-4:reason|=4
        if step>=10:reason|=8
        if reason:break
    if reason&8:raise RuntimeError('Native fitting step limit reached')
    profile=silhouette(parameters,profile_size);profile[:,1]*=-1
    public=parameters.copy();public[[1,4]]*=-1
    if public[2]<=0 or public[3]<=0 or not np.isfinite(public).all():raise ValueError('Nonphysical capillary fit')
    return dict(parameters=public,reconstruction=profile,capillary_length=np.array([public[2]/np.sqrt(public[3])]),history=np.array(history),objectives=np.array(objectives),profile_size=profile_size,reason=reason,warnings=warnings)
