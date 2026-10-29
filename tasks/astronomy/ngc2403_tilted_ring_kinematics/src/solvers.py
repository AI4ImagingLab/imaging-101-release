"""Explicit published initialization and inverse updates.

Extracted selected BBarolo GPL-2.0-or-later mathematics. No upstream imports.
"""
import math
import numpy as np
from .preprocessing import PIXEL_ARCSEC,search_mask,first_moment
from .physics_model import internal_rings, cloud_cube, beam_kernel, beam_convolve, azimuthal_loss, CloudRandom


def projected_velocity(coordinates, parameters):
    """Source float32 LOS model; p=(Vsys,Vrot,Vrad,PA,inc,x0,y0)."""
    p=np.asarray(parameters,dtype=np.float32)
    x,y=(np.asarray(coordinates,dtype=np.float32)-p[5:7]).T
    phi,inc=np.deg2rad(p[[3,4]].astype(float))
    xp=(-x.astype(float)*np.sin(phi)+y.astype(float)*np.cos(phi)).astype(np.float32)
    yp=((-x.astype(float)*np.cos(phi)-y.astype(float)*np.sin(phi))/np.cos(inc)).astype(np.float32)
    radius=np.sqrt(xp*xp+yp*yp,dtype=np.float32)
    cos_theta=xp/radius;sin_theta=yp/radius
    return (float(p[0])+(p[1]*cos_theta+p[2]*sin_theta).astype(float)*np.sin(inc)).astype(np.float32)


def prepass_derivatives(coordinates, parameters):
    """Analytic source partials for its two free prepass variables Vrot and PA."""
    p=np.asarray(parameters,dtype=np.float32)
    phi,inc=np.deg2rad(p[[3,4]].astype(float))
    cosphi=np.float32(np.cos(phi));sinphi=np.float32(np.sin(phi))
    cosinc=np.float32(np.cos(inc));sininc=np.float32(np.sin(inc))
    x,y=(np.asarray(coordinates,dtype=np.float32)-p[5:7]).T
    xp=-x*sinphi+y*cosphi;yp=(-x*cosphi-y*sinphi)/cosinc
    radius=np.sqrt(xp*xp+yp*yp,dtype=np.float32)
    c=xp/radius;s=yp/radius;c2=c*c;s2=s*s;si2=sininc*sininc
    dv=(sininc*c).astype(np.float32)
    dp=(np.pi/180*float(p[1])*(1-(si2*s2).astype(float))*s.astype(float)*float(sininc)/float(cosinc)
       -np.pi/180*float(p[2])*(1-(si2*c2).astype(float))*c.astype(float)*float(sininc)/float(cosinc)).astype(np.float32)
    return np.stack((dv,dp),axis=1)


def prepass_normal_equations(coordinates, velocity, weights, parameters):
    """Unnormalised weighted normal equations and squared residual, float64."""
    residual=(np.asarray(velocity,dtype=np.float32)-projected_velocity(coordinates,parameters)).astype(float)
    jac=prepass_derivatives(coordinates,parameters).astype(float)
    w=np.asarray(weights,dtype=float)
    vector=np.array([np.add.accumulate(residual*jac[:,j]*w)[-1] for j in range(2)])
    matrix=np.empty((2,2))
    for j in range(2):
        for i in range(j+1):
            matrix[j,i]=matrix[i,j]=np.add.accumulate(jac[:,i]*(jac[:,j]*w))[-1]
    chi=np.add.accumulate(residual*residual*w)[-1]
    return float(chi),matrix,vector


def _gauss_jordan_inverse(matrix):
    """Source scaled-normal-equation Gauss-Jordan pivot convention."""
    a=np.array(matrix,dtype=float,copy=True);n=a.shape[0];per=np.arange(n)
    for j in range(n):
        row=j+int(np.argmax(abs(a[j:,j])))
        if a[row,j]==0:raise np.linalg.LinAlgError('singular prepass system')
        if row>j:
            a[[j,row]]=a[[row,j]];per[[j,row]]=per[[row,j]]
        even=1/a[j,j]
        a[:,j]*=even;a[j,j]=even
        for k in list(range(j))+list(range(j+1,n)):
            mjk=a[j,k]
            for i in list(range(j))+list(range(j+1,n)):
                a[i,k]-=a[i,j]*mjk
            a[j,k]=-even*mjk
    out=a.copy()
    for k in range(n):out[:,per[k]]=a[:,k]
    return out


def _damped_prepass_fit(coordinates, velocity, weights, initial, tolerance=.001):
    """Original scaled LM inner fit: decrease lambda /10, reject by *10."""
    p=np.array(initial,dtype=np.float32,copy=True);lam=.001*10
    for iteration in range(1,101):
        chi,matrix,vector=prepass_normal_equations(coordinates,velocity,weights,p)
        diagonal=np.sqrt(np.diag(matrix))
        if np.any(diagonal<=0):raise ValueError('degenerate velocity-field initialization')
        if lam>1e-10:lam/=10
        while True:
            normal=matrix/diagonal[:,None]/diagonal[None,:]
            normal[np.diag_indices(2)]=1+lam
            inv=_gauss_jordan_inverse(normal)
            trial=p.copy()
            for j,index in enumerate([1,3]):
                change=0.
                for i in range(2):change+=vector[i]*inv[j,i]/diagonal[j]/diagonal[i]
                trial[index]=np.float32(float(p[index])+change)
            residual=(np.asarray(velocity,dtype=float)-projected_velocity(coordinates,trial).astype(float))
            trial_chi=float(np.add.accumulate(weights.astype(float)*residual*residual)[-1])
            if trial_chi<chi or lam>1e10:break
            lam*=10
        if lam<=1e10:p=trial
        if abs(chi-trial_chi)<=tolerance*trial_chi or lam>1e10:
            return p,iteration
    raise RuntimeError('Original prepass LM exhausted 100 iterations')


def _select_prepass_data(moment, parameters, inner, outer):
    """Changing projected annulus, minor-axis exclusion and cos² weighting."""
    p=np.asarray(parameters,dtype=np.float32)
    coordinates=np.indices(moment.shape).transpose(1,2,0)[...,::-1].reshape(-1,2).astype(np.float32)
    # Native traversal is x-major then y, not the array's natural row order.
    coordinates=coordinates[np.lexsort((coordinates[:,1],coordinates[:,0]))]
    x,y=(coordinates-p[5:7]).T
    phi,inc=np.deg2rad(p[[3,4]].astype(float))
    sinp=np.float32(np.sin(phi));cosp=np.float32(np.cos(phi));cosi=np.float32(np.cos(inc))
    xp=-x*sinp+y*cosp;yp=(-x*cosp-y*sinp)/cosi
    radius=np.sqrt(xp*xp+yp*yp,dtype=np.float32)
    theta=np.where(radius>=.1,np.arctan2(yp.astype(float),xp.astype(float))/(np.pi/180),0).astype(np.float32)
    cosine=np.abs(np.cos(theta.astype(float)*np.pi/180)).astype(np.float32)
    values=moment[coordinates[:,1].astype(int),coordinates[:,0].astype(int)]
    use=np.isfinite(values)&(radius>inner)&(radius<outer)&(cosine>np.float32(np.sin(np.pi/12)))
    coordinates=coordinates[use];values=values[use]
    weights=np.power(cosine[use].astype(float),2).astype(np.float32)
    if values.size<=2:
        return coordinates,values,weights,np.float32(np.finfo(np.float32).max)
    residual=values-projected_velocity(coordinates,p)
    total=np.add.accumulate((residual*residual*weights).astype(np.float32),dtype=np.float32)[-1]
    q=np.float32(np.sqrt(np.float32(total/np.float32(values.size-2))))
    return coordinates,values,weights,q


def initialize_rings(moment):
    """Original full 40-ring 2D prepass; minimally qualified 41st copy bound.

    Returns initial (41,3) [rotation,dispersion,PA], plus all fitted 2D states.
    The unmatched last ring keeps the actual user-supplied initialization.
    """
    radii=(np.arange(15,1216,30,dtype=np.float32)/PIXEL_ARCSEC).astype(np.float32)
    width=np.float32(30/PIXEL_ARCSEC)
    states=[];counts=[];qualities=[];statuses=[]
    for radius in radii[:-1]:
        p=np.array([132.8,120,0,123.7,60,77,77],dtype=np.float32)
        inner=np.float32(max(0,float(radius)-.5*float(width)))
        outer=np.float32(float(radius)+.5*float(width))
        x,y,w,q=_select_prepass_data(moment,p,inner,outer)
        stopped=False
        for h in range(1,101):
            chi=float(q)
            fitted,_=_damped_prepass_fit(x,y,w,p)
            delta=(fitted-p).astype(np.float32)
            flip=np.float32(1)
            for _ in range(150):
                trial=(flip*delta+p).astype(np.float32)
                x,y,w,newq=_select_prepass_data(moment,trial,inner,outer)
                q=newq
                if q<chi:
                    p=trial;break
                if 2*h>100:
                    stopped=bool(np.all(abs(flip*delta)<np.float32(.1)))
                else:
                    stopped=(q==chi==0) or abs(float(q)-chi)/chi<.001
                if stopped:
                    q=np.float32(chi);break
                flip=np.float32(flip*(-1 if flip>0 else -.5))
            else:raise RuntimeError('Prepass outer backtracking did not stop')
            if stopped:break
        else:raise RuntimeError('Original prepass outer loop exhausted 100 iterations')
        states.append(p.copy());counts.append(len(y));qualities.append(q);statuses.append(h)
    states=np.array(states,dtype=np.float32)
    initial=np.tile(np.array([120,8,123.7],dtype=np.float32),(41,1))
    initial[1:40,0]=states[1:,1];initial[1:40,2]=states[1:,3]
    initial[0,2]=initial[1,2]
    return initial,dict(parameters=states,counts=np.array(counts),q=np.array(qualities),status=np.array(statuses))


def ring_objective(cube,mask,ring_index,initial,noise,seed=-1,free_angle=True,
                   repair_random=None,global_rotation_max=1120.,global_angle_limits=(-360.,360.),smooth=True):
    """Selected bounded/repairing annular Monte-Carlo AZIM objective closure.

    The caller supplies legitimately observation-derived initialization/noise.
    Trial arrays are mutated by the actual source bound rules before scoring.
    """
    center=15+30*ring_index
    edges=np.array([max(center-15,0),center+15],dtype=np.float32)
    initial=np.asarray(initial,dtype=np.float32)
    random=CloudRandom(-1) if repair_random is None else repair_random
    kernel=beam_kernel()
    lower_rotation=max(float(np.float32(initial[0]-np.float32(1000))),0)
    upper_rotation=min(float(np.float32(initial[0]+np.float32(1000))),global_rotation_max)
    lower_angle=max(float(np.float32(initial[2]-np.float32(15))),global_angle_limits[0])
    upper_angle=min(float(np.float32(initial[2]+np.float32(15))),global_angle_limits[1])
    def objective(trial):
        if trial[0]>upper_rotation:
            trial[0]=np.float32(upper_rotation-float(random.canonical()[0])*1000)
        elif trial[0]<lower_rotation:
            trial[0]=np.float32(lower_rotation+float(random.canonical()[0])*1000)
        if trial[1]>1000:
            trial[1]=np.float32(1000-float(np.float32(random.canonical()[0]*initial[1])))
        elif trial[1]<np.float32(.1):
            trial[1]=np.float32(float(np.float32(.1))+float(np.float32(random.canonical()[0]*initial[1])))
        if free_angle:trial[2]=np.float32(np.clip(trial[2],lower_angle,upper_angle))
        parameters=trial if free_angle else np.r_[trial,initial[2]].astype(np.float32)
        rings,meta,geometry=internal_rings(edges,parameters)
        model=cloud_cube(rings,meta,seed=seed)
        if smooth:model=beam_convolve(model,kernel)
        nx,ny,nz,bx,by=map(int,meta[[1,2,3,4,5]])
        obs=cube[:,by:by+ny,bx:bx+nx];valid=mask[:,by:by+ny,bx:bx+nx]
        loss_meta=np.array([nx,ny,nz,bx,by,2,noise,2,1,2])
        _,summary=azimuthal_loss(model,obs,valid,geometry,loss_meta)
        return float(summary[-1])
    return objective


def nelder_mead(objective,initial,steps,tolerance=.001,trace=None,diagnostic_iterations=None):
    """Original float32 downhill simplex, including its tie and count rules.

    diagnostic_iterations only truncates an explicitly labelled stage test;
    it is never the standard reconstruction stopping rule.
    """
    initial=np.asarray(initial,dtype=np.float32);steps=np.asarray(steps,dtype=np.float32)
    ndim=initial.size;mpts=ndim+1
    points=np.repeat(initial[None],mpts,axis=0)
    for i in range(1,mpts):points[i,i-1]+=steps[i-1]
    values=np.array([objective(point.copy()) for point in points],dtype=np.float32)
    psum=np.add.accumulate(points,axis=0,dtype=np.float32)[-1]
    evaluations=0;iteration=0
    def attempt(hi,factor):
        fac1=(1-factor)/ndim;fac2=fac1-factor
        trial=(psum.astype(float)*fac1-points[hi].astype(float)*fac2).astype(np.float32)
        value=objective(trial)
        if value<float(values[hi]):
            values[hi]=np.float32(value)
            psum[:]=(psum+(trial-points[hi])).astype(np.float32)
            points[hi]=trial
        return np.float32(value)
    while True:
        if trace is not None:trace.append(dict(points=points.copy(),values=values.copy(),sum=psum.copy(),evaluations=evaluations))
        lo=0
        if values[0]>values[1]:hi,second=0,1
        else:hi,second=1,0
        for i in range(mpts):
            if values[i]<=values[lo]:lo=i
            if values[i]>values[hi]:second,hi=hi,i
            elif values[i]>values[second] and i!=hi:second=i
        rtol=2*abs(float(np.float32(values[hi]-values[lo])))/(abs(float(values[hi]))+abs(float(values[lo]))+1e-10)
        if rtol<tolerance:
            return dict(parameters=points[lo].copy(),loss=float(values[lo]),converged=True,evaluations=evaluations,iterations=iteration)
        if evaluations>=200*ndim or (diagnostic_iterations is not None and iteration>=diagnostic_iterations):
            return dict(parameters=points[lo].copy(),loss=float(values[lo]),converged=False,evaluations=evaluations,iterations=iteration)
        evaluations+=2;iteration+=1
        trial_value=attempt(hi,-1.)
        if trial_value<=values[lo]:attempt(hi,2.)
        elif trial_value>=values[second]:
            saved=values[hi]
            trial_value=attempt(hi,.5)
            if trial_value>=saved:
                for i in range(mpts):
                    if i!=lo:
                        points[i]=(np.float32(.5)*(points[i]+points[lo])).astype(np.float32)
                        psum[:]=points[i]
                        values[i]=np.float32(objective(psum))
                evaluations+=ndim
                psum[:]=np.add.accumulate(points,axis=0,dtype=np.float32)[-1]
        else:evaluations-=1


def reconstruct(cube,seed=-1,smooth=True,progress=None):
    """Full native41-ring selected two-stage reconstruction from observations.

    Returns (41,3) float32 [Vrot km/s,dispersion km/s,PA deg], and computed
    diagnostics. No field/reference arrays are read. PA regularization follows
    the selected source's observed low-scatter auto-median branch. An unexpected
    >3-degree scatter raises rather than silently substituting another method.
    """
    cube=np.asarray(cube,dtype=np.float32)
    if cube.shape!=(62,150,150) or not np.isfinite(cube).all():
        raise ValueError('Expected the complete finite62x150x150 native cube')
    mask,statistics=search_mask(cube)
    moment=first_moment(cube,mask)
    initial,prepass=initialize_rings(moment)
    rotation_max=float(np.float32(np.max(initial[:,0])+np.float32(1000)))
    angle_limits=(max(float(np.float32(initial[:,2].min()-15)),-360.),min(float(np.float32(initial[:,2].max()+15)),360.))
    repair_random=CloudRandom(-1)
    work=cube.copy();noise=np.float32(statistics['spread'])
    scaling=10/float(cube.max())
    all_states=[];all_losses=[];all_counts=[];all_iterations=[]
    current=initial.copy()
    max_rotation=max(abs((289.610969-61*5.15286)-132.8),abs(289.610969-132.8))/np.sin(np.pi/3)
    for stage in [1,2]:
        if stage==2:
            center=np.float32(np.median(current[:,2]))
            mad=np.float32(np.median(abs(current[:,2]-center)))
            scatter=np.float32(mad/np.float32(.6744888))
            if scatter>3:
                raise RuntimeError('Unexpected PA scatter selects unbound Bezier branch; do not silently replace it')
            current[:,2]=center
        work=(work.astype(float)*scaling).astype(np.float32)
        noise=np.float32(float(noise)*scaling)
        fitted=current.copy();losses=[];counts=[];iterations=[]
        for ir in range(41):
            point=current[ir].copy()
            free_angle=stage==1
            selected=point.copy() if free_angle else point[:2].copy()
            steps=[30 if point[0]>40 else 5,20 if point[1]>80 else 10 if point[1]>20 else .2 if point[1]<=2 else 5]
            if free_angle:steps.append(10)
            objective=ring_objective(work,mask,ir,point,noise,seed,free_angle,repair_random,rotation_max,angle_limits,smooth)
            result=nelder_mead(objective,selected,np.array(steps,dtype=np.float32))
            if not result['converged']:
                raise RuntimeError(f'Stage{stage} ring{ir+1} exhausted original simplex budget; no output masking')
            fitted[ir,:len(selected)]=result['parameters']
            if fitted[ir,0]>max_rotation:
                raise RuntimeError(f'Stage{stage} ring{ir+1} exceeds native spectral support; no output masking')
            losses.append(result['loss']);counts.append(result['evaluations']);iterations.append(result['iterations'])
            if progress is not None:progress(stage,ir+1,fitted[ir].copy(),result)
        current=fitted
        all_states.append(current.copy());all_losses.append(losses);all_counts.append(counts);all_iterations.append(iterations)
        work=(work.astype(float)/scaling).astype(np.float32)
        noise=np.float32(float(noise)/scaling)
    diagnostics=dict(initial=initial,prepass_parameters=prepass['parameters'],prepass_counts=prepass['counts'],prepass_q=prepass['q'],
                     mask=mask,moment=moment,stage_fields=np.array(all_states),stage_losses=np.array(all_losses),
                     stage_evaluations=np.array(all_counts),stage_iterations=np.array(all_iterations),
                     pa_scatter_deg=float(scatter),noise_Jy_beam=statistics['spread'])
    return current,diagnostics
