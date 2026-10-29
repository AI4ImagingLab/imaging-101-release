"""Local joint residual, analytic chain rule and native damped Gauss--Newton."""
import numpy as np
from scipy.linalg import solve
from .physics_model import octantoid,ao_operator,lightcurve_operator,occultation_operator,mesh_penalties

def initial_parameters():
    """311 unknowns:300harmonics,3spin,4AOoffsets,2loggains,2occultation offsets."""
    parameters=np.zeros(311);parameters[0]=np.log(43.56)/.2821
    parameters[200]=np.log(35.64)/.2821-parameters[0]
    parameters[300:303]=[(90-52)*np.pi/180,272*np.pi/180,48*np.pi/8.40060]
    parameters[309:311]=[0.,-10.]
    return parameters

def octantoid_penalty(coefficients):
    """Scalar sum l/lmax*(b_lm²+c_lm²) and its300derivatives."""
    degree=np.concatenate([np.full(2*l+1,l/9) for l in range(10)])
    a=np.asarray(coefficients).reshape(3,100);gradient=np.zeros_like(a)
    gradient[1:]=2*degree*a[1:]
    return float(np.sum(degree*a[1:]**2)),gradient.ravel()

def residual_and_jacobian(parameters,data,derivative=True,weights=(8.,2.,.1,30.,5.,.5,1.)):
    """Complete46595residuals; J=-d(residual)/d(parameters), native row order.

    Weights are residual multipliers LC,AO,OC,convexity,octantoid,dihedral,COM.
    The selected311unknown conditional inverse includes all nuisance fields.
    """
    p=np.asarray(parameters,dtype=np.float64);mesh,shape_jac=octantoid(p[:300],padding=3)
    angles=np.r_[p[300:303],0.];lcw,aow,ocw,cw,ow,dw,comw=weights
    residual=np.zeros(46595);jac=np.zeros((46595,311)) if derivative else None
    for curve in range(30):
        rows=np.flatnonzero(data['lc_curve_index']==curve)
        model,parts=lightcurve_operator(mesh,angles,data['lc_observer'][rows],data['lc_sun'][rows],data['lc_time'][rows],bool(data['lc_relative'][curve]))
        residual[rows]=lcw*(data['lc_brightness'][rows]-model)
        if derivative:
            coordinate=np.concatenate(parts,axis=1);jac[rows,:303]=lcw*(coordinate@shape_jac)
    n=len(data['ao_frequency']);base=923;nt=2*n
    for exposure in range(2):
        model,parts=ao_operator(mesh,angles,data['ao_observer'][exposure],data['ao_sun'][exposure],data['ao_up'][exposure],data['ao_time'][exposure,0],data['ao_distance'][exposure,0],data['ao_frequency'],p[303+2*exposure:305+2*exposure],derivative)
        transfer=np.exp(p[307+exposure])*data['ao_psf_fourier'][exposure]
        prediction=transfer*model;difference=aow*(data['ao_fourier'][exposure]-prediction)
        rows_real=slice(base+exposure*n,base+(exposure+1)*n);rows_imag=slice(base+nt+exposure*n,base+nt+(exposure+1)*n)
        residual[rows_real]=difference.real;residual[rows_imag]=difference.imag
        if derivative:
            # Chain XYZ blocks separately to avoid a giant complex Cartesian matrix.
            real=np.zeros((n,303));imag=np.zeros((n,303))
            for axis in range(3):
                real+=parts[2*axis]@shape_jac[402*axis:402*(axis+1),:]
                imag+=parts[2*axis+1]@shape_jac[402*axis:402*(axis+1),:]
            real[:,300:303]+=parts[6];imag[:,300:303]+=parts[7]
            joint=aow*transfer[:,None]*(real+1j*imag)
            jac[rows_real,:303]=joint.real;jac[rows_imag,:303]=joint.imag
            shift=aow*transfer[:,None]*(parts[8]+1j*parts[9])
            jac[rows_real,303+2*exposure:305+2*exposure]=shift.real;jac[rows_imag,303+2*exposure:305+2*exposure]=shift.imag
            jac[rows_real,307+exposure]=aow*prediction.real;jac[rows_imag,307+exposure]=aow*prediction.imag
        del parts
    start=923+4*n
    values,parts=occultation_operator(mesh,angles,data['oc_up'][0],data['oc_observer'][0],data['oc_velocity'][0],data['oc_time'],p[309:311],data['oc_chords'],data['oc_chord_type'])
    residual[start:start+72]=ocw*values
    if derivative:
        jac[start:start+72,:303]=-ocw*(np.concatenate(parts[:4],axis=1)@shape_jac)
        jac[start:start+72,309:311]=-ocw*parts[4]
    penalty,parts=mesh_penalties(mesh);oreg,doreg=octantoid_penalty(p[:300])
    residual[-4:]=[cw*penalty[0],ow*oreg,dw*penalty[1],comw*penalty[2]]
    if derivative:
        jac[-4,:303]=-cw*(parts[0]@shape_jac)
        jac[-3,:300]=-ow*doreg
        jac[-2,:303]=-dw*(parts[1]@shape_jac)
        jac[-1,:303]=-comw*(parts[2]@shape_jac)
    return residual,jac


def reconstruct(data,max_iterations=100,weights=(8.,2.,.1,30.,5.,.5,1.),progress=None):
    """Native additive LM: diagonal damping, factor10, two accepted decreases<.1.

    Rejected trials keep the same linearization/normal equations; accepted
    trials recompute all facets, visibility, observations and derivatives.
    This is the selected custom update, not a generic substitute optimizer.
    """
    parameters=initial_parameters();damping=.1;accepted=True
    decrease=previous_decrease=10.;history=[];start=None
    for iteration in range(max_iterations):
        if accepted:
            residual,jacobian=residual_and_jacobian(parameters,data,True,weights)
            objective=float(residual@residual);normal=jacobian.T@jacobian;rhs=jacobian.T@residual
            del jacobian
        damped=normal.copy();damped.flat[::312]+=damping*normal.diagonal()
        update=solve(damped,rhs,assume_a='sym',lower=True,check_finite=True)
        trial=parameters+update
        trial_residual,_=residual_and_jacobian(trial,data,False,weights)
        trial_objective=float(trial_residual@trial_residual)
        accepted=trial_objective<objective
        record=dict(iteration=iteration,objective=objective,trial_objective=trial_objective,damping=damping,accepted=accepted,step_norm=float(np.linalg.norm(update)))
        history.append(record)
        if progress is not None:progress(record)
        if accepted:
            previous_decrease=decrease;decrease=objective-trial_objective
            parameters=trial;damping/=10
        else:damping*=10
        if previous_decrease<.1 and decrease<.1:break
        if damping>1e6:break
    mesh,_=octantoid(parameters[:300])
    output=dict(reconstruction=mesh[None],spin=np.array([[90-np.rad2deg(parameters[300]),np.rad2deg(parameters[301])%360,48*np.pi/parameters[302]]]),ao_offsets=parameters[303:307].reshape(1,2,2),ao_log_gains=parameters[307:309].reshape(1,2),occultation_offset=parameters[309:311].reshape(1,1,2))
    return output,dict(parameters=parameters,history=history)
