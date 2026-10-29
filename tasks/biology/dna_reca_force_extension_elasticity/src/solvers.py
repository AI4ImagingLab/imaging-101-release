"""Two-condition shared elasticity fit, preserving the author tutorial."""
import numpy as np
from scipy.optimize import least_squares
from .physics_model import ewlc_marko_siggia_force,ewlc_marko_siggia_force_jac
def residual_and_jac(parameters,data):
    """p=(Lp,Lc_control,St,Lc_RecA); residual measured-minus-model pN."""
    lp,lc,st,lr=parameters;res=[];jac=[]
    for name,contour,column in [('control',lc,1),('reca',lr,3)]:
        mask=data['mask_'+name];x=data['distance_'+name][mask]
        q=ewlc_marko_siggia_force(x,lp,contour,st,4.11)
        derivatives=np.array(ewlc_marko_siggia_force_jac(x,lp,contour,st,4.11))
        block=np.zeros((len(x),4));block[:,0]=-derivatives[0];block[:,column]=-derivatives[1];block[:,2]=-derivatives[2]
        res.append(data['force_'+name][mask]-q);jac.append(block)
    # Original global-Jacobian selection of the four free columns is Fortran
    # contiguous. Preserve this LAPACK arithmetic convention for parity.
    return np.concatenate(res),np.asfortranarray(np.concatenate(jac))
def reconstruct(data):
    """Return full author TRF least-squares result; no regularizer or priors."""
    return least_squares(lambda p:residual_and_jac(p,data)[0],np.array([50.,2.7,1500.,3.]),jac=lambda p:residual_and_jac(p,data)[1],bounds=([39.,.00034,1.,.00034],[80.,np.inf,np.inf,np.inf]),method='trf',ftol=1e-8,xtol=1e-8,gtol=1e-8)
def submission(fit,data):
    p=fit.x;out={'persistence_length':p[:1][None],'stretch_modulus':p[2:3][None],'contour_lengths':p[[1,3]][None]}
    for name,lc in [('control',p[1]),('reca',p[3])]:
        out['predicted_force_'+name]=ewlc_marko_siggia_force(data['distance_'+name][data['mask_'+name]],p[0],lc,p[2],4.11)[None]
    return out
