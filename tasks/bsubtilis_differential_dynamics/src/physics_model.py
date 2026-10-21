"""Explicit DDM relaxation model and its analytic physical derivatives."""
import numpy as np

def structure_model(tau,parameters):
    """p=(A ADU²,B ADU²,Gamma s^-1,beta dimensionless), tau seconds."""
    A,B,Gamma,beta=parameters
    return A*(1-np.exp(-((tau*Gamma)**beta)))+B

def model_jacobian(tau,parameters):
    A,B,Gamma,beta=parameters
    u=(tau*Gamma)**beta;e=np.exp(-u)
    return np.column_stack((1-e,np.ones_like(tau),A*e*beta*u/Gamma,A*e*u*np.log(tau*Gamma)))

def intermediate_scattering(tau,Gamma,beta):
    """Normalized effective intermediate scattering f=exp[-(Gamma*tau)^beta]."""
    return np.exp(-((tau*Gamma)**beta))


def isotropic_speed_density(relative_speed,beta=1.35):
    """3-D isotropic characteristic-model density, not observed trajectories.
    P(n)=2n/pi integral_0^infinity exp(-t^beta)t sin(t n)dt.
    n=v/v0 is dimensionless. Mirrors the tutorial scientific interpretation.
    """
    from scipy.integrate import quad
    return np.array([2*n/np.pi*quad(lambda t:np.exp(-t**beta)*t*np.sin(t*n),0,np.inf,epsabs=1.49e-8,epsrel=1.49e-8)[0] for n in np.asarray(relative_speed)])
