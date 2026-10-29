"""Native OML, two-stage log-current fit, and Druyvesteyn inversion.

Adapted from PlasmaPy diagnostics.langmuir (BSD; see LICENSE). General SciPy
curve_fit preserves its original inverse-temperature parameterization.
"""
import numpy as np
from scipy.optimize import curve_fit
from .preprocessing import unique_characteristic,PROBE_AREA_M2
from .physics_model import linear_inverse_temperature,ion_current,electron_current,normalized_eedf,densities

def fit_temperature(bias,electron):
    """Fit log positive current using native unconstrained LM defaults."""
    use=electron>0
    fit,cov=curve_fit(linear_inverse_temperature,bias[use],np.log(electron[use]),p0=None,bounds=(-np.inf,np.inf))
    return fit,cov

def reconstruct(bias,current,area=PROBE_AREA_M2):
    """Return declared arrays plus separately identified stage diagnostics.

    All voltages are V, current A, temperature eV, densities m^-3, area m^2.
    No hidden assets are read. Windows are selected from the observed sweep.
    """
    v,i=unique_characteristic(bias,current)
    gradient=np.gradient(i,v)
    ip=int(np.argmax(gradient)); iff=int(np.argmin(abs(i)))
    vp=float(v[ip]);vf=float(v[iff]);ies=float(i[ip]);iis=float(np.min(i))
    ionmask=v<vf
    ionfit=np.polyfit(v[ionmask],(i[ionmask]*1e3)**2,1)
    ions=ion_current(v,ionfit)
    firstmask=(v>vf)&(v<vp)
    firstfit,firstcov=fit_temperature(v[firstmask],i[firstmask]-ions[firstmask])
    temp1=float(firstfit[2])
    secondmask=(v>vf+1.5*temp1)&(v<vp-.2*temp1)
    fit,cov=fit_temperature(v[secondmask],i[secondmask]-ions[secondmask])
    temperature=float(fit[2])
    ne,ni,noml=densities(iis,ies,temperature,ionfit[0],area)
    energy,p,deriv,deriv2=normalized_eedf(v,i,vp,vf)
    values=dict(V_P=vp,V_F=vf,I_es=ies,I_is=iis,n_e=ne,n_i=ni,T_e=temperature,n_i_OML=noml)
    output={k:np.array([[x]],dtype=np.float64) for k,x in values.items()}
    output.update(reconstruction=p[None],energy=energy[None])
    diagnostics=dict(bias=v,current=i,ion_fit=ionfit,ion_current=ions,electron_fit=fit,electron_current=electron_current(v,fit),first_fit=firstfit,first_temperature=temp1,first_mask=firstmask,second_mask=secondmask,gradient=deriv,second_derivative=deriv2,covariance=cov)
    return output,diagnostics
