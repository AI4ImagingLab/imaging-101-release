"""Explicit cylindrical-probe current and distribution mathematics."""
import numpy as np

ELEMENTARY_CHARGE=1.602176634e-19
ELECTRON_MASS=9.1093837139e-31
# Astropy/PlasmaPy's He-4+ mass: neutral isotopic mass minus one electron.
HELIUM_ION_MASS=6.645568142497802e-27

def linear_inverse_temperature(voltage,x0,y0,temperature):
    """Log current, natural log of amperes, with temperature in eV."""
    return y0+(voltage-x0)/temperature

def ion_current(voltage,fit):
    """OML square-current fit uses mA squared; returned current is A."""
    return -1e-3*np.sqrt(np.clip(fit[0]*voltage+fit[1],0,None))

def electron_current(voltage,fit):
    """Retarding-region extrapolation in A, before the display-only mask."""
    return np.exp(linear_inverse_temperature(voltage,*fit))

def distribution_current(voltage,energy,probability,plasma_potential,density,area):
    """Isotropic electron collection integral, for forward physics checks.

    Energy is ascending eV; probability is normalized per eV, voltage V,
    density /m^3, and area m^2. Positive barriers remove lower energies.
    This helper evaluates the supplied finite spectral support only and adds
    no unobserved tail. It is not used to invent observations or references.
    """
    energy=np.asarray(energy,dtype=float);p=np.asarray(probability,dtype=float)
    if energy.ndim!=1 or p.shape!=energy.shape or np.any(energy<=0) or np.any(np.diff(energy)<=0):
        raise ValueError('Expected positive ascending energy and matching density')
    barrier=np.maximum(plasma_potential-np.asarray(voltage,dtype=float),0)
    kernel=np.maximum(energy[None,:]-barrier[:,None],0)/np.sqrt(energy[None,:])
    flux=np.trapezoid(kernel*p[None,:],x=energy,axis=1)
    return area*ELEMENTARY_CHARGE*density*np.sqrt(2*ELEMENTARY_CHARGE/ELECTRON_MASS)*flux/4

def normalized_eedf(bias,current,plasma_potential,floating_potential):
    """Native Druyvesteyn estimate, descending energy in eV and density/eV.

    Differentiate on the entire native voltage grid twice before selecting
    the strict floating-to-plasma interval. Preserve signed derivative values.
    """
    first=np.gradient(current,bias)
    second=np.gradient(first,bias)
    mask=(bias>floating_potential)&(bias<plasma_potential)
    energy=plasma_potential-bias[mask]
    p=second[mask]*np.sqrt(energy)
    scale=abs(np.trapezoid(p,x=energy))
    if scale<=0 or not np.isfinite(scale):raise ValueError('Degenerate EEDF normalization')
    return energy,p/scale,first,second

def densities(ion_saturation,electron_saturation,temperature,ion_slope,area,ion_mass=HELIUM_ION_MASS):
    """Return electron LM, ion Bohm/LM and cylindrical OML densities /m^3."""
    kinetic=temperature*ELEMENTARY_CHARGE
    ne=4*electron_saturation/(area*ELEMENTARY_CHARGE*np.sqrt(8*kinetic/(np.pi*ELECTRON_MASS)))
    ni=abs(ion_saturation)/(.6*ELEMENTARY_CHARGE*area*np.sqrt(kinetic/ion_mass))
    ni_oml=np.sqrt(-ion_slope*1e-6*np.pi**2*ion_mass/(area**2*ELEMENTARY_CHARGE**3*2))
    return ne,ni,ni_oml
