"""Local intensity-space radiative transfer and asymmetric spectrometer response.
Adapted from iFit by Ben Esse, MIT; see LICENSE and private provenance.
"""
import numpy as np
from scipy.special import gamma
from scipy.interpolate import griddata
MODEL_SPACING=.01
def instrument_lineshape(fwem,k,aw,ak):
    """Return400 normalized samples on[-2,2)nm. FWEM is twice the width w."""
    x=np.arange(-2,2,MODEL_SPACING);w=.5*fwem
    a=k/(2*w*gamma(1/k))
    g=np.array([np.exp(-np.power(np.abs(v/(w-aw)),k-ak)) if v<=0 else np.exp(-np.power(np.abs(v/(w+aw)),k+ak)) for v in x])
    g=g*a
    return np.divide(g,sum(g))
def optical_depth(parameters,cross_sections):
    """Return each absorber optical depth (3,N),SO2/O3/Ring order."""
    return np.array([np.multiply(cross_sections[i],parameters[i]) for i in range(3)])
def forward(wavelength,parameters,model_wavelength,solar,cross_sections):
    """Beer-Lambert product, one ILS convolution, shift/stretch and cubic sampling.
    Parameter order: SO2,O3,Ring,bg[degree3..0],shift[linear,constant],FWEM,k,aw,ak.
    """
    p=parameters;grid=model_wavelength
    frs=np.multiply(solar,np.polyval(p[3:7],grid))
    plume=np.zeros((3,len(grid)));sky=np.zeros_like(plume)
    plume[0]=np.multiply(cross_sections[0],p[0])
    sky[1]=np.multiply(cross_sections[1],p[1]);sky[2]=np.multiply(cross_sections[2],p[2])
    total=np.sum(np.vstack([plume,sky]),axis=0)
    raw=np.multiply(frs,np.exp(-total))
    convolved=np.convolve(raw,instrument_lineshape(*p[9:13]),'same')
    shift=np.polyval(p[7:9],grid-min(grid))
    return griddata(np.add(grid,shift),convolved,wavelength,method='cubic')
