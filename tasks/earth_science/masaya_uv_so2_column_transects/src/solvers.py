"""Published independent LM spectral fits; no upstream package dependency."""
import numpy as np
from scipy.optimize import curve_fit
from .physics_model import forward
from .preprocessing import preprocess
INITIAL=np.array([1e16,1e19,.1,0.,0.,0.,1.,0.,-.1,.6,2.,0.,0.])
def fit_spectrum(wavelength,intensity,model_wavelength,solar,cross_sections,initial=None):
    """Return13 parameters,covariance,modeled spectrum using original curve_fit defaults."""
    f=lambda x,*p:forward(x,p,model_wavelength,solar,cross_sections)
    p,cov=curve_fit(f,wavelength,intensity,INITIAL if initial is None else initial)
    return p,cov,f(wavelength,*p)
def reconstruct(data):
    """Fit every measured row. Return SO2(360,),details dict with fresh spectra/parameters."""
    x,y=preprocess(data['wavelength'],data['spectra'],data['dark'])
    ps=[];fits=[];errors=[]
    for row in y:
        p,cov,f=fit_spectrum(x,row,data['model_wavelength'],data['solar'],data['cross_sections'])
        ps.append(p);fits.append(f);errors.append(np.sqrt(np.diag(cov)))
    parameters=np.array(ps)
    return parameters[:,0],{'parameters':parameters,'fit':np.array(fits),'parameter_errors':np.array(errors),'wavelength':x,'corrected':y}

def doas_comparison(data,details):
    """Fresh fixed-registration,I0-corrected linear DOAS diagnostic (not QDOAS).
    Uses first20measured spectra as FRS and mean fitted ILS/registration.
    Relative columns differ from the absolute iFit output if FRS contains SO2.
    """
    from scipy.interpolate import griddata
    from .physics_model import instrument_lineshape
    p=details['parameters'].mean(axis=0);g=instrument_lineshape(*p[9:13]);grid=data['model_wavelength'];solar=data['solar']
    smooth=np.convolve(solar,g,'same');columns=[]
    scales=np.array([1e18,1e19,.1])
    shifted=grid+np.polyval(p[7:9],grid-min(grid))
    for sigma,scale in zip(data['cross_sections'],scales):
        depth=-np.log(np.convolve(solar*np.exp(-sigma*scale),g,'same')/smooth)
        columns.append(griddata(shifted,depth,details['wavelength'],method='cubic'))
    x=(details['wavelength']-315)/5
    design=np.array(columns+[x**3,x**2,x,np.ones_like(x)]).T
    frs=details['corrected'][:20].mean(axis=0)
    measured=-np.log(details['corrected']/frs)
    coefficients=np.linalg.lstsq(design,measured.T,rcond=None)[0]
    return coefficients[0]*scales[0],measured,(design@coefficients).T
