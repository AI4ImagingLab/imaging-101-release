import numpy as np
from scipy.special import erf
BIN_PS=50.
ROI_START=33
ROI_STOP=948

def prediction_native(parameters,area):
    p=np.asarray(parameters,dtype='float64');x=np.arange(1000,dtype='float64')-ROI_START;result=np.zeros(1000)
    for gi in [6,9]:
        scale=p[gi]/(2*np.sqrt(np.log(2.)));mean=p[gi+1];irf_weight=p[gi+2];component=np.zeros(1000)
        for li in [0,2,4]:
            tau=p[li];weight=p[li+1]
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                a=np.exp(-(x-mean-scale*scale/(4*tau))/tau)*(1-erf(.5*scale/tau-(x-mean)/scale))
                b=np.exp(-(x+1-mean-scale*scale/(4*tau))/tau)*(1-erf(.5*scale/tau-(x+1-mean)/scale))
                component+=.5*weight*(a-b-erf((x-mean)/scale)+erf((x+1-mean)/scale))
        result+=irf_weight*component
    return (area-916*p[12])*result+p[12]

def residual(parameters,observed,area):
    expected=prediction_native(parameters,area)
    # Original callback overwrites its last data residual with the Gaussian normalization term.
    return np.r_[(observed[33:947]-expected[33:947])/np.sqrt(observed[33:947]+1.),1e4*(parameters[8]+parameters[11]-1.)]

def package(parameters,predicted):
    p=np.asarray(parameters);li=np.argsort(p[:,[0,2,4]],axis=1);gi=np.argsort(p[:,[6,9]],axis=1)
    def take(cols,idx):return np.take_along_axis(p[:,cols],idx,axis=1)
    return dict(reconstruction=take([0,2,4],li)[None]*50,intensity=take([1,3,5],li)[None],irf_fwhm_ps=take([6,9],gi)[None]*50,irf_shift_ps=take([7,10],gi)[None]*50,irf_weight=take([8,11],gi)[None],background=p[:,12][None],prediction=np.asarray(predicted)[None])
