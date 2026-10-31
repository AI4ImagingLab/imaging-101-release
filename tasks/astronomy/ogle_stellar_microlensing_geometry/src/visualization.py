"""Scientific comparisons using computed results."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import predict
from .preprocessing import magnitude_from_flux
def comparison_figure(data,result,details,upstream):
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    t=data['time'];grid=np.linspace(t.min(),t.max(),2400)
    p=result['reconstruction'][0];f=result['source_blend_flux'][0]
    ax[0,0].errorbar(t,data['magnitude'],data['magnitude_error'],fmt='.',ms=2,alpha=.5,label='Measured OGLE-IV')
    ax[0,0].plot(grid,magnitude_from_flux(predict(grid,p,f)),label='Fresh fit');ax[0,0].invert_yaxis();ax[0,0].set(ylabel='I magnitude',xlabel='HJD - 2450000 [day]',title='Complete measured light curve');ax[0,0].legend()
    residual=(details['flux']-details['model'])/details['error']
    ax[0,1].scatter(t,residual,s=8);ax[0,1].axhline(0,color='k',lw=.7);ax[0,1].set(xlabel='HJD - 2450000 [day]',ylabel='Residual / supplied sigma',title='All 383 residuals')
    ax[1,0].hist(residual,bins=25,density=True,alpha=.7,label='Measured residuals');x=np.linspace(-4,4,200);ax[1,0].plot(x,np.exp(-x*x/2)/np.sqrt(2*np.pi),label='Unit normal guide');ax[1,0].set(xlabel='Residual / sigma',ylabel='Density',title='Noise-model diagnostic');ax[1,0].legend()
    ax[1,1].plot(t,details['model']-upstream['model_flux'],'.');ax[1,1].set(xlabel='HJD - 2450000 [day]',ylabel='Fresh - original flux [m=22 units]',title='Independent original-code comparison')
    return fig
