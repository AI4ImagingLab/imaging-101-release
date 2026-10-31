import numpy as np
import matplotlib.pyplot as plt
from .physics_model import R,RSTEP,INITIAL,model_grid,scattering_splines,forward,transform
def comparisons(data,fresh,upstream):
    k=model_grid(data);chi=np.interp(k,data['k'],data['chi']);sp=scattering_splines(data)
    pred=forward(fresh['parameters'][0],k,sp);r=RSTEP*np.arange(1024)
    fig,ax=plt.subplots(1,3,figsize=(16,4.4),layout='constrained')
    ax[0].plot(k,k*k*chi,label='Measured Cu foil, 10 K',lw=.8)
    ax[0].plot(k,k*k*upstream['prediction'][0],'--',label='Independent upstream')
    ax[0].plot(k,k*k*pred,label='Fresh local',lw=.8)
    ax[0].set(xlim=(2,20),xlabel='k (Angstrom^-1)',ylabel='k^2 chi(k) (Angstrom^-2)');ax[0].legend(fontsize=7)
    ax[1].plot(r,np.abs(transform(chi)),label='Measured transform')
    ax[1].plot(r,np.abs(transform(forward(INITIAL,k,sp))),label='Initial shell',alpha=.5)
    ax[1].plot(r,np.abs(upstream['transform'][0]),'--',label='Upstream fit')
    ax[1].plot(r,np.abs(transform(pred)),label='Fresh fit',lw=.9)
    ax[1].axvspan(1.4,3.,alpha=.1,color='gray',label='Fixed fit interval')
    ax[1].set(xlim=(0,6),xlabel='Fourier radius (Angstrom; uncorrected phase)',ylabel='|chi(R)| (Angstrom^-3)');ax[1].legend(fontsize=7)
    ax[2].plot(R,upstream['reconstruction'][0],'--',label='Upstream Gaussian density')
    ax[2].plot(R,fresh['reconstruction'][0],label='Fresh inferred density',lw=.9)
    ax[2].set(xlabel='Cu-Cu distance (Angstrom)',ylabel='Neighbors per Angstrom');ax[2].legend(fontsize=7)
    return fig
