import numpy as np
import matplotlib.pyplot as plt
from .physics_model import Z,forward,INITIAL
def comparisons(data,result,upstream):
    """Plot observations, both independently computed fits, initialization, and SLD profiles."""
    q,r,e,dq=data
    fig,ax=plt.subplots(1,3,figsize=(16,4.4),layout='constrained')
    ax[0].errorbar(q,r,yerr=e,fmt='.',ms=2,alpha=.45,label='Real PLATYPUS data')
    ax[0].plot(q,forward(INITIAL,q,dq),label='Initial slab model',alpha=.6)
    ax[0].plot(q,upstream['prediction'][0],'--',label='Independent upstream run')
    pred=forward(result['parameters'][0],q,dq)
    ax[0].plot(q,pred,label='Fresh local reconstruction',lw=1)
    ax[0].set(xlabel='Q (Angstrom^-1)',ylabel='Reflectivity',yscale='log');ax[0].legend(fontsize=7)
    ax[1].plot(Z,upstream['reconstruction'][0],'--',label='Upstream')
    ax[1].plot(Z,result['reconstruction'][0],label='Fresh local',lw=1)
    ax[1].set(xlabel='Depth from Si/oxide (Angstrom)',ylabel='SLD (10^-6 Angstrom^-2)');ax[1].legend()
    ax[2].plot(q,(np.log10(r)-np.log10(pred))/(e/r/np.log(10)),'.',ms=3)
    ax[2].axhline(0,color='k',lw=.5);ax[2].set(xlabel='Q (Angstrom^-1)',ylabel='Weighted log residual')
    return fig
