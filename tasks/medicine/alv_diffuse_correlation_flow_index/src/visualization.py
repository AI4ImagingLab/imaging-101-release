import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import prepare
def curve_rmse(predicted,weighted,mask):
    return np.array([np.sqrt(np.mean((p[m]-y[m])**2)) for p,y,m in zip(predicted,weighted,mask)])
def make_figures(data,fresh,original,tighter,sensitivity):
    weighted,mask=prepare(data);tau=data['lag_times'];x=np.arange(390)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,i in zip(axes.flat,[0,100,200,300]):
        ax.semilogx(tau,weighted[i],'.',ms=3,label='Count-weighted measured g2')
        ax.semilogx(tau[mask[i]],weighted[i,mask[i]],'.',ms=4,label='Fixed fit support')
        ax.semilogx(tau,fresh['predicted_g2'][0,i],label='Fresh CDE')
        ax.set(xlim=(1e-8,.01),ylim=(.8,1.75),title=f'Fixed record {i+1}; early invalid hardware bins retained',xlabel='Correlation lag (s)',ylabel='Dimensionless g2');ax.legend(fontsize=7)
    fig2,axes=plt.subplots(3,1,figsize=(12,8),layout='constrained')
    axes[0].plot(x,original['flow_index'][0]*1e9,label='Original CDE');axes[0].plot(x,fresh['flow_index'][0]*1e9,'--',label='Fresh CDE');axes[0].set(ylabel='Flow index (10^-9 cm²/s)',title='All390measurements; unknown sample, no physiological event labels');axes[0].legend()
    axes[1].plot(x,fresh['beta'][0]);axes[1].axhline(.54,color='red',ls='--',label='Original upper fit bound');axes[1].set(ylabel='Coherence beta');axes[1].legend()
    axes[2].plot(x,curve_rmse(fresh['predicted_g2'][0],weighted,mask));axes[2].set(xlabel='Acquisition record index',ylabel='Masked g2 RMS residual')
    fig3,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    im=axes[0].imshow(np.where(mask,fresh['predicted_g2'][0]-weighted,np.nan),aspect='auto',cmap='RdBu_r',vmin=-.2,vmax=.2);fig3.colorbar(im,ax=axes[0],label='g2 residual');axes[0].set(title='All fit residuals; outside support blank',xlabel='Lag-bin index',ylabel='Record index')
    axes[1].plot(x,(fresh['flow_index'][0]/original['flow_index'][0]-1)*100,label='Fresh/default original');axes[1].plot(x,(tighter['flow_index'][0]/original['flow_index'][0]-1)*100,alpha=.6,label='Tighter/default original');axes[1].set(title='Numerical convergence comparison',xlabel='Record index',ylabel='Flow-index difference (%)');axes[1].legend(fontsize=7)
    axes[2].plot(x,sensitivity['flow_index'][0]/fresh['flow_index'][0],label='Header n1.332 / selected n1.4');axes[2].set(title='Fresh optical-assumption sensitivity',xlabel='Record index',ylabel='Flow-index ratio');axes[2].legend(fontsize=7)
    return [fig,fig2,fig3]
