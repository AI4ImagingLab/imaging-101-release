"""Fresh measured/reference comparisons; pyplot imported without backend changes."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import predict,association_constants,concentrations
from .preprocessing import ACQUISITION
def make_figures(data,fit,reference):
    """Return three labeled figures; never read or substitute cached solver results."""
    q=predict(fit.x,data['injection_volumes']); old=np.array([reference[f'output_dQ_{i}'] for i in range(6)])
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,ax in enumerate(axes.flat):
        x=np.arange(1,15);ax.scatter(x[1:],data['heats'][i,1:],s=23,color='black',label='Measured, fit set')
        ax.scatter(x[:1],data['heats'][i,:1],marker='x',s=50,color='gray',label='First injection, excluded')
        ax.plot(x,q[i],label='Fresh local ML');ax.plot(x,old[i],'--',label='Original author ML')
        ax.set(title=f'{ACQUISITION[i,0]:.5f} °C',xlabel='Injection number',ylabel='Heat (microcal)')
    axes[0,0].legend(fontsize=8);fig.suptitle('Real S100A14–zinc calorimetry: all 84 injections')
    fig2,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for i in range(6):axes[0].plot(np.arange(2,15),q[i,1:]-data['heats'][i,1:],'.-',label=f'{ACQUISITION[i,0]:.1f} °C')
    axes[0].axhline(0,color='black',lw=.7);axes[0].set(xlabel='Injection number',ylabel='Residual (microcal)',title='Model discrepancy is retained');axes[0].legend(fontsize=7)
    k=association_constants(fit.x[0],fit.x[1],ACQUISITION[:,0]+273.15)
    axes[1].plot(ACQUISITION[:,0],k,'o-');axes[1].set(xlabel='Actual temperature (°C)',ylabel='Association constant (M⁻¹)',title='Shared van’t Hoff relation')
    for i,a in enumerate(ACQUISITION):
        _,t=concentrations(data['injection_volumes'][i],a[1],a[2],a[3]);axes[2].plot(np.arange(1,15),t[1:]*fit.x[3+2*i]+fit.x[4+2*i],label=f'{a[0]:.1f} °C')
    axes[2].set(xlabel='Injection number',ylabel='Dilution contribution (microcal)',title='Fitted nuisance contribution');axes[2].legend(fontsize=7)
    fig3,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    err=np.abs(q-old);im=axes[0].imshow(err,aspect='auto');fig3.colorbar(im,ax=axes[0],label='Absolute difference (microcal)');axes[0].set(title=f'Fresh/original max difference {err.max():.2g} microcal',xlabel='Injection index',ylabel='Experiment index')
    rms=np.sqrt(np.mean((q[:,1:]-data['heats'][:,1:])**2,axis=1));ref_rms=np.sqrt(np.mean((old[:,1:]-data['heats'][:,1:])**2,axis=1));x=np.arange(6)
    axes[1].bar(x-.16,ref_rms,.32,label='Original ML');axes[1].bar(x+.16,rms,.32,label='Fresh local ML');axes[1].set(xticks=x,xticklabels=['5','10','17','25','30','35'],xlabel='Nominal temperature (°C)',ylabel='Observation RMSE (microcal)',title='Independent executions, same observations');axes[1].legend()
    return [fig,fig2,fig3]
