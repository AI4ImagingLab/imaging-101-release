import matplotlib.pyplot as plt
import numpy as np

def comparison(t,g,r,p,pred,upstream,alphas,res,pen,selected,adaptive):
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for j,color in enumerate(['tab:blue','tab:orange']):
        axes[0,0].semilogx(t,g[:,j],'.',ms=2,color=color,label=f'Measured replicate{j+1}');axes[0,0].semilogx(t,pred[:,j],color=color)
        axes[0,1].semilogx(r,p[j],color=color,label=f'Fresh fixed alpha replicate{j+1}');axes[0,1].semilogx(r,upstream[j],'--',color=color,label=f'Fresh upstream replicate{j+1}');axes[0,1].semilogx(r,adaptive[j],':',color=color,label=f'Fresh L-curve replicate{j+1}')
        axes[1,0].plot(np.log(res[:,j]),np.log(pen[:,j]),color=color,label=f'Replicate{j+1}');idx=np.argmin(abs(alphas-selected[j]));axes[1,0].plot(np.log(res[idx,j]),np.log(pen[idx,j]),'x',color=color,ms=8)
        axes[1,1].semilogx(t,g[:,j]-pred[:,j],color=color,label=f'Replicate{j+1}')
    axes[0,0].set(xlabel='Lag (s)',ylabel='Intensity correlation',title='Measured data and fresh fixed-alpha fits');axes[0,1].set(xlabel='Effective hydrodynamic radius (nm)',ylabel='Intensity-weighted bin mass',title='Populations under source solvent convention');axes[1,0].set(xlabel='Log augmented residual norm',ylabel='Log curvature norm',title='Full fresh L-curve comparison');axes[1,1].set(xlabel='Lag (s)',ylabel='Measured minus predicted',title='Residuals on every reported lag')
    for ax in axes.flat:ax.legend(fontsize=7)
    return fig
