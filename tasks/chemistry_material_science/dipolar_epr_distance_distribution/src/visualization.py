
"""Maintainer scientific comparisons, computed from local modules and supplied arrays.
No backend changes, filesystem writes, upstream imports, or hidden asset loading.
"""
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';GRAY='#666666'
def relative_error(estimate,reference):
    norm=np.linalg.norm(reference.ravel())
    if norm<=0:raise ValueError('Zero reference norm')
    return float(np.linalg.norm((estimate-reference).ravel())/norm)
def setup():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'svg.fonttype':'none','pdf.fonttype':42})
def finish(fig):
    fig.align_labels()
    return fig

def make_figures(data,result,reference):
    from .preprocessing import prepare
    from .physics_model import DISTANCE
    from .solvers import solve
    setup();t,y=prepare(data['time'],data['echo']);p=result['reconstruction'][0];fit=result['fitted_echo'][0]
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    ax[0,0].plot(t,y,'.',color=GRAY,label='Phase-corrected observation');ax[0,0].plot(t,fit,color=BLUE,label='Fresh two-Gaussian fit');ax[0,0].set(xlabel='Evolution time (microsecond)',ylabel='Normalized echo (1)',title='A  Measured dipolar evolution');ax[0,0].legend(fontsize=8)
    ax[0,1].plot(DISTANCE,reference['reconstruction'][0],'--',color=ORANGE,label='Reproduced upstream');ax[0,1].plot(DISTANCE,p,color=BLUE,label='Fresh local');ax[0,1].set(xlabel='Spin distance (nm)',ylabel='Probability density (1/nm)',title='B  Full distance distribution');ax[0,1].legend()
    ax[1,0].plot(t,y-fit,color=BLUE);ax[1,0].axhline(0,color=GRAY,lw=.7);ax[1,0].set(xlabel='Evolution time (microsecond)',ylabel='Signed echo residual (1)',title='C  Model residual')
    ax[1,1].hist(y-fit,bins=20,color=BLUE,alpha=.8);ax[1,1].set(xlabel='Signed echo residual (1)',ylabel='Count',title='D  Residual distribution')
    fig2,aa=plt.subplots(1,3,figsize=(13,3.7),layout='constrained');traces={'time':t,'phased_echo':y,'fit':fit,'distance':DISTANCE,'fresh_distribution':p,'upstream_distribution':reference['reconstruction'][0]};rows=[]
    for fraction,color in zip([.5,.75,1],[ORANGE,GREEN,BLUE]):
        count=int(len(t)*fraction);q,v,pars=solve(data['time'][:count],data['echo'][:count]);aa[0].plot(DISTANCE,q,color=color,label=f'{count} observed echoes');traces[f'echo_count_{count}_distribution']=q;rows.append([count,float(np.trapezoid(q*DISTANCE,DISTANCE)),relative_error(q,p)])
    for shift,color in zip([-.016,0,.016],[ORANGE,BLUE,GREEN]):
        # A timing-origin perturbation shifts both the admissible refocusing interval and fitted origin.
        # Here the observable time increments are scaled to model timing calibration error.
        factor=1+shift;q,v,pars=solve(data['time']*factor,data['echo']);aa[1].plot(DISTANCE,q,color=color,label=f'Time scale {factor:.3f}');traces[f'time_scale_{factor:.3f}_distribution']=q
    residual=y-fit;aa[2].plot(t[:-1],residual[:-1]*residual[1:],'.',ms=3,color=BLUE);aa[2].axhline(0,color=GRAY,lw=.7)
    aa[0].set(xlabel='Distance (nm)',ylabel='Density (1/nm)',title='A  Limited evolution time');aa[1].set(xlabel='Distance (nm)',ylabel='Density (1/nm)',title='B  Timing calibration sensitivity');aa[2].set(xlabel='Time (microsecond)',ylabel='Lag-one residual product (1)',title='C  Correlated-error diagnostic');aa[0].legend(fontsize=8);aa[1].legend(fontsize=8)
    return {'reconstruction':finish(fig),'robustness':finish(fig2)},traces,{'upstream_relative_error':relative_error(p,reference['reconstruction'][0]),'echo_rmse':float(np.sqrt(np.mean(residual**2))),'acquisition_reduction_count_mean_nm_relative_error':rows,'parameters':result['parameters'][0].tolist()}
