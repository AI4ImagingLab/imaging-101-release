
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
    from .preprocessing import phase_echo
    from .physics_model import RELAXATION_TIME,decay_kernel
    from .solvers import solve
    from scipy.optimize import nnls
    setup();t=data['echo_time'][1:];y=phase_echo(data['echo']);p=result['reconstruction'][0];fit=result['fitted_echo'][0];k=decay_kernel(t)
    avg=nnls(np.vstack([k,.01*np.eye(255)]),np.r_[y.mean(axis=0),np.zeros(255)])[0]
    fig,ax=plt.subplots(2,2,figsize=(11,7.6),layout='constrained')
    m=ax[0,0].pcolormesh(RELAXATION_TIME,np.arange(len(p)),p,shading='nearest',cmap='viridis');ax[0,0].set(xscale='log',xlabel='Relaxation time (s)',ylabel='Acquisition index',title='A  Every measured scan');fig.colorbar(m,ax=ax[0,0],label='Discrete amplitude (a.u.)')
    for i,color in zip([0,59,119],[ORANGE,GREEN,BLUE]):ax[0,1].semilogx(t,y[i],'.',ms=2,color=color,alpha=.5);ax[0,1].semilogx(t,fit[i],color=color,label=f'Scan {i}')
    ax[0,1].set(xlabel='Echo time (s)',ylabel='Echo amplitude (a.u.)',title='B  First, middle and final acquisitions');ax[0,1].legend(fontsize=8)
    ax[1,0].semilogx(RELAXATION_TIME,reference['reconstruction'][0].mean(axis=0),'--',color=ORANGE,label='Mean reproduced upstream');ax[1,0].semilogx(RELAXATION_TIME,p.mean(axis=0),color=BLUE,label='Mean individual inversions');ax[1,0].semilogx(RELAXATION_TIME,avg,color=GREEN,label='Inversion of mean phased echo');ax[1,0].set(xlabel='Relaxation time (s)',ylabel='Amplitude (a.u.)',title='C  Averaging and inversion');ax[1,0].legend(fontsize=8)
    residual=y-fit;lim=abs(residual).max();im=ax[1,1].pcolormesh(t,np.arange(len(p)),residual,cmap='RdBu_r',vmin=-lim,vmax=lim,shading='nearest');ax[1,1].set(xscale='log',xlabel='Echo time (s)',ylabel='Acquisition index',title='D  Signed residual, all scans');fig.colorbar(im,ax=ax[1,1],label='Residual (a.u.)')
    fig2,aa=plt.subplots(1,3,figsize=(13,3.7),layout='constrained');traces={'echo_time':t,'relaxation_time':RELAXATION_TIME,'phased_echo':y,'fresh_spectrum':p,'fresh_fit':fit,'average_inversion':avg,'upstream_spectrum':reference['reconstruction'][0]};rows=[]
    for alpha,color in zip([1e-6,1e-4,1e-2],[ORANGE,BLUE,GREEN]):
        q,v=solve(data['echo_time'],data['echo'],alpha);aa[0].semilogx(RELAXATION_TIME,q.mean(axis=0),color=color,label=f'alpha={alpha:g}');rows.append([alpha,relative_error(v,y),float(np.linalg.norm(q))]);traces[f'alpha_{alpha:g}_mean_spectrum']=q.mean(axis=0)
    mass=p.sum(axis=1);logmean=(p@np.log10(RELAXATION_TIME))/mass;aa[1].plot(mass,color=BLUE);aa[2].plot(10**logmean,color=GREEN)
    aa[0].set(xlabel='Relaxation time (s)',ylabel='Mean amplitude (a.u.)',title='A  Regularization sensitivity');aa[0].legend(fontsize=8);aa[1].set(xlabel='Acquisition index',ylabel='Total reconstructed amplitude (a.u.)',title='B  Integrated mobile signal');aa[2].set(xlabel='Acquisition index',ylabel='Geometric mean T2 (s)',title='C  Relaxation evolution')
    return {'reconstruction':finish(fig),'robustness':finish(fig2)},traces,{'upstream_relative_error':relative_error(p,reference['reconstruction'][0]),'observation_relative_error':relative_error(fit,y),'averaged_vs_mean_individual_relative_error':relative_error(avg,p.mean(axis=0)),'alpha_relative_residual_amplitude_norm':rows}
