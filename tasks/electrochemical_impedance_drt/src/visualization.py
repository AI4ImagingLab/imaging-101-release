
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
    from .solvers import solve
    from .physics_model import operators
    setup();f=data['frequency'];z=data['impedance'];tau=1/f;p=result['reconstruction'][0];fit=result['fitted_impedance'][0]
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    ax[0,0].plot(z.real,-z.imag,'o',ms=3,color=GRAY,label='Measured LIB impedance');ax[0,0].plot(fit.real,-fit.imag,color=BLUE,label='Fresh PWL fit');ax[0,0].set(xlabel="Re(Z) (ohm)",ylabel="-Im(Z) (ohm)",title='A  Complex impedance');ax[0,0].legend()
    ax[0,1].semilogx(tau,reference['reconstruction'][0],'--',color=ORANGE,label='Reproduced upstream');ax[0,1].semilogx(tau,p,color=BLUE,label='Fresh local');ax[0,1].set(xlabel='Relaxation time (s)',ylabel='DRT (ohm)',title='B  Relaxation spectrum');ax[0,1].legend()
    ax[1,0].semilogx(f,(fit-z).real*1e3,color=BLUE,label='Real residual');ax[1,0].semilogx(f,(fit-z).imag*1e3,color=ORANGE,label='Imaginary residual');ax[1,0].axhline(0,color=GRAY,lw=.7);ax[1,0].set(xlabel='Frequency (Hz)',ylabel='Signed residual (milliohm)',title='C  Observation consistency');ax[1,0].legend()
    ax[1,1].loglog(f,abs(z),'o',ms=3,color=GRAY,label='Measured');ax[1,1].loglog(f,abs(fit),color=BLUE,label='Fresh local');ax[1,1].set(xlabel='Frequency (Hz)',ylabel='Magnitude (ohm)',title='D  Full acquisition bandwidth');ax[1,1].legend()
    fig2,aa=plt.subplots(1,3,figsize=(13,3.7),layout='constrained');traces={'frequency':f,'observed':z,'tau':tau,'fresh_spectrum':p,'fresh_fit':fit,'upstream_spectrum':reference['reconstruction'][0]};rows=[]
    for lam,color in zip([1e-5,1e-3,1e-1],[ORANGE,BLUE,GREEN]):
        x,y=solve(f,z,lam);a,l,_=operators(f);res=np.linalg.norm(y-z);rough=np.linalg.norm(l@x)
        aa[0].semilogx(tau,x[2:],color=color,label=f'lambda={lam:g}');aa[1].plot(rough,res,'o',color=color,label=f'{lam:g}');rows.append([lam,res,rough]);traces[f'lambda_{lam:g}_spectrum']=x[2:]
    for stride,style in [(1,'-'),(2,'--'),(4,':')]:
        x,y=solve(f[::stride],z[::stride]);aa[2].semilogx(1/f[::stride],x[2:],style,label=f'{len(f[::stride])} measured frequencies');traces[f'stride_{stride}_spectrum']=x[2:]
    aa[0].set(xlabel='Relaxation time (s)',ylabel='DRT (ohm)',title='A  Regularization sensitivity');aa[1].set(xlabel='Derivative norm (ohm)',ylabel='Complex residual norm (ohm)',title='B  Bias-smoothness tradeoff');aa[2].set(xlabel='Relaxation time (s)',ylabel='DRT (ohm)',title='C  Reduced frequency acquisition')
    for a in aa:a.legend(fontsize=8)
    return {'reconstruction':finish(fig),'robustness':finish(fig2)},traces,{'upstream_relative_error':relative_error(p,reference['reconstruction'][0]),'observation_relative_error':relative_error(fit,z),'regularization_scan':rows,'series_L_H_R_ohm':result['series_parameters'][0].tolist()}
