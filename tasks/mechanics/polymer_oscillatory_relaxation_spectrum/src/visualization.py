
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
    from .solvers import solve,ReSpectConfig
    setup();w,c,d=solve(data);n=len(w);g=np.r_[np.interp(w,data['angular_frequency'],data['storage_modulus']),np.interp(w,data['angular_frequency'],data['loss_modulus'])];fig,ax=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    ax[0,0].loglog(c.s,np.exp(c.H),color=BLUE,label='Fresh continuous spectrum');ax[0,0].fill_between(c.s,np.exp(c.H-2.5*c.dH),np.exp(c.H+2.5*c.dH),color=BLUE,alpha=.15,label='Lambda-spread band');ax[0,0].loglog(c.s,reference['reconstruction'][0],'--',color=ORANGE,label='Reproduced upstream');ax[0,0].errorbar(d.tau,d.g,fmt='o',ms=4,color=GREEN,label='Fresh discrete Maxwell weights');ax[0,0].set(xlabel='Relaxation time (s)',ylabel='Spectrum / mode modulus (Pa)',title='A  Continuous and discrete spectra');ax[0,0].legend(fontsize=7)
    for offset,color,label in [(0,BLUE,"Storage G'"),(n,ORANGE,"Loss G''")]:
        ax[0,1].loglog(w,g[offset:offset+n],'o',ms=3,color=color,label=label+' measured');ax[0,1].loglog(w,c.G_fit[offset:offset+n],color=color);ax[0,1].loglog(w,d.G_fit[offset:offset+n],'--',color=color)
    ax[0,1].set(xlabel='Angular frequency (rad/s)',ylabel='Modulus (Pa)',title='B  Solid: continuous; dashed: discrete fit');ax[0,1].legend(fontsize=7)
    ax[1,0].semilogx(c.lam,c.log_P,color=BLUE);ax[1,0].axvline(c.lam_C,color=ORANGE,ls='--',label='Selected posterior mean');ax[1,0].set(xlabel='Regularization lambda (1)',ylabel='Relative log evidence (1)',title='C  Bayesian regularization selection');ax[1,0].legend(fontsize=7)
    ax[1,1].plot(d.wt_base,d.AIC_bst,'o-',color=BLUE);ax[1,1].set(xlabel='Base placement weight (1)',ylabel='Best AIC (1)',title=f'D  Discrete model selection: {d.N} modes')
    fig2,aa=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');traces={'angular_frequency':w,'observed_moduli':g,'relaxation_time':c.s,'fresh_spectrum':np.exp(c.H),'continuous_fit':c.G_fit,'discrete_fit':d.G_fit,'maxwell_modes':np.array([d.g,d.tau]),'lambda_scan':c.lam,'log_evidence':c.log_P,'AIC_base':d.wt_base,'AIC':d.AIC_bst};rows=[]
    for factor,color in zip([.1,1,10],[ORANGE,BLUE,GREEN]):
        _,cc,dd=solve(data,ReSpectConfig(lam_C=c.lam_C*factor));aa[0].loglog(cc.s,np.exp(cc.H),color=color,label=f'lambda x{factor:g}');aa[1].semilogx(w,(cc.G_fit[:n]/g[:n]-1)*100,color=color,label=f'x{factor:g}');traces[f'lambda_factor_{factor:g}_spectrum']=np.exp(cc.H);rows.append([factor,dd.N,float(np.linalg.norm(cc.G_fit/g-1))])
    for stride,style in [(1,'-'),(2,'--'),(4,':')]:
        reduced={k:v[::stride] for k,v in data.items()};_,cc,dd=solve(reduced);aa[2].loglog(cc.s,np.exp(cc.H),style,label=f'{len(reduced["angular_frequency"])} native points');traces[f'stride_{stride}_spectrum']=np.exp(cc.H);traces[f'stride_{stride}_relaxation_time']=cc.s
    traces['upstream_spectrum']=reference['reconstruction'][0]
    traces['log_spectrum_spread']=c.dH;traces['lcurve_residual']=c.rho;traces['lcurve_curvature']=c.eta;traces['mode_time_uncertainty']=d.dtau
    fig3,ab=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    ab[0].loglog(c.rho,c.eta,'o-',color=BLUE);ab[0].set(xlabel='Relative residual norm (1)',ylabel='Log-spectrum curvature norm (1)',title='A  Computed L-curve')
    from matplotlib.ticker import NullFormatter
    ab[0].set_xticks([.2,.5,1,2],['0.2','0.5','1','2']);ab[0].xaxis.set_minor_formatter(NullFormatter())
    ab[1].plot(np.arange(1,d.N+1),d.dtau/d.tau,'o',color=GREEN);ab[1].axhline(1,color=GRAY,ls='--',lw=.7);ab[1].set(yscale='log',xlabel='Mode index',ylabel='Local time uncertainty / time (1)',title='B  Mode-time identifiability')
    for channel,offset,color in [("Storage",0,BLUE),("Loss",n,ORANGE)]:
        ab[2].semilogx(w,(c.G_fit[offset:offset+n]/g[offset:offset+n]-1)*100,color=color,label=channel+' continuous');ab[2].semilogx(w,(d.G_fit[offset:offset+n]/g[offset:offset+n]-1)*100,'--',color=color,label=channel+' discrete')
    ab[2].axhline(0,color=GRAY,lw=.6);ab[2].set(xlabel='Angular frequency (rad/s)',ylabel='Signed relative residual (%)',title='C  All-channel observation consistency');ab[2].legend(fontsize=7)
    aa[0].set(xlabel='Relaxation time (s)',ylabel='Spectrum (Pa)',title='A  Regularization sensitivity');aa[1].set(xlabel='Angular frequency (rad/s)',ylabel="Signed storage residual (%)",title='B  Residual versus regularization');aa[1].axhline(0,color=GRAY,lw=.6);aa[2].set(xlabel='Relaxation time (s)',ylabel='Spectrum (Pa)',title='C  Frequency sampling reduction')
    for aax in aa:aax.legend(fontsize=8)
    return {'reconstruction':finish(fig),'robustness':finish(fig2),'inverse_diagnostics':finish(fig3)},traces,{'upstream_relative_error':relative_error(result['reconstruction'],reference['reconstruction']),'continuous_relative_residual_norm':float(np.linalg.norm(c.G_fit/g-1)),'discrete_relative_residual_norm':float(np.linalg.norm(d.G_fit/g-1)),'lambda_selected':c.lam_C,'discrete_mode_count':d.N,'lambda_factor_mode_count_relative_residual':rows}
