"""Measured-noise diagnostics and freshly computed model/sample sensitivity."""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri
from .preprocessing import DT_PS
from .physics_model import curvature,variance_components
from .solvers import reconstruct

def relative_error(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def experiments(x,out,diagnostic):
    start=time.perf_counter();variants={50:out};diags={50:diagnostic}
    uncertainties={50:curvature(diagnostic['theta'],x,diagnostic['scales'],DT_PS)}
    for m in [10,25]:
        variants[m],diags[m]=reconstruct(x[:m]);uncertainties[m]=curvature(diags[m]['theta'],x[:m],diags[m]['scales'],DT_PS)
    additive,add_diag=reconstruct(x,noise_model='additive')
    source={}
    for m,v in variants.items():
        for k,a in v.items():source[f'count{m}_{k}']=a
        for k,a in uncertainties[m].items():source[f'count{m}_{k}']=a
        source[f'count{m}_nll_history']=diags[m]['objective_history']
    for k,a in additive.items():source['additive_'+k]=a
    measurements=dict(method='Fresh selected full joint BFGS; independent Hessian diagnostics do not replace it',sample_counts={str(m):dict(noise=variants[m]['noise'][0].tolist(),err_noise=uncertainties[m]['err_noise'].tolist(),success=diags[m]['success'],message=diags[m]['message'],nll=diags[m]['nll'],gradient_inf_per_observation=diags[m]['gradient_inf_per_observation']) for m in variants},additive=dict(nll=add_diag['nll'],success=add_diag['success'],message=add_diag['message'],noise=additive['noise'][0].tolist()),experiment_runtime_s=time.perf_counter()-start)
    return variants,uncertainties,additive,source,measurements

def make_figures(x,out,diagnostic,reference,variants,uncertainties,additive,source):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    t=np.arange(x.shape[1])*DT_PS;mu=out['reconstruction'][0];z=out['expected_waveforms'][0];std=out['noise_std'][0];res=x-z;standard=res/std
    parts=variance_components(z,out['noise'][0],DT_PS);source.update(raw_waveforms_pA=x,time_ps=t,variance_components_pA2=parts,standardized_residual=standard,aligned_residual_variance_pA2=np.var(res,axis=0,ddof=1))
    figs={};fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ax=axs[0,0];ax.plot(t,x.T,alpha=.18,lw=.6,color='#0072B2');ax.set(title='A  All 50 measured air traces: full native window',xlabel='Relative sampling time (ps)',ylabel='Photocurrent (pA)')
    ax=axs[0,1];ax.plot(t,reference['reconstruction'][0],lw=3,alpha=.4,color='.4',label='Original latent waveform');ax.plot(t,mu,label='Fresh latent waveform');ax.plot(t,x.mean(axis=0),'--',label='Raw sample mean');ax.fill_between(t,mu-uncertainties[50]['err_mu'],mu+uncertainties[50]['err_mu'],alpha=.15,label='Conditional curvature ±1 SE')
    ax.set(title='B  Latent waveform in first-trace amplitude/delay gauge',xlabel='Relative sampling time (ps)',ylabel='Photocurrent (pA)');ax.legend(fontsize=8)
    inset=ax.inset_axes([.04,.62,.32,.25]);inset.plot(t,mu-x.mean(axis=0),lw=.8);inset.axhline(0,color='.6',lw=.5);inset.tick_params(labelsize=7);inset.set_title('Latent minus raw mean (pA)',fontsize=8);inset.set_xlabel('Time (ps)',fontsize=7)
    source['latent_minus_raw_mean_pA']=mu-x.mean(axis=0)
    ax=axs[1,0];indices=np.arange(50)
    ax.plot(indices,reference['amplitude'][0],color='.5',lw=3,alpha=.35,label='Original');ax.errorbar(indices,out['amplitude'][0],yerr=uncertainties[50]['err_a'],fmt='o-',ms=3,label='Fresh ± conditional SE');ax.set(title='C  Amplitude drift across repeated acquisition',xlabel='Scan index (1)',ylabel='Relative amplitude (1)');ax.legend(fontsize=8)
    ax=axs[1,1];ax.plot(indices,reference['delay'][0]*1e3,color='.5',lw=3,alpha=.35,label='Original');ax.errorbar(indices,out['delay'][0]*1e3,yerr=uncertainties[50]['err_eta']*1e3,fmt='o-',ms=3,label='Fresh ± conditional SE');ax.set(title='D  Timing drift across repeated acquisition',xlabel='Scan index (1)',ylabel='Delay relative to first scan (fs)');ax.legend(fontsize=8)
    figs['reconstruction']=fig
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ax=axs[0,0]
    for i,label in enumerate(['Additive','Amplitude','Timing jitter']):ax.plot(t,parts[i].mean(axis=0),label=label)
    ax.plot(t,parts.sum(axis=0).mean(axis=0),color='black',lw=1.4,label='Total model variance');ax.plot(t,np.var(x,axis=0,ddof=1),color='.65',lw=.8,label='Raw observed variance');ax.plot(t,np.var(res,axis=0,ddof=1),color='#CC79A7',lw=.8,label='Residual sample variance');ax.set(yscale='log',title='A  Signal-dependent variance: drift removed',xlabel='Relative sampling time (ps)',ylabel='Variance (pA²)');ax.legend(fontsize=7)
    ax=axs[0,1];ax.plot(t,standard.T,alpha=.1,lw=.5,color='#0072B2');ax.axhline(0,color='.5',lw=.7);ax.set(title='B  All standardized residual traces',xlabel='Relative sampling time (ps)',ylabel='Residual / bias-corrected noise SD (1)')
    ax=axs[1,0];r=np.sort(standard.ravel());q=ndtri((np.arange(len(r))+.5)/len(r));ax.plot(q,r,'.',ms=2,alpha=.5);limits=[min(q.min(),r.min()),max(q.max(),r.max())];ax.plot(limits,limits,'--',color='.5');ax.set(title='C  Normal QQ: fitted-residual diagnostic',xlabel='Standard-normal quantile (1)',ylabel='Measured residual quantile (1)')
    source['normal_quantiles']=q;source['sorted_standardized_residuals']=r
    ax=axs[1,1];history=diagnostic['objective_history'];ax.plot(np.arange(len(history)),history-history[-1]);ax.set(yscale='symlog',title='D  Canonical BFGS likelihood history',xlabel='Accepted iterate index (1)',ylabel='NLL minus final NLL (1)');ax.set_yscale('symlog',linthresh=1e-6);ax.text(.98,.95,'Precision-loss stop; gtol unmet',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    figs['residuals']=fig
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained');counts=[10,25,50]
    labels=[('Additive amplitude','pA',1.),('Multiplicative amplitude','1',1.),('Timing jitter','fs',1e3)]
    for k,(title,unit,factor) in enumerate(labels):
        ax=axs.flat[k];values=np.array([variants[m]['noise'][0,k] for m in counts])*factor;err=np.array([uncertainties[m]['err_noise'][k] for m in counts])*factor
        ax.errorbar(counts,values,yerr=err,fmt='o-',capsize=4,label='Fresh ± conditional curvature SE');ax.scatter([50],[reference['noise'][0,k]*factor],marker='x',s=70,color='black',label='Original full-data estimate');ax.set(title=chr(65+k)+'  '+title,xlabel='First M acquired scans (1)',ylabel=f'Noise amplitude ({unit})',xticks=counts);ax.legend(fontsize=8)
    ax=axs[1,1];add_standard=(x-additive['expected_waveforms'][0])/additive['noise_std'][0]
    ax.plot(t,np.sqrt(np.mean(standard**2,axis=0)),label='Full three-source model');ax.plot(t,np.sqrt(np.mean(add_standard**2,axis=0)),label='Additive-only model');ax.axhline(1,color='.5',lw=.8,ls='--');ax.set(title='D  Noise-model sensitivity across the waveform',xlabel='Relative sampling time (ps)',ylabel='RMS standardized residual across scans (1)');ax.legend(fontsize=8)
    source['additive_standardized_residuals']=add_standard;figs['sensitivity']=fig
    return figs
