"""Review figures use computed inputs and identify their scientific roles."""
import numpy as np
import matplotlib.pyplot as plt

def observation_figure(v,i,out,d):
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(v,i*1e3,'k.',ms=3,label='Measured, author-smoothed')
    for k,c in [('V_F','tab:orange'),('V_P','tab:blue')]:ax[0].axvline(float(out[k][0,0]),color=c,label=k)
    ax[0].set(xlabel='Bias (V)',ylabel='Signed current (mA)',title='Complete 304-point observation');ax[0].legend(fontsize=8)
    ax[1].plot(v,d['gradient']*1e3);ax[1].set(xlabel='Bias (V)',ylabel='dI/dV (mA/V)',title='Potential from full-sweep gradient')
    ax[2].plot(v,d['second_derivative']*1e3);ax[2].axvspan(float(out['V_F'][0,0]),float(out['V_P'][0,0]),alpha=.15)
    ax[2].set(xlabel='Bias (V)',ylabel='d²I/dV² (mA/V²)',title='Curvature before support selection')
    fig.tight_layout();return fig

def fit_figure(v,i,out,d):
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    vf=float(out['V_F'][0,0]);mask=v<vf
    ax[0].plot(v[mask],(i[mask]*1e3)**2,'k.',label='Measured ion region')
    ax[0].plot(v[mask],np.polyval(d['ion_fit'],v[mask]),label='Fresh OML fit')
    ax[0].set(xlabel='Bias (V)',ylabel='Squared current (mA²)',title='Ion collection');ax[0].legend(fontsize=8)
    for key,label in [('first_mask','First window'),('second_mask','Refined window')]:
        m=d[key];cur=i[m]-d['ion_current'][m];ax[1].plot(v[m],np.log(cur),'.',label=label)
    m=d['first_mask'];ax[1].plot(v[m],np.log(d['electron_current'][m]),label='Fresh refined fit')
    ax[1].set(xlabel='Bias (V)',ylabel='ln electron current (A)',title='Two native fit windows');ax[1].legend(fontsize=8)
    model=d['ion_current']+d['electron_current'];ax[2].plot(v[m],(model[m]-i[m])*1e6,'.-')
    ax[2].axhline(0,color='k',lw=.7);ax[2].set(xlabel='Bias (V)',ylabel='Predicted - measured (µA)',title='Retarding-region current residual')
    fig.tight_layout();return fig

def distribution_figure(out,upstream):
    e=out['energy'][0];p=out['reconstruction'][0];r=upstream['eedf'];temp=float(out['T_e'][0,0])
    ideal=np.sqrt(e)*np.exp(-e/temp);ideal/=abs(np.trapezoid(ideal,x=e))
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(e,r,'k-',lw=3,alpha=.35,label='Reproduced upstream, reference')
    ax[0].plot(e,p,'o',ms=3,label='Fresh local inverse');ax[0].plot(e,ideal,'--',label='Maxwellian from recovered T')
    ax[0].set(xlabel='Energy (eV)',ylabel='Normalized density (eV⁻¹)',title='Distribution and physical model');ax[0].legend(fontsize=8)
    ax[1].plot(e,p-r,'.-');ax[1].set(xlabel='Energy (eV)',ylabel='Fresh - original (eV⁻¹)',title='Independent upstream comparison')
    names=['n_e','n_i','n_i_OML'];ax[2].bar(names,[out[k][0,0] for k in names]);ax[2].set(ylabel='Density (m⁻³)',title='Model-dependent density estimates')
    fig.tight_layout();return fig

def sensitivity_figure(records):
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    levels=sorted(set(r['noise_fraction'] for r in records))
    for a,key,label in zip(ax,['T_e','V_P','eedf_difference'],['Temperature (eV)','Plasma potential (V)','EEDF L2 difference (common energy)']):
        a.boxplot([[r[key] for r in records if r['noise_fraction']==s] for s in levels],tick_labels=[f'{100*s:g}%' for s in levels])
        a.set(xlabel='Added noise / observed current range',ylabel=label,title='Computed perturbation sensitivity')
    fig.tight_layout();return fig
