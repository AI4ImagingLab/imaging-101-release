"""Publication review plots from measured data and genuinely computed results."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import predict,diagonalize,magnetization
from .preprocessing import preliminary_profile

COLORS=['#176b9b','#d56b2d','#388263']
def metrics(prediction,data):
    rows=[]
    for k in range(2):
        chi=0
        for t in range(3):
            m=data['fit_mask'][t];y=data['intensity'][t,m];x=prediction['neutron_spectrum'][0,k,t,m]
            c=float(np.sum(((x-y)/data['reported_error'][t,m])**2));chi+=c
            rows.append(dict(model=k,temperature=float(data['temperature'][t]),nfit=int(sum(m)),nrmse=float(np.sqrt(np.mean((x-y)**2))/np.ptp(y)),ncc=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))),chi2=c))
        rows.append(dict(model=k,total_chi2=chi,conditional_chi2_per_dof=chi/274,source_display_chi2=chi/524))
    return rows

def figures(prediction,diagnostics,data,upstream):
    plt.rcParams.update({'figure.dpi':110,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    figs=[];q=prediction['reconstruction'][0];spec=prediction['neutron_spectrum'][0];width=prediction['resolution_coefficients'][0]
    def new(*args,**kwargs):
        f,ax=plt.subplots(*args,**kwargs);figs.append(f);return f,ax
    f,ax=new(1,3,figsize=(13,3.5))
    for t,a in enumerate(ax):
        a.errorbar(data['energy'][t],data['intensity'][t],data['reported_error'][t],fmt='.',ms=3,color=COLORS[t],alpha=.7)
        m=data['fit_mask'][t];a.scatter(data['energy'][t,m],data['intensity'][t,m],s=5,c='k',label=f'Fit samples: {sum(m)}')
        a.set(title=f'Measured powder cut: {data["temperature"][t]:g} K',xlabel='Energy transfer (meV)',ylabel='Intensity (author units)');a.legend(fontsize=8)
    f.tight_layout()
    f,ax=new(1,2,figsize=(11,3.6));x=data['energy'][0];p=diagnostics['voigt_parameters']
    ax[0].errorbar(x,data['intensity'][0],data['reported_error'][0],fmt='.',ms=3,label='Measured 1.8 K');ax[0].plot(x,preliminary_profile(x,*p),color=COLORS[1],label='Fresh 17-parameter Voigt fit');ax[0].set(xlim=(-1,7),xlabel='Energy transfer (meV)',ylabel='Intensity (author units)',title='Active same-data peak estimation');ax[0].legend(fontsize=8)
    centers=p[1:15:3];fw=p[2:15:3];unc=np.sqrt(np.diag(diagnostics['voigt_covariance']));ax[1].errorbar(centers,fw,xerr=unc[1:15:3],yerr=unc[2:15:3],fmt='o',color=COLORS[0],label='Fitted peaks; elastic excluded from line');xx=np.linspace(-.2,7,100);ax[1].plot(xx,width[0]*xx+width[1],color=COLORS[1],label='Fresh inelastic width regression');ax[1].set(xlabel='Relative peak energy (meV)',ylabel='Gaussian FWHM (meV)',title='Resolution is inferred, not supplied');ax[1].legend(fontsize=8);f.tight_layout()
    f,ax=new(3,2,figsize=(12,10),sharex=True)
    for t in range(3):
        for k in range(2):
            a=ax[t,k];E=data['energy'][t];a.errorbar(E,data['intensity'][t],data['reported_error'][t],fmt='.',ms=3,color='#777777',label='Measured');a.plot(E,spec[k,t],color=COLORS[k],label='Fresh complete refinement');a.plot(E,upstream['final_spectra'][k,t],ls='--',lw=1,color='k',label='Independent original rerun');a.set(xlim=(-6,7),ylim=(-.0003,.03),title=f'{["Easy-plane","Easy-axis"][k]} hypothesis, {data["temperature"][t]:g} K',ylabel='Intensity (author units)',xlabel='Energy transfer (meV)')
            if t==0:a.legend(fontsize=8)
            a.text(.02,.88,'Elastic peak off-scale; full data in overview',transform=a.transAxes,fontsize=7,color='#555555')
    f.suptitle('Both neutron-consistent branches; each uses its own fitted expansion',y=1.002);f.tight_layout()
    f,ax=new(1,3,figsize=(13,3.7))
    for t,a in enumerate(ax):
        m=data['fit_mask'][t];E=data['energy'][t,m]
        for k in range(2):a.plot(E,(spec[k,t,m]-data['intensity'][t,m])/data['reported_error'][t,m],'.',ms=4,color=COLORS[k],label=['Easy-plane','Easy-axis'][k])
        a.axhspan(-1,1,color='grey',alpha=.15);a.axhline(0,color='k',lw=.7);a.set(title=f'{data["temperature"][t]:g} K: retained residuals',xlabel='Energy transfer (meV)',ylabel='Residual / reported error');a.legend(fontsize=8)
    f.tight_layout()
    f,ax=new(1,2,figsize=(11,3.8))
    for k,a in enumerate(ax):
        E=data['energy'][0];a.plot(E,data['intensity'][0],'.',color='#777777',ms=3,label='Measured');a.plot(E,upstream['initial_spectra'][k,0],ls=':',color='#ad4455',label='Published coarse starting hypothesis');a.plot(E,spec[k,0],color=COLORS[k],label='Fresh refined');keep=(E>=.5)&(E<=7);height=1.08*max(np.max(data['intensity'][0,keep]),np.max(upstream['initial_spectra'][k,0,keep]),np.max(spec[k,0,keep]));a.set(xlim=(.5,7),ylim=(0,height),title=f'Model {k+1}: initial vs refined inelastic peaks',xlabel='Energy transfer (meV)',ylabel='Intensity (author units)');a.legend(fontsize=8)
    f.tight_layout()
    f,ax=new(1,2,figsize=(11,3.8))
    fields=np.r_[np.arange(0,.7,.05),np.arange(.7,7.1,.1)];mag=[]
    for k,a in enumerate(ax):
        curves=np.array([[-magnetization(q[k,:6],.5,[h,0,0])[0],-magnetization(q[k,:6],.5,[0,0,h])[2]] for h in fields]);mag.append(curves)
        for j,(axis,color) in enumerate([('ab',COLORS[0]),('c',COLORS[1])]):
            a.plot(data['field_'+axis],data['moment_'+axis],'.',ms=3,alpha=.4,color=color,label=f'Independent measurement: {axis}');a.plot(fields,curves[:,j],color=color,label=f'Fresh single-ion: {axis}')
        a.set(xlim=(0,7),ylim=(0,9),title=['Easy-plane: selected by measured contrast','Easy-axis: neutron ambiguity rejected'][k],xlabel='Magnetic field (T)',ylabel='Physical moment (Bohr magnetons / Er)');a.legend(fontsize=7)
    f.suptitle('Author supplied 0.5 K directional data; single-ion model omits exchange',y=1.01);f.tight_layout()
    f,ax=new(1,2,figsize=(11,3.8))
    for k,a in enumerate(ax):
        _,e,_=diagonalize(q[k,:6]);observed=p[4:15:3]
        for j,ev in enumerate(e[::2][:5]):a.hlines(ev,j-.3,j+.3,color=COLORS[k],lw=3);a.text(j,ev+.07,f'{ev:.3f}',ha='center',fontsize=8)
        a.plot(np.arange(1,5),observed,'kx',ms=7,label='Phenomenological measured relative peaks');a.set(xlim=(-.6,4.6),ylim=(-.2,6.2),xlabel='Kramers doublet index',ylabel='Energy above ground (meV)',title=f'Model {k+1}: measured low-energy manifold');a.legend(fontsize=8,loc='lower right')
    f.tight_layout()
    f,ax=new(1,2,figsize=(11,3.7));rows=metrics(prediction,data)
    for k in range(2):
        r=[v for v in rows if v['model']==k and 'nrmse' in v];ax[0].bar(np.arange(3)+(k-.5)*.34,[v['nrmse'] for v in r],width=.34,color=COLORS[k],label=['Easy-plane','Easy-axis'][k]);last=[v for v in rows if v['model']==k and 'total_chi2' in v][0];ax[1].bar(np.arange(2)+(k-.5)*.34,[last['source_display_chi2'],last['conditional_chi2_per_dof']],width=.34,color=COLORS[k])
    ax[0].set(xticks=np.arange(3),xticklabels=['1.8 K','15 K','100 K'],ylabel='Dynamic-range NRMSE',title='Residuals on fixed fit masks');ax[0].legend();ax[1].set(xticks=np.arange(2),xticklabels=['Source display: chi2 / 524','Conditional nominal: chi2 / 274'],ylabel='Weighted residual statistic',title='Systematic discrepancy remains');f.tight_layout()
    f,ax=new(1,2,figsize=(11,3.7))
    for k in range(2):
        rel=(q[k]-upstream['final_parameters'][k])/abs(upstream['final_parameters'][k]);ax[0].plot(np.arange(15),rel,'o-',ms=3,color=COLORS[k],label=f'Model {k+1} fresh vs original');ax[1].plot(np.arange(15),upstream['default_to_extended_relative'][k],'o-',ms=3,color=COLORS[k],label=f'Model {k+1}: original budget sensitivity')
    ax[0].axhline(0,color='k',lw=.7);ax[0].set(xlabel='Parameter column',ylabel='Signed relative difference',title='Independent original parity; zero means equality');ax[0].legend(fontsize=8);ax[1].set(xlabel='Parameter column',ylabel='Absolute relative change',title='Default budget vs converged native method');ax[1].legend(fontsize=8);f.tight_layout()
    f,ax=new(1,2,figsize=(11,3.8))
    for k in range(2):
        _,e,v=diagonalize(q[k,:6]);weights=np.exp(-e[None,:]/(.0861733*data['temperature'][:,None]));weights/=weights.sum(axis=1)[:,None];im=ax[k].imshow(weights[:,::2]+weights[:,1::2],aspect='auto',origin='lower',cmap='viridis',vmin=0,vmax=1);f.colorbar(im,ax=ax[k],label='Doublet population');ax[k].set(xticks=range(8),yticks=range(3),yticklabels=['1.8','15','100'],xlabel='Doublet index (higher levels are extrapolations)',ylabel='Temperature (K)',title=f'Model {k+1}: thermal doublet populations')
    f.tight_layout();return figs
