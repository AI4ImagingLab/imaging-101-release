"""Scientific comparisons; all plotted estimates supplied by fresh callers."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import structure_model,isotropic_speed_density
COLORS=['#0072B2','#D55E00']
LABELS=['01umE recording','16umE recording']

def figures(frames,results,upstream,diagnostics):
    """Return four distinct review figures and exact numeric plotting arrays."""
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fs=[];data={}
    fig,axes=plt.subplots(2,3,figsize=(13,7.4),layout='constrained')
    for c,a in enumerate(results):
        q=a['q'][a['selected']];tau=a['tau'];pars=a['parameters'][a['selected']];d=a['data'][a['selected']]
        ax=axes[c,0];im=ax.imshow(frames[0,c,0],cmap='gray',vmin=0,vmax=255,extent=[-.5,127.5,127.5,-.5]);ax.set(title=LABELS[c]+' / native frame 0',xlabel='x (pixel)',ylabel='y (pixel)');fig.colorbar(im,ax=ax,label='Intensity (ADU)',shrink=.75)
        ax=axes[c,1];im=ax.pcolormesh(tau,q,d,cmap='viridis',shading='nearest');ax.set(xscale='log',xlabel='Delay (s)',ylabel='q (rad/µm)',title='Full-movie structure function');fig.colorbar(im,ax=ax,label='D (ADU²)',shrink=.75)
        ax=axes[c,2]
        for j in [0,12,25,38]:
            color=plt.cm.plasma(j/38);ax.semilogx(tau,d[j],'.',color=color,ms=3);ax.semilogx(tau,structure_model(tau,pars[j]),'-',color=color,label=f'q={q[j]:.2f}')
        ax.set(xlabel='Delay (s)',ylabel='D (ADU²)',title='Fresh compressed-exponential fits');ax.legend(fontsize=8)
        data[f'condition{c}_q']=q;data[f'condition{c}_tau']=tau;data[f'condition{c}_D']=d;data[f'condition{c}_parameters']=pars;data[f'condition{c}_native_frame0']=frames[0,c,0]
    fs.append(('native_structure_fits',fig))
    fig,axes=plt.subplots(2,2,figsize=(11,7.8),layout='constrained')
    for c,a in enumerate(results):
        q=data[f'condition{c}_q'];p=data[f'condition{c}_parameters'];ref=upstream[c];col=COLORS[c]
        axes[0,0].plot(q,p[:,2],'o',mfc='none',color=col,label=f'{LABELS[c]}, fresh');axes[0,0].plot(q,a['speed']*q,'--',color=col,label=f'v={a["speed"]:.2f} µm/s');axes[0,0].plot(q,ref[:,2],'.',color=col,ms=2)
        axes[0,1].plot(q,p[:,0],'-',color=col,label=LABELS[c]+' A');axes[0,1].plot(q,p[:,1],'--',color=col,label=LABELS[c]+' fitted B')
        axes[1,0].plot(q,p[:,3],'-',color=col,label=LABELS[c]);axes[1,0].plot(q,ref[:,3],'.',color=col,ms=2)
        for j,name in enumerate(['A','B','Gamma','beta']):
            axes[1,1].scatter(ref[:,j],p[:,j],s=11,alpha=.65,label=f'{c}: {name}')
    axes[0,0].set(xlabel='q (rad/µm)',ylabel='Gamma (s⁻¹)',title='Rate spectra and effective ballistic scales');axes[0,0].legend(fontsize=8)
    axes[0,1].set(xlabel='q (rad/µm)',ylabel='A, fitted B (ADU²)',title='Contrast and effective background');axes[0,1].legend(fontsize=8)
    axes[1,0].set(xlabel='q (rad/µm)',ylabel='Compression exponent beta',title='Variable / fixed compression conventions');axes[1,0].legend(fontsize=8)
    axes[1,1].set(xscale='symlog',yscale='symlog',xlabel='Original-method parameter (native units)',ylabel='Fresh parameter (same units)',title='Independent original run / fresh parity');axes[1,1].plot([0,1000],[0,1000],'k--',lw=1);axes[1,1].legend(fontsize=7,ncol=2)
    fs.append(('relaxation_parameter_spectra',fig))
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for c,a in enumerate(results):
        q=data[f'condition{c}_q'];tau=a['tau'];p=data[f'condition{c}_parameters'];d=data[f'condition{c}_D']
        for j in range(len(q)):
            f=1-(d[j]-p[j,1])/p[j,0];axes[c].semilogx(q[j]*tau,f,'.',color=plt.cm.plasma(j/38),ms=2,alpha=.6)
        qt=np.geomspace(.005,1,200);axes[c].semilogx(qt,np.exp(-(qt*a['speed'])**1.35),'k-',label='Common beta=1.35 model');axes[c].set(xlim=(.005,1),ylim=(-.12,1.1),xlabel='q × delay (s rad/µm)',ylabel='Effective correlation f',title=LABELS[c]);axes[c].legend(fontsize=8)
        data[f'condition{c}_scaled_correlation']=1-(d-p[:,1,None])/p[:,0,None]
    norm=plt.Normalize(data['condition0_q'].min(),data['condition0_q'].max());fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='plasma'),ax=axes[:2],label='q (rad/µm)',shrink=.7)
    n=np.linspace(0,7.5,100);density=isotropic_speed_density(n);axes[2].plot(n,density,color='#009E73');axes[2].set(xlabel='v / characteristic scale',ylabel='Dimensionless density',title='Isotropic 3-D model interpretation\nbeta=1.35; no trajectory measurements',ylim=(0,None));data['model_relative_speed']=n;data['model_speed_density']=density
    fs.append(('scaled_correlations_model_distribution',fig))
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    names=['Canonical','beta=1','beta=1.35','Unweighted']
    for c,a in enumerate(results):
        curves=diagnostics[c];q=data[f'condition{c}_q'];tau=a['tau'];d=data[f'condition{c}_D'];errors=[];speeds=[]
        for name,p in zip(names,curves):
            fitted=np.array([structure_model(tau,v) for v in p]);errors.append(np.linalg.norm(fitted-d)/np.linalg.norm(d));speeds.append(np.exp(np.mean(np.log(p[:,2]/q))))
            axes[c,0].plot(q,p[:,2],label=name)
        axes[c,0].set(xlabel='q (rad/µm)',ylabel='Gamma (s⁻¹)',title=LABELS[c]+' / controlled refits');axes[c,0].legend(fontsize=8)
        axes[c,1].bar(np.arange(4),errors,color=[COLORS[c],'#999999','#009E73','#CC79A7']);axes[c,1].set(xticks=np.arange(4),xticklabels=names,ylabel='Unweighted curve relative L2',title='Same full-movie curves; computed fits')
        for i,(err,v) in enumerate(zip(errors,speeds)):axes[c,1].text(i,err,f'v={v:.2f}',ha='center',va='bottom',fontsize=8)
        data[f'condition{c}_diagnostic_parameters']=np.array(curves);data[f'condition{c}_diagnostic_fit_relative_l2']=np.array(errors);data[f'condition{c}_diagnostic_speed']=np.array(speeds)
    fs.append(('controlled_model_weight_comparisons',fig))
    return fs,data
