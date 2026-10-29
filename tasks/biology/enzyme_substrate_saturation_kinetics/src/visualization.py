"""Review plots from actual measured observations and freshly computed results."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import rate,profile,jacobian
from .solvers import intersections
LABELS=('Direct NLS','LB','Weighted LB','ECB (upstream bug)','Hanes-Woolf','Eadie-Hofstee')
COLORS=('#176b87','#b93838','#2c8d62','#b78520','#8756a4','#687b89')

def observations(s,v,fresh,original):
    fig,axs=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    grid=np.geomspace(s.min(),s.max(),350)
    for g,ax in enumerate(axs.flat):
        ax.scatter(s,v[g],c='black',s=25,label='Measured')
        ax.plot(grid,rate(grid,fresh[g]),color=COLORS[0],label='Fresh direct NLS')
        ax.plot(s,rate(s,original[g]),'x',color='#d36a26',label='Original R run')
        ax.set(xscale='log',xlabel='ONPG (mM)',ylabel='Rate (micromolar/min)',title=f'Group {g+1}')
        ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=8)
    fig.suptitle('All 80 measured initial rates: active reconstruction and original reference')
    return fig

def parameters(comparison):
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained');x=np.arange(8)
    for j,ax in enumerate(axs):
        for m,label in enumerate(LABELS):ax.bar(x+(m-2.5)*.13,comparison[m,:,j],width=.125,color=COLORS[m],label=label)
        ax.axhline(0,color='black',lw=.6);ax.set_xticks(x,np.arange(1,9));ax.set(xlabel='Experimental group',ylabel=('Km (mM)','Vmax (micromolar/min)')[j])
    axs[0].legend(fontsize=8,ncol=2)
    fig.suptitle('Six methods on all complete series; negative estimates are retained')
    return fig

def errors(s,v,comparison):
    predictions=np.array([[rate(s,p) for p in method] for method in comparison])
    rmse=np.sqrt(np.mean((predictions-v[None])**2,axis=-1))
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    im=axs[0].imshow(np.log10(1+rmse),aspect='auto',cmap='magma')
    axs[0].set_yticks(range(6),LABELS);axs[0].set_xticks(range(8),range(1,9));axs[0].set(xlabel='Group',title='Observed-rate RMSE (numbers in micromolar/min)')
    for m in range(6):
        for g in range(8):axs[0].text(g,m,f'{rmse[m,g]:.1f}',ha='center',va='center',fontsize=8,color='black' if np.log10(1+rmse[m,g])>.7*np.max(np.log10(1+rmse)) else 'white')
    fig.colorbar(im,ax=axs[0],label='log10(1 + RMSE / [1 micromolar/min])')
    residual=predictions[0]-v
    lim=np.max(abs(residual));im=axs[1].imshow(residual,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    axs[1].set_yticks(range(8),range(1,9));axs[1].set_xticks(range(10),[str(x) for x in s],rotation=45)
    axs[1].set(xlabel='ONPG (mM)',ylabel='Group',title='Fresh direct fit minus measured rate')
    fig.colorbar(im,ax=axs[1],label='Residual (micromolar/min)')
    return fig,rmse

def paper_comparison(s,v,comparison):
    g=5;fig,axs=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    x=np.round(1/s,4);y=np.round(1/v[g],4)
    for ax in axs[:2]:
        ax.scatter(x,y,c='black',label='Measured reciprocal')
        grid=np.linspace(0,20,300)
        for m in [1,2]:
            k,vm=comparison[m,g];ax.plot(grid,1/vm+k/vm*grid,color=COLORS[m],label=LABELS[m])
        ax.set(xlabel='1 / ONPG (1/mM)',ylabel='1 / rate (min/micromolar)')
    axs[0].set_title('Paper Table2 / Figure4: group6')
    axs[1].set(xlim=(0,.5),ylim=(0,.015),title='High-weight region, expanded')
    axs[0].legend(fontsize=8)
    grid=np.linspace(0,30,350);axs[2].scatter(s,v[g],c='black',label='Measured')
    for m in [0,1,2]:axs[2].plot(grid,rate(grid,comparison[m,g]),color=COLORS[m],label=LABELS[m])
    axs[2].set(xlabel='ONPG (mM)',ylabel='Rate (micromolar/min)',title='Fresh fitted response curves');axs[2].legend(fontsize=8)
    return fig

def deletion(s,v,full,removed):
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');grid=np.linspace(0,30,400)
    axs[0].scatter(s,v[1],c='black',label='All measured points')
    axs[0].scatter(s[7],v[1,7],s=100,facecolors='none',edgecolors='red',label='Author-removed point')
    axs[0].plot(grid,rate(grid,full[0,1]),label='Complete primary',color=COLORS[0])
    axs[0].plot(grid,rate(grid,removed[0,1]),label='Author removal',color='#db7b22')
    axs[0].set(xlabel='ONPG (mM)',ylabel='Rate (micromolar/min)',title='Group2: original datum remains primary');axs[0].legend(fontsize=8)
    x=np.arange(6)
    for j,ax in enumerate(axs[1:]):
        ax.bar(x-.18,full[:,1,j],.36,label='Complete');ax.bar(x+.18,removed[:,1,j],.36,label='Author removal')
        ax.set_xticks(x,LABELS,rotation=45,ha='right');ax.set_ylabel(('Km (mM)','Vmax (micromolar/min)')[j]);ax.axhline(0,color='black',lw=.6)
    axs[1].legend(fontsize=8)
    return fig

def profiles(s,v,primary):
    fig,axs=plt.subplots(2,4,figsize=(15,7),layout='constrained');grid=np.geomspace(.05,1000,500)
    for g,ax in enumerate(axs.flat):
        _,rss=profile(s,v[g],grid);base=np.sum((v[g]-rate(s,primary[g]))**2)
        ax.semilogx(grid,(rss-base)/max(base,1e-15),color=COLORS[0]);ax.axvline(primary[g,0],color='#db7b22',ls='--')
        ax.set(xlabel='Fixed Km (mM)',ylabel='(profile RSS - fitted RSS) / fitted RSS',title=f'Group {g+1}',ylim=(-.05,2))
        ax.grid(alpha=.15)
    fig.suptitle('Positive-Km profiles: Vmax eliminated analytically; not a posterior or confidence limit')
    return fig

def sensitivity_plot(s,v,primary,loo):
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for j,ax in enumerate(axs[0]):
        rel=100*(loo[:,:,j]/primary[:,None,j]-1);lim=np.nanmax(abs(rel))
        im=ax.imshow(rel,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(10),s,rotation=45);ax.set_yticks(range(8),range(1,9))
        ax.set(xlabel='Omitted concentration (mM)',ylabel='Group',title=('Km','Vmax')[j]+' change after removing one measured point')
        fig.colorbar(im,ax=ax,label='Change (%)')
    se=[];condition=[]
    for observed,p in zip(v,primary):
        jac=jacobian(s,p);rss=np.sum((observed-rate(s,p))**2)
        se.append(np.sqrt(np.diag(np.linalg.inv(jac.T@jac)*rss/(len(s)-2))))
        condition.append(np.linalg.cond(jac*p[None,:]))
    se=np.array(se)
    for j,ax in enumerate(axs[1]):
        ax.errorbar(np.arange(1,9),primary[:,j],yerr=1.96*se[:,j],fmt='o',capsize=4,color=COLORS[0]);ax.set(xlabel='Group',ylabel=('Km (mM)','Vmax (micromolar/min)')[j],title='Estimate ±1.96 linearized SE (assumption-dependent)')
    return fig,se,np.asarray(condition)

def ecb_diagnostic(s,v,comparison,corrected):
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    grid=np.linspace(0,5,100)
    for slope,intercept in [(3,6),(2,8)]:axs[0].plot(grid,slope*grid+intercept)
    axs[0].scatter([2],[12],label='Correct intersection',color='green');axs[0].scatter([2],[14],label='Original formula',color='red')
    axs[0].set(xlabel='Km (mM)',ylabel='Vmax (micromolar/min)',title='Independent exact case: S=(2,4), v=(6,8)');axs[0].legend(fontsize=8)
    for values,label,color in [(comparison[3,:,1],'Original ECB (bug)',COLORS[3]),(corrected[:,1],'Corrected geometry','#338460'),(comparison[0,:,1],'Direct NLS',COLORS[0])]:axs[1].plot(range(1,9),values,'o-',label=label,color=color)
    axs[1].set(xlabel='Group',ylabel='Vmax (micromolar/min)',title='Full actual data: bug does affect ECB');axs[1].legend(fontsize=8)
    p=intersections(s,v[5]);q=intersections(s,v[5],True)
    finite=np.isfinite(q[:,1])&np.isfinite(p[:,1])
    axs[2].scatter(q[finite,1],p[finite,1],s=18,color=COLORS[3]);lo=min(q[finite,1].min(),p[finite,1].min());hi=max(q[finite,1].max(),p[finite,1].max())
    axs[2].plot([lo,hi],[lo,hi],'--',color='gray');axs[2].set(xlabel='Correct pair Vmax (micromolar/min)',ylabel='Original pair Vmax (micromolar/min)',title=f'Group6: {finite.sum()}/45 finite intersections')
    axs[2].text(.03,.95,f'{(~finite).sum()} parallel pair(s): infinite, not plotted',transform=axs[2].transAxes,va='top',fontsize=9)
    return fig
