"""Presentation figures from supplied computed arrays; no cached solver substitution."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import NAMES
from .solvers import reconstruct

COLORS=('#2563eb','#c2410c','#0f766e','#9333ea','#be185d')

def comparison_figures(raw,result,diagnostics,upstream):
    """Return four figures and all computed comparison arrays. No disk side effects."""
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110})
    figures=[]
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,d in enumerate(diagnostics):
        ax=axes.flat[i];temp=d['native_temperature'];v=d['native_vectors'];code=d['native_code']
        for c,marker,label in [(0,'o','Zero field'),(1,'s','In field'),(2,'^','pTRM check'),(3,'D','Tail check')]:
            selected=code==c
            if selected.any():ax.scatter(temp[selected],np.linalg.norm(v[selected],axis=1),s=21,marker=marker,label=label,alpha=.8)
        ax.set(title=f'{NAMES[i]} · {len(temp)} real readings',xlabel='Treatment temperature (°C)',ylabel='Measured magnitude / initial magnitude')
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.85,'OBSERVATION → UNKNOWN\n\nMeasured magnetic vectors after paired\nthermal zero / laboratory-field treatments\n\n→ original cooling-field strength (µT)\n\nAll five natural specimens are real.\nThe observations are normalized,\nauthor-processed laboratory readings.\nNo physical ground truth is asserted.',va='top',linespacing=1.7)
    legend={}
    for axis in axes.flat[:5]:
        handles,labels=axis.get_legend_handles_labels();legend.update(zip(labels,handles))
    fig.legend(list(legend.values()),list(legend),loc='outside lower center',ncol=4)
    fig.suptitle('Real thermal acquisition: observations and laboratory checks',fontsize=16);figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,d in enumerate(diagnostics):
        ax=axes.flat[i];ax.scatter(d['x'],d['y'],color='#94a3b8',s=22,label='Other temperature');k=d['indices'];ax.scatter(d['x'][k],d['y'][k],color=COLORS[i],s=36,label='Fixed author interval')
        grid=np.linspace(0,max(d['x'])*1.02,100)
        ax.plot(grid,d['intercept']+d['slope']*grid,color=COLORS[i],lw=2,label='Fresh inverse')
        ax.fill_between(grid,d['scat_low'][0]+d['scat_low'][1]*grid,d['scat_high'][0]+d['scat_high'][1]*grid,color=COLORS[i],alpha=.08,label='SCAT acceptance box')
        for kind,marker in [('ptrm','^'),('tail','D')]:
            p=d['checks'][kind]['points']
            if len(p):ax.scatter(p[:,0],p[:,1],marker=marker,s=25,facecolors='none',edgecolors='#1f2937',label='pTRM check' if kind=='ptrm' else 'Tail check')
        ax.set(title=f'{NAMES[i]}: {d["field"]:.3f} ± {d["sigma_field"]:.3f} µT',xlabel='Gained TRM / initial NRM',ylabel='Remaining NRM / initial NRM',ylim=(0,max(d['y'])*1.12))
    ax=axes.flat[-1];k=np.arange(5);ax.bar(k-.18,upstream['reconstruction'][0],.35,label='Reproduced upstream',color='#94a3b8');ax.bar(k+.18,result[0],.35,label='Fresh local',color=COLORS);ax.set_xticks(k,NAMES,rotation=30,ha='right');ax.set(ylabel='Recovered field (µT)',title='Method-reference parity; no GT');ax.legend(fontsize=8)
    legend={}
    for axis in axes.flat[:5]:
        handles,labels=axis.get_legend_handles_labels();legend.update(zip(labels,handles))
    fig.legend(list(legend.values()),list(legend),loc='outside lower center',ncol=6,fontsize=8)
    fig.suptitle('Recover cooling-field strength from the thermal reciprocity slope',fontsize=16);figures.append(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,d in enumerate(diagnostics):
        k=d['indices'];axes[0,0].plot(d['temperature'][k],d['y'][k]-d['prediction'][k],'.-',color=COLORS[i],label=NAMES[i]);axes[0,1].scatter(d['frac'],d['beta'],s=70,color=COLORS[i]);axes[0,1].annotate(NAMES[i],(d['frac'],d['beta']),xytext=(4,3),textcoords='offset points',fontsize=8)
    axes[0,0].axhline(0,color='black',lw=.7);axes[0,0].set(xlabel='Fitted temperature (°C)',ylabel='Normalized NRM residual',title='Fresh observation-space residuals');axes[0,0].legend(fontsize=8)
    axes[0,1].axhline(.1,color='black',ls='--',label='SCAT β threshold');axes[0,1].axvline(.5,color='gray',ls=':');axes[0,1].set(xlabel='FRAC: segment / total vector change',ylabel='Relative slope uncertainty β',title='Fresh QC; original selection fixed before extraction')
    axes[1,0].bar(np.arange(5)-.16,[d['mad'] for d in diagnostics],.32,label='Anchored PCA',color='#2563eb');axes[1,0].bar(np.arange(5)+.16,[d['free_mad'] for d in diagnostics],.32,label='Free PCA',color='#fb923c');axes[1,0].set_xticks(range(5),NAMES,rotation=25);axes[1,0].set(ylabel='Maximum angular deviation (degrees)',title='Directional coherence is a diagnostic');axes[1,0].legend()
    hist=upstream['expected_historical'][0];axes[1,1].bar(range(5),100*(result[0]/hist-1),color=COLORS);axes[1,1].axhline(0,color='black',lw=.7);axes[1,1].set_xticks(range(5),NAMES,rotation=25);axes[1,1].set(ylabel='Difference from historical context (%)',title='Model bias remains despite numerical agreement')
    fig.suptitle('Fit quality, directional checks and physical limitations',fontsize=16);figures.append(fig)
    trimmed,_=reconstruct(raw,trim=1)
    ols=[];scalar=[]
    for i,d in enumerate(diagnostics):
        k=d['indices'];x=d['x'][k];y=d['y'][k];slope=np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2);ols.append(abs(slope)*raw['laboratory_field'][0,i])
        # Deliberately scalar subtraction ignores vector directions: a computed
        # diagnostic, not a physically equivalent reconstruction method.
        temp=d['native_temperature'];code=d['native_code'];v=d['native_vectors'];sx=[]
        for temperature in d['temperature'][k]:
            z=np.flatnonzero((temp==temperature)&(code==0))[0]
            ins=np.flatnonzero((temp==temperature)&(code==1))
            sx.append(0. if temperature==0 else abs(np.linalg.norm(v[ins[0]])-np.linalg.norm(v[z])))
        sx=np.asarray(sx);slope=np.sign(np.sum((sx-sx.mean())*(y-y.mean())))*np.std(y,ddof=1)/np.std(sx,ddof=1);scalar.append(abs(slope)*raw['laboratory_field'][0,i])
    ols=np.array([ols]);scalar=np.array([scalar]);fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');k=np.arange(5)
    for offset,values,color,label in [(-.27,result,'#2563eb','Published SMA'),(-.09,trimmed,'#f59e0b','Drop interval endpoints'),(.09,ols,'#0f766e','OLS diagnostic'),(.27,scalar,'#be185d','Scalar-difference diagnostic')]:axes[0].bar(k+offset,values[0],.18,label=label,color=color)
    axes[0].set_xticks(k,NAMES,rotation=25);axes[0].set(ylabel='Recovered field (µT)',title='All variants recomputed from bound observations');axes[0].legend(fontsize=7)
    for values,color,label in [(trimmed,'#f59e0b','Trim interval'),(ols,'#0f766e','OLS'),(scalar,'#be185d','Scalar difference')]:axes[1].plot(k,100*(values[0]/result[0]-1),'o-',color=color,label=label)
    axes[1].axhline(0,color='black',lw=.7);axes[1].set_xticks(k,NAMES,rotation=25);axes[1].set(ylabel='Change from published SMA (%)',title='Temperature selection and physical vector model matter');axes[1].legend(fontsize=8)
    fig.suptitle('Fresh sensitivity experiments; diagnostic alternatives are explicitly labeled',fontsize=15);figures.append(fig)
    return figures,dict(trimmed=trimmed,ols=ols,scalar_difference=scalar)
