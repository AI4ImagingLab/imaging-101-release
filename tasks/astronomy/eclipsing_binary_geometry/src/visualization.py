"""Maintainer figures; all predictions are computed from supplied parameters."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .physics_model import orbit,blocked,P,T0
from .solvers import models,residuals

NAMES=['WASP','NITES','OED','BYU','KPNO','RV']
PARAMETERS=['r1','r2','inclination (deg)','sqrt(e) sin(omega)','sqrt(e) cos(omega)','mean disk S_J','K1 (km/s)']

def group_fits(groups,point,source,first=0):
    fresh=models(point,groups);upstream=models(source,groups)
    _,fz,_=residuals(point,groups,True);_,uz,_=residuals(source,groups,True)
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for row,gi in enumerate(range(first,first+3)):
        t,y,s,wt,flag=groups[gi];x=((t-T0)/P+.5)%1-.5 if gi<4 else ((t-t.min())*24 if gi==4 else ((t-T0)/P)%1);sort=np.argsort(x);unit='km/s' if gi==5 else 'mag';xlabel='Phase relative to primary conjunction' if gi<4 else ('Hours since first group observation' if gi==4 else 'Orbital phase')
        # Display binning is not used by the fit or evaluator.
        stride=max(1,len(t)//1400);ids=np.arange(0,len(t),stride)
        axes[row,0].plot(x[ids],y[ids],'.',ms=2,alpha=.45,label=f'Observed (display every {stride} rows)')
        axes[row,0].plot(x[sort],upstream[gi][sort]+uz[gi],'--',lw=1.4,label='Original V2 posterior median')
        axes[row,0].plot(x[sort],fresh[gi][sort]+fz[gi],lw=1.2,label='Fresh posterior median')
        axes[row,0].set(title=NAMES[gi],ylabel=unit,xlabel=xlabel)
        residual=y-fresh[gi]-fz[gi];axes[row,1].plot(x[ids],residual[ids],'.',ms=2,alpha=.5)
        axes[row,1].axhline(0,color='k',lw=.7);axes[row,1].set(title='Fresh residual; all rows scored',ylabel=unit,xlabel=xlabel)
        if gi<4:
            axes[row,0].set_xlim(-.04,.04);axes[row,1].set_xlim(-.04,.04)
        if gi<5:axes[row,0].invert_yaxis()
    axes[0,0].legend(fontsize=8);return fig

def posterior_panels(chain,source_quantiles):
    samples=chain[:,500:].reshape(-1,7);fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
    for j,ax in enumerate(axes.flat):
        if j==7:ax.axis('off');continue
        ax.hist(samples[:,j],bins=35,density=True,alpha=.7,color='steelblue')
        ax.axvline(source_quantiles[1,j],color='darkorange',label='Original median')
        ax.axvspan(source_quantiles[0,j],source_quantiles[2,j],color='darkorange',alpha=.15,label='Original 16–84% interval')
        ax.set(title=PARAMETERS[j],ylabel='Marginal density')
        if j==6:ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0,0].legend(fontsize=7);return fig

def traces(chain):
    fig,axes=plt.subplots(7,1,figsize=(12,12),sharex=True,layout='constrained')
    for j,ax in enumerate(axes):
        ax.plot(chain[::5,:,j].T,alpha=.35,lw=.5);ax.axvline(500,color='black',ls='--');ax.set_ylabel(PARAMETERS[j],fontsize=8)
    axes[0].set_title('Fresh chain: ten displayed walkers; all fifty sampled')
    axes[-1].set_xlabel('Ensemble update; first 500 discarded');return fig

def correlations(chain):
    samples=chain[:,500:].reshape(-1,7);corr=np.corrcoef(samples.T)
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained');im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(7),PARAMETERS,rotation=35,ha='right');ax.set_yticks(range(7),PARAMETERS)
    for i in range(7):
        for j in range(7):ax.text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_title('Fresh posterior correlations, finite retained chain');fig.colorbar(im,ax=ax,label='Pearson correlation');return fig

def fixed_point_ablations(point,groups):
    nominal=models(point,groups);cases={'Selected conditional model':nominal}
    no_ltt=list(nominal);r1,r2,_,_,_,sj,_=point;t=groups[4][0];direct=groups[4][4]==1
    sep,front,rv,dx,dy=orbit(t[direct],point,False,True);ratio=sj*(r2/r1)**2;flux=np.ones(len(sep));a=np.flatnonzero(front);b=np.flatnonzero(~front)
    flux[a]-=blocked(r1,r2,sep[a],(0,0),dx[a],dy[a])/(1+ratio)
    flux[b]-=blocked(r2,r1,sep[b],(.1076,.2857),-dx[b],-dy[b])*ratio/(1+ratio)
    no_ltt[4]=-2.5*np.log10(np.interp(t,t[direct],flux));no_ltt[5]=orbit(groups[5][0],point,False)[2];cases['Omit infrared/RV light travel time']=no_ltt
    dark=list(nominal);dark[4]=np.zeros(len(t));cases['Set infrared secondary brightness to zero']=dark
    no_ld=[]
    for gi,(tt,y,s,wt,flag) in enumerate(groups):
        if gi==5:no_ld.append(nominal[5]);continue
        mask=flag==1;d,f,_,x,z=orbit(tt[mask],point,gi==4,True);rat=ratio if gi==4 else 0.;ff=np.ones(len(d));aa=np.flatnonzero(f);bb=np.flatnonzero(~f)
        ff[aa]-=blocked(r1,r2,d[aa],(0,0),x[aa],z[aa])/(1+rat)
        if rat:ff[bb]-=blocked(r2,r1,d[bb],(0,0),-x[bb],-z[bb])*rat/(1+rat)
        if gi in (1,4):mm=-2.5*np.log10(np.interp(tt,tt[mask],ff))
        elif gi==0:mm=np.zeros(len(tt));mm[mask]=-2.5*np.log10(ff)
        else:mm=-2.5*np.log10(ff)
        no_ld.append(mm)
    cases['Replace all limb darkening by uniform disks']=no_ld
    result={}
    for name,preds in cases.items():
        chi=[]
        for m,(tt,y,s,wt,flag) in zip(preds,groups):
            d=y-m;z=np.sum(d*wt)/np.sum(wt);chi.append(float(np.sum((d-z)**2*wt)))
        result[name]=np.array(chi)
    return result

def quality_figure(source_chi,fresh_chi,ablations):
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');x=np.arange(6)
    axes[0].bar(x-.17,source_chi,.34,label='Original V2 median');axes[0].bar(x+.17,fresh_chi,.34,label='Fresh median');axes[0].set_xticks(x,NAMES);axes[0].set_ylabel('Profiled chi-square');axes[0].legend()
    labels=list(ablations);values=[v.sum() for v in ablations.values()];axes[1].barh(range(len(labels)),np.array(values)-values[0]);axes[1].set_yticks(range(len(labels)),labels,fontsize=8);axes[1].set_xlabel('Change in total chi-square at fixed fresh point');axes[1].set_title('Computed conditional ablations; no refitting')
    return fig
