"""Scientific review plots from actual computed arrays; no reconstruction I/O."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from .physics_model import cell_basis,structure_factors
from .preprocessing import dspacings

def style():
    plt.rcParams.update({'figure.dpi':120,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})

def image(ax,x,title,label='Relative density contrast',limit=None):
    limit=float(np.max(abs(x))) if limit is None else limit
    artist=ax.imshow(x.T,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,extent=(0,1,0,1))
    ax.set(title=title,xlabel='Fractional a',ylabel='Fractional c')
    plt.colorbar(artist,ax=ax,label=label,shrink=.8)

def export(fig,name,directory,numeric):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:fig.savefig(directory/(name+'.'+suffix))
    np.savez_compressed(directory/(name+'_numeric.npz'),**numeric)

def plot_observations(data):
    fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    I=data['intensity'];s=data['sigma_intensity'];d=dspacings(data['hkl'])
    sc=ax[0,0].scatter(data['hkl'][:,0],data['hkl'][:,2],c=I,s=5,cmap='viridis',norm=SymLogNorm(.001))
    ax[0,0].set(xlabel='Miller h',ylabel='Miller l',title='All4508 measured reciprocal locations');fig.colorbar(sc,ax=ax[0,0],label='Signed relative intensity')
    ax[0,1].hist(I,bins=120);ax[0,1].set(yscale='log',xlabel='Signed relative intensity',ylabel='Reflection count',title=f'All measurements; {np.sum(I<0)} negative values retained')
    ax[1,0].scatter(1/d,I,s=4);ax[1,0].set(yscale='symlog',xlabel='Reciprocal spacing [1/angstrom]',ylabel='Signed relative intensity',title='Finite measured spatial-frequency coverage')
    ax[1,1].scatter(I,s,s=4);ax[1,1].set(xscale='symlog',yscale='log',xlabel='Signed relative intensity',ylabel='Reported standard uncertainty',title='Reported uncertainty; no target-derived calibration')
    fig.suptitle('Real indexed diffraction: phases and absolute density are unavailable',fontsize=16)
    return fig,{'hkl':data['hkl'],'intensity':I,'uncertainty':s,'spacing_angstrom':d}

def plot_density(rho):
    fig=plt.figure(figsize=(15,9),layout='constrained');gs=fig.add_gridspec(2,3)
    ax=fig.add_subplot(gs[:,0],projection='3d')
    # A labeled display selection only; all voxels remain scored and stored.
    indices=np.argwhere(rho>=np.quantile(rho,.995));values=rho[tuple(indices.T)]
    xyz=(indices/np.array(rho.shape))@cell_basis().T
    sc=ax.scatter(*xyz.T,c=values,s=7,cmap='magma');fig.colorbar(sc,ax=ax,shrink=.55,label='Relative density contrast')
    ax.set(xlabel='Cartesian x [angstrom]',ylabel='Cartesian y [angstrom]',title='Top0.5% displayed voxels\nFull-cell scoring')
    ax.text2D(.05,.90,'Vertical: Cartesian z [angstrom]',transform=ax.transAxes,fontsize=8)
    basis=cell_basis();corners=np.array([[i,j,k] for i in [0,1] for j in [0,1] for k in [0,1]])
    for i,u in enumerate(corners):
        for v in corners[i+1:]:
            if np.sum(abs(u-v))==1:
                p=np.array([u,v])@basis.T;ax.plot(*p.T,color='.6',lw=.7)
    for axis,b in enumerate([0,6,12,18]):image(fig.add_subplot(gs[axis//2,axis%2+1]),rho[:,b,:],f'Complete section at b={b}/24')
    fig.suptitle('Complete periodic relative electron density; no absolute electron count',fontsize=16)
    return fig,{'density':rho,'displayed_indices':indices,'displayed_values':values,'cartesian_angstrom':xyz}

def plot_parity(fresh,old,modern,aligned_old):
    fig,ax=plt.subplots(2,3,figsize=(16,9),layout='constrained');b=12
    lim=max(np.max(abs(x[:,b,:])) for x in [old,modern,fresh])
    for a,x,title in zip(ax[0],[old,modern,fresh],['Unmodified author NumPy1.26','Qualified author NumPy2.4','Fresh local NumPy2.4']):image(a,x[:,b,:],title,limit=lim)
    image(ax[1,0],(fresh-modern)[:,b,:],'Fresh minus same-backend author\nExactly zero everywhere')
    image(ax[1,1],(aligned_old-old)[:,b,:],'Aligned fresh minus original old backend')
    ax[1,2].hist((aligned_old-old).ravel(),bins=120,histtype='step',label='Old-backend difference');ax[1,2].hist((fresh-modern).ravel(),bins=120,histtype='step',label='Same-backend difference')
    ax[1,2].set(yscale='log',xlabel='Relative density difference',ylabel='All-voxel count',title='Complete165888-voxel comparison');ax[1,2].legend(fontsize=8)
    fig.suptitle('Separate source backends: roundoff changes the nonlinear long trajectory',fontsize=15)
    return fig,{'fresh':fresh,'old':old,'modern':modern,'aligned_fresh_to_old':aligned_old}

def plot_histories(runs,original):
    fig,ax=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for name,result in runs.items():
        h=result['history'];ax[0].plot(h[:,0],h[:,1],label='Fresh '+name,lw=1)
    for name,h in original.items():
        label='GUI20% stop:24 updates' if name=='bound20_history' else name
        ax[1].plot(h[:,0],h[:,1],label=label,lw=1)
    for a in ax:a.axhline(5,color='.4',ls='--',label='Source early-stop5%');a.set(xlabel='Actual completed update',ylabel='Source normalized Rcf [%]');a.legend(fontsize=7)
    ax[0].set_title('All actual fresh trajectories');ax[1].set_title('Independently executed original trajectories')
    fig.suptitle('Fixed finite budgets: no certified convergence or density truth',fontsize=15)
    return fig,{**{'fresh_'+k:v['history'] for k,v in runs.items()},**{'source_'+k:v for k,v in original.items()}}

def plot_controls(runs,names,title):
    fig,ax=plt.subplots(2,len(names),figsize=(5*len(names),9),layout='constrained')
    limit=max(np.max(abs(runs[n]['reconstruction'][:,12,:])) for n in names)
    for k,name in enumerate(names):
        rho=runs[name]['reconstruction'];image(ax[0,k],rho[:,12,:],name,limit=limit)
        image(ax[1,k],np.mean(rho,axis=1),'Full-b mean projection: '+name)
    fig.suptitle(title+'\nSource heuristic origins; no display registration',fontsize=14)
    return fig,{n:runs[n]['reconstruction'] for n in names}

def plot_consistency(data,runs,random_density,quality):
    fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained');I=data['intensity']
    primary=runs['primary']['reconstruction'];pred=abs(structure_factors(primary,data['hkl']))**2
    ax[0,0].scatter(I,pred,s=5);ax[0,0].set(xscale='symlog',yscale='symlog',xlabel='All signed measured intensities',ylabel='Predicted intensity',title='All4508 observations; negative noise floor remains')
    ax[0,1].hist(pred-I,bins=100);ax[0,1].set(yscale='log',xlabel='Predicted minus signed measured intensity',ylabel='Count',title='Full measured residual, no positive-only score')
    image(ax[1,0],random_density[:,12,:],'Fresh initial random phases; not reconstructed structure')
    labels=list(quality);x=np.arange(len(labels));ax[1,1].bar(x,[quality[n]['nrmse'] for n in labels]);ax[1,1].set_xticks(x,labels,rotation=30,ha='right');ax[1,1].set(ylabel='Aligned complete-map range NRMSE',title='Method agreement also rejects arbitrary phases')
    fig.suptitle('Positive-modulus agreement alone cannot recover missing phases',fontsize=15)
    return fig,{'measured':I,'predicted':pred,'random_phase_density':random_density,**{'quality_'+n:np.array(list(v.values())) for n,v in quality.items()}}

def plot_snapshots(result):
    steps=list(result['snapshots']);fig,ax=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    limit=max(np.max(abs(result['snapshots'][s][:,12,:])) for s in steps)
    for a,s in zip(ax.flat,steps):image(a,result['snapshots'][s][:,12,:],f'Fresh update{s}; before source origin roll',limit=limit)
    fig.suptitle('The complete real-space method is actively recomputed',fontsize=15)
    return fig,{'update_'+str(s):result['snapshots'][s] for s in steps}
