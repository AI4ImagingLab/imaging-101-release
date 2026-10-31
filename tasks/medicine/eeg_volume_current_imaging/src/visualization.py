from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct
from .physics_model import forward
def error(x,y):return float(np.linalg.norm(x-y)/np.linalg.norm(y))
def projected_power(positions,power,axes):
    coordinates,inverse=np.unique(positions[:,axes],axis=0,return_inverse=True)
    total=np.zeros(len(coordinates));np.add.at(total,inverse,power)
    return coordinates,total
def run_experiments(data,upstream):
    nominal,stage=reconstruct(data);values=[];solutions=[]
    y=data['measurements']-data['measurements'].mean(axis=0)
    for lam in [1/900,1/90,1/9,1,10]:
        out,_=reconstruct(data,lambda2=lam);j=out['reconstruction'][0];solutions.append(j)
        values.append([lam,np.linalg.norm(j),error(forward(j,data['lead_field']),y)])
    x=nominal['reconstruction'][0];pred=forward(x,data['lead_field']);power=np.mean(np.sum(x*x,axis=1),axis=1)
    return {'nominal':x,'predicted':pred,'observed':y,'power':power,'sweep':np.asarray(values),'solutions':np.asarray(solutions),'singular':stage['singular_values'],'metrics':{'upstream_relative_l2':error(nominal['reconstruction'],upstream['reconstruction']),'observation_residual':error(pred,y),'explained_variance':1-error(pred,y)**2}}
def make_figures(data,e):
    figs=[];time=data['times'];pos=data['positions']*1000
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    potential_limit=float(max(np.max(abs(e['observed'])),np.max(abs(e['predicted'])))*1e6)
    im=ax[0,0].imshow(e['observed']*1e6,aspect='auto',extent=[time[0],time[-1],64,0],cmap='RdBu_r',vmin=-potential_limit,vmax=potential_limit);ax[0,0].set(title='Real observed average-reference EEG',xlabel='Time (s)',ylabel='Electrode index');fig.colorbar(im,ax=ax[0,0],label='Potential (µV)')
    im=ax[0,1].imshow(e['predicted']*1e6,aspect='auto',extent=[time[0],time[-1],64,0],cmap='RdBu_r',vmin=-potential_limit,vmax=potential_limit);ax[0,1].set(title='Fresh source field forward prediction',xlabel='Time (s)',ylabel='Electrode index');fig.colorbar(im,ax=ax[0,1],label='Potential (µV)')
    for idx,color in zip([0,20,40],['tab:blue','tab:orange','tab:green']):
        ax[1,0].plot(time,e['observed'][idx]*1e6,color=color,alpha=.6,lw=.7,label=f'Measured {idx}')
        ax[1,0].plot(time,e['predicted'][idx]*1e6,color=color,ls='--',lw=1,label=f'Predicted {idx}')
    ax[1,0].set(xlabel='Time (s)',ylabel='Potential (µV)',title='Measured and predicted electrode traces');ax[1,0].legend(fontsize=7)
    ax[1,1].semilogy(e['singular'],'o-',ms=3);ax[1,1].set(title='Whitened normalized acquisition singular values',xlabel='Mode index',ylabel='Singular value');figs.append(fig)
    fig=plt.figure(figsize=(12,8),constrained_layout=True)
    for i,(axes,title) in enumerate([((0,1),'Axial'),((0,2),'Coronal'),((1,2),'Sagittal')]):
        coordinates,total=projected_power(pos,e['power'],axes)
        ax=fig.add_subplot(2,2,i+1);sc=ax.scatter(coordinates[:,0],coordinates[:,1],c=total*1e18,s=30,cmap='magma');ax.set(title=title+' current power summed through volume',xlabel='xyz'[axes[0]]+' (mm)',ylabel='xyz'[axes[1]]+' (mm)',aspect='equal');fig.colorbar(sc,ax=ax,label='Summed mean squared moment (nA² m²)')
    ax=fig.add_subplot(2,2,4,projection='3d');stride=3;vectors=np.mean(e['nominal'],axis=2);magn=np.linalg.norm(vectors,axis=1);unit=vectors/np.maximum(magn[:,None],1e-30);ax.quiver(pos[::stride,0],pos[::stride,1],pos[::stride,2],unit[::stride,0],unit[::stride,1],unit[::stride,2],length=6,color='teal');ax.set(title='Mean current directions on full spatial grid',xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)');figs.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);s=e['sweep']
    ax[0,0].loglog(s[:,0],s[:,2],'o-');ax[0,0].set(title='Actually reconstructed regularization sweep',xlabel='Regularization lambda²',ylabel='Relative EEG residual')
    ax[0,1].loglog(s[:,0],s[:,1]*1e9,'o-');ax[0,1].set(title='Minimum-norm tradeoff',xlabel='Regularization lambda²',ylabel='Full current norm (nA m)')
    for i,name in enumerate(['Weak regularization','Strong regularization']):
        x=e['solutions'][0 if i==0 else -1];p=np.mean(np.sum(x*x,axis=1),axis=1)*1e18;coordinates,total=projected_power(pos,p,(0,1));sc=ax[1,i].scatter(coordinates[:,0],coordinates[:,1],c=total,s=25,cmap='magma');ax[1,i].set(title=name+' depth-summed source power',xlabel='x (mm)',ylabel='y (mm)',aspect='equal');fig.colorbar(sc,ax=ax[1,i],label='Summed mean squared moment (nA² m²)')
    figs.append(fig);return figs
def export(figures,e,directory):
    p=Path(directory);p.mkdir(parents=True,exist_ok=True)
    for i,f in enumerate(figures,1):
        for ext in ['png','svg','pdf']:f.savefig(p/f'figure_{i}.{ext}',dpi=140)
    np.savez_compressed(p/'numeric_sources.npz',power=e['power'],sweep=e['sweep'],predicted=e['predicted'],observed=e['observed'],singular=e['singular'])
    (p/'metrics.json').write_text(json.dumps(e['metrics'],indent=2),encoding='utf8')
