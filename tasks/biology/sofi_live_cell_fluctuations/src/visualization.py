import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def relative_l2(a,b):
    denominator=np.linalg.norm(np.asarray(b,dtype=float).ravel())
    if denominator<=0:raise ValueError('Zero reference norm')
    return float(np.linalg.norm((np.asarray(a,dtype=float)-b).ravel())/denominator)

def export_figures(figures,directory,values):
    """Export rendered evidence and numerical source values to caller directory."""
    target=Path(directory);target.mkdir(parents=True,exist_ok=True)
    for name,fig in figures.items():
        for ext in ['png','svg','pdf']:fig.savefig(target/(name+'.'+ext),dpi=180,bbox_inches='tight')
    np.savez_compressed(target/'source_values.npz',**values)

def style():
    plt.rcParams.update({'font.size':11,'axes.titlesize':11,'axes.labelsize':10,
        'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','pdf.fonttype':42})

def panel(ax,image,title,cmap='gray',vmin=None,vmax=None):
    im=ax.imshow(image,cmap=cmap,vmin=vmin,vmax=vmax,origin='upper')
    ax.set_title(title);ax.set_xlabel('Detector column (pixel)');ax.set_ylabel('Detector row (pixel)')
    return im

from .preprocessing import temporal_mean
from .solvers import reconstruct

def run_experiments(data,fresh):
    """Actually compute four recording lengths; full-length result is reused."""
    counts=np.array([25,50,100,200]);series={}
    for n in counts:series[int(n)]=fresh if n==200 else reconstruct(data,frames=int(n))
    errors=np.array([[relative_l2(series[int(n)][k],fresh[k]) for k in ['cumulant2','reconstruction']] for n in counts])
    return {'frames':counts,'length_errors':errors,'short_reconstruction':series[25]['reconstruction'],
        'mean_images':np.stack([temporal_mean(b) for b in data[0]]),
        'block_trace':data[0].mean(axis=(2,3)),'negative_fraction':np.mean(fresh['reconstruction'][0]-3*fresh['cumulant2'][0]**2<0,axis=(1,2))}

def figures(data,fresh,upstream,experiments):
    """Four scientific plates: order, spatial detail, length stability, dynamics."""
    style();figs={};mean=experiments['mean_images'];c2=np.sqrt(np.maximum(fresh['cumulant2'][0],0));c4=np.sign(fresh['reconstruction'][0])*np.abs(fresh['reconstruction'][0])**.25
    fig,axes=plt.subplots(3,3,figsize=(12,11),layout='constrained')
    limits=[(0,np.percentile(mean,99.5)),(0,np.percentile(c2,99.5)),(np.percentile(c4,.5),np.percentile(c4,99.5))]
    for i,block in enumerate([1,10,20]):
        for j,(value,title) in enumerate(zip([mean,c2,c4],['Temporal mean (ADU)','sqrt C2 (ADU)','Fourth root M4 (ADU)'])):
            im=panel(axes[i,j],value[i],f'Block {block}: {title}',vmin=limits[j][0],vmax=limits[j][1]);fig.colorbar(im,ax=axes[i,j],shrink=.75)
    fig.suptitle('Live-cell actin: moment-SOFI enhances actin fluctuation contrast',fontsize=14);figs['01_fluctuation_orders']=fig
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for j,(v,title) in enumerate(zip([mean,c2,c4],['Widefield mean','Second cumulant root','Fourth moment root'])):
        im=panel(axes[0,j],v[0,175:275,175:275],title+'; fixed center ROI');im.set_extent((174.5,274.5,274.5,174.5));fig.colorbar(im,ax=axes[0,j],shrink=.8)
        line=v[0,225,175:275];axes[1,j].plot(np.arange(175,275),line,color='#0072B2',lw=1.5);axes[1,j].set(xlabel='Detector column (pixel)',ylabel='Display intensity (ADU)',title='Row 225, unnormalized profile')
    fig.suptitle('Spatial detail without intensity rescaling in the saved reconstruction',fontsize=14);figs['02_filament_detail']=fig
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for j,(name,color) in enumerate(zip(['Second order','Fourth order'],['#0072B2','#D55E00'])):axes[0].plot(experiments['frames'],experiments['length_errors'][:,j],'-o',label=name,color=color)
    axes[0].set(xlabel='Frames used per block',ylabel='Relative L2 to 200-frame statistic',title='Finite-record statistical convergence');axes[0].legend()
    early=experiments['short_reconstruction'][0,0];early=np.sign(early)*abs(early)**.25
    for ax,v,title in zip(axes[1:],[early,c4[0]],['25 frames, fourth moment root','200 frames, fourth moment root']):
        im=panel(ax,v,title,vmin=limits[2][0],vmax=limits[2][1]);fig.colorbar(im,ax=ax,shrink=.8,label='ADU')
    figs['03_recording_length']=fig
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for i,(b,color) in enumerate(zip([1,10,20],['#0072B2','#009E73','#D55E00'])):axes[0].plot(experiments['block_trace'][i],label=f'Block {b}',color=color)
    axes[0].set(xlabel='Frame within block',ylabel='Field mean (ADU)',title='Actual time traces: drift and biological dynamics');axes[0].legend()
    err=np.abs(fresh['reconstruction'][0,0]-upstream['output_fourth_moment'][0]);im=panel(axes[1],err,'Fresh vs original fourth moment');fig.colorbar(im,ax=axes[1],label='ADU to fourth power')
    figs['04_dynamics_and_parity']=fig
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    fourth_cumulant=fresh['reconstruction'][0]-3*fresh['cumulant2'][0]**2
    signed=np.sign(fourth_cumulant)*np.abs(fourth_cumulant)**.25
    im=panel(axes[0],c4[0],'Selected fourth moment, fourth root',vmin=0,vmax=np.percentile(c4,99.5));fig.colorbar(im,ax=axes[0],shrink=.8,label='ADU')
    lim=np.percentile(np.abs(signed),99)
    im=panel(axes[1],signed[0],'Fourth cumulant: finite-record noise','coolwarm',-lim,lim);fig.colorbar(im,ax=axes[1],shrink=.8,label='signed fourth root (ADU)')
    axes[2].bar([1,10,20],100*np.mean(fourth_cumulant<0,axis=(1,2)),width=4,color='#D55E00');axes[2].set(xticks=[1,10,20],xlabel='Source block',ylabel='Negative fourth cumulant pixels (%)',title='Fourth-cumulant sign instability')
    figs['05_cumulant_sampling_limitation']=fig
    return figs
