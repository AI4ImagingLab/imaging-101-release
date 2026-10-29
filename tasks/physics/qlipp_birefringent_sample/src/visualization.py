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

import torch
from .solvers import reconstruct

def angular_error(a,b,retardance):
    delta=np.arctan2(np.sin(2*(a-b)),np.cos(2*(a-b)))/2;w=np.sin(retardance)**2
    return float(np.sqrt(np.sum(w*delta**2)/np.sum(w)))

def run_experiments(data,fresh):
    """Actually rerun background ablation and independent dark-offset perturbations."""
    torch.set_num_threads(1);uncorrected=reconstruct(data,correct_background=False)
    offsets=np.array([80.58,100.58,120.58]);sensitivity=[]
    for d in offsets:
        r=fresh if d==100.58 else reconstruct(data,black_level=float(d))
        sensitivity.append([relative_l2(r['reconstruction'],fresh['reconstruction']),angular_error(r['orientation'],fresh['orientation'],fresh['reconstruction'])])
    return {'uncorrected_retardance':uncorrected['reconstruction'],'uncorrected_orientation':uncorrected['orientation'],
        'dark_offsets':offsets,'dark_sensitivity':np.array(sensitivity),
        'dop_above_one':np.mean(fresh['depolarization'][0]>1,axis=(2,3)),
        'z_median_retardance':np.median(fresh['reconstruction'][0],axis=(2,3)),
        'z_median_transmittance':np.median(fresh['transmittance'][0],axis=(2,3))}

def figures(data,fresh,upstream,e):
    style();figs={};delta=fresh['reconstruction'][0];theta=fresh['orientation'][0];z=8
    fig,axes=plt.subplots(1,4,figsize=(13,3.5),layout='constrained')
    for c,ax in enumerate(axes):
        im=panel(ax,data['intensities'][0,0,c,z],f'Analyzer State{c}',vmin=0,vmax=47511);fig.colorbar(im,ax=ax,shrink=.8,label='ADU')
    fig.suptitle('Four analyzer measurements at recorded plane 8',fontsize=14);figs['01_analyzer_observations']=fig
    fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
    for i in range(3):
        im=panel(axes[i,0],delta[i,z],f'Position {i}: retardance + axes','magma',0,.6);fig.colorbar(im,ax=axes[i,0],shrink=.8,label='radian')
        yy,xx=np.mgrid[8:128:16,8:256:16];a=theta[i,z,yy,xx];keep=delta[i,z,yy,xx]>.03
        axes[i,0].quiver(xx[keep],yy[keep],np.cos(a[keep]),np.sin(a[keep]),color='white',pivot='middle',headwidth=0,headlength=0,headaxislength=0,scale=18,width=.004)
        for j,(key,title,limits) in enumerate([('transmittance','Relative transmission',(.45,.9)),('depolarization','Polarization retention',(.9,1.03))],1):
            im=panel(axes[i,j],fresh[key][0,i,z],f'P{i}: {title}','viridis',*limits);fig.colorbar(im,ax=axes[i,j],shrink=.8,label='unitless')
    fig.suptitle('Calibrated polarimetry of a real anisotropic specimen',fontsize=14);figs['02_quantitative_polarimetry']=fig
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,v,title in zip(axes[0],[e['uncorrected_retardance'][0,0,z],delta[0,z],e['uncorrected_retardance'][0,0,z]-delta[0,z]],['No background correction','Fresh calibrated retardance','Uncorrected minus corrected']):
        im=panel(ax,v,title,'coolwarm' if 'minus' in title else 'magma',None if 'minus' in title else 0,None if 'minus' in title else .6);fig.colorbar(im,ax=ax,shrink=.8,label='radian')
    im=panel(axes[1,0],theta[0,z],'Slow-axis angle, pi periodic','twilight',0,np.pi);fig.colorbar(im,ax=axes[1,0],shrink=.8,label='radian')
    im=panel(axes[1,1],fresh['reconstruction'][0,0,z]-upstream['output_retardance'][0,z],'Fresh minus original retardance','coolwarm');fig.colorbar(im,ax=axes[1,1],shrink=.8,label='radian')
    axes[1,2].hist(fresh['depolarization'].ravel(),bins=70,color='#0072B2');axes[1,2].axvline(1,color='black',ls='--');axes[1,2].set(xlabel='Retained polarization P',ylabel='Voxel count',title='P > 1 reveals noise/model limitations')
    figs['03_background_and_numerical_parity']=fig
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');planes=np.arange(16)
    for i,color in enumerate(['#0072B2','#009E73','#D55E00']):
        axes[0].plot(planes,e['z_median_retardance'][i],'-o',ms=3,color=color,label=f'Position {i}');axes[1].plot(planes,100*e['dop_above_one'][i],'-o',ms=3,color=color)
    axes[0].set(xlabel='Recorded defocus plane',ylabel='Median retardance (radian)',title='Full defocus-series response');axes[0].legend(fontsize=9)
    axes[1].set(xlabel='Recorded defocus plane',ylabel='P > 1 pixels (%)',title='No exceptional-pixel suppression')
    axes[2].plot(e['dark_offsets'],e['dark_sensitivity'][:,0],'-o',color='#CC79A7');axes[2].axvline(100.58,ls='--',color='black');axes[2].set(xlabel='Assumed camera dark offset (ADU)',ylabel='Relative L2 retardance change',title='Calibration sensitivity, fixed raw data')
    figs['04_defocus_and_calibration']=fig;return figs
