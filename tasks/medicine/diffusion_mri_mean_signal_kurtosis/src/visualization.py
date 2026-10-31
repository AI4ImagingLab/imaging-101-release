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

from .preprocessing import shell_average,signal_mask
from .physics_model import predict
from .solvers import reconstruct,fit_shells

def gaussian_fit(means,counts,bvalues,mask):
    """Lower-order Gaussian log-signal baseline with matching observed weights."""
    B=np.column_stack((-bvalues,np.ones_like(bvalues)));out=np.zeros(means.shape[:-1]+(2,))
    for idx in np.ndindex(mask.shape):
        if not mask[idx]:continue
        y=np.maximum(means[idx],.0001);w=counts*y*y;normal=B.T@(w[:,None]*B)
        out[idx]=np.linalg.pinv(normal)@(B.T@(w*np.log(y)))
    return out[...,0],np.exp(out[...,1])

def run_experiments(data,fresh):
    """OLS, reduced-shell, Gaussian-order, support and cross-slice checks."""
    raw=data['signals'][0];b=data['bvalues'][0];mask=signal_mask(raw,b);means,counts,shells=shell_average(raw,b)
    ols=reconstruct(data,weighted=False);reduced=reconstruct(data,shell_selection=[0,600,1000,2000]);D0,S0=gaussian_fit(means,counts,shells,mask)
    predicted=predict(fresh['diffusivity'][0],fresh['reconstruction'][0],fresh['baseline'][0],shells)
    gaussian=S0[...,None]*np.exp(-D0[...,None]*shells)
    per_slice=np.array([[np.sqrt(np.mean((predicted[i][mask[i]]-means[i][mask[i]])**2)),np.sqrt(np.mean((gaussian[i][mask[i]]-means[i][mask[i]])**2)),np.mean(fresh['reconstruction'][0,i][mask[i]]<0)] for i in range(3)])
    return {'shell_means':means,'shells':shells,'mask':mask,'ols_kurtosis':ols['reconstruction'][0],
        'reduced_kurtosis':reduced['reconstruction'][0],'gaussian_diffusivity':D0,'prediction':predicted,
        'gaussian_prediction':gaussian,'slice_metrics':per_slice,'mask_counts':np.array([(raw[...,b==0].mean(-1)>t).sum() for t in [20,40,60,80]])}

def mri_panel(ax,*args,**kwargs):
    im=panel(ax,*args,**kwargs);ax.set_xlabel('Native x index (voxel)');ax.set_ylabel('Native y index (voxel)');return im

def figures(data,fresh,upstream,e):
    style();figs={};D=fresh['diffusivity'][0];K=fresh['reconstruction'][0];m=e['mask'];means=e['shell_means']
    fig,axes=plt.subplots(3,3,figsize=(11,10),layout='constrained')
    for i,sl in enumerate([8,9,10]):
        for j,(v,title,cm,lim) in enumerate([(means[...,0],'b=0 (ADU)','gray',(0,400)),(D*1000,'D (10^-3 mm^2/s)','viridis',(0,2)),(K,'Mean-signal K','magma',(0,1.5))]):
            im=mri_panel(axes[i,j],v[i].T,f'Native slice {sl}: {title}',cm,*lim);fig.colorbar(im,ax=axes[i,j],shrink=.8)
    fig.suptitle('Real human-brain full fields: diffusivity and non-Gaussian signal curvature',fontsize=14);figs['01_native_slice_maps']=fig
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for j,(v,title) in enumerate([(K,'Weighted all 16 shells'),(e['ols_kurtosis'],'Unweighted all 16 shells'),(e['reduced_kurtosis'],'Weighted b=0,600,1000,2000')]):
        im=mri_panel(axes[0,j],v[1].T,title,'magma',0,1.5);fig.colorbar(im,ax=axes[0,j],shrink=.8,label='K (unitless)')
    for ax,v,title in zip(axes[1:][0],[(K[1]<0)&m[1],np.abs(K[1]-upstream['output_kurtosis'][1]),e['reduced_kurtosis'][1]-K[1]],['Negative K: model limitation','Fresh-original |K difference|','Reduced minus full-shell K']):
        if 'minus' in title:
            bound=max(float(np.max(np.abs(v))),1e-12)
            im=mri_panel(ax,v.T,title,'coolwarm',-bound,bound)
        elif '|K difference|' in title:
            im=mri_panel(ax,v.T,title,'viridis',0,max(float(np.max(v)),1e-12))
        else:
            im=mri_panel(ax,v.T,title,'viridis',0,1)
        fig.colorbar(im,ax=ax,shrink=.8)
    figs['02_weighting_and_model_limits']=fig
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');rois=[(38,44,38,44),(44,50,52,58),(50,56,30,36)]
    for ax,(x0,x1,y0,y1) in zip(axes,rois):
        selected=e['mask'][1,x0:x1,y0:y1];obs=means[1,x0:x1,y0:y1][selected];pred=e['prediction'][1,x0:x1,y0:y1][selected];gauss=e['gaussian_prediction'][1,x0:x1,y0:y1][selected]
        ax.plot(e['shells'],obs.mean(0),'o',color='black',ms=4,label='Observed shell mean')
        ax.plot(e['shells'],pred.mean(0),color='#0072B2',label='Fresh kurtosis model');ax.plot(e['shells'],gauss.mean(0),'--',color='#D55E00',label='Gaussian baseline')
        ax.set(xlabel='b (s/mm^2)',ylabel='Signal (ADU)',title=f'Fixed ROI x{x0}:{x1}, y{y0}:{y1}; n={selected.sum()}')
    axes[0].legend(fontsize=8);figs['03_signal_model_evidence']=fig
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained');x=np.arange(3);w=.35
    axes[0].bar(x-w/2,e['slice_metrics'][:,0],w,label='Kurtosis',color='#0072B2');axes[0].bar(x+w/2,e['slice_metrics'][:,1],w,label='Gaussian',color='#D55E00');axes[0].set(xticks=x,xticklabels=[8,9,10],xlabel='Native slice',ylabel='Shell-signal RMSE (ADU)',title='Forward consistency');axes[0].legend()
    axes[1].bar(x,100*e['slice_metrics'][:,2],color='#CC79A7');axes[1].set(xticks=x,xticklabels=[8,9,10],xlabel='Native slice',ylabel='Negative K within support (%)',title='Unclipped failures remain visible')
    axes[2].plot([20,40,60,80],e['mask_counts'],'o-',color='#009E73');axes[2].axvline(40,color='black',ls='--');axes[2].set(xlabel='b=0 support threshold (ADU)',ylabel='Included voxels over three slices',title='Observation-only support sensitivity')
    figs['04_cross_slice_and_support']=fig;return figs
