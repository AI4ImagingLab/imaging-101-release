
"""Evidence-bearing spatial and numerical review figures; no backend changes on import."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import load_data
from .solvers import solve
def metrics(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    n=np.linalg.norm(a)*np.linalg.norm(b)
    return (float(np.dot(a,b)/n) if n else 0.),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'figure.dpi':110,'savefig.dpi':180,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def plate(ax,image,title,cmap='viridis',**kwargs):
    artist=ax.imshow(image,origin='lower',cmap=cmap,**kwargs);ax.set_title(title,pad=8)
    ax.set_xlabel('East / column (pixel)');ax.set_ylabel('North / row (pixel)');return artist

from scipy.ndimage import gaussian_filter
from .solvers import grid_catalogue
from .physics_model import inverse,forward
def review(data,upstream):
    """Compute the standard method, calibration ablation, resolution sweep and32 shape-noise nulls."""
    style();e,b=solve(data);g1,g2,count,weight=grid_catalogue(data);ref=upstream['reconstruction'][0]
    figs=[];tables=[];sources={}
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    lim=max(np.max(abs(e)),np.max(abs(b)))
    values=[(count,'Reliable galaxy counts','cividis',{}),(g1,'Calibrated shear gamma1','RdBu_r',{'vmin':-.6,'vmax':.6}),(g2,'Calibrated shear gamma2','RdBu_r',{'vmin':-.6,'vmax':.6}),(ref,'Original KS93 + Gaussian','RdBu_r',{'vmin':-lim,'vmax':lim}),(e,'Fresh local E convergence','RdBu_r',{'vmin':-lim,'vmax':lim}),(b,'Fresh local B diagnostic','RdBu_r',{'vmin':-lim,'vmax':lim})]
    for ax,(v,title,cmap,kw) in zip(axs.flat,values):
        im=plate(ax,v,title,cmap,**kw);fig.colorbar(im,ax=ax,shrink=.8,label='galaxies' if v is count else 'dimensionless')
    fig.suptitle('KiDS-450 | real galaxy shapes to mass-map modes',fontsize=16);figs.append(('01_mass_maps',fig))
    fig,axs=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
    sweep=[]
    for ax,sigma in zip(axs,[1.,2.,4.]):
        ee,bb=solve(data,sigma=sigma);sweep.append(ee);im=plate(ax,ee,f'Gaussian sigma = {sigma:g} pixels','RdBu_r',vmin=-lim,vmax=lim)
        tables.append({'experiment':'sigma='+str(sigma),'E_rms':float(np.std(ee)),'B_rms':float(np.std(bb)),'NRMSE_vs_original':metrics(ee,ref)[1]})
    fig.colorbar(im,ax=list(axs),shrink=.85,label='Dimensionless convergence (shared scale)');fig.suptitle('Resolution–shape-noise tradeoff',fontsize=16);figs.append(('02_smoothing',fig))
    rng=np.random.default_rng(20261020);null=[]
    ellipticity=data['ellipticity'][:,0]+1j*data['ellipticity'][:,1]
    for iteration in range(32):
        angle=rng.uniform(0,np.pi,len(ellipticity));rot=ellipticity*np.exp(2j*angle)
        d=dict(data);d['ellipticity']=np.c_[rot.real,rot.imag];null.append(solve(d)[0])
    noise=np.std(null,axis=0,ddof=1);snr=np.divide(e,noise,out=np.zeros_like(e),where=noise>0)
    pd=dict(data);pd['ellipticity']=data['psf_ellipticity'];pe,pb=solve(pd)
    fig,axs=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,v,title,cmap in zip(axs,[noise,snr,pe],['32 orientation-null standard deviations','E / orientation-null sigma','Independent PSF ellipticity diagnostic'],['cividis','RdBu_r','RdBu_r']):
        im=plate(ax,v,title,cmap);fig.colorbar(im,ax=ax,shrink=.85,label='S/N' if v is snr else 'dimensionless')
    fig.suptitle('Noise and instrumental-systematic context | no true mass map available',fontsize=14);figs.append(('03_null_psf',fig))
    f1,f2=forward(e,b);smooth1=gaussian_filter(g1,2,mode='wrap');smooth2=gaussian_filter(g2,2,mode='wrap')
    ne,nb=solve(data,calibrate=False)
    fig,axs=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
    diff=ne-e;im=plate(axs[0],diff,'Ablation: omit ensemble calibration','RdBu_r');fig.colorbar(im,ax=axs[0],label='Delta convergence')
    residual=np.hypot(f1-(smooth1-smooth1.mean()),f2-(smooth2-smooth2.mean()))
    im=plate(axs[1],residual,'Forward closure after smoothing','magma');fig.colorbar(im,ax=axs[1],label='Shear residual norm')
    axs[2].hist(e.ravel(),bins=40,histtype='step',lw=1.7,label='E map');axs[2].hist(b.ravel(),bins=40,histtype='step',lw=1.7,label='B map');axs[2].hist(np.asarray(null).ravel(),bins=40,histtype='step',lw=1.7,label='Orientation null (32x)',weights=np.full(np.asarray(null).size,1/32));axs[2].set_xlabel('Dimensionless convergence');axs[2].set_ylabel('Pixel count / null realization');axs[2].legend(fontsize=8)
    fig.suptitle('Calibration mechanism and forward consistency',fontsize=16);figs.append(('04_mechanism',fig))
    tables.append({'experiment':'no ensemble calibration','E_rms':float(np.std(ne)),'B_rms':float(np.std(nb)),'NRMSE_vs_original':metrics(ne,ref)[1]})
    sources.update(E=e,B=b,gamma1=g1,gamma2=g2,count=count,weight=weight,noise_sigma=noise,snr=snr,psf_E=pe,smoothing_sweep=np.asarray(sweep),forward_residual=residual,calibration_ablation=ne,orientation_nulls=np.asarray(null))
    return figs,tables,sources
