from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import height_gradient,effective_height

def metrics(a,r):
    a=a.astype(float);r=r.astype(float)
    if np.ptp(r)<=0 or np.linalg.norm(r)==0:raise ValueError('degenerate reference')
    ncc=float(np.vdot(a,r)/(np.linalg.norm(a)*np.linalg.norm(r))) if np.linalg.norm(a)>0 else 0.
    return {'ncc':float(np.clip(ncc,-1,1)),'nrmse':float(np.sqrt(np.mean((a-r)**2))/np.ptp(r))}
def phase_residual(height,diagnostics):
    c=diagnostics['carriers'];g=height_gradient(height,c['calibration']);q=c['frequencies'];pred=-effective_height()*np.array([v[1]*g[0]+v[0]*g[1] for v in q]);observed=diagnostics['phases'];circle=np.exp(1j*(pred-observed));offset=np.angle(np.mean(circle,axis=(1,2)));error=np.angle(circle*np.exp(-1j*offset[:,None,None]));return float(np.sqrt(np.mean(error**2))),error

def figures(reference,measurements,fresh,upstream,diagnostics,variants):
    plots={};spacing=diagnostics['carriers']['calibration'];extent=[0,fresh.shape[1]*spacing*1000,fresh.shape[0]*spacing*1000,0]
    f,ax=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for a,v,title in zip(ax[:2],[reference,measurements],['Flat-surface reference','Measured refracted pattern']):
        im=a.imshow(v,cmap='gray',extent=extent);a.set(xlabel='x (mm)',ylabel='y (mm)',title=title);f.colorbar(im,ax=a,label='DN')
    im=ax[2].imshow(np.log1p(abs(np.fft.fftshift(np.fft.fft2(reference-reference.mean())))),cmap='magma');p=diagnostics['carriers']['peaks'];ax[2].plot(p[:,1],p[:,0],'co',mfc='none',ms=12);ax[2].set(title='Measured carrier spectrum',xlabel='FFT column',ylabel='FFT row');f.colorbar(im,ax=ax[2],label='log(1 + Fourier amplitude)')
    plots['acquisition']=f
    f,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained');limit=float(np.max(abs(upstream)))*1e6
    arrays=[upstream,fresh,fresh-upstream,variants['wrapped_only'],variants['narrow'],variants['narrow']-fresh];titles=['Original modern FCD','Fresh local FCD','Fresh - original','Fresh without unwrapping','Fresh narrower aperture','Narrower - selected']
    for k,(a,v,title) in enumerate(zip(ax.flat,arrays,titles)):
        lim=max(float(np.max(abs(v)))*1e6,1e-8) if k in [2,5] else limit;im=a.imshow(v*1e6,cmap='RdBu_r',extent=extent,vmin=-lim,vmax=lim);a.set(title=title,xlabel='x (mm)',ylabel='y (mm)');f.colorbar(im,ax=a,label='Relative elevation (um)')
    plots['height_comparisons']=f
    f,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for i in range(2):
        for j,(v,title,unit) in enumerate([(np.angle(np.exp(1j*diagnostics['phases'][i])),f'Wrapped phase {i}','rad'),(diagnostics['phases'][i],f'Unwrapped phase {i}','rad'),(diagnostics['displacement'][i]*1000,['Apparent x shift','Apparent y shift'][i],'mm')]):
            im=ax[i,j].imshow(v,cmap='twilight' if j==0 else 'RdBu_r',extent=extent);ax[i,j].set(title=title,xlabel='x (mm)',ylabel='y (mm)');f.colorbar(im,ax=ax[i,j],label=unit)
    plots['phase_and_slopes']=f
    residual,error=phase_residual(fresh,diagnostics);g=height_gradient(fresh,spacing);raw=diagnostics['gradient'];f,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for i in range(2):
        im=ax[0,i].imshow(error[i],cmap='RdBu_r',extent=extent,vmin=-.6,vmax=.6);ax[0,i].set(title=f'Forward carrier {i} residual',xlabel='x (mm)',ylabel='y (mm)');f.colorbar(im,ax=ax[0,i],label='Wrapped error (rad)')
    x=(np.arange(fresh.shape[1])+.5)*spacing*1000;y=fresh.shape[0]//2
    ax[1,0].plot(x,upstream[y]*1e6,label='Original');ax[1,0].plot(x,fresh[y]*1e6,'--',label='Fresh');ax[1,0].plot(x,variants['wrapped_only'][y]*1e6,alpha=.6,label='No unwrap');ax[1,0].set(xlabel='x (mm)',ylabel='Elevation (um)',title='Central spatial section');ax[1,0].legend()
    for i,name in enumerate(['x','y']):
        observed=raw[i]-raw[i].mean();ax[1,1].plot(x,observed[y],alpha=.65,label='Observed '+name);ax[1,1].plot(x,g[i,y],'--',label='Integrated '+name)
    ax[1,1].set(xlabel='x (mm)',ylabel='Surface slope (dimensionless)',title='Gradient projection and mismatch');ax[1,1].legend(ncol=2,fontsize=8)
    plots['physical_consistency']=f
    return plots,error

def export(plots,directory,arrays,table):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);entries=[]
    for name,fig in plots.items():
        for ext in ['png','svg','pdf']:
            p=directory/(name+'.'+ext);fig.savefig(p,dpi=160);entries.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    p=directory/'numeric_sources.npz';np.savez_compressed(p,**arrays);entries.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});p=directory/'comparison_metrics.json';p.write_text(json.dumps(table,indent=2,allow_nan=False));entries.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (directory/'manifest.json').write_text(json.dumps({'figure_groups':list(plots),'provenance':'Active local benchmark and sensitivity reconstructions; separately labeled captured original source height','artifacts':entries},indent=2))
