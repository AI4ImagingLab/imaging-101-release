"""Scientific comparisons; no backend changes at import time."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import forward

def metrics(a,r):
    a=a.astype(float);r=r.astype(float)
    return {'ncc':float(np.vdot(a,r)/(np.linalg.norm(a)*np.linalg.norm(r))),'nrmse':float(np.sqrt(np.mean((a-r)**2))/np.ptp(r))}
def figures(data,angles,volume,reference,history,matrix,variants):
    result={};extent=[0,128*6.39,128*6.39,0]
    f,ax=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for a,i in zip(ax[0],(0,37,73)):
        im=a.imshow(data[:,:,i].T,cmap='gray',extent=extent);a.set_title(f'Measured angle {angles[i]:g} deg');f.colorbar(im,ax=a,label='Detector counts')
    ax[1,0].plot(angles,np.mean(data,axis=(0,1)),'o-',ms=3);ax[1,0].set(xlabel='Angle (degree)',ylabel='Mean detector counts',title='Exposure/projection variation')
    ax[1,1].pcolormesh(angles,(np.arange(128)+.5)*6.39,data[64],shading='nearest',cmap='gray');ax[1,1].invert_yaxis();ax[1,1].set(xlabel='Angle (degree)',ylabel='Detector y (nm)',title='Central measured sinogram')
    ax[1,2].plot(angles,np.ones(len(angles)),'|',ms=20);ax[1,2].axvspan(21,27,alpha=.2,color='red');ax[1,2].set(xlim=(-90,90),xlabel='Angle (degree)',yticks=[],title='Missing wedge and missing views')
    result['observations']=f
    f,ax=plt.subplots(3,3,figsize=(11,10),layout='constrained');vmax=float(np.max(reference))
    cuts=lambda v:[v[64],v[:,64,:],v[:,:,64]]
    for row,(v,name) in enumerate([(reference,'Original Tomviz'),(volume,'Fresh local'),(abs(volume-reference),'Absolute difference')]):
        for col,cut in enumerate(cuts(v)):
            im=ax[row,col].imshow(cut,cmap='magma' if row<2 else 'viridis',extent=extent,vmin=0,vmax=vmax if row<2 else max(float(np.max(abs(volume-reference))),1e-12));ax[row,col].set_title(name+' '+['x=64','z=64','y=64'][col]);f.colorbar(im,ax=ax[row,col],label='Response / voxel-length')
    for a in ax.flat:a.set(xlabel='Horizontal coordinate (nm)',ylabel='Vertical coordinate (nm)')
    result['volume_comparison']=f
    prediction=forward(volume,matrix);res=prediction-data
    f,ax=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for col,i in enumerate((0,37,73)):
        im=ax[0,col].imshow(prediction[:,:,i].T,cmap='gray',extent=extent);ax[0,col].set_title(f'Forward projection {angles[i]:g} deg');f.colorbar(im,ax=ax[0,col],label='Detector counts')
        m=float(np.max(abs(res[:,:,i])));im=ax[1,col].imshow(res[:,:,i].T,cmap='RdBu_r',vmin=-m,vmax=m,extent=extent);ax[1,col].set_title('Prediction - measurement');f.colorbar(im,ax=ax[1,col],label='Detector counts')
    result['observation_residuals']=f
    f,ax=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    ax[0,0].plot(history[:,0],history[:,1],'o-');ax[0,0].set(xlabel='Iteration',ylabel='Relative L2 projection residual',title='Full-volume convergence')
    ax[0,1].plot(angles,np.sqrt(np.sum(res**2,axis=(0,1))/np.sum(data**2,axis=(0,1))),'o-',ms=3);ax[0,1].set(xlabel='Angle (degree)',ylabel='Relative L2 residual',title='Angle-dependent model mismatch')
    ax[0,2].plot(volume[64,64],label='Fresh');ax[0,2].plot(reference[64,64],'--',label='Original');ax[0,2].set(xlabel='Detector y voxel',ylabel='Response / voxel-length',title='Central line');ax[0,2].legend()
    for a,(count,v) in zip(ax[1],variants.items()):
        im=a.imshow(v[0],cmap='magma',extent=extent,vmin=0,vmax=vmax);a.set_title(f'Central slice: {count} iterations');f.colorbar(im,ax=a,label='Response / voxel-length')
    result['convergence_sensitivity']=f
    return result,prediction

def export(figures,directory,arrays):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);entries=[]
    for name,fig in figures.items():
        for ext in ['png','svg','pdf']:
            path=directory/(name+'.'+ext);fig.savefig(path,dpi=160);entries.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    path=directory/'numeric_sources.npz';np.savez_compressed(path,**arrays);entries.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (directory/'manifest.json').write_text(json.dumps({'figures':list(figures),'sources':'All fresh arrays computed in the active notebook; original volume separately identified','artifacts':entries},indent=2))
