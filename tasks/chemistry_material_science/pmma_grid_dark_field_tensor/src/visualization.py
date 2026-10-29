"""Review figures and explicitly conditional uncertainty diagnostics."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import covariance_eigenvalues,average_coefficients,mode_ratios
from .preprocessing import PERIOD

def conditional_eigenvalue_uncertainty(coefficients,fit_covariance,covariance,average=13,period=PERIOD):
    """Cauchy upper bound, without assuming independence of overlapping fits."""
    xx,yy,xy=covariance;d=np.hypot(xx-yy,2*xy);d=np.maximum(d,np.finfo(float).tiny)
    derivative=np.array([.5*(1-(xx-yy)/d),.5*(1+(xx-yy)/d),-2*xy/d])
    transform=period**2/np.pi**2*np.array([[-3/8,1/8,-1/16,-1/16],[1/8,-3/8,-1/16,-1/16],[0,0,1/8,-1/8]])
    modes=np.einsum('cyx,ci->iyx',derivative,transform)
    logs=np.concatenate([-modes.sum(axis=0,keepdims=True),modes])
    result=np.zeros(xx.shape)
    for channel in range(2):
        means=average_coefficients(coefficients[channel],average)
        gradients=logs/means
        for row in range(xx.shape[0]):
            for col in range(xx.shape[1]):
                local=fit_covariance[channel,row*average:(row+1)*average,col*average:(col+1)*average]
                g=gradients[:,row,col];variance=np.einsum('i,abij,j->ab',g,local,g)
                result[row,col]+=np.nanmean(np.sqrt(np.maximum(variance,0)))
    return result

def figures(grid,sample,covariance,transmission,diagnostics,reference,sigma):
    """Return four multi-panel actual-data figure groups and numeric sources."""
    eig=covariance_eigenvalues(covariance);ratios=diagnostics['ratios'];predicted=mode_ratios(covariance,transmission)
    groups={}
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,a,title,unit in zip(axes.ravel(),[grid,sample,transmission,*covariance],['Reference grid / flat','Sample grid / flat','Transmission','Cxx','Cyy','Cxy'],['ratio','ratio','ratio','pixel squared','pixel squared','pixel squared']):
        im=ax.imshow(a,cmap='coolwarm' if title.startswith('C') else 'viridis');ax.set_title(title);fig.colorbar(im,ax=ax,label=unit);ax.set_xlabel('map column' if a.shape==(35,35) else 'ROI column');ax.set_ylabel('map row' if a.shape==(35,35) else 'ROI row')
    groups['01_measurements_fields']=fig
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for k in range(3):
        a=reference[k];b=covariance[k];ax=axes[0,k];ax.scatter(a.ravel(),b.ravel(),s=3);lo=min(a.min(),b.min());hi=max(a.max(),b.max());ax.plot([lo,hi],[lo,hi],color='black',lw=.8);ax.set_title(['Cxx','Cyy','Cxy'][k]);ax.set_xlabel('Original oracle (pixel squared)');ax.set_ylabel('Fresh local (pixel squared)');im=axes[1,k].imshow(b-a,cmap='coolwarm');fig.colorbar(im,ax=axes[1,k],label='fresh minus oracle (pixel squared)')
    groups['02_original_fresh_parity']=fig
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    residual=np.sqrt(np.mean((predicted[1:]-ratios[1:])**2,axis=0));isotropic=covariance.copy();isotropic[0]=isotropic[1]=(covariance[0]+covariance[1])/2;isotropic[2]=0
    constrained=[np.repeat(transmission[None],5,axis=0),mode_ratios(isotropic,transmission),predicted]
    errors=[float(np.linalg.norm(p-ratios)/np.linalg.norm(ratios)) for p in constrained]
    for i in range(4):
        ax=axes.ravel()[i];ax.scatter(ratios[i+1].ravel(),predicted[i+1].ravel(),s=3);ax.plot([0,1],[0,1],color='black',lw=.8);ax.set_title(['x mode','y mode','x-y mode','x+y mode'][i]);ax.set_xlabel('Measured correlation ratio');ax.set_ylabel('Gaussian tensor prediction')
    im=axes.ravel()[4].imshow(residual);fig.colorbar(im,ax=axes.ravel()[4],label='Directional ratio RMS mismatch');axes.ravel()[4].set_title('Stationarity / Gaussian closure')
    axes.ravel()[5].bar(['Transmission only','Isotropic','Signed tensor'],errors);axes.ravel()[5].tick_params(axis='x',rotation=25);axes.ravel()[5].set_ylabel('Relative directional-mode residual');axes.ravel()[5].set_title('Actual fresh constrained comparisons')
    groups['03_directional_closure']=fig
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,a,title,unit in zip(axes.ravel()[:4],[eig[0],eig[1],sigma,eig[0]/sigma],['Signed smaller eigenvalue','Signed larger eigenvalue','Conditional fit SD bound','Signed smaller eigenvalue / SD'],['pixel squared','pixel squared','pixel squared','dimensionless']):
        im=ax.imshow(a,cmap='coolwarm');ax.set_title(title);fig.colorbar(im,ax=ax,label=unit)
    negative=eig[0]<0;axes.ravel()[4].hist((abs(eig[0])/sigma)[negative],bins=25);axes.ravel()[4].axvline(1.96,color='black',ls='--');axes.ravel()[4].set_xlabel('Absolute negative eigenvalue / conditional SD');axes.ravel()[4].set_title(f'All {negative.sum()} negative locations retained')
    axes.ravel()[5].scatter(transmission.ravel(),eig[0].ravel(),s=5,c=residual.ravel());axes.ravel()[5].axhline(0,color='black');axes.ravel()[5].set_xlabel('Transmission');axes.ravel()[5].set_ylabel('Smaller eigenvalue (pixel squared)');axes.ravel()[5].set_title('Weak-scattering noise and boundary mismatch')
    groups['04_signed_uncertainty']=fig
    numeric=dict(grid=grid,sample=sample,reconstruction=covariance,transmission=transmission,oracle=reference,identified_ratios=ratios,predicted_ratios=predicted,eigenvalues=eig,conditional_sigma=sigma,comparison_residuals=np.array(errors),directional_rms=residual)
    return groups,numeric

def export_figures(groups,numeric,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);np.savez_compressed(directory/'numeric_sources.npz',**numeric);files=[]
    for name,fig in groups.items():
        for extension in ['png','svg','pdf']:
            path=directory/(name+'.'+extension);fig.savefig(path,dpi=160);files.append(path)
    files.append(directory/'numeric_sources.npz')
    manifest={'description':'Fresh active full-data reconstruction; original oracle separately identified; no PSD clipping. Conditional SD is not a calibrated camera confidence interval.','files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files]}
    (directory/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
