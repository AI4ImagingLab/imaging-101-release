"""Scientific opacity comparisons from computed arrays; no backend selection."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct


def make_figures(image,trimap,alpha,diagnostics,reference,default_reference,default_history):
    unknown=(trimap!=0)&(trimap!=1);known=~unknown
    numeric=dict(image=image,trimap=trimap,fresh_alpha=alpha,upstream_alpha=reference,
                 upstream_default_ichol_alpha=default_reference,fresh_history=diagnostics['history'],upstream_default_ichol_history=default_history)
    measurements=dict(iterations=len(diagnostics['history']),explicit_relative_residual=diagnostics['explicit_relative_residual'],source_max_difference=float(abs(alpha-reference).max()),source_relative_l2=float(np.linalg.norm(alpha-reference)/np.linalg.norm(reference)),unknown_pixels=int(unknown.sum()),known_pixel_error=float(abs(alpha[known]-trimap[known]).max()),sensitivity=[])
    figures=[]
    def panel(ax,array,title,cmap='gray',vmin=0,vmax=1,extent=None):
        artist=ax.imshow(array,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest',extent=extent);ax.set_title(title,fontsize=10);ax.set_xlabel('native column (pixel)');ax.set_ylabel('native row (pixel)')
        if array.ndim==2:plt.colorbar(artist,ax=ax,fraction=.045,pad=.03)
    fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    panel(axs[0,0],image,'Real CC0 photograph | complete native observation')
    panel(axs[0,1],trimap,'Independent authored trimap | gray means unknown')
    panel(axs[0,2],alpha,'Fresh local fractional opacity | not physical truth')
    panel(axs[1,0],unknown.astype(float),'38666 unknown pixels | all are reconstructed')
    panel(axs[1,1],reference,'Reproduced original Jacobi-CG opacity')
    panel(axs[1,2],alpha-reference,'Fresh minus original | actually computed','coolwarm',-1e-7,1e-7)
    fig.suptitle('Inverse compositing from real RGB and independent manual constraints',fontsize=15);figures.append(fig)
    # Native detail panels show the same physical region and shared opacity scale.
    region=(slice(180,310),slice(80,245));detail_extent=(79.5,244.5,309.5,179.5);numeric['hard_alpha']=(alpha>=.5).astype(float)
    fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    panel(axs[0,0],image[region],'Same native hair/boundary region | observation',extent=detail_extent)
    panel(axs[0,1],trimap[region],'Manual unknown labels are not fractional truth',extent=detail_extent)
    panel(axs[0,2],alpha[region],'Fresh fractional opacity | fine boundary',extent=detail_extent)
    panel(axs[1,0],numeric['hard_alpha'][region],'Thresholded fresh opacity | hard approximation',extent=detail_extent)
    panel(axs[1,1],(alpha-numeric['hard_alpha'])[region],'Fractional minus hard | boundary information','coolwarm',-.5,.5,extent=detail_extent)
    axs[1,2].hist(alpha[unknown],bins=60,color='#0072B2',alpha=.8,label='fresh continuous opacity');axs[1,2].set(xlabel='opacity (dimensionless)',ylabel='unknown-region pixel count',title='Actual fractional boundary distribution');axs[1,2].legend()
    fig.suptitle('Boundary evidence | no opacity ground truth is available',fontsize=15);figures.append(fig)
    fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for ax,step in zip(axs[0],[1,10,100]):
        stage=trimap.ravel().copy();stage[diagnostics['unknown']]=diagnostics['captured'][step];stage=np.clip(stage.reshape(trimap.shape),0,1);numeric['fresh_iterate_'+str(step)]=stage
        panel(ax,stage,f'Fresh selected CG | update {step}')
    h=diagnostics['history'];axs[1,0].semilogy(h[:,0],h[:,2],color='#0072B2',label='fresh Jacobi-CG');axs[1,0].semilogy(default_history[:,0],default_history[:,2],color='#E69F00',marker='o',label='reproduced original default ichol');axs[1,0].axhline(1e-7,color='black',ls='--',lw=1,label='unchanged residual tolerance');axs[1,0].set(xlabel='CG update',ylabel='recursive residual / rhs norm',title='Different preconditioners, same variational model');axs[1,0].legend(fontsize=8)
    panel(axs[1,1],alpha-default_reference,'Selected minus original default ichol','coolwarm',-4e-5,4e-5)
    axs[1,2].plot(alpha[225],color='#0072B2',lw=1,label='fresh selected Jacobi');axs[1,2].plot(default_reference[225],color='#E69F00',lw=1,ls='--',label='original default ichol');axs[1,2].set(xlabel='native column (pixel)',ylabel='opacity',title='Native row225 | solver precision control');axs[1,2].legend()
    fig.suptitle('Actual convergence and independently reproduced numerical controls',fontsize=15);figures.append(fig)
    sensitivities=[]
    for epsilon in [1e-5,1e-3]:
        t=time.perf_counter();a,d=reconstruct(image,trimap,epsilon=epsilon);elapsed=time.perf_counter()-t
        sensitivities.append(a);numeric['fresh_alpha_epsilon_'+str(epsilon)]=a
        measurements['sensitivity'].append(dict(epsilon=epsilon,seconds=elapsed,iterations=len(d['history']),rms_change=float(np.sqrt(np.mean((a-alpha)**2))),explicit_relative_residual=d['explicit_relative_residual']))
    fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for ax,a,e in zip(axs[0],[alpha]+sensitivities,[1e-7,1e-5,1e-3]):panel(ax,a,f'Fresh covariance regularizer epsilon={e:g}')
    for ax,a,e in zip(axs[1,:2],sensitivities,[1e-5,1e-3]):panel(ax,a-alpha,f'Fresh epsilon={e:g} minus canonical','coolwarm',-.2,.2)
    for a,e in zip([alpha]+sensitivities,[1e-7,1e-5,1e-3]):axs[1,2].plot(a[225],lw=1,label=f'epsilon={e:g}')
    axs[1,2].set(xlabel='native column (pixel)',ylabel='opacity',title='Regularization sensitivity | not calibrated uncertainty');axs[1,2].legend()
    fig.suptitle('Actual regularization sensitivity | same real observations and manual labels',fontsize=15);figures.append(fig)
    return figures,numeric,measurements


def export_figures(figures,numeric,measurements,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for i,fig in enumerate(figures,1):
        for suffix in ['png','svg','pdf']:fig.savefig(directory/f'figure_{i}.{suffix}',dpi=180,bbox_inches='tight')
    np.savez_compressed(directory/'numeric_sources.npz',**numeric)
    (directory/'metrics.json').write_text(json.dumps(measurements,indent=2,allow_nan=False),encoding='utf8')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file() and p.name!='hash_manifest.json'}
    (directory/'hash_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
