"""Maintainer comparisons from freshly supplied arrays, with numeric export."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import predicted_amplitudes
from .preprocessing import DISTANCES

def figure_groups(amplitudes,fresh,upstream,single,two,history,export=None):
    pred=predicted_amplitudes(fresh,DISTANCES);a=abs(amplitudes)
    figs=[]
    fig,axs=plt.subplots(2,4,figsize=(13,6),constrained_layout=True)
    for j,ax in enumerate(axs.flat):
        im=ax.imshow(a[j]**2,cmap='gray',vmin=0,vmax=1);ax.set_title(f'z={DISTANCES[j]:.0f} μm');ax.set_xlabel('column [pixel]');ax.set_ylabel('row [pixel]')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Normalized intensity [1]');fig.suptitle('Author supplied simulated observations: eight full planes');figs.append(('observations',fig))
    fig,axs=plt.subplots(2,4,figsize=(13,7),constrained_layout=True)
    for j,(title,u) in enumerate([('Single-plane backpropagation',single),('Two-plane, 30 sweeps',two),('Fresh eight-plane, 100 sweeps',fresh),('Reproduced upstream eight-plane',upstream)]):
        im=axs[0,j].imshow(abs(u),cmap='gray',vmin=0,vmax=1.35);axs[0,j].set_title(title,fontsize=10)
        pm=axs[1,j].imshow(np.angle(u),cmap='twilight',vmin=-np.pi,vmax=np.pi)
        for ax in axs[:,j]:ax.set_xlabel('column [pixel]');ax.set_ylabel('row [pixel]')
    fig.colorbar(im,ax=axs[0,:].tolist(),label='Field amplitude [1]');fig.colorbar(pm,ax=axs[1,:].tolist(),label='Wrapped phase [rad]');fig.suptitle('Computed comparison: amplitude above, wrapped phase below');figs.append(('reconstructions',fig))
    fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for row,j in enumerate([0,7]):
        axs[row,0].imshow(a[j],cmap='gray',vmin=0,vmax=1);axs[row,0].set_title(f'Measured amplitude, z={DISTANCES[j]:g} μm')
        axs[row,1].imshow(pred[j],cmap='gray',vmin=0,vmax=1);axs[row,1].set_title('Fresh predicted amplitude')
        im=axs[row,2].imshow(pred[j]-a[j],cmap='RdBu_r',vmin=-.15,vmax=.15);axs[row,2].set_title('Predicted minus measured')
        for ax in axs[row]:ax.set_xlabel('column [pixel]');ax.set_ylabel('row [pixel]')
    fig.colorbar(im,ax=axs[:,2].tolist(),label='Amplitude residual [1]');fig.suptitle('Observation consistency retains upstream camera normalization');figs.append(('residual_maps',fig))
    fig,axs=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    axs[0,0].semilogy(history[:,0],history[:,1],'-o');axs[0,0].set(xlabel='Full forward/reverse sweeps',ylabel='Amplitude L2 residual [1]',title='Fresh convergence')
    for title,u in [('Single',single),('Two-plane',two),('Eight-plane',fresh)]:
        p=predicted_amplitudes(u,DISTANCES);errors=np.linalg.norm((p-a).reshape(8,-1),axis=1)/np.linalg.norm(a.reshape(8,-1),axis=1)
        axs[0,1].plot(DISTANCES,errors,'-o',label=title);axs[1,0].plot(abs(u[256]),label=title)
    axs[0,1].set(xlabel='Detector distance [μm]',ylabel='Per-plane amplitude residual [1]',title='Plane-count ablation');axs[0,1].legend()
    axs[1,0].set(xlabel='Column [pixel], row=256',ylabel='Amplitude [1]',title='Fresh spatial cross sections');axs[1,0].legend()
    phase=np.exp(-1j*np.angle(np.vdot(upstream,fresh)));diff=abs(fresh*phase-upstream)
    im=axs[1,1].imshow(diff,cmap='magma',vmin=0,vmax=max(float(diff.max()),1e-12));axs[1,1].set(title=f'Fresh/upstream max error = {diff.max():.2g}',xlabel='column [pixel]',ylabel='row [pixel]');fig.colorbar(im,ax=axs[1,1],label='Absolute field difference [1]');figs.append(('diagnostics',fig))
    if export:
        out=Path(export);out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'numeric_sources.npz',amplitudes=a,predicted_amplitudes=pred,fresh=fresh,upstream=upstream,single=single,two=two,history=history,distances=DISTANCES)
        manifest={'groups':[],'source':'numeric_sources.npz','provenance':'Existing author simulations; fresh local reconstructions; independent reproduced upstream reference'}
        for name,fig in figs:
            files=[]
            for ext in ['png','svg','pdf']:
                p=out/f'{name}.{ext}';fig.savefig(p,dpi=160);files.append({'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            manifest['groups'].append({'name':name,'files':files,'numeric_consumer':'numeric_sources.npz'})
        manifest['source_sha256']=hashlib.sha256((out/'numeric_sources.npz').read_bytes()).hexdigest();(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return figs
