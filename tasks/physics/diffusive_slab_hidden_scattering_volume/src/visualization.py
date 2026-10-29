"""Computed scientific comparisons. Importing this module never selects a backend."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt

def metrics(estimate: np.ndarray,reference: np.ndarray) -> dict[str,float]:
    """Cosine and dynamic-range NRMSE for finite nondegenerate comparison arrays."""
    a=np.asarray(estimate,dtype=np.float64).ravel();b=np.asarray(reference,dtype=np.float64).ravel()
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.ptp(b)<=0 or np.linalg.norm(b)==0:raise ValueError('Undefined comparison')
    norm=np.linalg.norm(a)
    return dict(ncc=float(np.dot(a,b)/(norm*np.linalg.norm(b))) if norm else 0.,nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))

def export(fig,name: str,arrays: dict,caption: str,directory: Path | None=None):
    """Optionally export all formats and numerical source arrays outside the task."""
    if directory is None:return fig
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    record=dict(name=name,caption=caption,arrays={},files={})
    for key,value in arrays.items():
        a=np.asarray(value)
        if a.dtype.kind not in 'biufc' or not np.isfinite(a).all():raise ValueError(f'Invalid exported numeric array{key}')
        record['arrays'][key]=dict(shape=list(a.shape),dtype=str(a.dtype))
    for ext in ['png','svg','pdf']:
        path=directory/f'{name}.{ext}';fig.savefig(path,dpi=160,bbox_inches='tight')
        record['files'][ext]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path=directory/f'{name}.npz';np.savez_compressed(path,**arrays)
    record['files']['npz']=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest=directory/'figure_manifest.json'
    old=json.loads(manifest.read_text()) if manifest.exists() else []
    old=[r for r in old if r['name']!=name]+[record]
    manifest.write_text(json.dumps(old,indent=2),encoding='utf8')
    return fig

def observation_figure(raw):
    fig,ax=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    for axis,title,j in [(0,'Integrated counts (full512 bins)',0),(1,'Native time / y projection',1)]:
        data=raw.sum(axis=axis)
        im=ax[j].imshow(data,aspect='auto',cmap='magma')
        ax[j].set(title=title,xlabel='scan index',ylabel='x index' if not j else 'native time-bin index')
        fig.colorbar(im,ax=ax[j],label='counts')
    ax[2].plot(np.arange(512)*.016,raw.sum(axis=(1,2)),label='all1024 scan points')
    ax[2].axvline(416*.016,color='tab:red',ls='--',label='selected original window end')
    ax[2].set(title='Full photon arrival distribution',xlabel='source-nominal delay(ns)',ylabel='counts / native bin')
    ax[2].legend(fontsize=8)
    return fig

def kernel_figure(kernel):
    fig,ax=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    profile=kernel.sum(axis=(1,2));time=np.linspace(0,4/(3e8/1.12),256)*1e9
    ax[0].plot(time,profile);ax[0].set(title='Normalized double-pass diffusion',xlabel='model delay(ns)',ylabel='probability mass / model bin')
    im=ax[1].imshow(np.fft.fftshift(kernel.sum(axis=0)),cmap='magma',extent=[-.7,.7,.7,-.7]);fig.colorbar(im,ax=ax[1])
    ax[1].set(title='Spatially integrated impulse',xlabel='model offset x(m)',ylabel='model offset y(m)')
    ax[2].bar(['native/comment','resampled/comment','effective kernel'],[16,416*16/127,4/(3e8/1.12)/255*1e12],color=['#507b9a','#ddad56','#bd5962'])
    ax[2].set(title='Distinct clocks; no absolute calibration',ylabel='sample interval(ps)');ax[2].tick_params(axis='x',labelrotation=15)
    return fig

def comparison_figure(result):
    names=['observations','migration_only','deconvolution_only','reconstruction']
    titles=['Observed normalized counts','Migration only','Deconvolution only (signed)','Combined CDT (fresh)']
    fig,ax=plt.subplots(3,4,figsize=(14,10),layout='constrained')
    for j,(key,title) in enumerate(zip(names,titles)):
        a=result[key]
        for i,axis in enumerate([0,1,2]):
            im=ax[i,j].imshow(a.max(axis=axis),aspect='auto',cmap='coolwarm' if key=='deconvolution_only' else 'magma')
            ax[i,j].set(title=title if i==0 else ['','Depth/y projection','Depth/x projection'][i],xlabel='scan index',ylabel='lateral index' if not i else 'nominal depth index')
            fig.colorbar(im,ax=ax[i,j],shrink=.75)
    fig.suptitle('Every original scientific comparison retained; panels have individual intensity scales')
    return fig

def slice_figure(field):
    indices=[40,48,56,64,72,80]
    fig,ax=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for a,i in zip(ax.flat,indices):
        im=a.imshow(field[i],vmin=0,vmax=float(field.max()),cmap='magma',extent=[-.35,.35,.35,-.35])
        a.set(title=f'Nominal depth{i/127:.3f}m (index{i})',xlabel='nominal y(m)',ylabel='nominal x(m)')
    fig.colorbar(im,ax=list(ax.flat),label='relative response (common scale)')
    return fig

def parity_figure(fresh,upstream):
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    error=fresh-upstream
    im=ax[0].imshow(upstream.max(axis=0),cmap='magma');fig.colorbar(im,ax=ax[0]);ax[0].set_title('Original run (method reference)')
    im=ax[1].imshow(np.max(abs(error),axis=0),cmap='viridis');fig.colorbar(im,ax=ax[1]);ax[1].set_title('Full-depth max absolute difference')
    summary=metrics(fresh,upstream)
    ax[2].scatter(upstream.ravel()[::11],fresh.ravel()[::11],s=1,alpha=.4)
    ax[2].plot([0,upstream.max()],[0,upstream.max()],'k--',lw=.7)
    ax[2].set(xlabel='original response',ylabel='fresh response',title=f'NCC={summary["ncc"]:.9f}\nDR-NRMSE={summary["nrmse"]:.3g}')
    return fig

def sensitivity_figure(fields,labels,title):
    fig,ax=plt.subplots(2,len(fields),figsize=(13,7),layout='constrained')
    vmax=max(float(a.max()) for a in fields)
    for j,(field,label) in enumerate(zip(fields,labels)):
        im=ax[0,j].imshow(field.max(axis=0),vmin=0,vmax=vmax,cmap='magma');ax[0,j].set_title(label)
        profile=field.sum(axis=(1,2));ax[1,j].plot(np.arange(128)/127,profile)
        ax[1,j].set(xlabel='nominal depth(m)',ylabel='sum relative response',title=f'Peak nominal depth{np.argmax(profile)/127:.3f}m')
    fig.colorbar(im,ax=list(ax[0]),label='relative response (common scale)')
    fig.suptitle(title)
    return fig

def controls_figure(fields):
    target=[.3,.5,.75]; recovered=[]
    fig,ax=plt.subplots(1,4,figsize=(14,3.7),layout='constrained')
    for j,field in enumerate(fields):
        peak=np.unravel_index(np.argmax(field),field.shape);recovered.append(peak[0]/127)
        im=ax[j].imshow(field.max(axis=1),aspect='auto',cmap='magma',extent=[-.35,.35,1,0])
        ax[j].axhline(target[j],color='cyan',ls='--',label='independent target')
        ax[j].set(title=f'Known point{target[j]:.2f}m',xlabel='nominal y(m)',ylabel='nominal depth(m)');ax[j].legend(fontsize=8)
    ax[3].plot(target,recovered,'o-',label='fresh recovered peak');ax[3].plot([.2,.8],[.2,.8],'k--',label='identity')
    ax[3].set(xlabel='independent target depth(m)',ylabel='nominal recovered depth(m)',title='Resolution control, not real-data truth');ax[3].legend(fontsize=8)
    return fig
