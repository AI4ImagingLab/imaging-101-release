"""Scientific native-image, geometry, deprojection and sensitivity comparisons."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .physics_model import forward_profile,ARCMIN_KPC

def make_figures(data,output,stages,upstream,experiments,export_dir=None):
    figures=[];m=stages['measured'];edges=data['radial_edges'];r=(edges[1:]+edges[:-1])/2
    density=output['reconstruction'][0];q=output['emissivity'][0]
    def finish(fig,title):
        fig.suptitle(title,fontsize=14);fig.tight_layout(rect=(0,0,1,.95));figures.append(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    for ax,key,label in zip(axs,['counts','exposure','instrument_background'],['Observed counts [count]','On-axis exposure [s]','Instrument background [count]']):
        image=data[key];im=ax.imshow(image,origin='lower',cmap='magma',norm=LogNorm(vmin=max(image[image>0].min(),1e-3),vmax=image.max()))
        ax.plot(*m['center'],'c+',ms=10);ax.set(xlabel='x [pixel]',ylabel='y [pixel]',title=label);fig.colorbar(im,ax=ax,shrink=.75)
    finish(fig,'1. Complete real ROSAT acquisition; center from count image')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    ax=axs[0];ax.errorbar(m['outer_centers'],m['outer_profile'],m['outer_error'],fmt='.',ms=3,label='Real outer annuli')
    ax.axhline(m['background'],color='C3',label='Independent constant fit');ax.axvspan(30,40,color='C3',alpha=.1,label='Background region')
    ax.set(xlabel='Radius [arcmin]',ylabel='Surface rate [count s$^{-1}$ arcmin$^{-2}$]',yscale='log');ax.legend(fontsize=8)
    ax=axs[1];ax.errorbar(r,m['profile_raw'],m['error_raw'],fmt='o',ms=3,label='Before sky subtraction');ax.errorbar(r,m['profile'],m['error'],fmt='.',label='After subtraction')
    ax.set(xlabel='Radius [arcmin]',ylabel='Surface rate [count s$^{-1}$ arcmin$^{-2}$]',yscale='log');ax.legend()
    finish(fig,'2. Exposure correction, independent sky measurement and propagated errors')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));v=stages['volume']
    im=axs[0].imshow(np.where(v>0,np.log10(np.maximum(v,1)),np.nan),origin='lower',cmap='viridis');fig.colorbar(im,ax=axs[0],label='log10 volume [cm$^3$]')
    axs[0].set(xlabel='Emitting shell',ylabel='Projected annulus',title='Upper triangular line-of-sight mixing')
    axs[1].plot(r,stages['correction'],'o-');axs[1].set(xlabel='Radius [arcmin]',ylabel='Exterior emission fraction',title='Finite-field boundary contribution')
    finish(fig,'3. Explicit spherical geometry and exterior assumption')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,fresh,old,error,label in [(axs[0],density,upstream['output_density'],stages['density_uncertainty'],'nH [cm$^{-3}$]'),(axs[1],q,upstream['output_emissivity'],stages['emissivity_std'],'nH$^2$ [cm$^{-6}$]')]:
        ax.plot(r*ARCMIN_KPC,old,'k--',label='Reproduced original');ax.errorbar(r*ARCMIN_KPC,fresh,error,fmt='o',ms=3,label='Fresh cleaned; source uncertainty')
        ax.set(xlabel='Radius [kpc]',ylabel=label,yscale='log');ax.legend(fontsize=8)
    finish(fig,'4. Original and fresh 35-shell physical fields; neither is ground truth')
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for quantity,label in [(q,'Projected q'),(density**2,'Projected nH squared')]:
        prediction=forward_profile(quantity,edges)+stages['correction']*m['profile']
        axs[0].plot(r,prediction,label=label);axs[1].plot(r,(prediction-m['profile'])/m['error'],'o-',ms=3,label=label)
    axs[0].errorbar(r,m['profile'],m['error'],fmt='k.',label='Real annuli');axs[0].set(yscale='log',ylabel='Surface rate [count s$^{-1}$ arcmin$^{-2}$]');axs[0].legend()
    axs[1].axhline(0,color='k',lw=.7);axs[1].axhspan(-1,1,alpha=.1,color='gray');axs[1].set(xlabel='Radius [arcmin]',ylabel='Residual / count uncertainty');axs[1].legend()
    finish(fig,'5. Forward consistency against every observed annulus')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for name,values in experiments.items():
        axs[0].plot(r,values[0],label=name);axs[1].plot(r,(values[0]-density)/density,label=name)
    axs[0].plot(r,density,'k--',label='Standard fresh');axs[0].set(yscale='log',xlabel='Radius [arcmin]',ylabel='nH [cm$^{-3}$]');axs[0].legend(fontsize=8)
    axs[1].axhline(0,color='k',lw=.7);axs[1].set(xlabel='Radius [arcmin]',ylabel='Relative density change');axs[1].legend(fontsize=8)
    finish(fig,'6. Measured sensitivity to exterior correction, smoothing and MC integration')
    if export_dir is not None:
        path=Path(export_dir);path.mkdir(parents=True,exist_ok=True)
        numeric={**{'input_'+k:v for k,v in data.items()},'radius':r,'density':density,'emissivity':q,'volume':v,'exterior_fraction':stages['correction'],**{'measured_'+k:np.asarray(value) for k,value in m.items()},**{'upstream_'+k:value for k,value in upstream.items()},**{'experiment_'+name.replace(' ','_'):value[0] for name,value in experiments.items()}}
        np.savez_compressed(path/'numeric_sources.npz',**numeric)
        for i,fig in enumerate(figures,1):
            for ext in ('png','svg','pdf'):fig.savefig(path/f'figure_{i:02}.{ext}',dpi=150,bbox_inches='tight')
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.suffix in ('.png','.svg','.pdf','.npz')}
        (path/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return figures
