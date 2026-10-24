"""Scientific figures from actual freshly computed maps; no backend mutation."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import defocus_derivative
def metrics(x,r):
    a=x.ravel();b=r.ravel();den=np.linalg.norm(a)*np.linalg.norm(b)
    if np.ptp(b)<=0:raise ValueError('Degenerate reference')
    return dict(ncc=float(np.dot(a,b)/den) if den else 0.,nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
def review_figures(unflipped,flipped,fresh,upstream,comparisons,figure_dir):
    """Export five figure groups and exact numeric sources; return(figures,record).

    fresh and comparisons contain actual reconstruct() dictionaries. upstream
    contains independently frozen standard-method estimates, never truth.
    """
    directory=Path(figure_dir);directory.mkdir(parents=True,exist_ok=True)
    extent=np.array([-.5,511.5,511.5,-.5])*5.1066/1000
    figures=[];numeric={};record={'comparison_to_original':{},'observation_relative_l2':{},'comparisons':{}}
    def panel(ax,image,title,unit,cmap='viridis',symmetric=False,limits=None):
        numeric['panel_'+str(len(numeric))]=np.asarray(image)
        if symmetric:
            limit=float(np.max(np.abs(image)));limits=(-limit,limit) if limit else (-1.,1.)
        kwargs=dict(vmin=limits[0],vmax=limits[1]) if limits else {}
        im=ax.imshow(image,extent=extent,origin='upper',cmap=cmap,**kwargs)
        ax.set(title=title,xlabel='x (µm)',ylabel='y (µm)')
        ax.tick_params(labelsize=8);ax.title.set_fontsize(10)
        ax.figure.colorbar(im,ax=ax,shrink=.78,label=unit)
    fig,axes=plt.subplots(2,3,figsize=(13,7.4),constrained_layout=True)
    for row,stack in enumerate([unflipped,flipped]):
        lo=float(stack[[2,3,4]].min());hi=float(stack[[2,3,4]].max())
        for col,index in enumerate([2,3,4]):panel(axes[row,col],stack[index],('Unflipped' if row==0 else 'Flipped')+f' df={[-199.065,0,199.065][col]} µm','detector a.u.','gray',limits=(lo,hi))
    fig.suptitle('1 | Real aligned focal observations, native central ROI');figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7.4),constrained_layout=True)
    for ax,key in zip(axes.ravel()[:4],['phase_B','phase_E','Bx','By']):
        panel(ax,fresh[key],{'phase_B':'Magnetic phase','phase_E':'Electrostatic phase','Bx':'Projected induction Bx','By':'Projected induction By'}[key],'rad' if 'phase' in key else 'T·nm','viridis' if 'phase' in key else 'RdBu_r',symmetric='phase' not in key)
    magnitude=np.hypot(fresh['Bx'],fresh['By']);panel(axes[1,1],magnitude,'Projected induction magnitude','T·nm')
    ax=axes[1,2];step=20;y,x=np.mgrid[0:512:step,0:512:step];bx=fresh['Bx'][::step,::step];by=fresh['By'][::step,::step]
    ax.quiver(x*5.1066/1000,y*5.1066/1000,bx,by,magnitude[::step,::step],cmap='viridis',angles='xy',scale_units='xy',scale=300)
    ax.set(xlim=extent[:2],ylim=extent[2:],xlabel='x (µm)',ylabel='y (µm)',title='Induction direction (arrows)')
    fig.suptitle('2 | Fresh phase separation and path-integrated induction; no thickness division');figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7.4),constrained_layout=True)
    for row,key in enumerate(['phase_B','phase_E']):
        ref=upstream[key];m=metrics(fresh[key],ref);record['comparison_to_original'][key]=m
        limits=(float(ref.min()),float(ref.max()))
        panel(axes[row,0],ref,'Reproduced original '+key,'rad',limits=limits)
        panel(axes[row,1],fresh[key],f'Fresh local; NCC={m["ncc"]:.12f}','rad',limits=limits)
        panel(axes[row,2],fresh[key]-ref,f'Difference; max={np.max(abs(fresh[key]-ref)):.2e}','rad','RdBu_r',True)
    for key in ['Bx','By']:record['comparison_to_original'][key]=metrics(fresh[key],upstream[key])
    fig.suptitle('3 | Independent original-code run versus fresh local extraction (method reference, not truth)');figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7.4),constrained_layout=True)
    for row,label in enumerate(['B','E']):
        observed=fresh['difference_'+label]/398130
        predicted=defocus_derivative(fresh['phase_'+label],fresh['infocus'])
        residual=float(np.linalg.norm(predicted-observed)/np.linalg.norm(observed));record['observation_relative_l2'][label]=residual
        limit=max(float(abs(observed).max()),float(abs(predicted).max()))*1e6
        panel(axes[row,0],observed*1e6,label+' measured central focus derivative','10⁻⁶ a.u./nm','RdBu_r',limits=(-limit,limit))
        panel(axes[row,1],predicted*1e6,label+' independent spectral TIE prediction','10⁻⁶ a.u./nm','RdBu_r',limits=(-limit,limit))
        panel(axes[row,2],(predicted-observed)*1e6,f'Residual; relative L2={residual:.4f}','10⁻⁶ a.u./nm','RdBu_r',True)
    fig.suptitle('4 | Observation closure exposes finite-defocus, bandwidth and boundary limitations');figures.append(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,7.4),constrained_layout=True)
    groups=[('standard',fresh)]+list(comparisons.items())
    maximum=max(float(np.hypot(r['Bx'],r['By']).max()) for _,r in groups)
    for ax,(name,r) in zip(axes.ravel(),groups):
        mag=np.hypot(r['Bx'],r['By']);delta=float(np.linalg.norm(mag-magnitude)/np.linalg.norm(magnitude))
        record['comparisons'][name]={'magnitude_relative_change':delta,'peak_T_nm':float(mag.max())}
        panel(ax,mag,f'{name}\nrelative change={delta:.3f}','T·nm',limits=(0,maximum))
    for ax in axes.ravel()[len(groups):]:ax.set_visible(False)
    fig.suptitle('5 | Fresh focus, regularization and boundary sensitivity; shared physical color scale');figures.append(fig)
    for key,value in fresh.items():numeric['fresh_'+key]=value
    for key,value in upstream.items():numeric['upstream_'+key]=value
    for name,r in comparisons.items():
        for key in ['phase_B','phase_E','Bx','By']:numeric[name+'_'+key]=r[key]
    numeric.update(unflipped=unflipped,flipped=flipped)
    np.savez_compressed(directory/'numeric_sources.npz',**numeric)
    (directory/'metrics.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for index,fig in enumerate(figures,1):
        for ext in ['png','svg','pdf']:fig.savefig(directory/f'figure_{index}.{ext}',dpi=160)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return figures,record
