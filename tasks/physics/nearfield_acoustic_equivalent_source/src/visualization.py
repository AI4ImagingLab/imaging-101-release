"""Active scientific review. Never switches Matplotlib's backend."""
from pathlib import Path
import json,time,hashlib,importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt
from .preprocessing import load_pressure
from .physics_model import forward
from .solvers import solve,difference

def review(task_dir,figure_dir):
    """Actively reconstruct and export five groups; return (figures,summary)."""
    task=Path(task_dir);dest=Path(figure_dir);dest.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    p=load_pressure(task/'data/raw_data.npz');up=dict(np.load(task/'notebooks/reference_outputs/upstream_methods.npz',allow_pickle=False))
    fresh={};diag={};timing={};nums={'pressure':p};summary={}
    for method in ['LN','TV','TGV']:
        start=time.perf_counter();fresh[method],diag[method]=solve(p,method,return_history=True);timing[method]=time.perf_counter()-start
        ref=up[method];q=fresh[method];res=forward(q)-p
        summary[method]={'seconds':timing[method],'relative_complex_l2_original':float(np.linalg.norm(q-ref)/np.linalg.norm(ref)),'observation_relative_l2':float(np.linalg.norm(res)/np.linalg.norm(p)),'lambda':diag[method]['lambdaa']}
        nums['fresh_'+method]=q;nums['original_'+method]=ref;nums['predicted_'+method]=forward(q)
    spec=importlib.util.spec_from_file_location('nah_independent_judge',task/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    np.savez(task/'output/reconstruction.npz',reconstruction=fresh['TGV'][None]);report,code=judge.evaluate(task/'output/reconstruction.npz')
    if code:raise RuntimeError(report)
    summary['judge']=report;figs=[]
    def panel(ax,a,title,unit,cmap='viridis',limits=None,receiver=False):
        # Extents represent actual pixel centers including inclusive endpoints.
        h,w=a.shape;dx=.825/(w-1);dy=.6/(h-1)
        if limits is None and cmap=='RdBu_r':
            peak=float(np.max(np.abs(a)));limits=(-peak,peak)
        if limits is None and np.min(a)>=0 and np.max(a)==0:limits=(0.,1e-7)
        im=ax.imshow(a,origin='lower',extent=[-dx/2,.825+dx/2,-dy/2,.6+dy/2],cmap=cmap,**({} if limits is None else {'vmin':limits[0],'vmax':limits[1]}));ax.set(title=title,xlabel='x (m)',ylabel='y (m)');plt.colorbar(im,ax=ax,label=unit,shrink=.8)
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for a,v,title,unit,cmap in zip(ax.ravel(),[abs(p),np.angle(p),p.real,p.imag],['Measured composed magnitude','Measured composed phase','Measured real pressure','Measured imaginary pressure'],['U','rad','U','U'],['viridis','twilight','RdBu_r','RdBu_r']):panel(a,v,title,unit,cmap)
    fig.suptitle('1. Real author-processed pressure: 108 complex samples');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained');q=fresh['TGV'];ref=up['TGV'];peak=float(abs(ref).max())
    for a,v,title,unit,cmap,limits in zip(ax.ravel(),[abs(ref),abs(q),abs(q-ref),q.real,q.imag,np.angle(q)],['Original second-order baseline','Fresh local second-order baseline','Absolute complex discrepancy','Fresh real source strength','Fresh imaginary source strength','Fresh source phase'],['U m³/(Pa s)']*5+['rad'],['viridis']*3+['RdBu_r','RdBu_r','twilight'],[(0,peak),(0,peak),None,None,None,(-np.pi,np.pi)]):panel(a,v,title,unit,cmap,limits)
    fig.suptitle('2. Equivalent sources; original method output is not ground truth');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for j,m in enumerate(['LN','TV','TGV']):
        panel(ax[0,j],abs(fresh[m]),f'Fresh {m}: source magnitude','U m³/(Pa s)',limits=(0,max(abs(v).max() for v in fresh.values())))
        panel(ax[1,j],abs(forward(fresh[m])-p),f'{m}: pressure residual {summary[m]["observation_relative_l2"]:.4f}','U',limits=(0,max(abs(forward(v)-p).max() for v in fresh.values())))
    fig.suptitle('3. Same real observations, three actively computed regularized inverses');figs.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained');pred=forward(q)
    panel(ax[0,0],abs(pred),'Fresh TGV reprojected magnitude','U');panel(ax[0,1],np.angle(pred),'Fresh TGV reprojected phase','rad','twilight',(-np.pi,np.pi))
    ax[1,0].plot(p.real.ravel(),pred.real.ravel(),'o',label='real');ax[1,0].plot(p.imag.ravel(),pred.imag.ravel(),'x',label='imaginary');lo=min(p.real.min(),p.imag.min());hi=max(p.real.max(),p.imag.max());ax[1,0].plot([lo,hi],[lo,hi],'k--');ax[1,0].set(xlabel='Measured pressure (U)',ylabel='Predicted pressure (U)',title='No fitted phase or gain');ax[1,0].legend()
    ax[1,1].plot(abs(p).ravel(),label='measured');ax[1,1].plot(abs(pred).ravel(),label='predicted');ax[1,1].plot(abs(pred-p).ravel(),label='complex residual');ax[1,1].set(xlabel='Receiver row-major index',ylabel='Pressure (U)',title='All 108 measurements');ax[1,1].legend();fig.suptitle('4. Independent pressure-space consistency');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained');sensitivity=[]
    for j,snr in enumerate([30.,35.,40.]):
        qs=q if snr==35 else solve(p,'TGV',snr_db=snr)[0];nums['TGV_snr_'+str(int(snr))]=qs
        r=float(np.linalg.norm(forward(qs)-p)/np.linalg.norm(p));sensitivity.append({'snr_db':snr,'observation_relative_l2':r});panel(ax[0,j],abs(qs),f'SNR tuning {snr:g} dB; residual {r:.4f}','U m³/(Pa s)')
    iterations=[];residuals=[];penalties=[]
    for it,state in diag['TGV']['history'].items():
        qi=state['q'][0,0];iterations.append(it);residuals.append(float(np.linalg.norm(forward(qi)-p)/np.linalg.norm(p)));penalties.append(float(torch.linalg.norm(difference(torch.from_numpy(qi)),dim=-1).sum()));nums['iteration_'+str(it)]=qi
    ax[1,0].semilogx(iterations,residuals,'o-');ax[1,0].set(xlabel='Iteration',ylabel='Relative pressure residual',title='Fresh second-order trajectory')
    ax[1,1].semilogx(iterations,penalties,'o-');ax[1,1].set(xlabel='Iteration',ylabel='Sum of derivative vector norms',title='Index-unit derivative penalty')
    ax[1,2].bar(['LN','TV','TGV'],[timing[m] for m in ['LN','TV','TGV']]);ax[1,2].set(ylabel='Seconds (CPU, one thread)',title='Measured fresh reconstruction runtime');fig.suptitle('5. Regularization sensitivity and actual iteration history');figs.append(fig)
    nums.update(iterations=np.array(iterations),iteration_residuals=np.array(residuals),iteration_penalties=np.array(penalties));summary['sensitivity']=sensitivity
    np.savez_compressed(dest/'numeric_sources.npz',**nums);(dest/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    manifest={}
    for i,fig in enumerate(figs,1):
        for ext in ['png','svg','pdf']:
            target=dest/f'figure_{i}.{ext}';fig.savefig(target,dpi=150);manifest[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
        plt.close(fig)  # Notebook explicitly displays each labeled group once.
    for name in ['numeric_sources.npz','metrics.json']:manifest[name]=hashlib.sha256((dest/name).read_bytes()).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');return figs,summary
