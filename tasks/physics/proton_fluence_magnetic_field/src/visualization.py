"""Scientific comparisons from actually computed inputs; no reconstruction caches."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

def image_metrics(estimate, reference):
    """Cosine NCC, dynamic-range NRMSE, relative L2; no gain/shift fitting."""
    a=np.asarray(estimate,dtype=np.float64).ravel();b=np.asarray(reference,dtype=np.float64).ravel()
    norm=np.linalg.norm(b);scale=np.ptp(b)
    if norm==0 or scale==0:raise ValueError('Degenerate reference')
    anorm=np.linalg.norm(a)
    return dict(ncc=float(np.dot(a,b)/(anorm*norm)) if anorm else 0.,
                nrmse=float(np.sqrt(np.mean((a-b)**2))/scale),
                relative_l2=float(np.linalg.norm(a-b)/norm))

def field_metrics(estimate, reference):
    return {name:image_metrics(estimate[i],reference[i]) for i,name in enumerate(['Bx','By'])}

def _image(fig,ax,array,x,y,title,unit='G cm',signed=False,limit=None,norm=None):
    if signed:
        limit=float(np.max(np.abs(array))) if limit is None else limit
        limit=max(limit,np.finfo(float).tiny)
        kw=dict(cmap='RdBu_r',vmin=-limit,vmax=limit)
    else:
        kw=dict(cmap='viridis',norm=norm) if limit is None else dict(cmap='viridis',vmin=0,vmax=limit)
    im=ax.imshow(array.T,origin='lower',extent=[x[0,0],x[-1,0],y[0,0],y[0,-1]],aspect='equal',**kw)
    ax.set(title=title,xlabel='x (cm)',ylabel='y (cm)')
    fig.colorbar(im,ax=ax,label=unit,shrink=.82,extend='both' if signed else ('max' if limit is not None else 'neither'))
    return im

def make_figures(data, truth, upstream, fresh, linear, forwards, directory=None):
    """Return six figures and full numeric metrics; optional PNG/SVG/PDF exports.

    All fields use x/y array axes. forwards maps names to genuinely computed
    counts arrays: truth4/8/16/32, fresh4/8/16/32, wrong32 and linear16.
    """
    x,y,f=data['x'],data['y'],data['fluence'];b=fresh['reconstruction']
    colors={'Upstream nonlinear':'#656c78','Fresh nonlinear':'#167f78','Linear Poisson':'#bf5b25'}
    methods={'Upstream nonlinear':upstream,'Fresh nonlinear':b,'Linear Poisson':linear}
    scores={k:field_metrics(v,truth) for k,v in methods.items()}
    ray_scores={k:image_metrics(v,f) for k,v in forwards.items()}
    figs=[]
    with plt.rc_context({'font.size':10,'axes.titlesize':11,'figure.dpi':120,'savefig.facecolor':'white'}):
        fig,ax=plt.subplots(2,2,figsize=(11.5,8),layout='constrained')
        _image(fig,ax[0,0],f,x,y,'Bound author observations','counts/pixel')
        _image(fig,ax[0,1],f/np.mean(f)-1,x,y,'Strong, signed relative fluence contrast','f / mean(f) − 1',signed=True)
        ax[1,0].hist(f.ravel(),bins=70,color='#326a91');ax[1,0].set(xlabel='Counts/pixel',ylabel='Number of pixels',yscale='log',title=f'Total stored counts: {f.sum():,.0f}')
        ix=np.unravel_index(np.argmax(f),f.shape)[0]
        ax[1,1].plot(y[0,:],f[ix,:],color='#326a91',label=f'x = {x[ix,0]:.4f} cm')
        ax[1,1].axhline(np.mean(f),ls='--',color='#9a5844',label='Uniform incident normalization')
        ax[1,1].set(xlabel='y (cm)',ylabel='Counts/pixel',title='Observed bright-structure line profile');ax[1,1].legend(fontsize=8)
        ax[1,1].xaxis.set_major_locator(MaxNLocator(5))
        fig.suptitle('1  Data and nonlinear imaging contrast',fontsize=15);figs.append(fig)

        fig,ax=plt.subplots(3,2,figsize=(10.5,11),layout='constrained');vmax=np.max(np.abs(truth))
        for row,(name,field) in enumerate([('Actual author field',truth),('Independent original run',upstream),('Fresh local full run',b)]):
            for j,comp in enumerate(['Bx','By']):_image(fig,ax[row,j],field[j],x,y,f'{name}: {comp}',signed=True,limit=vmax)
        fig.suptitle('2  Both signed field components; shared truth scale\nTriangle colorbars mark display saturation; full values enter metrics',fontsize=14);figs.append(fig)

        fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        _image(fig,ax[0,0],linear[0],x,y,'Fresh weak-contrast Poisson: Bx',signed=True,limit=vmax)
        _image(fig,ax[0,1],b[0]-truth[0],x,y,'Nonlinear Bx error (own signed scale)',signed=True)
        _image(fig,ax[0,2],linear[0]-truth[0],x,y,'Linear Bx error (own signed scale)',signed=True)
        iy=np.unravel_index(np.argmax(np.abs(truth[0])),truth[0].shape)[1]
        ax[1,0].plot(x[:,0],truth[0,:,iy],color='black',lw=2,label='Actual field')
        for name,field in methods.items():ax[1,0].plot(x[:,0],field[0,:,iy],color=colors[name],ls='--' if 'Upstream' in name else '-',label=name)
        ax[1,0].set(xlabel='x (cm)',ylabel='Bx (G cm)',title=f'Absolute field, y={y[0,iy]:.4f} cm');ax[1,0].legend(fontsize=7)
        positions=np.arange(2)
        for j,(name,field) in enumerate(methods.items()):
            values=[scores[name][c]['nrmse'] for c in ['Bx','By']]
            bars=ax[1,1].bar(positions+(j-1)*.24,values,width=.23,label=name,color=colors[name])
            ax[1,1].bar_label(bars,labels=[f'{v:.4f}' for v in values],padding=3,fontsize=7,rotation=90)
            ax[1,2].bar(positions+(j-1)*.24,[np.sqrt(np.mean(field[k]**2))/np.sqrt(np.mean(truth[k]**2)) for k in range(2)],width=.23,label=name,color=colors[name])
        for a in ax[1,1:]:a.set_xticks(positions,['Bx','By'])
        ax[1,1].set(ylabel='Dynamic-range NRMSE',title='Full-domain errors; colors as line legend',yscale='log',ylim=(.001,.3))
        ax[1,0].xaxis.set_major_locator(MaxNLocator(5))
        ax[1,2].axhline(1,color='black',ls='--');ax[1,2].set(ylabel='Reconstructed / actual RMS',title='Amplitude diagnostic; no fitted gain')
        fig.suptitle('3  Nonlinear transport versus its weak-contrast approximation',fontsize=15);figs.append(fig)

        fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        steps=np.arange(1,len(fresh['residual_norm'])+1)
        ax[0,0].semilogy(steps/1000,fresh['residual_norm'],color='#167f78');ax[0,0].axhline(.001,color='#b24040',ls='--',label='Native tolerance')
        ax[0,0].set(xlabel='Euler updates (thousands)',ylabel='Euclidean log-mass residual',title=f"{fresh['iterations']:,} updates: {fresh['stopping']}");ax[0,0].legend(fontsize=8)
        ax[0,0].xaxis.set_major_locator(MaxNLocator(5))
        selected=sorted(fresh['snapshots'])
        for k,c in enumerate(['Bx','By']):ax[0,1].loglog([s+1 for s in selected],[image_metrics(fresh['snapshots'][s][k],truth[k])['nrmse'] for s in selected],marker='o' if k==0 else 'x',ls='-' if k==0 else '--',mfc='none',label=c)
        ax[0,1].set(xlabel='Euler updates',ylabel='Dynamic-range NRMSE',title='Actual computed trajectory snapshots');ax[0,1].legend()
        _image(fig,ax[1,0],fresh['det'],x,y,'Map Jacobian determinant','dimensionless',norm=LogNorm(vmin=np.min(fresh['det']),vmax=np.max(fresh['det'])))
        _image(fig,ax[1,1],np.expm1(fresh['Fn']),x,y,'Pulled-back relative mass residual','f(T) det(H) / f0 − 1',signed=True)
        fig.suptitle('4  Relaxation budget and injective-map physics',fontsize=15);figs.append(fig)

        fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for a,key,title in [(ax[0,0],None,'Actual observed counts'),(ax[0,1],'fresh32','Fresh field →32² rays/pixel'),(ax[0,2],'linear16','Linear field →16² rays/pixel')]:
            _image(fig,a,f if key is None else forwards[key],x,y,title,'counts/pixel',limit=np.max(f))
        _image(fig,ax[1,0],forwards['fresh32']-f,x,y,'Fresh ray forward error','counts/pixel',signed=True)
        ix=np.unravel_index(np.argmax(f),f.shape)[0]
        for array,label,col in [(f,'Observed','black'),(forwards['fresh32'],'Fresh nonlinear','#167f78'),(forwards['linear16'],'Linear','#bf5b25')]:ax[1,1].plot(y[0,:],array[ix,:],label=label,color=col)
        ax[1,1].set(xlabel='y (cm)',ylabel='Counts/pixel',title='Independent detector-bin forward check');ax[1,1].legend(fontsize=8)
        ax[1,1].xaxis.set_major_locator(MaxNLocator(5))
        labels=['Fresh nonlinear','Linear Poisson'];values=[ray_scores['fresh32']['nrmse'],ray_scores['linear16']['nrmse']]
        ax[1,2].bar(labels,values,color=['#167f78','#bf5b25']);ax[1,2].set(ylabel='Count-image dynamic-range NRMSE',title='No gain, alignment or crop')
        fig.suptitle('5  Independent ray-bin forward model; all pixels and count conservation',fontsize=15);figs.append(fig)

        fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        stride=10;xx=x[::stride,::stride];yy=y[::stride,::stride]
        quiver=ax[0,0].quiver(xx,yy,truth[0,::stride,::stride],truth[1,::stride,::stride],color='#326a91')
        ax[0,0].quiverkey(quiver,.80,.93,10000,'10000 G cm',labelpos='S',fontproperties={'size':8})
        ax[0,0].set(xlabel='x (cm)',ylabel='y (cm)',title='Actual field: every10th pixel; beam in',aspect='equal')
        _image(fig,ax[0,1],forwards['truth32'],x,y,'Actual field, specified beam sign','counts/pixel',limit=np.max(f))
        _image(fig,ax[0,2],forwards['wrong32'],x,y,'Actual field, reversed beam control','counts/pixel',limit=np.max(f))
        _image(fig,ax[1,0],forwards['truth32']-f,x,y,'Approximate-model / stored-data discrepancy','counts/pixel',signed=True)
        levels=[4,8,16,32]
        for prefix,label,col in [('truth','Actual field','#656c78'),('fresh','Fresh inferred field','#167f78')]:
            ax[1,1].semilogx(levels,[ray_scores[f'{prefix}{n}']['nrmse'] for n in levels],base=2,marker='o',label=label,color=col)
        ax[1,1].set(xlabel='Rays per pixel axis',ylabel='Count-image dynamic-range NRMSE',title='Forward discretization refinement');ax[1,1].legend(fontsize=8)
        ax[1,2].bar(['Into screen','Reversed beam'],[ray_scores['truth32']['ncc'],ray_scores['wrong32']['ncc']],color=['#326a91','#b95145'])
        ax[1,2].set(ylim=(0,1.05),ylabel='Count-image cosine NCC',title='Handedness control, unchanged field')
        fig.suptitle('6  Physical sign audit and forward-model discrepancy\nStored author counts are not claimed to be exact predictions of this approximate forward model',fontsize=14);figs.append(fig)
    metrics={'field':scores,'ray_forward':ray_scores,
             'iterations':fresh['iterations'],'stopping':fresh['stopping'],
             'runtime_seconds':fresh['runtime_seconds'],
             'final_residual_norm':float(fresh['residual_norm'][-1]),
             'rms_log_mass_residual':float(np.sqrt(np.mean(fresh['Fn']**2))),
             'negative_determinant_count':fresh['negative_determinant_count']}
    if directory is not None:
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for i,fig in enumerate(figs,1):
            for extension in ['png','svg','pdf']:fig.savefig(directory/f'figure{i}.{extension}',dpi=180)
        np.savez_compressed(directory/'figure_numeric_source.npz',x=x,y=y,fluence=f,actual_truth=truth,
                            upstream=upstream,fresh=b,linear=linear,det=fresh['det'],Fn=fresh['Fn'],
                            residual_norm=fresh['residual_norm'],**{f'forward_{k}':v for k,v in forwards.items()},
                            **{f'snapshot_{k}':v for k,v in fresh['snapshots'].items()})
        (directory/'figure_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
        provenance={'producer':'Caller-computed local fields; upstream explicitly separate; independent author truth unchanged. See associated notebook or CLI execution record.',
                    'numeric_file':'figure_numeric_source.npz','metrics_file':'figure_metrics.json',
                    'figure_consumers':{'figure1':['fluence','x','y'],'figure2':['actual_truth','upstream','fresh'],
                    'figure3':['actual_truth','linear','fresh','upstream'],'figure4':['residual_norm','snapshot_*','det','Fn'],
                    'figure5':['fluence','forward_fresh32','forward_linear16'],
                    'figure6':['actual_truth','fluence','forward_truth*','forward_fresh*','forward_wrong32']}}
        (directory/'figure_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return figs,metrics
