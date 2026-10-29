
"""Evidence-bearing spatial and numerical review figures; no backend changes on import."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import load_data
from .solvers import solve,solve_sfim
def metrics(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    n=np.linalg.norm(a)*np.linalg.norm(b)
    return (float(np.dot(a,b)/n) if n else 0.),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'figure.dpi':110,'savefig.dpi':180,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def plate(ax,image,title,cmap='viridis',**kwargs):
    artist=ax.imshow(image,origin='lower',cmap=cmap,**kwargs);ax.set_title(title,pad=8)
    ax.set_xlabel('Column (PAN pixel)');ax.set_ylabel('Row (PAN pixel)');return artist

from .physics_model import interpolate23,lowpass,degrade
def rgb(x):return np.clip(x[...,[2,1,0]],0,1)
def review(data,upstream):
    """Compute original GSA, interpolation, equal-band ablation and detail-gain sweep."""
    style();fresh=solve(data);up=interpolate23(data['multispectral']);pan=data['panchromatic'][...,0];ref=upstream['reconstruction'][0].astype(float)/255
    figs=[];tables=[];sources={}
    fig,axs=plt.subplots(2,3,figsize=(12.5,8),layout='constrained')
    for ax,v,title in zip(axs[0],[pan,rgb(up),rgb(fresh)],['Native high-resolution PAN','23-tap interpolation only','Fresh GSA | bands3,2,1 display']):
        im=plate(ax,v,title,'gray',vmin=0,vmax=1)
    for ax,v,title in zip(axs[1],[rgb(ref)[80:160,80:160],rgb(fresh)[80:160,80:160],abs(fresh-ref).mean(-1)],['Original GSA: detail crop','Fresh GSA: same detail crop','Pre-quantization vs original uint8']):
        im=plate(ax,v,title,'magma');
        if v.ndim==2:fig.colorbar(im,ax=ax,shrink=.8,label='Mean absolute normalized radiometry')
    fig.suptitle('GF2 Beijing supplied native-resolution observations | registered ROI',fontsize=15);figs.append(('01_native_fusion',fig))
    gains=[0.,.5,1.,1.5];variants=[solve(data,detail_gain=g) for g in gains]
    fig,axs=plt.subplots(1,4,figsize=(15,4.1),layout='constrained')
    for ax,g,v in zip(axs,gains,variants):plate(ax,rgb(v[80:176,80:176]),f'Detail gain = {g:g}')
    fig.suptitle('Fresh spatial-detail sweep | identical radiometry/display stretch',fontsize=16);figs.append(('02_detail_gain',fig))
    unmatched=solve(data,adaptive=False)
    sfim_failure=solve_sfim(data)
    fig,axs=plt.subplots(1,4,figsize=(16,4.2),layout='constrained')
    plate(axs[0],rgb(unmatched),'Equal-band intensity ablation')
    plate(axs[1],rgb(sfim_failure),'SFIM dark-pixel failure')
    difference=np.sqrt(np.mean((fresh-up)**2,-1));im=plate(axs[2],difference,'Injected detail amplitude','magma');fig.colorbar(im,ax=axs[2],label='RMS normalized radiometry')
    for c,col in enumerate(['#0072B2','#009E73','#D55E00','#CC79A7']):
        axs[3].plot(fresh[128,:,c],color=col,lw=1,label=f'Fresh band{c+1}');axs[3].plot(up[128,:,c],color=col,lw=.9,ls='--')
    axs[3].set(xlabel='Column (PAN pixel)',ylabel='Normalized radiometry',title='Solid GSA / dashed interpolation');axs[3].legend(fontsize=8)
    fig.suptitle('Regression mechanism, equal-band ablation and retained SFIM failure',fontsize=16);figs.append(('03_mechanism',fig))
    def diagnostics(v):
        gray=v.mean(-1);pg=np.hypot(*np.gradient(pan));vg=np.hypot(*np.gradient(gray))
        contrast=float(np.corrcoef(pg.ravel(),vg.ravel())[0,1]);consistency=float(np.sqrt(np.mean((degrade(v)-data['multispectral'])**2)))
        ncc,nrmse=metrics(np.uint8(v*255),upstream['reconstruction'][0])
        return {'NCC_vs_original':ncc,'NRMSE_vs_original':nrmse,'gradient_correlation_PAN':contrast,'box_model_MS_RMSE':consistency,'clipped_fraction':float(np.mean((v<=0)|(v>=1)))}
    for g,v in zip(gains,variants):tables.append({'experiment':f'detail gain={g:g}',**diagnostics(v)})
    tables.append({'experiment':'equal-band intensity',**diagnostics(unmatched)})
    tables.append({'experiment':'SFIM failure comparison',**diagnostics(sfim_failure)})
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
    axs[0].plot(gains,[r['gradient_correlation_PAN'] for r in tables[:4]],'o-',color='#0072B2');axs[0].set(xlabel='Detail gain',ylabel='Gradient magnitude correlation with native PAN',title='Spatial consistency diagnostic')
    axs[1].plot(gains,[r['box_model_MS_RMSE'] for r in tables[:4]],'s--',color='#D55E00');axs[1].set(xlabel='Detail gain',ylabel='Normalized radiometry RMSE',title='Box-model low-resolution consistency')
    fig.suptitle('Spatial/spectral tradeoff | diagnostics, not ground-truth accuracy',fontsize=14);figs.append(('04_tradeoff',fig))
    sources.update(fresh_normalized=fresh,interpolation=up,equal_band_ablation=unmatched,sfim_failure_comparison=sfim_failure,detail_variants=np.asarray(variants),detail_gains=np.array(gains))
    return figs,tables,sources
