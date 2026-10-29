
"""Evidence-bearing spatial and numerical review figures; no backend changes on import."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import load_data
from .solvers import solve
def metrics(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    n=np.linalg.norm(a)*np.linalg.norm(b)
    return (float(np.dot(a,b)/n) if n else 0.),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'figure.dpi':110,'savefig.dpi':180,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def plate(ax,image,title,cmap='viridis',**kwargs):
    artist=ax.imshow(image,origin='lower',cmap=cmap,**kwargs);ax.set_title(title,pad=8)
    ax.set_xlabel('Solar west / column (pixel)');ax.set_ylabel('North / row (pixel)');return artist

from .physics_model import response_matrix,forward
def review(data,upstream):
    """Compute the full48x48x20 inverse, propagated errors and three discrepancy/weighting alternatives."""
    style();dem,error,resolution,chi,fit=solve(data);ref=upstream['reconstruction'][0]
    matrix,logt,width=response_matrix(data['response'],data['response_log_temperature'],data['temperature_edges'])
    dt=10**logt*np.log(10**width);emd=dem*dt;total=emd.sum(-1);meanlog=(emd*logt).sum(-1)/total
    figs=[];tables=[];sources={}
    fig,axs=plt.subplots(2,3,figsize=(12.5,8),layout='constrained')
    for c,ax in enumerate(axs.flat):
        im=plate(ax,np.log10(data['count_rates'][...,c]),f'AIA {[94,131,171,193,211,335][c]} angstrom','magma');fig.colorbar(im,ax=ax,shrink=.8,label='log10(DN / s / pixel)')
    fig.suptitle('Real AIA active-region observations | 2014-01-01',fontsize=16);figs.append(('01_observations',fig))
    fig,axs=plt.subplots(2,3,figsize=(12.5,8),layout='constrained');chosen=[3,6,10,14]
    logdem=np.log10(np.maximum(dem,1e-30));lo,hi=np.percentile(logdem[...,chosen],[2,98])
    for ax,j in zip(axs.flat,chosen):
        im=plate(ax,np.log10(np.maximum(ref[...,6],1e-30)) if j==3 else logdem[...,j],('Original DEM at logT = '+f'{logt[6]:.3f}') if j==3 else f'Fresh DEM at logT = {logt[j]:.3f}','magma',vmin=lo,vmax=hi);fig.colorbar(im,ax=ax,shrink=.8,label='log10(DEM / cm^-5 K^-1)')
    im=plate(axs[1,1],np.log10(total),'Integrated emission measure','magma');fig.colorbar(im,ax=axs[1,1],shrink=.8,label='log10(EM / cm^-5)')
    im=plate(axs[1,2],meanlog,'Emission-weighted logT','viridis');fig.colorbar(im,ax=axs[1,2],shrink=.8,label='Mean log10(T/K)')
    for row,col,label in [(10,10,'A'),(24,24,'B'),(38,38,'C')]:
        axs[0,1].plot(col,row,'o',ms=4,mfc='none',mec='white');axs[0,1].text(col+1,row+1,label,color='white',fontsize=9)
    fig.suptitle('Original / fresh DEM and temperature-resolved plasma maps',fontsize=16);figs.append(('02_thermal_maps',fig))
    fig,axs=plt.subplots(2,2,figsize=(11,8.5),layout='constrained')
    for row,col,color,label in [(10,10,'#0072B2','Pixel A'),(24,24,'#D55E00','Pixel B'),(38,38,'#009E73','Pixel C')]:
        value=dem[row,col];err=error[row,col];axs[0,0].plot(logt,value,color=color,label=label)
        axs[0,0].fill_between(logt,np.maximum(value-err,1),value+err,color=color,alpha=.13)
    axs[0,0].set_yscale('log');axs[0,0].set_ylim(bottom=1e17);axs[0,0].set_xlabel('log10(T/K)');axs[0,0].set_ylabel('DEM (cm^-5 K^-1)');axs[0,0].set_title('DEM and propagated vertical 1-sigma');axs[0,0].legend()
    for c in range(6):axs[0,1].scatter(data['count_rates'][...,c].ravel()[::8],fit[...,c].ravel()[::8],s=5,alpha=.4,label=str([94,131,171,193,211,335][c]))
    mn=min(data['count_rates'].min(),fit.min());mx=max(data['count_rates'].max(),fit.max());axs[0,1].plot([mn,mx],[mn,mx],color='black',ls='--',lw=1);axs[0,1].set(xscale='log',yscale='log',xlabel='Observed DN / s / pixel',ylabel='Fresh predicted DN / s / pixel',title='Six-channel forward count rates');axs[0,1].legend(fontsize=8,ncol=2)
    im=plate(axs[1,0],chi,'Chi-square per channel | positivity relaxation','cividis');fig.colorbar(im,ax=axs[1,0],label='Mean squared standardized residual')
    im=plate(axs[1,1],np.median(resolution,axis=-1),'Median temperature-resolution sigma','viridis');fig.colorbar(im,ax=axs[1,1],label='log10(T/K)')
    fig.suptitle('Inference uncertainty and observational closure',fontsize=16);figs.append(('03_uncertainty_closure',fig))
    variants=[]
    for label,kw in [('Half discrepancy',{'reg_tweak':.5}),('Standard',{}),('Double discrepancy',{'reg_tweak':2.}),('EM-loci weights',{'gloci':True})]:
        vals=(dem,error,resolution,chi,fit) if not kw else solve(data,**kw)
        ncc,nrmse=metrics(vals[0],ref);tables.append({'experiment':label,'NCC_vs_original':ncc,'NRMSE_vs_original':nrmse,'mean_chi2_per_channel':float(vals[3].mean()),'negative_bins':int(np.sum(vals[0]<0))});variants.append(vals[0])
    fig,axs=plt.subplots(1,4,figsize=(15,4.2),layout='constrained');vmin,vmax=np.percentile(np.log10(np.maximum(dem[...,10],1e-30)),[1,99])
    for ax,label,v in zip(axs,[x['experiment'] for x in tables],variants):
        im=plate(ax,np.log10(np.maximum(v[...,10],1e-30)),label,'magma',vmin=vmin,vmax=vmax)
    fig.colorbar(im,ax=list(axs),label=f'log10(DEM), logT={logt[10]:.3f}',shrink=.8);fig.suptitle('Regularization and weighting sensitivity | fresh inversions',fontsize=16);figs.append(('04_regularization',fig))
    sources.update(DEM=dem,vertical_sigma=error,temperature_sigma=resolution,chi2_per_channel=chi,fitted_counts=fit,total_em=total,mean_log_temperature=meanlog,regularization_variants=np.asarray(variants))
    return figs,tables,sources
