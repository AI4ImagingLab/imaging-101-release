"""Scientific review figures from freshly computed inverse results."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import model_jac,initial_guess,physical_fields
from .solvers import fit
from .generate_data import perturb_spectra

CHANNELS=['Remanent field','Bias-odd field','Midpoint, configuration 0','Midpoint, configuration 1']
UNITS=['µT','µT','MHz relative to 2.87 GHz','MHz relative to 2.87 GHz']
def sensitivity_experiment(observations,indices=None):
    """Refit real selected spectra and three perturbed observations; no reference input."""
    if indices is None:indices=np.array([0,479,143520,143999,47*480+80,150*480+240,150*480+30,220*480+350,80*480+390,120*480+120,180*480+260,260*480+70])
    data=observations['spectra'][0].reshape(2,2,-1,51)[:,:,indices]
    frequency=observations['frequency_hz'][0]/np.float32(1e9)
    sigmas=np.array([0.,.0001,.0003,.001])
    params=np.empty((len(sigmas),2,2,len(indices),6),dtype='float32')
    for k,sigma in enumerate(sigmas):
        noisy=data if sigma==0 else perturb_spectra(data,sigma,20261022+k)
        for pol in range(2):
            for window in range(2):
                guess=initial_guess(noisy[pol,window],frequency[window])
                params[k,pol,window]=fit(frequency[window],noisy[pol,window],guess,10000)[0]
    fields=np.stack([physical_fields(p,1,len(indices))[0,:,0] for p in params])
    return dict(indices=indices,sigmas=sigmas,parameters=params,fields=fields,rmse=np.sqrt(np.mean((fields-fields[0])**2,axis=-1)))

def review_figures(observations,fresh,default,qualified,upstream_default,sensitivity,export_dir=None):
    """Return six fully computed acquisition/field/fit/residual/budget/noise figures."""
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.dpi':110,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    figs=[];sources=[]
    data=observations['spectra'][0];freq=observations['frequency_hz'][0]/np.float32(1e9)
    image=fresh['reconstruction'][0];ref=qualified['reconstruction'][0]
    def mapshow(fig,ax,a,title,cmap='viridis',vmin=None,vmax=None,unit='',extend='neither'):
        im=ax.imshow(a,cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True)
        ax.set_title(title);ax.set_xlabel('Column [binned pixel]');ax.set_ylabel('Row [binned pixel]')
        fig.colorbar(im,ax=ax,shrink=.82,pad=.025,extend=extend,label=unit)
    fig,axs=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    for k,name in enumerate(['Laser illumination','Reflected LED']):
        mapshow(fig,axs[0,k],observations['calibration_images'][0,k],name,'gray',unit='Camera intensity [a.u.]')
    mean=data.mean(axis=(0,1,4));mapshow(fig,axs[0,2],mean,'Mean measured fluorescence',unit='Normalized intensity')
    locations=[(150,240),(105,60),(47,80)]
    for ax,(row,col) in zip(axs[1],locations):
        for pol in range(2):
            for window in range(2):
                ax.plot(freq[window],data[pol,window,row,col],'.-',lw=.7,ms=2,label=f'Config {pol}, window {window}')
        ax.set(title=f'Real spectra at ({row}, {col})',xlabel='Frequency [GHz]',ylabel='Normalized fluorescence');ax.legend(fontsize=6)
    fig.suptitle('1 · Real acquisition: every native pixel contributes to the complete field')
    figs.append(fig);sources.append(dict(calibration=observations['calibration_images'],mean_fluorescence=mean,locations=np.array(locations),location_spectra=np.stack([data[:,:,r,c] for r,c in locations]),frequency_GHz=freq))
    fig,axs=plt.subplots(4,3,figsize=(13,13),constrained_layout=True)
    for k in range(4):
        transform=lambda a: (a-2.87)*1e3 if k>=2 else a
        rr=transform(ref[k]);ff=transform(image[k]);diff=ff-rr
        lo,hi=float(rr.min()),float(rr.max())
        if k==0:lo,hi=-max(abs(lo),abs(hi)),max(abs(lo),abs(hi))
        cmap='RdBu_r' if k==0 else 'viridis'
        for ax,a,title in zip(axs[k,:2],[rr,ff],['Qualified original','Fresh local']):
            name=CHANNELS[k] if k<2 else f'D{k-2} midpoint'
            mapshow(fig,ax,a,f'{name} · {title}',cmap,lo,hi,UNITS[k],extend='both')
        limit=max(float(abs(diff).max()),1e-12)
        difference_title='Fresh − original' if np.any(diff) else 'Fresh − original: exactly zero'
        mapshow(fig,axs[k,2],diff,difference_title,'RdBu_r',-limit,limit,'µT' if k<2 else 'MHz')
    fig.suptitle('2 · Four unknown fields: standard-method reference is not measured truth')
    figs.append(fig);sources.append(dict(original_fields=ref,fresh_fields=image,full_error=image-ref))
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for window in range(2):
        for col,(row,pixel) in enumerate(locations):
            ax=axs[window,col];idx=row*480+pixel;p=fresh['parameters'][0,window,idx:idx+1]
            dense=np.linspace(freq[window,0],freq[window,-1],501,dtype='float32')
            y,_=model_jac(dense,p)
            ax.plot(freq[window],data[0,window,row,pixel],'o',ms=3,mfc='none',label='Real observation')
            ax.plot(dense,y[0],lw=1.5,label='Fresh three-line fit')
            for j,shift in enumerate([.002158,0,-.002158]):
                center=p[0,0]-shift
                ax.axvline(center,color='gray',ls=':',lw=.6)
            ax.set(title=f'({row}, {pixel}), window {window}',xlabel='Frequency [GHz]',ylabel='Normalized fluorescence')
            ax.legend(fontsize=7)
    fig.suptitle('3 · The full nitrogen-14 model: weak pixels and a structured-residual region retained')
    figs.append(fig);sources.append(dict(locations=np.array(locations),parameters=fresh['parameters'][0,:,np.array([r*480+c for r,c in locations])],frequency_GHz=freq))
    residual=np.sqrt(fresh['sse'].reshape(2,2,300,480)/51)
    widths=fresh['parameters'][...,1].reshape(2,2,300,480)
    iterations=fresh['iterations'].reshape(2,2,300,480)
    fig,axs=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    mapshow(fig,axs[0,0],residual.max(axis=(0,1)),'Worst-window residual RMS',unit='Normalized fluorescence')
    mapshow(fig,axs[0,1],iterations.max(axis=(0,1)),'Maximum iteration count',unit='Iteration')
    mapshow(fig,axs[0,2],np.sum(np.isclose(widths,.005,rtol=0,atol=1e-9),axis=(0,1)),'Number of upper-width-bound fits',unit='Count among four fits')
    ax=axs[1,0]
    for p in range(2):
        for w in range(2):ax.hist(residual[p,w].ravel(),bins=100,histtype='step',label=f'C{p}/W{w}')
    ax.set(xlabel='Spectral residual RMS',ylabel='Pixel count',yscale='log',title='All 576000 fits');ax.legend(fontsize=7)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax=axs[1,1];ax.hist(iterations.ravel(),bins=np.geomspace(1,4000,50),histtype='step')
    ax.set(xscale='log',yscale='log',xlabel='Iteration count',ylabel='Curve count',title='Long trajectories are retained')
    ax=axs[1,2];ax.scatter(widths.ravel()[::53]*1e3,residual.ravel()[::53],s=2,alpha=.25,rasterized=True)
    ax.set(xlabel='Half-width [MHz]',ylabel='Residual RMS',title='Display every 53rd fit; metrics use all')
    fig.suptitle('4 · Model discrepancy and constrained solutions remain visible')
    figs.append(fig);sources.append(dict(residual_rms=residual,width_GHz=widths,iterations=iterations))
    fig,axs=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    budget_error=default['reconstruction'][0]-image
    for k in range(2):
        lim=max(float(abs(budget_error[k]).max()),1e-12)
        mapshow(fig,axs[0,k],budget_error[k],f'{CHANNELS[k]}: fresh 1000 − 10000','RdBu_r',-lim,lim,'µT')
    cap=upstream_default['states_fr_pol'].reshape(2,2,300,480)
    mapshow(fig,axs[0,2],np.sum(cap!=0,axis=(0,1)),'Original default unfinished fits',unit='Count among four fits')
    for k,ax in enumerate(axs[1,:2]):
        vals=[np.sqrt(np.mean((upstream_default['reconstruction'][0,k]-ref[k])**2)),np.sqrt(np.mean(budget_error[k]**2)),np.sqrt(np.mean((image[k]-ref[k])**2))]
        ax.bar(['Original default','Fresh default','Fresh primary'],vals,color=['#5b7083','#d2944a','#248b87'])
        ax.set(yscale='log',ylabel='Full-field RMSE [µT]',title=CHANNELS[k]);ax.tick_params(axis='x',rotation=15)
    ax=axs[1,2]
    for k in range(4):
        e=budget_error[k].ravel()*(1e3 if k>=2 else 1)
        ax.plot(np.linspace(0,1,len(e)),np.sort(abs(e)),label=CHANNELS[k])
    ax.set_yscale('symlog',linthresh=1e-7)
    ax.set(xlim=(.998,1.00001),xlabel='Full-field quantile (upper 0.2% displayed)',ylabel='Absolute change [µT; D in MHz]',title='Budget sensitivity: nonzero tail');ax.legend(fontsize=6)
    fig.suptitle('5 · Supported iteration budget: default failures are compared, not discarded')
    figs.append(fig);sources.append(dict(default_fields=default['reconstruction'],primary_fields=fresh['reconstruction'],original_default_fields=upstream_default['reconstruction'],original_default_states=cap,budget_difference=budget_error))
    fig,axs=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    for k,ax in enumerate(axs.flat):
        factor=1e3 if k>=2 else 1
        err=(sensitivity['fields'][:,k]-sensitivity['fields'][0,k])*factor
        sigma=sensitivity['sigmas']
        ax.plot(sigma,np.sqrt(np.mean(err**2,axis=-1)),'o-',label='Selected-pixel RMSE')
        ax.plot(sigma,np.max(abs(err),axis=-1),'s--',label='Selected-pixel maximum')
        ax.set(xlabel='Added Gaussian noise σ [normalized intensity]',ylabel='Field change ['+('MHz' if k>=2 else 'µT')+']',title=CHANNELS[k])
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('6 · Fresh noise sensitivity: all four fields, 12 fixed real pixels, no new primary data')
    figs.append(fig);sources.append(sensitivity)
    if export_dir is not None:
        path=Path(export_dir);path.mkdir(parents=True,exist_ok=True)
        for k,(fig,arrays) in enumerate(zip(figs,sources),1):
            for ext in ['png','svg','pdf']:fig.savefig(path/f'figure_{k:02d}.{ext}',bbox_inches='tight')
            np.savez_compressed(path/f'figure_{k:02d}_source.npz',**arrays)
        (path/'provenance.json').write_text(json.dumps({'figures':6,'data':'real Harvard Dataverse dipole observations, complete field','original':'qualified source 10000 and unchanged source default1000','fresh':'local explicit LM run inside executed notebook','noise':'seed20261022+k, sigma0/1e-4/3e-4/1e-3,12 fixed real pixels','source_files':[f'figure_{k:02d}_source.npz' for k in range(1,7)]},indent=2),encoding='utf8')
    return figs

