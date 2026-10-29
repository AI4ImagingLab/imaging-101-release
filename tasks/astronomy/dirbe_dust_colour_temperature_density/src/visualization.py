"""Scientific comparisons; returns actual numeric plot sources with each figure."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import forward,bandpass_prediction,jacobian
from .solvers import reconstruct,fit_pixel
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110})
COLORS=['#246a91','#cc643c','#5d8d58']
def image(ax,a,title,unit,cmap='viridis',vmin=None,vmax=None):
    q=ax.imshow(a,origin='lower',aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax);ax.set_title(title);ax.set_xlabel('Longitude sampling column');ax.set_ylabel('Latitude sampling row');plt.colorbar(q,ax=ax,label=unit,shrink=.82)
def acquisition(data):
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,w in enumerate(data['wavelengths']):
        image(axes[0,i],data['intensity'][i],f'{w:g} μm measured brightness','MJy/sr')
        image(axes[1,i],data['uncertainty'][i],f'{w:g} μm weekly-mean scatter','MJy/sr','magma')
    fig.suptitle('Real DIRBE Taurus field • shared 0.7° beam, native sky samples',fontweight='bold')
    return fig,{'intensity':data['intensity'],'uncertainty':data['uncertainty']}
def physics(data,maps):
    fig,ax=plt.subplots(1,3,figsize=(15,4.4),layout='constrained');wl=np.geomspace(60,650,250);spectra=[]
    locations=[(5,8),(16,24),(26,40)]
    for i,(y,x) in enumerate(locations):
        spectrum=forward(wl,maps[0,y:y+1,x:x+1],maps[1,y:y+1,x:x+1])[:,0,0];spectra.append(spectrum)
        ax[0].plot(wl,spectrum,color=COLORS[i],label=f'({y},{x}), T={maps[0,y,x]:.1f} K');ax[0].errorbar(data['wavelengths'],data['intensity'][:,y,x],yerr=data['uncertainty'][:,y,x],fmt='o',color=COLORS[i],capsize=3)
    ax[0].set(xlabel='Wavelength / μm',ylabel='Radiance / MJy sr⁻¹',xscale='log',title='Fitted spectra and actual observations');ax[0].legend(fontsize=8)
    for i,r in enumerate(data['spectral_response']):ax[1].plot(data['response_wavelength'],r,label=f'{data["wavelengths"][i]:g} μm',color=COLORS[i])
    ax[1].set(xlabel='Wavelength / μm',ylabel='Relative response',title='Measured full instrument passbands');ax[1].legend()
    J=jacobian(data['wavelengths'],maps[0],maps[1]);J=J/data['uncertainty'][...,None];J=J*np.stack([maps[0],maps[1]],axis=-1)[None];s=np.linalg.svd(np.moveaxis(J,0,-2),compute_uv=False);condition=s[...,0]/s[...,1]
    image(ax[2],condition,'Relative-parameter Jacobian conditioning','condition number','magma')
    return fig,{'spectral_wavelengths':wl,'spectra':np.array(spectra),'condition':condition}
def components(maps,author):
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,(name,unit,scale) in enumerate([('Colour temperature','K',1),('Apparent dust surface density','g/m²',1000)]):
        lo=min(maps[i].min(),author[i].min())*scale;hi=max(maps[i].max(),author[i].max())*scale
        image(axes[i,0],author[i]*scale,name+'\nOriginal PTS fit',unit,vmin=lo,vmax=hi)
        image(axes[i,1],maps[i]*scale,name+'\nFresh local fit',unit,vmin=lo,vmax=hi)
        bound=max(float(np.max(np.abs((maps[i]-author[i])*scale))),1e-18)
        image(axes[i,2],(maps[i]-author[i])*scale,'Fresh − original',unit,'RdBu_r',-bound,bound)
    fig.suptitle('Independent original-method comparison • neither map is physical truth',fontweight='bold')
    return fig,{'fresh':maps,'original':author,'difference':maps-author}
def residuals(data,pred):
    r=pred-data['intensity'];z=r/data['uncertainty'];fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained');bound=np.max(np.abs(z))
    for i,w in enumerate(data['wavelengths']):
        image(axes[0,i],z[i],f'{w:g} μm standardized residual','relative scatter','RdBu_r',-bound,bound)
        axes[1,i].scatter(data['intensity'][i].ravel(),pred[i].ravel(),s=5,alpha=.35,color=COLORS[i]);lo=data['intensity'][i].min();hi=data['intensity'][i].max();axes[1,i].plot([lo,hi],[lo,hi],color='black',ls='--');axes[1,i].set(xlabel='Measured / MJy sr⁻¹',ylabel='Predicted / MJy sr⁻¹',title=f'RMS={np.sqrt(np.mean(r[i]**2)):.3f} MJy/sr')
    fig.suptitle('Monochromatic fit • residual structure tests the approximation',fontweight='bold')
    return fig,{'prediction':pred,'residual':r,'standardized_residual':z}
def colour_diagnostic(data,maps,pred):
    bp=bandpass_prediction(data['wavelengths'],maps[0],maps[1],data['response_wavelength'],data['spectral_response']);factor=bp/pred
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,w in enumerate(data['wavelengths']):
        image(axes[0,i],factor[i],f'{w:g} μm predicted colour factor K','dimensionless','coolwarm',.90,1.10)
        axes[1,i].hist(((pred-data['intensity'])/data['uncertainty'])[i].ravel(),bins=40,alpha=.65,label='Selected monochromatic fit',density=True,color=COLORS[0]);axes[1,i].hist(((bp-data['intensity'])/data['uncertainty'])[i].ravel(),bins=40,alpha=.55,label='Same fields, full passband',density=True,color=COLORS[1]);axes[1,i].set(xlabel='Residual / weekly-mean scatter',ylabel='Density',title=f'{w:g} μm: passband discrepancy');axes[1,i].legend(fontsize=7)
    fig.suptitle('Known bandpass bias • full-response prediction is a diagnostic, not the fitted model',fontweight='bold')
    return fig,{'bandpass_prediction':bp,'colour_factor':factor}
def sensitivity(data,maps):
    betas=np.linspace(1.5,2.5,7);results=[];rms=[]
    for beta in betas:
        m,p=reconstruct(data,beta);results.append(m);rms.append(np.sqrt(np.mean(((p-data['intensity'])/data['uncertainty'])**2)))
    results=np.array(results);fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,beta in enumerate([1.5,2.,2.5]):image(ax[0,i],results[[0,3,6][i],0],f'Fresh refit: emissivity β={beta:g}','K',vmin=results[:,0].min(),vmax=results[:,0].max())
    ax[1,0].plot(betas,results[:,0].mean(axis=(1,2)),marker='o',color=COLORS[0]);ax[1,0].set(xlabel='Assumed emissivity index β',ylabel='Mean colour temperature / K',title='Temperature–emissivity coupling')
    ax[1,1].plot(betas,1000*results[:,1].mean(axis=(1,2)),marker='o',color=COLORS[1]);ax[1,1].set(xlabel='Assumed emissivity index β',ylabel='Mean apparent column / g m⁻²',title='Opacity assumption changes inferred mass')
    ax[1,2].plot(betas,rms,marker='o',color=COLORS[2]);ax[1,2].set(xlabel='Assumed emissivity index β',ylabel='Standardized residual RMS',title='Three-band goodness of fit')
    fig.suptitle('Seven actual field reconstructions • fixed observations, changed spectral assumption',fontweight='bold')
    return fig,{'betas':betas,'refit_maps':results,'standardized_rms':np.array(rms)}
