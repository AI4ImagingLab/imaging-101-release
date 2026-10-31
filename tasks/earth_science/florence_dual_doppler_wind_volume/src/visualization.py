import numpy as np
import matplotlib.pyplot as plt
from .physics_model import forward,fall_speed,projection,divergence
from .solvers import reconstruct
NAMES=['Eastward u','Northward v','Upward w'];COLORS=['#246a91','#cc643c','#5d8d58']
def image(ax,array,data,title,unit='m/s',bound=None,cmap='RdBu_r'):
    if bound is None:bound=max(float(np.nanmax(abs(array))),1e-12)
    p=ax.imshow(array,origin='lower',extent=[data['x'][0]/1000,data['x'][-1]/1000,data['y'][0]/1000,data['y'][-1]/1000],aspect='equal',cmap=cmap,vmin=0 if cmap=='viridis' else -bound,vmax=bound);ax.set(xlabel='Eastward x / km',ylabel='Northward y / km',title=title);plt.colorbar(p,ax=ax,label=unit,shrink=.8)
def acquisition(d):
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained');bound=np.max(abs(d['radial_velocity']*d['validity']))
    for r,name in enumerate(['NEXRAD A','NEXRAD B']):
        for j,k in enumerate([1,4,7]):
            a=np.where(d['validity'][r,k]>0,d['radial_velocity'][r,k],np.nan);image(ax[r,j],a,d,f'{name} • z={d["z"][k]/1000:g} km',bound=bound)
    fig.suptitle('Actual measured Doppler velocities • blank voxels carry no radial constraint',fontweight='bold')
    return fig,{'radial_velocity':d['radial_velocity'],'validity':d['validity']}
def physics(d):
    a=projection(d['azimuth'],d['elevation']);fall=fall_speed(d['reflectivity'],d['z']);k=4;fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for r,name in enumerate(['NEXRAD A','NEXRAD B']):
        image(ax[r,0],a[r,2,k],d,name+' vertical projection','dimensionless',bound=1)
        image(ax[r,1],np.where(d['validity'][r,k]>0,abs(fall[r,k]),np.nan),d,name+' downward particle speed','m/s',bound=10,cmap='viridis')
        image(ax[r,2],d['validity'][r,k],d,name+' accepted beam geometry','binary weight',bound=1,cmap='viridis')
    fig.suptitle('Weak vertical sensitivity and empirical fall-speed correction • z=4 km',fontweight='bold')
    return fig,{'projection':a,'terminal_speed':fall}
def volume(d,m):
    fig,ax=plt.subplots(3,3,figsize=(15,12),layout='constrained')
    for c in range(3):
        bound=np.max(abs(m[c]))
        for j,k in enumerate([1,4,7]):image(ax[c,j],m[c,k],d,f'{NAMES[c]} • z={d["z"][k]/1000:g} km',bound=bound)
    fig.suptitle('Fresh complete wind-volume inversion • component scales fixed across heights',fontweight='bold')
    return fig,{'winds':m}
def comparison(d,m,a):
    k=4;fig,ax=plt.subplots(3,3,figsize=(15,12),layout='constrained')
    for c in range(3):
        bound=max(np.max(abs(m[c])),np.max(abs(a[c])));difference=m[c,k]-a[c,k];eb=max(np.max(abs(difference)),1e-12)
        image(ax[c,0],a[c,k],d,NAMES[c]+'\nOriginal JAX backend',bound=bound);image(ax[c,1],m[c,k],d,NAMES[c]+'\nFresh local SciPy backend',bound=bound);image(ax[c,2],difference,d,'Fresh − original',bound=eb)
    fig.suptitle('Independent converged-method comparison • no physical wind truth',fontweight='bold')
    return fig,{'original':a,'fresh':m,'difference':m-a}
def residuals(d,m,p):
    zero=forward(np.zeros_like(m),d);r=(p-d['radial_velocity'])*d['validity'];zr=(zero-d['radial_velocity'])*d['validity'];div=divergence(m,d);k=4;fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained');bound=max(np.max(abs(r)),1e-12)
    for radar,name in enumerate(['NEXRAD A','NEXRAD B']):
        image(ax[radar,0],np.where(d['validity'][radar,k]>0,r[radar,k],np.nan),d,name+' fresh radial residual',bound=bound)
        mask=d['validity'][radar]>0;ax[radar,1].hist(zr[radar][mask],bins=45,density=True,alpha=.6,label='Zero air wind',color=COLORS[0]);ax[radar,1].hist(r[radar][mask],bins=45,density=True,alpha=.6,label='Fresh wind volume',color=COLORS[1]);ax[radar,1].set(xlabel='Predicted − observed radial / m s⁻¹',ylabel='Density',title=name+' valid-voxel fit improvement');ax[radar,1].legend(fontsize=8)
    image(ax[0,2],div[k]*1e3,d,'Fresh anelastic divergence','10⁻³ s⁻¹',bound=np.max(abs(div))*1e3)
    ax[1,2].plot(np.sqrt(np.mean(m[2]**2,axis=(1,2))),d['z']/1000,color=COLORS[2],marker='o');ax[1,2].set(xlabel='Vertical-wind spatial RMS / m s⁻¹',ylabel='Height / km',title='Explicit w=0 analysis-plane boundaries')
    return fig,{'prediction':p,'radial_residual':r,'zero_wind_residual':zr,'anelastic_divergence':div}
def sensitivity(d,standard):
    priors=np.array([1e-5,3e-5,1e-4,3e-4,1e-3]);maps=[];rms=[];vertical=[]
    for q in priors:
        m,p,s=reconstruct(d,prior_weight=q);maps.append(m);rms.append(np.sqrt(np.sum((p-d['radial_velocity'])**2*d['validity'])/d['validity'].sum()));vertical.append(np.sqrt(np.mean(m[2]**2)))
    maps=np.array(maps);mass=np.array([64.,256.,1024.]);massmaps=[]
    for q in mass:massmaps.append(reconstruct(d,mass_weight=q)[0])
    massmaps=np.array(massmaps);fig,ax=plt.subplots(2,3,figsize=(15,9),layout='constrained');bound=np.max(abs(maps[:,2]))
    for j,idx in enumerate([0,2,4]):image(ax[0,j],maps[idx,2,4],d,f'Upward w • horizontal prior={priors[idx]:g}',bound=bound)
    ax[1,0].semilogx(priors,rms,marker='o',color=COLORS[0]);ax[1,0].set(xlabel='Horizontal prior coefficient / s² m⁻²',ylabel='Valid radial residual RMS / m s⁻¹',title='Data fit changes with regularization')
    ax[1,1].semilogx(priors,vertical,marker='o',color=COLORS[1]);ax[1,1].set(xlabel='Horizontal prior coefficient / s² m⁻²',ylabel='Vertical-wind RMS / m s⁻¹',title='Indirect prior sensitivity of w')
    for c in range(3):ax[1,2].semilogx(mass,np.sqrt(np.mean((massmaps[:,c]-standard[c])**2,axis=(1,2,3))),marker='o',label=NAMES[c],color=COLORS[c])
    ax[1,2].set(xlabel='Continuity coefficient / s²',ylabel='Whole-volume RMS change / m s⁻¹',title='Three actual continuity-weight refits');ax[1,2].legend(fontsize=8)
    fig.suptitle('Computed sensitivity, not wind uncertainty truth • weakly observed vertical flow remains conditional',fontweight='bold')
    return fig,{'prior_weights':priors,'prior_maps':maps,'radial_rms':np.array(rms),'vertical_rms':np.array(vertical),'mass_weights':mass,'mass_maps':massmaps}
