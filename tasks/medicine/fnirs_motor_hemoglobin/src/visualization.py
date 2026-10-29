"""Fresh controlled experiments and evidence plots; no plotting backend mutation."""
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct,invert_density
from .preprocessing import optical_density,SAMPLE_RATE_HZ
from .physics_model import forward,absorption_matrix

def metrics(estimate,reference):
    out={}
    for i,name in enumerate(('hbo','hbr')):
        x=estimate[:,:,i,:];y=reference[:,:,i,:]
        nx=np.linalg.norm(x);ny=np.linalg.norm(y);dr=np.ptp(y)
        if ny<=0 or dr<=0:raise ValueError('Undefined reference metric')
        out[name+'_ncc']=float(np.sum(x*y)/(nx*ny)) if nx>0 else 0.
        out[name+'_nrmse']=float(np.sqrt(np.mean((x-y)**2))/dr)
    return out

def experiments(obs,reference):
    intensity=obs['intensity'];e=obs['extinction'];d=obs['distances']
    fresh=reconstruct(intensity,e,d);od=optical_density(intensity)
    a=absorption_matrix(e,d)
    single=np.einsum('bpi,bpt->bpit',a[:,:,0,:],od[:,:,0,:])/np.sum(a[:,:,0,:]**2,axis=-1)[:,:,None,None]
    regularized=[]
    for level in (0.001,0.01,0.1,1.):
        at=np.swapaxes(a,-1,-2);h=at@a
        lam=level*np.trace(h,axis1=-2,axis2=-1)/2
        regularized.append(np.linalg.solve(h+lam[...,None,None]*np.eye(2),at@od))
    ppf_values=np.array([4.8,5.4,6.,6.6,7.2]);ppf=np.stack([reconstruct(intensity,e,d,v) for v in ppf_values])
    decim_factors=np.array([1,2,4,8,16]);decim_errors=[]
    for k in decim_factors:
        coarse=reconstruct(intensity[...,::k],e,d)
        original_at_coarse=fresh[...,::k]
        decim_errors.append(float(np.linalg.norm(coarse-original_at_coarse)/np.linalg.norm(original_at_coarse)))
    short=d[0]<.015
    nuisance=fresh[:,short,:,:].mean(axis=1,keepdims=True)
    centered=nuisance-nuisance.mean(axis=-1,keepdims=True)
    coef=np.sum(fresh*centered,axis=-1,keepdims=True)/np.sum(centered**2,axis=-1,keepdims=True)
    corrected=fresh-coef*centered
    residual=forward(fresh,e,d)-od
    measurements=metrics(fresh,reference)
    measurements.update(forward_relative_rms=float(np.linalg.norm(residual)/np.linalg.norm(od)),max_absolute_upstream_difference=float(np.max(np.abs(fresh-reference))),matrix_condition_number=float(np.linalg.cond(a[0,0])),short_channels=int(short.sum()),single_wavelength_relative_error=float(np.linalg.norm(single-fresh)/np.linalg.norm(fresh)),short_regression_relative_change=float(np.linalg.norm(corrected-fresh)/np.linalg.norm(fresh)),ppf_relative_changes=[float(np.linalg.norm(v-fresh)/np.linalg.norm(fresh)) for v in ppf],decimation_relative_changes=decim_errors,regularized_relative_changes=[float(np.linalg.norm(v-fresh)/np.linalg.norm(fresh)) for v in regularized])
    return dict(fresh=fresh,reference=reference,od=od,residual=residual,single=single,regularized=np.stack(regularized),ppf=ppf,ppf_values=ppf_values,decim_factors=decim_factors,decim_errors=np.array(decim_errors),corrected=corrected,measurements=measurements)

def figures(obs,result):
    plt.rcParams.update({'font.size':9,'axes.titlesize':11,'axes.labelsize':9})
    time=np.arange(obs['intensity'].shape[-1])/SAMPLE_RATE_HZ;c=result['fresh'][0]*1e6
    figs={};fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    z=obs['intensity'][0,:,0,:];im=ax[0,0].imshow(z,aspect='auto',origin='lower',extent=[-0.5/SAMPLE_RATE_HZ,time[-1]+0.5/SAMPLE_RATE_HZ,-.5,67.5],cmap='viridis');fig.colorbar(im,ax=ax[0,0],label='760 nm intensity (a.u.)');ax[0,0].set(title='Complete native recording',xlabel='Time (s)',ylabel='Pair index')
    # All concentration values displayed, no percentile clipping.
    vmax=np.max(np.abs(c))
    for i,name in enumerate(('HbO','HbR')):
        im=ax[1,i].imshow(c[:,i,:],aspect='auto',origin='lower',extent=[-0.5/SAMPLE_RATE_HZ,time[-1]+0.5/SAMPLE_RATE_HZ,-.5,67.5],cmap='RdBu_r',vmin=-vmax,vmax=vmax);fig.colorbar(im,ax=ax[1,i],label='Relative concentration (uM)');ax[1,i].set(title=name+' — complete spectral inversion',xlabel='Time (s)',ylabel='Pair index')
    for w,label in enumerate(('760 nm','850 nm')):ax[0,1].plot(time,result['od'][0,0,w],lw=.6,label=label)
    ax[0,1].set(title='Pair 0: logarithmic measured attenuation',xlabel='Time (s)',ylabel='Optical density');ax[0,1].legend()
    figs['recording']=fig
    fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    colors=['#0072B2','#D55E00']
    for j,label in enumerate(('HbO','HbR')):
        ax[0,0].plot(time,c[0,j],color=colors[j],lw=.8,label=label+' fresh')
        ax[0,0].plot(time,result['reference'][0,0,j]*1e6,color=colors[j],lw=.5,ls='--',label=label+' original')
    ax[0,0].set(title='Fresh vs original: pair 0',xlabel='Time (s)',ylabel='Relative concentration (uM)');ax[0,0].legend(ncol=2,fontsize=8)
    diff=(result['fresh']-result['reference']).reshape(-1)*1e6
    ax[0,1].hist(diff,bins=50,color='#0072B2');ax[0,1].set(title='Full-record numerical parity',xlabel='Fresh minus original (uM)',ylabel='Array elements')
    for i,label in enumerate(('HbO','HbR')):
        ax[1,0].plot(time,result['single'][0,0,i]*1e6,color=colors[i],lw=.6,ls='--',label=label+' one wavelength')
        ax[1,0].plot(time,c[0,i],color=colors[i],lw=.6,label=label+' two wavelengths')
    ax[1,0].set(title='Removing a wavelength loses identifiability',xlabel='Time (s)',ylabel='Relative concentration (uM)');ax[1,0].legend(fontsize=8,ncol=2)
    ax[1,1].plot(time,result['residual'][0,0,0],lw=.5,color='#009E73');ax[1,1].set(title='Pair 0 signed forward residual (roundoff)',xlabel='Time (s)',ylabel='Predicted minus measured OD')
    figs['parity_identifiability']=fig
    fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    ax[0,0].plot(result['ppf_values'],result['measurements']['ppf_relative_changes'],'o-',color='#0072B2');ax[0,0].set(title='Pathlength uncertainty changes physical scale',xlabel='Assumed PPF (dimensionless)',ylabel='Relative concentration change')
    ax[0,1].plot(result['decim_factors'],result['decim_errors'],'o-',color='#D55E00');ax[0,1].set_xscale('log',base=2);ax[0,1].set(title='Acquisition reduction changes baseline gauge',xlabel='Retain every k-th native sample',ylabel='Relative error at retained samples')
    pos=obs['positions'][0]*1000;rms=np.sqrt(np.mean(c**2,axis=-1));vmax=rms.max()
    for j,label in enumerate(('HbO','HbR')):
        im=ax[1,j].scatter(pos[:,0],pos[:,1],c=rms[:,j],s=45,cmap='viridis',vmin=0,vmax=vmax,edgecolor='white',lw=.4);fig.colorbar(im,ax=ax[1,j],label='Temporal RMS concentration (uM)');ax[1,j].set(title=label+' path midpoint map (not cortical depth)',xlabel='Head x (mm)',ylabel='Head y (mm)',aspect='equal')
    figs['sensitivity_channel_maps']=fig
    return figs
