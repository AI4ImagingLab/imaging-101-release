import numpy as np
import matplotlib.pyplot as plt
from .physics_model import harmonics,spherical_coefficients
from .solvers import reconstruct
COLORS=['#276b91','#cc643c','#5d8d58','#8f65a6']
def map_image(ax,image,d,title,limits=None,cmap='magma',unit='Stellar-normalized brightness'):
    if limits is None:limits=(image.min(),image.max())
    p=ax.imshow(image,origin='lower',extent=[-90,90,-90,90],aspect='auto',cmap=cmap,vmin=limits[0],vmax=limits[1]);ax.set(title=title,xlabel='East longitude / degree',ylabel='North latitude / degree');plt.colorbar(p,ax=ax,shrink=.8,label=unit)
def acquisition(d):
    t=d['time']*24;f=d['measurements'];sigma=d['uncertainty'];fig=plt.figure(figsize=(14,8),layout='constrained');g=fig.add_gridspec(2,2);axes=[fig.add_subplot(g[0,:]),fig.add_subplot(g[1,0]),fig.add_subplot(g[1,1])]
    for ax in axes:ax.errorbar(t,1e6*(f-1),yerr=sigma*1e6,fmt='.',ms=4,elinewidth=.7,color=COLORS[0]);ax.set(xlabel='Exposure-center time / hour',ylabel='System flux above unity / ppm')
    axes[0].set_title('Existing author noiseless simulation with simulated instrument uncertainty');axes[1].set(xlim=(-1.05,-.4),title='Ingress resolves broad surface structure');axes[2].set(xlim=(.4,1.05),title='Egress scans the opposite limb')
    fig.suptitle('3.31 micrometre unresolved eclipse acquisition - 126 original samples',fontweight='bold');return fig,{'time_hours':t,'flux':f,'uncertainty':sigma}
def physics(d,s):
    coeff=s['pca_coefficients'];H=harmonics(d['latitude'],d['longitude']);fig,ax=plt.subplots(2,3,figsize=(15,9),layout='constrained');eigenmaps=[]
    for i in range(3):
        p=np.zeros(5);p[i+2]=1;image=H@spherical_coefficients(p,coeff);eigenmaps.append(image);bound=max(abs(image.min()),abs(image.max()));map_image(ax[0,i],image,d,f'Eigenmap {i+1}',(-bound,bound),'RdBu_r','Unit mode brightness')
    for i in range(3):ax[1,0].plot(d['time']*24,s['scores'][i],label=f'Eigencurve {i+1}',color=COLORS[i])
    ax[1,0].set(xlabel='Time / hour',ylabel='Unit-amplitude integrated flux',title='Centered orthogonal light curves');ax[1,0].legend()
    ev=np.maximum(s['eigenvalues'][:8],1e-20);ax[1,1].semilogy(np.arange(1,9),ev/ev.sum(),marker='o',color=COLORS[0]);ax[1,1].axvline(3.5,color=COLORS[1],ls='--',label='Selected 3 modes');ax[1,1].set(xlabel='Mode rank',ylabel='Fraction of nonuniform variance',title='Acquisition spectrum limits information');ax[1,1].legend()
    ax[1,2].plot(d['time']*24,s['areas'].sum(axis=1)/np.pi,color=COLORS[2]);ax[1,2].set(xlabel='Time / hour',ylabel='Visible projected disk fraction',title='400 equal-area tiles, circular occultation')
    fig.suptitle('Explicit geometric forward model and data-ranked spatial subspace',fontweight='bold');return fig,{'eigenmaps':np.array(eigenmaps),'scores':s['scores'],'eigenvalues':s['eigenvalues'],'visible_areas':s['areas']}
def comparison(d,m,original_maps):
    fig,ax=plt.subplots(2,3,figsize=(15,9),layout='constrained');limits=(min(m.min(),original_maps.min()),max(m.max(),original_maps.max()))
    for i in range(3):map_image(ax[0,i],original_maps[i],d,f'Qualified original posterior - seed {20261022+i}',limits)
    map_image(ax[1,0],m,d,'Fresh explicit local posterior mean',limits);diff=m-original_maps[0];bound=max(np.max(abs(diff)),1e-12);map_image(ax[1,1],diff,d,'Fresh - original seed 20261022',(-bound,bound),'RdBu_r')
    map_image(ax[1,2],original_maps.std(axis=0,ddof=1),d,'Independent posterior-mean replica SD',cmap='viridis',unit='Monte Carlo mean variation')
    fig.suptitle('Three independent original chains and a fresh reconstruction - no spatial truth',fontweight='bold');return fig,{'fresh':m,'original_maps':original_maps,'difference':diff,'replica_sd':original_maps.std(axis=0,ddof=1)}
def residuals(d,s):
    X=s['design'];uniform=np.linalg.lstsq(X[:,:2],d['measurements'],rcond=None)[0];u=X[:,:2]@uniform;p=s['prediction'];t=d['time']*24;sig=d['uncertainty'];fig,ax=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    ax[0,0].errorbar(t,(d['measurements']-1)*1e6,yerr=sig*1e6,fmt='.',color=COLORS[0],label='Author simulated samples');ax[0,0].plot(t,(p-1)*1e6,color=COLORS[1],label='Fresh 3PC posterior');ax[0,0].plot(t,(u-1)*1e6,color=COLORS[2],ls='--',label='Best uniform map');ax[0,0].set(xlabel='Time / hour',ylabel='Flux above unity / ppm',title='Computed full light-curve comparison');ax[0,0].legend(fontsize=8)
    for model,label,color in [(u,'Uniform',COLORS[2]),(p,'Fresh 3PC',COLORS[1])]:ax[0,1].plot(t,(model-d['measurements'])/sig,label=label,color=color)
    ax[0,1].axhspan(-1,1,color='gray',alpha=.15,label='One supplied sigma');ax[0,1].set(xlabel='Time / hour',ylabel='Predicted - observed / sigma',title='Truncation mismatch remains despite noiseless input');ax[0,1].legend(fontsize=8)
    bins=np.linspace(min(((u-d['measurements'])/sig).min(),((p-d['measurements'])/sig).min()),max(((u-d['measurements'])/sig).max(),((p-d['measurements'])/sig).max()),31);ax[1,0].hist((u-d['measurements'])/sig,bins=bins,alpha=.6,color=COLORS[2],label='Uniform');ax[1,0].hist((p-d['measurements'])/sig,bins=bins,alpha=.7,color=COLORS[1],label='Fresh 3PC');ax[1,0].set(xlabel='Standardized residual',ylabel='Samples',title='Observed-fit improvement');ax[1,0].legend()
    rms=np.array([np.sqrt(np.mean(((u-d['measurements'])/sig)**2)),np.sqrt(np.mean(((p-d['measurements'])/sig)**2))]);ax[1,1].bar(['Best uniform','Fresh 3PC'],rms,color=[COLORS[2],COLORS[1]]);ax[1,1].axhline(1,color='gray',ls='--',label='Supplied sigma scale');ax[1,1].set(ylabel='Uncertainty-normalized residual RMS',title='No claim that a truncated map is physical truth');ax[1,1].legend()
    fig.suptitle('Fresh predicted measurements, residuals and a computed uniform-map comparison',fontweight='bold');return fig,{'fresh_prediction':p,'uniform_prediction':u,'fresh_standardized_residual':(p-d['measurements'])/sig,'standardized_rms':rms}
def posterior(d,s,original):
    chain=s['chain'];flat=chain.reshape(-1,chain.shape[-1]);X=s['design'];W=X/d['uncertainty'][:,None];cov=np.linalg.inv(W.T@W);exact=np.linalg.solve(W.T@W,W.T@(d['measurements']/d['uncertainty']));sd=np.sqrt(np.diag(cov));co=spherical_coefficients(flat,s['pca_coefficients']);C=np.cov(co,rowvar=False);H=harmonics(d['latitude'],d['longitude']);std=np.sqrt(np.einsum('...i,ij,...j->...',H,C,H));corr=np.corrcoef(flat,rowvar=False);fig,ax=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    map_image(ax[0,0],std,d,'Fresh conditional posterior SD',cmap='viridis',unit='Brightness SD, not truth error');im=ax[0,1].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax[0,1].set(xticks=np.arange(5),yticks=np.arange(5),xticklabels=['c0','b','PC1','PC2','PC3'],yticklabels=['c0','b','PC1','PC2','PC3'],title='Fresh retained-parameter correlation');plt.colorbar(im,ax=ax[0,1],shrink=.8)
    stepmean=chain.mean(axis=1);cum=np.cumsum(stepmean,axis=0)/np.arange(1,len(chain)+1)[:,None]
    for k in [0,2,4]:ax[0,2].plot(np.arange(len(chain)),(cum[:,k]-exact[k])/sd[k],label=['c0','b','PC1','PC2','PC3'][k])
    ax[0,2].axhline(0,color='gray',lw=.7);ax[0,2].set(xlabel='Retained iteration',ylabel='Running-mean error / posterior SD',title='Independent analytic Gaussian check');ax[0,2].legend(fontsize=8)
    for k in range(5):ax[1,0].plot(np.arange(len(chain))[::10],(stepmean[::10,k]-exact[k])/sd[k],alpha=.7,label=['c0','b','PC1','PC2','PC3'][k])
    ax[1,0].set(xlabel='Retained iteration',ylabel='Mean minus analytic / posterior SD',title='Actual long-chain traces');ax[1,0].legend(fontsize=7,ncol=3)
    alltaus=np.vstack([original['autocorrelation'],s['autocorrelation']]);lengths=np.r_[original['retained_lengths'],len(s['chain'])];walkers=np.r_[original['walkers'],s['chain'].shape[1]];ratios=lengths[:,None]/alltaus;alleff=lengths[:,None]*walkers[:,None]/alltaus
    for i,name in enumerate(['Original 22','Original 23','Original 24','Fresh local']):ax[1,1].plot(np.arange(5),ratios[i],marker='o',label=name,color=COLORS[i]);ax[1,2].plot(np.arange(5),alleff[i],marker='o',label=name,color=COLORS[i])
    for a in ax[1,1:]:a.set(xticks=np.arange(5),xticklabels=['c0','b','PC1','PC2','PC3']);a.legend(fontsize=7)
    ax[1,1].axhline(50,color='black',ls='--',lw=1,label='Required 50');ax[1,1].set(ylabel='Retained length / autocorrelation time',title='Whole 32-walker ensembles: length / tau',ylim=(45,max(85,ratios.max()*1.05)));ax[1,1].legend(fontsize=7);ax[1,2].set(ylabel='Effective samples',title='Independent-chain sampling adequacy')
    fig.suptitle('Measured posterior uncertainty and convergence, separated from model mismatch',fontweight='bold');return fig,{'posterior_sd':std,'parameter_correlation':corr,'running_mean':cum,'analytic_mean':exact,'analytic_covariance':cov,'autocorrelation':alltaus,'effective_sample_size':alleff,'retained_lengths':lengths,'walkers':walkers,'retained_length_over_tau':ratios}
def sensitivity(d,standard):
    counts=np.array([1,2,3,4,5]);images=[];fit=[];minimum=[]
    for k in counts:
        m,s=reconstruct(d,components=int(k));images.append(m);fit.append(np.sqrt(np.mean(((s['prediction']-d['measurements'])/d['uncertainty'])**2)));minimum.append(s['effective_sample_size'].min())
    images=np.array(images);fig,ax=plt.subplots(2,3,figsize=(15,9),layout='constrained');limits=(images.min(),images.max())
    for j,k in enumerate([0,2,4]):map_image(ax[0,j],images[k],d,f'Fresh posterior mean - {counts[k]} PCs',limits)
    ax[1,0].plot(counts,fit,marker='o',color=COLORS[0]);ax[1,0].set(xlabel='Retained eigencurves',ylabel='Standardized observation RMS',title='Additional modes reduce truncation mismatch')
    ax[1,1].plot(counts,np.sqrt(np.mean((images-standard)**2,axis=(1,2))),marker='o',color=COLORS[1]);ax[1,1].set(xlabel='Retained eigencurves',ylabel='Brightness RMS change vs 3PC',title='Spatial structure depends on selected subspace')
    ax[1,2].plot(counts,minimum,marker='o',color=COLORS[2]);ax[1,2].set(xlabel='Retained eigencurves',ylabel='Minimum effective samples',title='Same 4000-step sampler, actual refits')
    fig.suptitle('Five computed model-order refits - only the declared 3PC field defines the task',fontweight='bold');return fig,{'components':counts,'maps':images,'standardized_rms':np.array(fit),'minimum_effective_samples':np.array(minimum)}
