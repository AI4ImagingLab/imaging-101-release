"""Numerical metrics and five scientific figure groups; backend is caller-owned."""
import numpy as np
from .physics_model import dilation
from .solvers import reconstruct

def metrics(estimate,reference):
    x,r=np.ravel(estimate),np.ravel(reference);den=np.linalg.norm(x)*np.linalg.norm(r)
    return dict(ncc=float(np.dot(x,r)/den) if den else 0.0,nrmse=float(np.sqrt(np.mean((x-r)**2))/np.ptp(r)))

def make_figures(raw,probe,truth,fresh,plain,blind,blind_tip,blind_loss):
    """Return [(name,Figure),...], numeric_sources dictionary, metric table."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    groups=[];extent=(-7.5,442.5,442.5,-7.5)
    names=['Raw AFM image','Erosion','Substrate-constrained','Blind BTR']
    methods=[raw,plain,fresh,blind]
    table={name:metrics(v,truth) for name,v in zip(names,methods)}
    for name,v in zip(names,methods):table[name]['observation_rmse']=float(np.sqrt(np.mean((dilation(v,probe)-raw)**2)))
    def im(ax,a,title,cmap='viridis',vmin=None,vmax=None):
        h=ax.imshow(a,extent=extent,cmap=cmap,vmin=vmin,vmax=vmax);ax.set_title(title,fontsize=10);ax.set_xlabel('x (A)');ax.set_ylabel('y (A)');return h
    fig,axs=plt.subplots(3,4,figsize=(13,9),layout='constrained')
    for row,frame in enumerate([0,7,19]):
        for col,(label,arr) in enumerate([('True sample',truth),('Observed contact image',raw),('Fresh constrained surface',fresh),('Fresh blind BTR surface',blind)]):
            h=im(axs[row,col],arr[frame],f'{label}; frame {frame}',vmin=0,vmax=float(truth.max()));fig.colorbar(h,ax=axs[row,col],label='height (A)',shrink=.7)
    fig.suptitle('Independent author surfaces, bound observations and fresh local inversions');groups.append(('01_surface_comparison',fig))
    fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained');lim=float(np.max(abs(fresh-truth)))
    for ax,label,v in zip(axs[0],names[1:],methods[1:]):
        h=im(ax,v[7]-truth[7],label+' minus truth','RdBu_r',-lim,lim);fig.colorbar(h,ax=ax,label='height error (A)')
    colors=['#9d9d9d','#de8f05','#0173b2','#029e73']
    for ax,key,title in zip(axs[1],['ncc','nrmse','observation_rmse'],['Cosine similarity','Dynamic-range NRMSE','Forward residual RMSE (A)']):
        vals=[table[n][key] for n in names];ax.bar(np.arange(4),vals,color=colors);ax.set_xticks(np.arange(4),names,rotation=25,ha='right');ax.set_title(title)
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.4g}',ha='center',va='bottom',fontsize=8)
    fig.suptitle('Actual errors, physical consistency and explicit constrained-method change');groups.append(('02_errors_metrics',fig))
    true_forward=dilation(truth,probe);fresh_forward=dilation(fresh,probe);noise=raw-true_forward;residual=raw-fresh_forward
    fig,axs=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax,v,label in [(axs[0,0],noise[7],'Author noise: measured minus true forward'),(axs[0,1],residual[7],'Measured minus fresh re-dilation')]:
        h=im(ax,v,label,'RdBu_r',-10,10);fig.colorbar(h,ax=ax,label='residual (A)')
    bins=np.linspace(-14,14,65);axs[1,0].hist(noise.ravel(),bins,density=True,histtype='step',label='Author noise');axs[1,0].hist(residual.ravel(),bins,density=True,histtype='step',label='Fresh residual');u=np.linspace(-14,14,300);axs[1,0].plot(u,np.exp(-u*u/18)/(3*np.sqrt(2*np.pi)),label='N(0, 3 A)');axs[1,0].set_xlabel('residual (A)');axs[1,0].set_ylabel('density (1/A)');axs[1,0].legend(fontsize=8)
    axs[1,1].plot(np.sqrt(np.mean(noise**2,axis=(1,2))),label='True-model noise');axs[1,1].plot(np.sqrt(np.mean(residual**2,axis=(1,2))),label='Fresh-model residual');axs[1,1].axhline(3,color='k',ls=':',label='Acquisition sigma');axs[1,1].set_xlabel('frame');axs[1,1].set_ylabel('RMS residual (A)');axs[1,1].legend(fontsize=8)
    fig.suptitle('Forward-model verification on exact observations and noise-scale audit');groups.append(('03_observation_consistency',fig))
    fig=plt.figure(figsize=(12,8),layout='constrained');ax=fig.add_subplot(221);h=ax.imshow(probe,cmap='magma');ax.set_title('Independent known reflected double probe');ax.set_xlabel('probe column');ax.set_ylabel('probe row');fig.colorbar(h,ax=ax,label='reflected height (A)');ax=fig.add_subplot(222);h=ax.imshow(blind_tip,cmap='magma');ax.set_title('Fresh blind estimate: different 10 x 10 model');fig.colorbar(h,ax=ax,label='reflected height (A)');ax=fig.add_subplot(223);ax.semilogy(blind_loss);ax.set_xlabel('epoch');ax.set_ylabel('sum of frame opening MSE (A^2)');ax.set_title('Fresh blind optimization, 30 frames / 200 epochs');ax=fig.add_subplot(224,projection='3d');yy,xx=np.mgrid[:30,:30]*15;ax.plot_surface(xx,yy,fresh[7],cmap='viridis');ax.set_xlabel('x (A)');ax.set_ylabel('y (A)');ax.set_zlabel('height (A)');ax.set_title('Fresh recovered sample topography')
    groups.append(('04_probe_and_geometry',fig))
    scales=np.array([.7,.85,1.,1.15,1.3]);scale_surfaces=np.stack([reconstruct(raw,probe*s) for s in scales]);sigma=np.array([0.,.35,.7,1.]);smooth_surfaces=np.stack([reconstruct(gaussian_filter(raw,(0,s,s)),probe) for s in sigma]);
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for key,ax in zip(['nrmse','ncc'],axs[0]):ax.plot(scales,[metrics(v,truth)[key] for v in scale_surfaces],'o-');ax.axvline(1,color='k',ls=':');ax.set_xlabel('multiplicative probe-height perturbation');ax.set_ylabel(key);ax.set_title('Independent calibration sensitivity')
    axs[1,0].plot(sigma,[metrics(v,truth)['nrmse'] for v in smooth_surfaces],'o-',label='Actual-truth NRMSE');axs[1,0].set_xlabel('Optional image prefilter sigma (pixels)');axs[1,0].set_ylabel('NRMSE');axs[1,0].set_title('Extra denoising ablation; sigma=0 is shipped');axs[1,1].plot(np.arange(30)*15,truth[7,15],label='True');axs[1,1].plot(np.arange(30)*15,raw[7,15],label='AFM');axs[1,1].plot(np.arange(30)*15,fresh[7,15],label='Fresh');axs[1,1].set_xlabel('x (A)');axs[1,1].set_ylabel('height (A)');axs[1,1].legend();axs[1,1].set_title('Paired lateral profile, frame 7 / row 15')
    groups.append(('05_sensitivity_profiles',fig))
    sources=dict(raw=raw,probe=probe,truth=truth,fresh=fresh,plain=plain,blind=blind,blind_tip=blind_tip,blind_loss=blind_loss,true_forward=true_forward,fresh_forward=fresh_forward,scales=scales,scale_surfaces=scale_surfaces,smoothing_sigma=sigma,smoothing_surfaces=smooth_surfaces)
    return groups,sources,table
