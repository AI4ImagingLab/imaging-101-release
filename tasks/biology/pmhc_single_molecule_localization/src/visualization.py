"""Fresh full-movie comparisons with native pixel/ADU conventions."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from .solvers import reconstruct,find_peaks_yx
from .preprocessing import bandpass_filter

def table_metrics(estimate,reference):
    x=estimate[0];y=reference[0]
    if not len(y):raise ValueError('Empty reference')
    matched=0;loss=np.zeros(4)
    for frame in range(10):
        a=x[x[:,0]==frame];b=y[y[:,0]==frame];n=0;current=np.zeros(4)
        if len(a) and len(b):
            dist=np.linalg.norm(a[:,None,1:3]-b[None,:,1:3],axis=-1)
            i,j=linear_sum_assignment(np.where(dist<=.5,dist,1e6));valid=dist[i,j]<=.5;i=i[valid];j=j[valid];n=len(i)
            current=[np.sum(dist[i,j]**2),np.sum((a[i,3]/b[j,3]-1)**2),np.sum((a[i,4]-b[j,4])**2),np.sum((a[i,5]-b[j,5])**2)]
        matched+=n;loss+=np.asarray(current)+(len(b)-n)
    rms=np.sqrt(loss/len(y))
    return dict(detection_f1=float(2*matched/(len(x)+len(y))),position_rmse_pixels=float(rms[0]),mass_relative_rmse=float(rms[1]),size_rmse_pixels=float(rms[2]),ecc_rmse=float(rms[3]))

def experiments(frames,original):
    fresh=reconstruct(frames);filtered=np.stack([bandpass_filter(v) for v in frames[0]])
    threshold_values=np.array([150.,300.,450.,600.]);radius_values=np.array([2,3,4,5]);noise_values=np.array([0.,25.,50.,100.])
    thresholds=[reconstruct(frames,signal_thresh=v) for v in threshold_values]
    radii=[reconstruct(frames,radius=int(v)) for v in radius_values]
    rng=np.random.default_rng(20261020)
    # Controlled ADU stress only; camera photon gain/noise is unknown.
    noisy=[reconstruct(frames.astype(float)+rng.normal(0,v,size=frames.shape)) for v in noise_values]
    no_bandpass=reconstruct(frames,bandpass=False)
    integer=fresh.copy();integer[0,:,1:3]=np.round(integer[0,:,1:3])
    results=dict(fresh=fresh,original=original,filtered=filtered,integer=integer,no_bandpass=no_bandpass,threshold_values=threshold_values,radius_values=radius_values,noise_values=noise_values)
    for i,t in enumerate(thresholds):results[f'threshold_{i}']=t
    for i,t in enumerate(radii):results[f'radius_{i}']=t
    for i,t in enumerate(noisy):results[f'noise_{i}']=t
    measurement=dict(canonical_count=int(fresh.shape[1]),counts_by_frame=[int(np.sum(fresh[0,:,0]==i)) for i in range(10)],upstream_comparison=table_metrics(fresh,original),max_absolute_upstream_difference=float(np.max(np.abs(fresh-original))),integer_comparison=table_metrics(integer,fresh),no_bandpass_comparison=table_metrics(no_bandpass,fresh),no_bandpass_count=int(no_bandpass.shape[1]),threshold_counts=[int(t.shape[1]) for t in thresholds],radius_counts=[int(t.shape[1]) for t in radii],noise_counts=[int(t.shape[1]) for t in noisy],threshold_metrics=[table_metrics(t,fresh) for t in thresholds],radius_metrics=[table_metrics(t,fresh) for t in radii],noise_metrics=[table_metrics(t,fresh) for t in noisy])
    results['measurements']=measurement;return results

def figures(frames,result):
    plt.rcParams.update({'font.size':9,'axes.titlesize':10})
    figs={};table=result['fresh'][0];original=result['original'][0]
    fig,axes=plt.subplots(2,5,figsize=(14,6),constrained_layout=True)
    vmin=float(frames.min());vmax=float(frames.max())
    for i,ax in enumerate(axes.flat):
        im=ax.imshow(frames[0,i],cmap='gray',vmin=vmin,vmax=vmax,origin='upper',extent=[-.5,127.5,127.5,-.5])
        p=table[table[:,0]==i];q=original[original[:,0]==i]
        ax.scatter(q[:,1],q[:,2],s=55,marker='s',facecolors='none',edgecolors='#D55E00',lw=.6,label='Original')
        ax.scatter(p[:,1],p[:,2],s=24,marker='+',color='#56B4E9',lw=.7,label='Fresh')
        ax.set(title=f'Frame{i}: {len(p)} features',xlabel='x (pixel)',ylabel='y (pixel)')
    axes[0,0].legend(fontsize=7,loc='upper right');fig.suptitle(f'All complete native frames — shared native ADU range [{vmin:g}, {vmax:g}]',fontsize=13);fig.colorbar(im,ax=axes.ravel().tolist(),label='Native camera intensity (ADU)',shrink=.85)
    figs['native_movie_localizations']=fig
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    p=table[table[:,0]==0][0];cx,cy=int(round(p[1])),int(round(p[2]));lo_x,hi_x=cx-5,cx+6;lo_y,hi_y=cy-5,cy+6;extent=[lo_x-.5,hi_x-.5,hi_y-.5,lo_y-.5]
    for ax,img,title in [(axes[0,0],frames[0,0],'Observed local display zoom (ADU)'),(axes[0,1],result['filtered'][0],'Bandpass local display zoom (ADU)')]:
        im=ax.imshow(img[lo_y:hi_y,lo_x:hi_x],origin='upper',cmap='viridis',extent=extent);fig.colorbar(im,ax=ax,label='ADU');ax.scatter([p[1]],[p[2]],color='#D55E00',marker='+',s=80,label='Subpixel moment');ax.scatter([cx],[cy],edgecolor='white',facecolor='none',s=50,label='Nearest pixel');ax.set(title=title,xlabel='x (pixel)',ylabel='y (pixel)')
    axes[0,1].legend(fontsize=7)
    axes[0,2].plot(np.arange(10),result['measurements']['counts_by_frame'],'o-',color='#0072B2');axes[0,2].set(title='Measured feature counts',xlabel='Frame index',ylabel='Retained features')
    axes[1,0].hist(table[:,1:3].reshape(-1)%1,bins=np.linspace(0,1,11),color='#0072B2');axes[1,0].set(title='Subpixel phases of all centers',xlabel='Fractional x/y coordinate (pixel)',ylabel='Coordinate measurements')
    axes[1,1].scatter(table[:,3],table[:,4],c=table[:,5],cmap='viridis',s=30);axes[1,1].set(title='All inferred feature moments',xlabel='Filtered integrated mass (ADU)',ylabel='Radius of gyration (pixel)');fig.colorbar(axes[1,1].collections[0],ax=axes[1,1],label='Eccentricity')
    delta=table[:,1:3]-original[:,1:3];axes[1,2].scatter(np.arange(len(table)),np.linalg.norm(delta,axis=1),s=10,color='#D55E00');axes[1,2].set(title='Independent original position parity',xlabel='Feature index',ylabel='Euclidean center difference (pixel)')
    figs['subpixel_moments_parity']=fig
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for col,(name,values,xlabel) in enumerate([('threshold',result['threshold_values'],'Signal threshold (ADU)'),('radius',result['radius_values'],'Aperture feature radius (pixel)'),('noise',result['noise_values'],'Added Gaussian stress SD (ADU)')]):
        metrics=result['measurements'][name+'_metrics'];counts=result['measurements'][name+'_counts']
        axes[0,col].plot(values,counts,'o-',color='#0072B2');axes[0,col].set(title=name.capitalize()+' sensitivity: count',xlabel=xlabel,ylabel='Retained features')
        axes[1,col].plot(values,[m['detection_f1'] for m in metrics],'o-',color='#D55E00',label='F1 vs canonical');axes[1,col].set(xlabel=xlabel,ylabel='Detection F1',ylim=(0,1.05));axes[1,col].legend(fontsize=8)
    fig.suptitle('Fresh complete-movie sensitivity — estimator agreement, not biological truth',fontsize=12);figs['threshold_support_noise']=fig
    return figs
