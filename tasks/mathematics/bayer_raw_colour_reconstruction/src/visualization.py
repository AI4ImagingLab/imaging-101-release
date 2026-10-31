"""Scientific native-scale comparisons; rendering never mutates reconstruction."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import sample_rgb,sampling_adjoint
from .solvers import interpolation_kernels,reconstruct

def compute_metrics(image,reference):
    x=np.asarray(image).ravel();r=np.asarray(reference).ravel()
    rn=float(np.linalg.norm(r));xn=float(np.linalg.norm(x));dr=float(np.ptp(r))
    if rn<=0 or dr<=0:raise ValueError('Degenerate reference')
    return {'ncc':float(x@r/(xn*rn)) if xn>0 else 0.,'nrmse':float(np.sqrt(np.mean((x-r)**2))/dr)}

def render_figures(mosaic,image,bilinear,reference,white_balance):
    """Return four actual figure groups and exact plotted numeric sources."""
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.dpi':110})
    gain=np.asarray(white_balance[:3],dtype=float)/white_balance[1]
    scale=float(np.percentile(np.maximum(image[::4,::4]*gain,0),99.5))
    def rgb(x):return np.clip(np.maximum(x*gain,0)/scale,0,1)**(1/2.2)
    numeric={'display_gain':gain,'display_scale':np.array(scale),'display_gamma':np.array(1/2.2),'overview_stride':np.array(4),'mosaic_preview':mosaic[::4,::4],'fresh_preview':image[::4,::4],'bilinear_preview':bilinear[::4,::4],'upstream_preview':reference[::4,::4]}
    figs=[]
    f,ax=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    averaged=mosaic.reshape(mosaic.shape[0]//4,4,mosaic.shape[1]//4,4).mean(axis=(1,3))
    numeric['mosaic_block_mean4']=averaged
    ax[0,0].imshow(np.clip(averaged/scale,0,1)**(1/2.2),cmap='gray',vmin=0,vmax=1);ax[0,0].set_title('Measured CFA (display4x4 block mean/gamma)')
    edges=np.linspace(-.02,1.05,100);numeric['sensor_histogram_edges']=edges
    channel_samples=[mosaic[0::2,0::2],np.r_[mosaic[0::2,1::2].ravel(),mosaic[1::2,0::2].ravel()],mosaic[1::2,1::2]]
    for name,color,values in zip('RGB',['tab:red','tab:green','tab:blue'],channel_samples):
        counts,_=np.histogram(values,bins=edges);numeric['sensor_histogram_'+name]=counts
        ax[0,1].stairs(counts,edges,color=color,label=name)
    ax[0,1].set(title='Actually measured colour-site distributions',xlabel='Normalized detector intensity',ylabel='Number of sensor samples',yscale='log');ax[0,1].legend()
    for a,x,title in [(ax[0,2],reference,'Independent original Malvar'),(ax[1,0],image,'Fresh local Malvar'),(ax[1,1],bilinear,'Fresh bilinear comparison')]:a.imshow(rgb(x[::4,::4]));a.set_title(title)
    diff=(image[::4,::4]-bilinear[::4,::4]).mean(axis=-1);numeric['overview_difference']=diff
    im=ax[1,2].imshow(diff,cmap='RdBu_r',vmin=-.04,vmax=.04);f.colorbar(im,ax=ax[1,2],label='Mean RGB difference (normalized ADU)');ax[1,2].set_title('Malvar minus bilinear; not error to truth')
    for a in [ax[0,0],ax[0,2],*ax[1,:]]:a.set_xticks([]);a.set_yticks([])
    f.suptitle('Real Canon A540: complete native reconstruction; display-only white balance/gamma',fontsize=14);figs.append(f)
    f,ax=plt.subplots(2,4,figsize=(15,8),layout='constrained')
    rois=[(800,1056,1450,1706),(1400,1656,1600,1856)]
    for row,(y0,y1,x0,x1) in enumerate(rois):
        sl=np.s_[y0:y1,x0:x1];numeric[f'roi{row}_bounds']=np.array([y0,y1,x0,x1]);numeric[f'roi{row}_mosaic']=mosaic[sl];numeric[f'roi{row}_fresh']=image[sl];numeric[f'roi{row}_bilinear']=bilinear[sl]
        ax[row,0].imshow(mosaic[sl],cmap='gray',vmin=0,vmax=1)
        ax[row,1].imshow(rgb(bilinear[sl]));ax[row,2].imshow(rgb(image[sl]))
        dd=(image[sl]-bilinear[sl]).mean(axis=-1);im=ax[row,3].imshow(dd,cmap='RdBu_r',vmin=-.04,vmax=.04);f.colorbar(im,ax=ax[row,3],label='Normalized ADU')
        for col,title in enumerate(['Native measured CFA','Fresh bilinear','Fresh Malvar','Signed method difference']):ax[row,col].set_title(title);ax[row,col].set_xlabel(f'Native ROI x={x0}:{x1}');ax[row,col].set_yticks([]);ax[row,col].set_xticks([])
    f.suptitle('Native256×256 regions: object edges and texture, same display scale; no RGB ground truth',fontsize=14);figs.append(f)
    f,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained');row=1320;cols=np.arange(700,1900)
    numeric['profile_row']=np.array(row);numeric['profile_columns']=cols;numeric['profile_fresh']=image[row,cols];numeric['profile_bilinear']=bilinear[row,cols];numeric['profile_observed']=mosaic[row,cols]
    for c,(name,color) in enumerate(zip('RGB',['tab:red','tab:green','tab:blue'])):
        ax[0,c].plot(cols,bilinear[row,cols,c],color='0.5',lw=.9,label='Fresh bilinear');ax[0,c].plot(cols,image[row,cols,c],color=color,lw=.8,label='Fresh Malvar')
        known=(cols%2==0) if c==0 else (cols%2==1) if c==1 else np.zeros_like(cols,dtype=bool)
        if np.any(known):ax[0,c].scatter(cols[known][::5],mosaic[row,cols[known]][::5],s=7,color='black',label='Measured channel samples')
        else:ax[0,c].text(.02,.03,'No blue samples on this sensor row',transform=ax[0,c].transAxes,fontsize=8)
        ax[0,c].set(title=f'Native row{row}, {name} channel',xlabel='Sensor column',ylabel='Normalized detector intensity');ax[0,c].legend(fontsize=8)
        values=(image[...,c]-bilinear[...,c]).ravel();q=np.array([0,.01,.1,.5,.9,.99,1]);v=np.quantile(values,q);numeric[f'difference_quantiles_{name}']=v;numeric['quantile_probabilities']=q
        ax[1,c].plot(q,v,'o-',color=color);ax[1,c].axhline(0,color='0.6',lw=.8);ax[1,c].set(title=f'{name}: signed method-difference quantiles',xlabel='Quantile probability (all native pixels)',ylabel='Normalized detector intensity')
    f.suptitle('Measured samples constrain one channel per pixel; absent blue samples on this row are not fabricated',fontsize=13);figs.append(f)
    f,ax=plt.subplots(2,4,figsize=(15,7.5),layout='constrained')
    for k,(kernel,title) in enumerate(zip(interpolation_kernels(),['Green at red/blue','Horizontal colour correction','Vertical colour correction','Opposite red/blue'])):
        numeric[f'kernel_{k}']=kernel;im=ax[0,k].imshow(kernel,cmap='RdBu_r',vmin=-.8,vmax=.8);f.colorbar(im,ax=ax[0,k],label='Linear coefficient');ax[0,k].set_title(title)
        for i in range(5):
            for j in range(5):
                if kernel[i,j]:ax[0,k].text(j,i,f'{kernel[i,j]:g}',ha='center',va='center',fontsize=8)
        imp=np.zeros((15,15));imp[6+k//2,6+k%2]=1;response=reconstruct(imp)[4:11,4:11]
        numeric[f'unit_impulse_{k}']=response
        im=ax[1,k].imshow(response.mean(axis=-1),cmap='RdBu_r',vmin=-.8,vmax=.8);f.colorbar(im,ax=ax[1,k],label='Mean RGB / unit detector impulse');ax[1,k].set_title(f'Unit-impulse response, Bayer phase{k}')
    for a in ax.flat:a.set_xticks([]);a.set_yticks([])
    f.suptitle('Independent operator diagnostics: signed gradient filters can spread sensor defects; not extra measured scenes',fontsize=13);figs.append(f)
    return figs,numeric
