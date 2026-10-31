"""Actual array visualizations; no backend changes on import."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

def style():
    plt.rcParams.update({'font.size':9,'figure.dpi':110,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'semibold'})

def profile_grid():
    return plt.subplots(4,6,figsize=(17,12),layout='constrained')

def preprocessing_profiles(raw,measured,coords,log=False):
    fig,axs=profile_grid()
    for k in range(8):
        for j,d in enumerate(['x','y','u']):
            ax=axs[k//2,3*(k%2)+j];key=f'm{k//4}_w{k%4}_{d}'
            x=raw[key+'_coords']-raw[key+'_mean'];y=raw[key+'_counts']
            if j==1:x=-x[::-1];y=y[::-1]
            y=y/y.max();g=measured[k,j]/measured[k,j].max()
            ax.plot(x,np.where(y>0,y,np.nan) if log else y,color='#a7adb2',lw=.8,label='Native centered signal / peak')
            ax.plot(coords,np.where(g>0,g,np.nan) if log else g,color='#007f85',lw=1.3,label='Processed observation / peak')
            ax.set(title=f'Group {k} · {d}',xlabel='Centered position (mm)',ylabel='Relative signal')
            if log:ax.set(yscale='log',ylim=(1e-4,1.5))
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2)
    return fig

def covariance_correction(S,source_v1_N,source_v2_N,fresh_N):
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,N,title in zip(axs,[source_v1_N,source_v2_N,fresh_N],
                          ['Original V1: wrong mode pairing','Corrected source V2','Fresh paired-mode extraction']):
        error=N@S@N.T-np.eye(4)
        image=ax.imshow(error,cmap='RdBu_r',vmin=-.2,vmax=.2)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{error[i,j]:.2g}',ha='center',va='center',fontsize=8)
        ax.set(title=title,xlabel='Normalized coordinate',ylabel='Normalized coordinate',xticks=range(4),yticks=range(4))
    fig.colorbar(image,ax=axs,label=r'$N\Sigma N^\mathsf{T}-I$ (dimensionless)')
    return fig

def fitted_profiles(measured,fresh,source_mc,coords,log=False):
    fig,axs=profile_grid()
    for k in range(8):
        for j,d in enumerate(['x','y','u']):
            ax=axs[k//2,3*(k%2)+j]
            ax.plot(coords,measured[k,j],'k.',ms=2,label='Measured observation')
            ax.plot(coords,source_mc[k,j],color='#ae9275',lw=.8,alpha=.8,label='Source V2: one million particle draw')
            ax.plot(coords,fresh[k,j],color='#007f85',lw=1.3,label='Fresh density: deterministic cell integral')
            ax.set(title=f'Group {k} · {d}',xlabel='Centered position (mm)',ylabel=r'Profile density ($\mathrm{mm}^{-1}$)')
            if log:ax.set(yscale='log',ylim=(1e-5,max(measured[k,j].max()*2,1e-3)))
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3)
    return fig

def corner(fresh_samples,source_samples,prior_covariance,normalized=False):
    fig,axs=plt.subplots(4,4,figsize=(12,12),layout='constrained')
    labels=[r'$z_0$',r'$z_1$',r'$z_2$',r'$z_3$'] if normalized else ['X (mm)',"X′ (mrad)",'Y (mm)',"Y′ (mrad)"]
    ranges=[(-3.5*np.sqrt(prior_covariance[i,i]),3.5*np.sqrt(prior_covariance[i,i])) for i in range(4)]
    actual_cov=np.cov(fresh_samples,rowvar=False)
    mean=fresh_samples.mean(axis=0)
    for i in range(4):
        for j in range(4):
            ax=axs[i,j]
            if i<j:ax.axis('off');continue
            if i==j:
                a,edges=np.histogram(fresh_samples[:,i],bins=65,range=ranges[i],density=True)
                b,_=np.histogram(source_samples[:,i],bins=edges,density=True)
                x=(edges[1:]+edges[:-1])/2
                ax.step(x,a,where='mid',color='#007f85',label='Fresh')
                ax.step(x,b,where='mid',color='#222222',ls='--',lw=.8,label='Reproduced source V2')
                sigma=np.sqrt(prior_covariance[i,i])
                ax.plot(x,np.exp(-.5*(x/sigma)**2)/(np.sqrt(2*np.pi)*sigma),color='#a67c52',ls=':',label='Conditional Gaussian prior')
                ax.set(ylabel='Marginal density')
            else:
                a,xe,ye=np.histogram2d(fresh_samples[:,j],fresh_samples[:,i],bins=65,range=[ranges[j],ranges[i]],density=True)
                b,_,_=np.histogram2d(source_samples[:,j],source_samples[:,i],bins=[xe,ye],density=True)
                ax.pcolormesh(xe,ye,np.ma.masked_less_equal(a.T,0),cmap='viridis',norm=LogNorm(),shading='flat')
                xc=(xe[1:]+xe[:-1])/2;yc=(ye[1:]+ye[:-1])/2
                ax.contour(xc,yc,b.T,levels=np.array([.1,.5,.9])*b.max(),colors='black',linewidths=.55,alpha=.6)
                theta=np.linspace(0,2*np.pi,200);circle=np.array([np.cos(theta),np.sin(theta)])
                for cov,center,ls in [(actual_cov,mean,'-'),(prior_covariance,np.zeros(4),':')]:
                    block=cov[np.ix_([j,i],[j,i])]
                    w,v=np.linalg.eigh(block);xy=2*(v@np.diag(np.sqrt(w))@circle)+center[[j,i],None]
                    ax.plot(xy[0],xy[1],color='white' if ls=='-' else '#e4b274',ls=ls,lw=1)
                ax.set(xlim=ranges[j],ylim=ranges[i])
            if i==3:ax.set_xlabel(labels[j])
            if j==0 and i!=0:ax.set_ylabel(labels[i])
    handles=[Line2D([],[],color='#007f85',label='Fresh histogram / marginals'),
             Line2D([],[],color='black',ls='--',label='Source V2 contours / marginals'),
             Line2D([],[],color='#a67c52',ls=':',label='Conditional Gaussian prior'),
             Line2D([],[],color='gray',label='Fresh 2-RMS ellipse (white on maps)')]
    fig.legend(handles=handles,loc='outside upper center',ncol=2)
    return fig

def metric_profiles(source,fresh,boundaries):
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for j,d in enumerate(['x','y','u']):
        ids=np.arange(j,24,3)
        axs[j].plot(range(8),np.asarray(source)[ids],'o--',color='#333333',label='Source V2 deterministic integration')
        axs[j].plot(range(8),np.asarray(fresh)[ids],'s-',color='#007f85',label='Fresh deterministic integration')
        axs[j].plot(range(8),np.asarray(boundaries)[ids],'x:',color='#b14d43',label='Frozen per-profile boundary')
        axs[j].set(title=f'All eight {d} profiles',xlabel='Transport group',ylabel='Relative profile L2 error',xticks=range(8))
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3)
    return fig

def entropy_tradeoff(rows):
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    epochs=np.arange(len(rows))
    axs[0].plot(epochs,[r['mean_error'] for r in rows],'o-',color='#007f85')
    axs[0].set(xlabel='Complete sweep (0 = support-constrained prior)',ylabel='Mean of all 24 relative profile errors',title='Complete physical reconstruction path')
    axs[1].plot([r['KL'] for r in rows],[r['mean_error'] for r in rows],'o-',color='#007f85')
    for i,r in enumerate(rows):axs[1].annotate(str(i),(r['KL'],r['mean_error']),xytext=(4,4),textcoords='offset points')
    axs[1].set(xlabel='Discrete KL relative to Gaussian midpoint prior',ylabel='Mean relative profile error',title='Measurement agreement requires leaving the prior')
    return fig
