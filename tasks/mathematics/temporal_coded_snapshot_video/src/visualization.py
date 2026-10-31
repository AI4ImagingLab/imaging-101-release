"""Scientific full-sequence review; display bounds never change stored outputs."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import observed_residuals

def review_figures(data,fresh_gap,fresh_admm,original_gap,original_admm,gap_diagnostics,admm_diagnostics):
    """Six figures from actual full-resolution fresh and original arrays."""
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':100})
    mask=data['mask'];obs=data['observations'];figures=[]
    fig,axes=plt.subplots(3,5,figsize=(14,8.5),layout='constrained')
    for k,ax in enumerate(axes.flat):
        im=ax.imshow(obs[k],cmap='gray',vmin=0,vmax=float(obs.max()));ax.set_title(f'Measured exposure {k+1}');ax.axis('off')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='Recorded integrated relative intensity')
    fig.suptitle('All 15 real integrated camera measurements; no high-speed video truth');figures.append(fig)

    fig,axes=plt.subplots(2,7,figsize=(15,5),layout='constrained')
    for k,ax in enumerate(axes.flat):
        im=ax.imshow(mask[k],cmap='gray',vmin=0,vmax=1);ax.set_title(f'Effective code {k+1}');ax.axis('off')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='Calibrated effective transmission')
    fig.suptitle('All 14 real-valued calibration masks; sweep reverses on alternating bursts');figures.append(fig)

    fig,axes=plt.subplots(15,14,figsize=(22,24))
    for k,ax in enumerate(axes.flat):
        ax.imshow(fresh_gap[k],cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
        if k<14:ax.set_title(f'Subframe {k+1}',fontsize=8)
        if k%14==0:ax.set_ylabel(f'Exposure {k//14+1}',fontsize=9)
        ax.text(.03,.05,str(k+1),transform=ax.transAxes,color='#e6a745',fontsize=6)
    fig.subplots_adjust(wspace=.015,hspace=.025,left=.04,right=.995,bottom=.01,top=.96)
    fig.suptitle('Every one of the 210 fresh GAP-TV chronological frames — display only [0,1], arrays retain overshoots',fontsize=14)
    figures.append(fig)

    selected=[0,13,14,27,105,209];fig,axes=plt.subplots(4,6,figsize=(16,10),layout='constrained')
    for row,(movie,label) in enumerate([(original_gap,'Reproduced original GAP100'),(fresh_gap,'Fresh local GAP100'),(original_admm,'Reproduced original ADMM200'),(fresh_admm,'Fresh local ADMM200')]):
        for col,k in enumerate(selected):
            ax=axes[row,col];ax.imshow(movie[k],cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Chronological frame {k+1}')
            if col==0:ax.set_ylabel(label,fontsize=9)
    fig.suptitle('Full-frame method comparisons on identical real measurements; none is actual video GT');figures.append(fig)

    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for ax,movie,label in [(axes[0,0],fresh_gap,'Fresh GAP-TV100'),(axes[0,1],fresh_admm,'Fresh ADMM-TV200')]:
        im=ax.imshow(movie[:,128,:].T,aspect='auto',cmap='gray',vmin=0,vmax=1,origin='upper');ax.set(title=label+' — full temporal row slice',xlabel='Chronological frame',ylabel='Detector column at row128')
        for k in range(14,210,14):ax.axvline(k-.5,color='#da8542',lw=.4,alpha=.5)
    for movie,label in [(fresh_gap,'Fresh GAP100'),(fresh_admm,'Fresh ADMM200')]:axes[1,0].plot(np.arange(1,211),np.mean(movie,axis=(1,2)),label=label)
    axes[1,0].set(xlabel='Chronological frame',ylabel='Mean relative intensity',title='Temporal signal and regularization differences');axes[1,0].legend()
    index=np.arange(15)
    for offset,movie,label,color in [(-.2,fresh_gap,'GAP100','#2077aa'),(.2,fresh_admm,'ADMM200','#da8542')]:
        residual=observed_residuals(movie,obs,mask);axes[1,1].bar(index+offset,residual,width=.38,label=label,color=color)
    axes[1,1].set(yscale='log',xlabel='Exposure index',ylabel='Relative observed L2 residual',title='All exposures: data consistency is not scene accuracy');axes[1,1].legend();figures.append(fig)

    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for diagnostics,ax,label in [(gap_diagnostics,axes[0,0],'GAP100'),(admm_diagnostics,axes[0,1],'ADMM200')]:
        for k,d in enumerate(diagnostics):ax.semilogy(np.arange(1,d['iterations']+1),d['relative_observation_residual_history'],lw=.7,label=f'{k+1}')
        ax.set(title='Every burst: actual fresh '+label+' history',xlabel='Outer iteration',ylabel='Relative observed L2 residual')
    for actual,original,label in [(fresh_gap,original_gap,'GAP100'),(fresh_admm,original_admm,'ADMM200')]:
        error=np.max(abs(actual-original),axis=(1,2));axes[1,0].plot(np.arange(1,211),error,label=label)
    axes[1,0].set(xlabel='Chronological frame',ylabel='Maximum pixel difference to original',title='Numerical extraction error across every frame');axes[1,0].legend()
    for movie,label in [(fresh_gap,'GAP100'),(fresh_admm,'ADMM200')]:
        outside=np.mean((movie<0)|(movie>1),axis=(1,2));axes[1,1].plot(np.arange(1,211),outside*100,label=label)
    axes[1,1].set(title='Unconstrained outputs: negatives/overshoots retained',xlabel='Chronological frame',ylabel='Pixels outside display [0,1] (%)');axes[1,1].legend();figures.append(fig)
    return figures
