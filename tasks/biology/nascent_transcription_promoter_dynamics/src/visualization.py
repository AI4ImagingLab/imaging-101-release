"""Figures from real traces and actually computed posterior quantities."""
import numpy as np
import matplotlib.pyplot as plt

def predicted_signal(result,lengths):
    p=result['posterior'][0];v=result['emission'][0];out=np.zeros_like(p);weights=np.r_[1/3,11/12,np.ones(17)]
    for i,L in enumerate(lengths):
        loading=v[0]+(v[1]-v[0])*p[i,:L]
        extended=np.r_[np.full(18,v[0]),loading]
        out[i,:L]=np.convolve(extended,weights,mode='full')[18:18+L]
    return out

def observation_figure(values,lengths):
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    a=values.copy();a[np.arange(86)[None,:]>=lengths[:,None]]=np.nan
    im=axs[0].imshow(a,aspect='auto',extent=(0,86*20,63,0));fig.colorbar(im,ax=axs[0],label='Measured fluorescence units');axs[0].set(xlabel='Time (s)',ylabel='Trace',title='All real author-selected observations')
    for i in (0,31,62):axs[1].plot(np.arange(lengths[i])*20,values[i,:lengths[i]],label=f'Trace {i}')
    axs[1].set(xlabel='Time (s)',ylabel='Measured fluorescence units',title='Distinct recorded nuclei');axs[1].legend();return fig

def posterior_figure(fresh,original,lengths):
    fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained');valid=np.arange(86)[None,:]<lengths[:,None]
    for ax,name,a in zip(axs,['Corrected original','Fresh local','Fresh minus corrected original'],[original['posterior'][0],fresh['posterior'][0],fresh['posterior'][0]-original['posterior'][0]]):
        im=ax.imshow(np.where(valid,a,np.nan),aspect='auto',cmap='coolwarm' if 'minus' in name else 'viridis');fig.colorbar(im,ax=ax,label='ON probability' if 'minus' not in name else 'Probability difference');ax.set(title=name,xlabel='Frame',ylabel='Trace')
    return fig

def parameter_figure(fresh,original):
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,name in zip(axs[:2],['transition','initial']):
        x=fresh[name][0];r=original[name][0];width=.35;pos=np.arange(x.size)
        ax.bar(pos-width/2,r.ravel(),width,label='Corrected original');ax.bar(pos+width/2,x.ravel(),width,label='Fresh');ax.set(title=name,ylabel='Probability');ax.legend(fontsize=8)
    x=np.arange(2);axs[2].bar(x-.175,original['emission'][0],.35,label='Corrected original');axs[2].bar(x+.175,fresh['emission'][0],.35,label='Fresh');axs[2].set(xticks=x,xticklabels=['OFF','ON'],ylabel='Fluorescence loading units',title=f"Noise SD fresh {fresh['noise'][0,0]:.2f}");axs[2].legend(fontsize=8);return fig

def residual_figure(values,lengths,fresh):
    pred=predicted_signal(fresh,lengths);fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for j,i in enumerate((0,31,62)):
        n=lengths[i];time=np.arange(n)*20
        axs[0,j].plot(time,values[i,:n],label='Measured');axs[0,j].plot(time,pred[i,:n],label='Posterior-mean forward');axs[0,j].set(title=f'Trace {i}',xlabel='Time (s)',ylabel='Fluorescence units');axs[0,j].legend(fontsize=8)
        axs[1,j].plot(time,values[i,:n]-pred[i,:n]);axs[1,j].axhline(0,color='gray');axs[1,j].set(xlabel='Time (s)',ylabel='Measured minus prediction')
    return fig

def entropy_figure(fresh,lengths):
    p=np.clip(fresh['posterior'][0],1e-15,1-1e-15);entropy=-p*np.log2(p)-(1-p)*np.log2(1-p);valid=np.arange(86)[None,:]<lengths[:,None]
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');im=axs[0].imshow(np.where(valid,entropy,np.nan),aspect='auto',vmin=0,vmax=1);fig.colorbar(im,ax=axs[0],label='Binary posterior entropy (bit)');axs[0].set(xlabel='Frame',ylabel='Trace',title='State ambiguity under the fitted model');axs[1].hist(p[valid],bins=30);axs[1].set(xlabel='Posterior ON probability',ylabel='Valid frames',title='No measured hidden-state labels');return fig
