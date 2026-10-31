"""Review figures; source Bayesian comparisons remain explicitly separate."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.special import spherical_jn

def native_observations(raw):
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    cmap=plt.colormaps['viridis'].copy();cmap.set_bad('white')
    for row,n in enumerate(['360','197']):
        q=raw['q_edges_'+n][0];e=raw['energy_edges_'+n][0]
        for group,title in enumerate(['290 K sample','260 K mobile reference','Empty container']):
            a=raw['signal_'+n][0,group];finite=a[np.isfinite(a)];lo,hi=np.percentile(finite,[1,99])
            image=axes[row,group].pcolormesh(e,q,a,shading='flat',cmap=cmap,norm=Normalize(lo,hi))
            axes[row,group].set(title=f'{float(n)/100:.2f} meV: {title}',xlabel='True energy transfer (meV)',ylabel='Native Q (1/Angstrom)')
            fig.colorbar(image,ax=axes[row,group],label='Native signal (U), 1-99% color limits')
    fig.suptitle('All six original observations; white is explicit missing acquisition support')
    return fig

def analysis_observations(data):
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,n in enumerate(['360','197']):
        d=data[n]
        for col,key,title,unit in [(0,'y','Sample - empty, analysis scale','U'),(1,'sigma','Propagated diagonal error','U'),(2,'resolution','Normalized 260 K - empty reference','dimensionless')]:
            image=axes[row,col].pcolormesh(d['energy'],d['q'],d[key],shading='nearest',cmap='viridis')
            axes[row,col].set(title=f'{float(n)/100:.2f} meV: {title}',xlabel='True energy transfer (meV)',ylabel='Q (1/Angstrom)')
            fig.colorbar(image,ax=axes[row,col],label=unit)
    fig.suptitle('Fixed complete analysis windows; empirical reference is not a static instrument response')
    return fig

def all_q_comparison(data,n,fresh,source,posterior):
    d=data[n];q=d['q'];columns=4;rows=int(np.ceil(len(q)/columns))
    fig,axes=plt.subplots(rows,columns,figsize=(16,3.1*rows),layout='constrained',squeeze=False)
    colors=['#d7e7f3','#a3c5df','#609ac5']
    bands=posterior['bands_'+n][0];mean=posterior['mean_'+n][0]
    for i,ax in enumerate(axes.flat):
        if i>=len(q):ax.set_visible(False);continue
        e=d['energy']
        for level,(low,high) in enumerate([(0,5),(1,4),(2,3)]):ax.fill_between(e,bands[i,low],bands[i,high],color=colors[level],label=['Source 99.7% band','Source 95% band','Source 68% band'][level])
        ax.errorbar(e,d['y'][i],yerr=d['sigma'][i],fmt='.',ms=2,color='#333333',alpha=.6,elinewidth=.4,label='Observed +/- diagonal error')
        ax.plot(e,mean[i],color='#27638e',lw=1.3,label='Source full posterior mean')
        ax.plot(e,source['predicted_'+n][0,i],color='#b17a24',ls=':',lw=1.2,label='Source best sampled point')
        ax.plot(e,fresh['predicted_'+n][0,i],color='#c73736',ls='--',lw=1.3,label='Fresh local point estimate')
        ax.set(title=f'Q = {q[i]:.4f} /Angstrom',xlabel='True energy transfer (meV)',ylabel='Analysis signal (U)')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle(f'{float(n)/100:.2f} meV: every selected Q row, source Bayesian bands and fresh point fit')
    return fig

def residual_diagnostics(data,fresh,source,posterior):
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,n in enumerate(['360','197']):
        d=data[n];local=(fresh['predicted_'+n][0]-d['y'])/d['sigma'];original=(source['predicted_'+n][0]-d['y'])/d['sigma']
        vmax=max(np.max(abs(local)),np.max(abs(original)))
        for col,a,title in [(0,original,'Source best sampled standardized residual'),(1,local,'Fresh point standardized residual')]:
            im=axes[row,col].pcolormesh(d['energy'],d['q'],a,shading='nearest',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
            axes[row,col].set(title=f'{float(n)/100:.2f} meV: {title}',xlabel='True energy transfer (meV)',ylabel='Q (1/Angstrom)');fig.colorbar(im,ax=axes[row,col],label='Residual / sigma')
        ax=axes[row,2]
        for a,label,color in [(original,'Source best sampled','#b17a24'),(local,'Fresh local point','#c73736')]:ax.plot(d['q'],np.sqrt(np.mean(a*a,axis=1)),'o-',label=label,color=color)
        bayes=(posterior['mean_'+n][0]-d['y'])/d['sigma'];ax.plot(d['q'],np.sqrt(np.mean(bayes*bayes,axis=1)),'s-',label='Source posterior mean',color='#27638e')
        ax.axhline(1,color='grey',ls=':');ax.set(xlabel='Q (1/Angstrom)',ylabel='RMS standardized residual',title='Every Q row retained');ax.legend(fontsize=8)
    fig.suptitle('Numerical fit quality; residual structure and diagonal-noise limitations remain visible')
    return fig

def parameter_and_model_limits(data,fresh,source,posterior):
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained');p=fresh['reconstruction'][0];s=source['reconstruction'][0]
    for index,(label,unit) in enumerate([('Translation linewidth coefficient','meV Angstrom^2'),('First rotational linewidth','meV'),('Mean-square displacement','Angstrom^2')]):
        ax=axes[0,index];x=posterior['parameter_histogram_centers'][index];density=posterior['parameter_histogram_density'][index]
        ax.plot(x,density,color='#27638e',label='Original joint posterior marginal');ax.axvline(s[index],color='#b17a24',ls=':',label='Source best sampled point');ax.axvline(p[index],color='#c73736',ls='--',label='Fresh local point')
        ax.set(title=label,xlabel=unit,ylabel='Source probability density');ax.legend(fontsize=8)
    for n,start,stop in [('360',4,16),('197',17,31)]:
        axes[1,0].plot(data[n]['q'],s[start:stop],':o',label=n+' source');axes[1,0].plot(data[n]['q'],p[start:stop],'--s',label=n+' fresh')
    axes[1,0].set(title='All 26 fitted backgrounds',xlabel='Q (1/Angstrom)',ylabel='Background (U)');axes[1,0].legend(fontsize=8)
    for n,idx in [('360',3),('197',16)]:
        axes[1,1].plot(posterior['parameter_histogram_centers'][idx],posterior['parameter_histogram_density'][idx],label=n+' source marginal');axes[1,1].axvline(p[idx],ls='--',label=n+' fresh point')
    axes[1,1].set(title='Both shared normalizers',xlabel='Normalizer (meV^-2/U)',ylabel='Source probability density');axes[1,1].legend(fontsize=8)
    for n in ['360','197']:
        q=data[n]['q'];mass=sum((2*l+1)*spherical_jn(l,q*2.48)**2 for l in range(3));axes[1,2].plot(q,mass,'o-',label=n+' retained l<=2 weight')
    axes[1,2].axhline(1,color='grey',ls=':');axes[1,2].set(title='Published rotational truncation, no renormalization',xlabel='Q (1/Angstrom)',ylabel='Retained continuous angular weight',ylim=(0,1.05));axes[1,2].legend(fontsize=8)
    fig.suptitle('Source Bayesian marginals versus fresh point inference; neither is independent physical truth')
    return fig
