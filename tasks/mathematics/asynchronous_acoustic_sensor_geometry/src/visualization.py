"""Maintainer figures with explicit fresh, original, surveyed and privileged roles."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import predict_measurements,residual_jacobian

def _distances(points):
    d=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
    return d[np.triu_indices(len(points),1)]

def review_figures(data,result,diagnostics,truth,upstream,repeat_result,repeat_diagnostics,clock_free_result,clock_free_diagnostics,privileged):
    """Return five actual figure groups; callers reconstruct before supplying results."""
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':115})
    fresh=result['reconstruction'][0];orig=upstream['primary'][:,:3];rep=repeat_result['reconstruction'][0]
    wrong=clock_free_result['reconstruction'][0];state=diagnostics['state'];figures=[]
    fig,ax=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for a,image,title in [(ax[0],(data['tdoa_successive']-data['emission_intervals'][1:,None])*1e6,'Measured successive delay minus emission interval (µs)'),(ax[1],data['tdoa_between']*1e6,'Measured inter-sensor delay (µs)')]:
        im=a.imshow(image,aspect='auto',cmap='coolwarm');fig.colorbar(im,ax=a,shrink=.8);a.set_title(title,fontsize=10)
    ax[0].set(xlabel='Microphone index',ylabel='Event interval');ax[1].set(xlabel='Event index',ylabel='Microphone 1–9')
    for j,label in enumerate('xyz'):ax[2].plot(np.arange(1,14),data['displacement'][:,j],'.-',label=label)
    ax[2].set(title='Real processed source displacements',xlabel='Interval',ylabel='Displacement (m)');ax[2].legend()
    fig.suptitle('Observations: real audio timing + independent motion information');figures.append(fig)

    fig=plt.figure(figsize=(15,5),layout='constrained');a=fig.add_subplot(121,projection='3d');b=fig.add_subplot(122)
    for sl,color,label,marker in [(slice(0,10),'#2077aa','Microphones','o'),(slice(10,24),'#da8542','Sources','^')]:
        a.scatter(*truth[sl].T,color=color,marker=marker,label=label+' surveyed')
        a.scatter(*fresh[sl].T,facecolors='none',edgecolors=color,marker=marker,label=label+' fresh')
        for p,q in zip(truth[sl],fresh[sl]):a.plot(*np.stack((p,q)).T,color=color,alpha=.45,lw=.7)
    a.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Joint recovered geometry: visible survey discrepancies');a.legend(fontsize=8,loc='upper left')
    for points,label,color in [(orig,'Original selected run','#777777'),(fresh,'Fresh joint fit','#2077aa'),(rep,'Fresh independent starts','#da8542')]:
        b.plot(np.arange(24),np.linalg.norm(points-truth,axis=1),'.-',label=label,color=color)
    b.axvline(9.5,color='k',ls=':',lw=.8);b.text(2,.98,'10 microphones',transform=b.get_xaxis_transform(),va='top');b.text(14,.98,'14 sources',transform=b.get_xaxis_transform(),va='top')
    b.set(xlabel='Ordered target point',ylabel='Euclidean error to survey (m)',title='Physical error is decimetre-scale, not a numerical tolerance');b.legend(fontsize=9)
    figures.append(fig)

    fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained');pred=predict_measurements(state,data['emission_intervals'])
    for a,array,title,xlab,ylab in [(ax[0,0],(pred[0]-data['tdoa_successive'])*1e6,'Fresh successive timing residual','Microphone','Interval'),(ax[0,1],(pred[1]-data['tdoa_between'])*1e6,'Fresh inter-sensor timing residual','Event','Microphone 1–9')]:
        im=a.imshow(array,aspect='auto',cmap='coolwarm');fig.colorbar(im,ax=a,label='µs',shrink=.8);a.set(title=title,xlabel=xlab,ylabel=ylab)
    for j,label in enumerate('xyz'):ax[1,0].plot(np.arange(1,14),100*(pred[2]-data['displacement'])[:,j],'.-',label=label)
    ax[1,0].set(title='All source-motion residuals',xlabel='Interval',ylabel='Residual (cm)');ax[1,0].legend()
    ids=np.arange(10);a=ax[1,1];a.bar(ids-.17,state[:10,3]*1e3,.34,label='Offset (ms)',color='#2077aa');t=a.twinx();t.plot(ids,state[:10,4]*1e6,'o-',label='Drift (ppm)',color='#da8542')
    a.set(title='Estimated nuisance clocks — no clock truth available',xlabel='Microphone',ylabel='Relative offset (ms)');t.set_ylabel('Fractional drift (ppm)');a.legend(loc='upper left');t.legend(loc='upper right');figures.append(fig)

    fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for info,label,color in [(diagnostics,'Fresh primary','#2077aa'),(repeat_diagnostics,'Fresh independent','#da8542')]:
        ok=[s for s in info['starts'] if s['converged'] and s['cost'] is not None]
        bad=[s for s in info['starts'] if not s['converged'] and s['cost'] is not None]
        ax[0].scatter([s['start'] for s in ok],[s['cost'] for s in ok],s=20,color=color,label=label+' converged')
        if bad:ax[0].scatter([s['start'] for s in bad],[s['cost'] for s in bad],s=20,color=color,marker='x',alpha=.5)
    ax[0].set(yscale='log',xlabel='Independent start',ylabel='Observed weighted objective',title='All finite starts; × = not converged');ax[0].legend(fontsize=8)
    h=np.asarray(diagnostics['history']);ax[1].plot(h[:,0]+1,h[:,1],'.-',color='#2077aa');ax[1].set(yscale='log',xlabel='Selected-candidate iteration',ylabel='Pre-update objective',title='Actual undamped Gauss–Newton path')
    r,j=residual_jacobian(state,data);weighted=j/np.sqrt(np.r_[np.full(256,1e-8),np.full(39,1e-4)])[:,None]
    column_norm=np.linalg.norm(weighted,axis=0);scaled=weighted/column_norm
    singular=np.linalg.svd(scaled,compute_uv=False);ax[2].semilogy(np.arange(1,86),singular,'.',color='#2077aa')
    ax[2].set(title='Column-normalized local sensitivity spectrum',xlabel='Descending singular-value index',ylabel='Singular value');ax[2].text(.04,.06,'Nominal weights; conditional local diagnostic.\nNot measured uncertainty or a confidence interval.',transform=ax[2].transAxes,fontsize=8);figures.append(fig)

    fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    labels=['Joint clocks','Clocks fixed zero'];cost=[diagnostics['cost'],clock_free_diagnostics['cost']]
    bars=ax[0].bar(labels,cost,color=['#2077aa','#da8542']);ax[0].set(ylabel='Observed weighted objective',title='Fresh wrong-clock hypothesis ablation')
    for bar,v in zip(bars,cost):ax[0].text(bar.get_x()+bar.get_width()/2,v,f'{v:.1f}',ha='center',va='bottom')
    ax[1].plot(np.linalg.norm(fresh-truth,axis=1),'.-',label='Joint fit');ax[1].plot(np.linalg.norm(wrong-truth,axis=1),'.-',label='Zero-clock ablation');ax[1].set(xlabel='Ordered point',ylabel='Survey error (m)',title='Both comparisons computed from raw data');ax[1].legend(fontsize=8)
    ref=_distances(truth)
    for points,label,color in [(orig,'Original joint','#777777'),(fresh,'Fresh joint','#2077aa'),(privileged['state'][:,:3],'Original TDOA-M: privileged frame','#ac5596')]:
        e=np.abs(_distances(points)-ref);ax[2].plot(np.sort(e),np.arange(1,len(e)+1)/len(e),label=label,color=color)
    ax[2].set(xlabel='Absolute pairwise-distance error (m)',ylabel='Fraction of all 276 point pairs',title='Rigid-invariant comparison; unequal information');ax[2].legend(fontsize=8)
    ax[2].text(.04,.08,'TDOA-M frame uses surveyed microphones.\nNo equal-information superiority claim.',transform=ax[2].transAxes,fontsize=8);figures.append(fig)
    return figures
