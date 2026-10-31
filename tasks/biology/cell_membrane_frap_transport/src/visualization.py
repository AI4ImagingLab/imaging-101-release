"""Maintainer plots and independently tested display metrics."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import compression_matrix
from .preprocessing import PIXEL_SIZE,TIME_STEP

def agreement(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    if np.ptp(b)<=0 or np.linalg.norm(b)<=0: raise ValueError('Degenerate reference')
    an=np.linalg.norm(a)
    return float(a@b/(an*np.linalg.norm(b))) if an else 0.,float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))

def acquisition_figure(data,result):
    movie=data['movie'][0];p=result['preprocessing']
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    ax[0].imshow(movie[0],cmap='gray',vmin=95,vmax=165)
    for key,color in [('frap_coordinates','tomato'),('control_coordinates','cyan')]:
        xy=data[key][0];ax[0].plot(xy[:,0],xy[:,1],color=color,lw=2,label=key.replace('_coordinates',''))
    ax[0].legend(fontsize=8);ax[0].set_title('Measured prebleach image and ROIs')
    times=np.arange(42)*float(TIME_STEP)
    ax[1].plot(times,p['control'].mean(axis=1),'o',ms=3,label='Control observation')
    ax[1].plot(times,p['control_fit'],label='Exponential fit')
    ax[1].set(xlabel='Acquisition time (s)',ylabel='Camera intensity (a.u.)',title='Independent photobleaching estimate');ax[1].legend(fontsize=8)
    ax[2].plot(times,p['profile'].mean(axis=1),label='Entire FRAP line')
    ax[2].plot(times,p['profile'][:,30:60].mean(axis=1),label='Central line segment')
    ax[2].axvline(30,color='gray',ls='--');ax[2].set(xlabel='Acquisition time (s)',ylabel='Camera intensity (a.u.)',title='Recovery mixed with imaging loss');ax[2].legend(fontsize=8)
    return fig

def recovery_figure(result,upstream):
    p=result['preprocessing'];c=compression_matrix(89)
    projected=p['normalized']@c.T@c*p['norm'][1]+p['norm'][0]
    fresh=result['reconstruction'][0]
    arrays=[p['profile'][1:],projected,upstream[0],fresh,fresh-projected]
    labels=['Measured line','Nine-mode observations','Reproduced upstream','Fresh local fit','Fresh fit − observations']
    fig,axes=plt.subplots(1,5,figsize=(15,4.2),layout='constrained')
    low,high=np.percentile(p['profile'][1:],[1,99])
    for i,(ax,a,label) in enumerate(zip(axes,arrays,labels)):
        lim=np.max(abs(a)) if i==4 else None
        im=ax.imshow(a,aspect='auto',origin='lower',extent=(0,88*float(PIXEL_SIZE),0,40*float(TIME_STEP)),cmap='coolwarm' if i==4 else 'viridis',vmin=-lim if i==4 else low,vmax=lim if i==4 else high)
        ax.set(title=label,xlabel='Arc length (µm)');fig.colorbar(im,ax=ax,label='Camera a.u.',shrink=.75)
    axes[0].set_ylabel('Time after bleaching (s)')
    return fig

def model_figure(results):
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    names=list(results)
    ax[0].bar(names,[results[k]['cost'] for k in names]);ax[0].set(title='Fresh model comparison',ylabel='Trace-normalized residual cost')
    ax[1].bar(names,[results[k]['r2'] for k in names]);ax[1].set(ylabel='Adjusted R²',title='Same complete acquisition')
    for name,r in results.items():
        ax[2].plot(np.arange(41)*30,r['reconstruction'][0,:,44],label=name)
    ax[2].set(xlabel='Time after bleaching (s)',ylabel='Camera a.u.',title='Center-position recovery');ax[2].legend(fontsize=8)
    for a in ax[:2]:a.tick_params(axis='x',labelrotation=25)
    return fig
