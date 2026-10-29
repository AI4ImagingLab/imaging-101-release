
"""Maintainer scientific comparisons, computed from local modules and supplied arrays.
No backend changes, filesystem writes, upstream imports, or hidden asset loading.
"""
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';GRAY='#666666'
def relative_error(estimate,reference):
    norm=np.linalg.norm(reference.ravel())
    if norm<=0:raise ValueError('Zero reference norm')
    return float(np.linalg.norm((estimate-reference).ravel())/norm)
def setup():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'svg.fonttype':'none','pdf.fonttype':42})
def finish(fig):
    fig.align_labels()
    return fig

def make_figures(data,result,reference):
    from .solvers import solve
    from .preprocessing import calibrate,cross_spectral_matrix
    from .physics_model import grid_positions
    setup();clean,dirty=solve(data);extent=[-.625,.025,-.325,.325];lim=max(clean.max(),dirty.max());fig,ax=plt.subplots(2,2,figsize=(10.5,8),layout='constrained')
    for a,image,title in [(ax[0,0],dirty,'A  Conventional delay-and-sum'),(ax[0,1],clean,'B  Fresh DAMAS, 100 iterations'),(ax[1,0],reference['reconstruction'][0],'C  Reproduced upstream DAMAS')]:
        im=a.imshow(image.T,origin='lower',extent=extent,vmin=0,vmax=lim,cmap='viridis');a.set(xlabel='x (m)',ylabel='y (m)',title=title);fig.colorbar(im,ax=a,label='Band mean-square pressure (Pa²)')
    residual=clean-reference['reconstruction'][0];rmax=max(abs(residual).max(),1e-20);im=ax[1,1].imshow(residual.T,origin='lower',extent=extent,vmin=-rmax,vmax=rmax,cmap='RdBu_r');ax[1,1].set(xlabel='x (m)',ylabel='y (m)',title='D  Signed local-minus-upstream');fig.colorbar(im,ax=ax[1,1],label='Difference (Pa²)')
    fig2,aa=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');traces={'microphone_position':data['microphone_position'],'fresh_damas':clean,'das':dirty,'upstream_damas':reference['reconstruction'][0]};rows=[]
    for iterations,color in zip([1,10,100],[ORANGE,GREEN,BLUE]):
        q,_=solve(data,iterations=iterations);aa[0].plot(np.linspace(-.6,0,13),q.sum(axis=1),color=color,label=f'{iterations} iterations');rows.append([iterations,float(q.sum()),float(q.max())]);traces[f'iterations_{iterations}_map']=q
    for speed,color in zip([336.04,346.04,356.04],[ORANGE,BLUE,GREEN]):
        q,_=solve(data,sound_speed=speed);aa[1].plot(np.linspace(-.6,0,13),q.sum(axis=1),color=color,label=f'c={speed:g} m/s');traces[f'speed_{speed:g}_map']=q
    for count,style in [(6400,':'),(12800,'--'),(25600,'-')]:
        q,_=solve(dict(data,microphone_signal=data['microphone_signal'][:count]));aa[2].plot(np.linspace(-.6,0,13),q.sum(axis=1),style,label=f'{count/51200:.3f} s');traces[f'samples_{count}_map']=q
    for aax,title in zip(aa,['A  Inverse-iteration convergence','B  Sound-speed sensitivity','C  Shorter acquisition duration']):aax.set(xlabel='x (m)',ylabel='Power summed over y (Pa²)',title=title);aax.legend(fontsize=8)
    return {'reconstruction':finish(fig),'robustness':finish(fig2)},traces,{'upstream_relative_error':relative_error(clean,reference['reconstruction'][0]),'das_sum_Pa2':float(dirty.sum()),'damas_sum_Pa2':float(clean.sum()),'iteration_total_peak_Pa2':rows,'peak_xy_m':grid_positions()[np.argmax(clean)].tolist()[:2]}
