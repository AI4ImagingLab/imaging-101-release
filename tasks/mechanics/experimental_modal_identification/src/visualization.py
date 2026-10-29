
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
    from .solvers import reconstruct,prepare,identify_poles,stabilization
    setup();f,h,dt=prepare(data['frequency'],data['accelerance']);fit=result['fitted_accelerance'][0];a=result['reconstruction'][0];fn=result['natural_frequency'][0];xi=result['damping_ratio'][0]
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    ax[0,0].semilogy(f,abs(h[0]),color=GRAY,label='Measured');ax[0,0].semilogy(f,abs(fit[0]),color=BLUE,label='Fresh LSCF/LSFD');ax[0,0].set(xlabel='Frequency (Hz)',ylabel='Accelerance magnitude (m/s²/N)',title='A  Full selected measurement band');ax[0,0].legend()
    ax[0,1].plot(f,np.angle(h[0],deg=True),color=GRAY,label='Measured');ax[0,1].plot(f,np.angle(fit[0],deg=True),color=BLUE,label='Fresh');ax[0,1].set(xlabel='Frequency (Hz)',ylabel='Phase (degree)',title='B  Complex response phase')
    positions=np.arange(1,7)
    for j in range(3):ax[1,0].plot(positions,a[:,j].real/np.max(abs(a[:,j])),'o-',label=f'Mode {j+1}: {fn[j]:.1f} Hz')
    ax[1,0].set(xlabel='Roving excitation index',ylabel='Normalized real residue (1)',title='C  First three spatial mode signatures');ax[1,0].legend(fontsize=8)
    ax[1,1].bar(np.arange(1,8),xi*100,color=BLUE);ax[1,1].set(xlabel='Mode index',ylabel='Damping ratio (%)',title='D  All seven damped modes')
    fig2,aa=plt.subplots(1,3,figsize=(13,3.7),layout='constrained');traces={'frequency':f,'observed_accelerance':h,'fresh_accelerance':fit,'natural_frequency':fn,'damping_ratio':xi,'modal_constants':a,'upstream_modal_constants':reference['reconstruction'][0]};rows=[];unstable_counts=[]
    for order,color in zip([30,45,60],[ORANGE,GREEN,BLUE]):
        q=reconstruct(data,order=order);unstable_counts.append(int(np.count_nonzero(q['damping_ratio']<=0)));aa[0].plot(np.arange(1,8),q['natural_frequency'][0]-fn,'o-',color=color,label=f'Maximum order {2*order}');aa[1].plot(np.arange(1,8),q['damping_ratio'][0]*100,'o-',color=color,label=f'{2*order}');traces[f'polynomial_order_{order}_frequency']=q['natural_frequency'];traces[f'polynomial_order_{order}_damping_ratio']=q['damping_ratio'];rows.append([order,relative_error(q['fitted_accelerance'][0],h)])
    for scale,color in zip([.99,1,1.01],[ORANGE,BLUE,GREEN]):
        pert=dict(data,frequency=data['frequency']*scale);q=reconstruct(pert);aa[2].plot(np.arange(1,8),100*(q['natural_frequency'][0]/fn-1),'o-',color=color,label=f'Frequency scale {scale:g}');traces[f'frequency_scale_{scale:g}']=q['natural_frequency']
    aa[0].set(xlabel='Mode index',ylabel='Natural frequency change (Hz)',title='A  Polynomial-order convergence');aa[1].set(xlabel='Mode index',ylabel='Damping ratio (%)',title='B  Damping identifiability');aa[2].set(xlabel='Mode index',ylabel='Frequency shift (%)',title='C  Frequency calibration sensitivity')
    if any(unstable_counts):aa[1].text(.25,.035,'Negative damping = unstable candidate',transform=aa[1].transAxes,fontsize=7,color='#a33b32')
    for aax in aa:aax.legend(fontsize=7)
    allp=identify_poles(f,h,dt);ft,xt,tf,tx=stabilization(allp);stable=(tf>0)&(tx>0)&(xt>0);fig3,ab=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ii,jj=np.where(stable&(ft>0)&(ft<5000));ab[0].scatter(ft[ii,jj],2*(jj+1),s=5,color=GRAY,label='Stable frequency and damping');ab[0].vlines(fn,0,120,color=BLUE,linestyle='--',linewidth=.8,label='Selected modes');ab[0].set(xlabel='Frequency (Hz)',ylabel='Polynomial denominator degree',title='A  Complete stabilization evidence');ab[0].legend(fontsize=8)
    residual=h-fit;ab[1].plot(f,np.sqrt(np.mean(abs(residual)**2,axis=0)),color=BLUE);ab[1].set(yscale='log',xlabel='Frequency (Hz)',ylabel='Six-location residual RMS (m/s²/N)',title='B  Unmodeled measured response')
    traces['stable_frequency_matrix']=ft;traces['stable_damping_matrix']=xt;traces['stability_frequency_pass']=tf;traces['stability_damping_pass']=tx
    return {'reconstruction':finish(fig),'robustness':finish(fig2),'inverse_diagnostics':finish(fig3)},traces,{'upstream_relative_error':relative_error(result['reconstruction'],reference['reconstruction']),'observation_relative_error':relative_error(fit,h),'order_relative_response_error':rows,'polynomial_order_unstable_modes':unstable_counts,'natural_frequency_Hz':fn.tolist(),'damping_ratio':xi.tolist()}
