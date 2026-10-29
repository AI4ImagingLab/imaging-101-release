"""Maintainer comparison figures; no global backend changes."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import jacobian,forward,cumulative_sensitivity
from .solvers import reconstruct,quadratic_solution
def compare(a,b):
    return float(np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))
def style():
    plt.rcParams.update({'figure.dpi':125,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','axes.labelcolor':'#24354b','text.color':'#24354b','figure.facecolor':'white','axes.facecolor':'#f8fafc','savefig.bbox':'tight'})
def acquisition(data):
    style();obs=data['apparent_conductivity'];x=data['station_coordinates'][:,0]
    fig,axs=plt.subplots(1,2,figsize=(12,3.9),layout='constrained')
    cmap=axs[0].imshow(obs.T,origin='lower',aspect='auto',extent=[x.min()-.5,x.max()+.5,-.5,5.5],cmap='magma')
    axs[0].set(xlabel='Along-transect position (m)',ylabel='Coil index',title='A  |  All field observations');fig.colorbar(cmap,ax=axs[0],label='Apparent conductivity (mS/m)')
    for i in range(6):axs[1].plot(x,obs[:,i],label=f'{"VCP" if i<3 else "HCP"} {data["coil_spacing"][i]:.2f} m',lw=1.5)
    axs[1].set(xlabel='Along-transect position (m)',ylabel='Apparent conductivity (mS/m)',title='B  |  Anomalous site retained');axs[1].axvline(8,color='#a65a44',ls=':',lw=1);axs[1].legend(ncol=2,fontsize=8)
    return fig,{'observations':obs,'station_x':x}
def sensitivity(data):
    style();fig,axs=plt.subplots(1,3,figsize=(13,3.6),layout='constrained');z=np.linspace(0,4,301)
    for i,(orientation,ls) in enumerate([('vcp','-'),('hcp','--')]):
        for s in (.32,.71,1.18):axs[0].plot(cumulative_sensitivity(z,s,orientation),z,ls,label=f'{orientation.upper()} {s:.2f}m')
    axs[0].invert_yaxis();axs[0].set(xlabel='Cumulative response',ylabel='Depth (m)',title='A  |  Broad depth kernels');axs[0].legend(fontsize=7)
    J=jacobian(data['layer_interfaces'],data['coil_spacing']);im=axs[1].imshow(J,aspect='auto',cmap='viridis');fig.colorbar(im,ax=axs[1],label='Sensitivity fraction');axs[1].set(xlabel='Layer index',ylabel='Coil index',title='B  |  Discrete acquisition')
    s=np.linalg.svd(J,compute_uv=False);axs[2].semilogy(np.arange(1,7),s,'o-',color='#138b95');axs[2].set(xlabel='Singular-value index',ylabel='Singular value',title='C  |  Six averages, seven layers');axs[2].text(.04,.12,'At least one unconstrained\nvertical combination',transform=axs[2].transAxes,fontsize=9)
    return fig,{'kernel_depth':z,'jacobian':J,'singular_values':s}
def reconstruction(data,fresh,upstream):
    style();fig,axs=plt.subplots(1,3,figsize=(13,3.7),layout='constrained');vmax=max(fresh.max(),upstream.max());extent=[-.5,29.5,6.5,-.5]
    for ax,image,label in zip(axs[:2],[upstream,fresh],['A  |  Reproduced author method','B  |  Fresh local reconstruction']):
        im=ax.imshow(image.T,aspect='auto',extent=extent,cmap='cividis',vmin=0,vmax=vmax);ax.set(xlabel='Station (m)',ylabel='Layer index, downward',title=label);fig.colorbar(im,ax=ax,label='Conductivity (mS/m)')
    error=fresh-upstream;lim=max(np.abs(error).max(),1e-12);im=axs[2].imshow(error.T,aspect='auto',extent=extent,cmap='RdBu_r',vmin=-lim,vmax=lim);fig.colorbar(im,ax=axs[2],label='Local − author (mS/m)');axs[2].set(xlabel='Station (m)',ylabel='Layer index, downward',title='C  |  Numerical port difference')
    return fig,{'fresh':fresh,'upstream':upstream,'difference':error}
def residuals(data,fresh):
    style();obs=data['apparent_conductivity'];pred=np.array([forward(v,data['layer_interfaces'],data['coil_spacing']) for v in fresh]);res=pred-obs
    fig,axs=plt.subplots(1,3,figsize=(13,3.7),layout='constrained');lim=np.abs(res).max();im=axs[0].imshow(res.T,aspect='auto',origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim);fig.colorbar(im,ax=axs[0],label='Predicted − observed (mS/m)');axs[0].set(xlabel='Station (m)',ylabel='Coil index',title='A  |  Forward residual map')
    axs[1].scatter(obs,pred,s=15,alpha=.7,color='#138b95');lo=min(obs.min(),pred.min());hi=max(obs.max(),pred.max());axs[1].plot([lo,hi],[lo,hi],color='#a65a44',ls='--');axs[1].set(xlabel='Observed (mS/m)',ylabel='Predicted (mS/m)',title='B  |  Every measurement')
    axs[2].bar(np.arange(30),np.sqrt(np.mean(res*res,axis=1)),color='#344c6e');axs[2].set(xlabel='Station (m)',ylabel='Residual RMS (mS/m)',title='C  |  Model limitation at site 8')
    return fig,{'prediction':pred,'residual':res,'station_rms':np.sqrt(np.mean(res*res,axis=1))}
def comparisons(data,fresh,gn):
    style();unbounded=quadratic_solution(data);fig,axs=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for ax,image,label in zip(axs,[gn,unbounded,fresh],['A  |  Author CSgn, different scaling','B  |  Same objective, unbounded','C  |  Selected bounded L2 method']):
        im=ax.imshow(image.T,aspect='auto',cmap='coolwarm',vmin=-35,vmax=175);fig.colorbar(im,ax=ax,label='Conductivity (mS/m)');ax.set(xlabel='Station (m)',ylabel='Layer index, downward',title=label)
    return fig,{'author_gn':gn,'same_objective_unbounded':unbounded,'selected':fresh}
def regularization(data):
    style();alphas=np.geomspace(.002,.7,9);solutions=[];rms=[];rough=[]
    for alpha in alphas:
        image,_=reconstruct(data,float(alpha));solutions.append(image);pred=np.array([forward(v,data['layer_interfaces'],data['coil_spacing']) for v in image]);rms.append(np.sqrt(np.mean((pred-data['apparent_conductivity'])**2)));rough.append(np.sqrt(np.mean(np.diff(image,axis=1)**2)))
    fig,axs=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    axs[0].loglog(alphas,rms,'o-',color='#138b95');axs[0].axvline(.07,color='#a65a44',ls=':');axs[0].set(xlabel='Regularization alpha',ylabel='Data RMS (mS/m)',title='A  |  Observation fit')
    axs[1].loglog(alphas,rough,'o-',color='#344c6e');axs[1].axvline(.07,color='#a65a44',ls=':');axs[1].set(xlabel='Regularization alpha',ylabel='Adjacent-layer RMS (mS/m)',title='B  |  Vertical roughness')
    for i in (0,4,8):axs[2].plot(solutions[i][14],np.arange(7),'-o',label=f'alpha={alphas[i]:.3g}')
    axs[2].invert_yaxis();axs[2].set(xlabel='Conductivity (mS/m)',ylabel='Layer index',title='C  |  Profile uncertainty, site 14');axs[2].legend(fontsize=8)
    return fig,{'alphas':alphas,'solutions':np.array(solutions),'data_rms':np.array(rms),'roughness_rms':np.array(rough)}
