"""Maintainer figures and metrics; does not select a plotting backend."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import matrix_quaternion,ideal_expectation

def process_metrics(estimate,reference):
    """Same global-phase convention as the independent Judge."""
    aligned=estimate*np.exp(-1j*np.angle(np.vdot(reference,estimate)))
    return {'phase_aligned_process_rmse':float(np.sqrt(np.mean(abs(aligned-reference)**2))), 'unitarity_error':float(np.linalg.norm(estimate[0].conj().T@estimate[0]-np.eye(2)))}

def observations(data):
    e=data['expectation'];fig,axes=plt.subplots(2,4,figsize=(13,5),sharex=True,sharey=True)
    for rank in range(2):
        for j,name in enumerate(['I','X','Y','Z']):
            ax=axes[rank,j];ax.plot(e[rank,j].real,'.-',label='real');ax.plot(e[rank,j].imag,'.-',label='imaginary');ax.set_title(f'Rank {rank}, controlled {name}');ax.set_xlabel('Measured direction index')
    axes[0,0].legend();axes[0,0].set_ylabel('Expectation (dimensionless)');axes[1,0].set_ylabel('Expectation (dimensionless)');fig.tight_layout();return fig

def process_comparison(matrices,names):
    fig,axes=plt.subplots(2,len(names),figsize=(4*len(names),6),squeeze=False)
    for j,(u,name) in enumerate(zip(matrices,names)):
        for i,(part,label) in enumerate([(u[0].real,'Real'),(u[0].imag,'Imaginary')]):
            im=axes[i,j].imshow(part,vmin=-1,vmax=1,cmap='RdBu_r');axes[i,j].set_title(f'{name}: {label}');axes[i,j].set_xticks([0,1]);axes[i,j].set_yticks([0,1]);fig.colorbar(im,ax=axes[i,j],shrink=.7)
            for a in range(2):
                for b in range(2):axes[i,j].text(b,a,f'{part[a,b]:.4f}',ha='center',va='center')
    fig.tight_layout();return fig

def residuals(data,u):
    q=matrix_quaternion(u[0]);model=ideal_expectation(q,data['polar_angle'],data['azimuth']);e=data['expectation'];fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].scatter(model.real.ravel(),e.real.ravel(),s=13,alpha=.6);axes[0,0].plot([-.6,.6],[-.6,.6],'k--');axes[0,0].set(xlabel='Ideal predicted real expectation',ylabel='Measured real expectation',title='Device attenuation remains visible')
    im=axes[0,1].imshow(abs(e-model).reshape(8,-1),aspect='auto');fig.colorbar(im,ax=axes[0,1],label='Absolute ideal-model residual');axes[0,1].set(xlabel='Direction index',ylabel='Rank/channel')
    # Diagnostic gain is fitted to observations, never supplied as public calibration.
    gain=np.vdot(model,e)/np.vdot(model,model);res=e-gain*model
    axes[1,0].hist(res.real.ravel(),bins=25,alpha=.6,label='real');axes[1,0].hist(res.imag.ravel(),bins=25,alpha=.6,label='imaginary');axes[1,0].legend();axes[1,0].set(xlabel='Residual after diagnostic global gain',ylabel='Count',title=f'Observed nuisance gain: {gain.real:.3f}{gain.imag:+.3f}i')
    axes[1,1].plot(np.sqrt(np.mean(abs(res)**2,axis=(0,1))),'.-');axes[1,1].set(xlabel='Direction index',ylabel='Residual RMS (dimensionless)',title='All 50 directions retained')
    fig.tight_layout();return fig,{'ideal_residual_rms':float(np.sqrt(np.mean(abs(e-model)**2))),'gain_adjusted_diagnostic_rms':float(np.sqrt(np.mean(abs(res)**2)))}

def droplet_surface(u,title):
    from .physics_model import basis
    theta,phi=np.meshgrid(np.linspace(0,np.pi,60),np.linspace(0,2*np.pi,120));q=matrix_quaternion(u[0]);b=basis(theta.ravel(),phi.ravel());f=(np.r_[1j*q[:3],q[3]]@b).reshape(theta.shape)
    radius=abs(f);fig=plt.figure(figsize=(6,5));ax=fig.add_subplot(projection='3d');colors=plt.cm.twilight((np.angle(f)+np.pi)/(2*np.pi))
    ax.plot_surface(radius*np.sin(theta)*np.cos(phi),radius*np.sin(theta)*np.sin(phi),radius*np.cos(theta),facecolors=colors,linewidth=0,shade=False);ax.set(xlabel='x',ylabel='y',zlabel='z',title=title);ax.set_box_aspect((1,1,1))
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-np.pi,np.pi),cmap='twilight'),ax=ax,shrink=.55,pad=.12,label='Complex phase (radian)')
    fig.tight_layout();return fig
