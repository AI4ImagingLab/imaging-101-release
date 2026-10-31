import numpy as np
import matplotlib.pyplot as plt
from .physics_model import residual_jacobians,wrap

def comparison_metrics(poses,reference,data):
    """Separate physical units; no arbitrary similarity alignment."""
    from .physics_model import weighted_error
    return {'position_max_error_m':float(np.max(np.linalg.norm(poses[:,:2]-reference[:,:2],axis=1))),'heading_max_error_rad':float(np.max(abs(wrap(poses[:,2]-reference[:,2])))),'weighted_observation_error':weighted_error(poses,data)}

def trajectory_figure(initial,fresh,original,data):
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,x,title in zip(axes,[initial,fresh,original],['Fresh measurement-only odometry','Fresh local canonical GN20','Original author g2o GN20 (reference)']):
        ax.plot(x[:,0],x[:,1],lw=1);ax.scatter(*x[0,:2],color='red',s=25,label='Identity anchor')
        ax.set(xlabel='x (m)',ylabel='y (m)',title=title);ax.set_aspect('equal');ax.legend(fontsize=7)
    return fig

def residual_figure(initial,fresh,data):
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for x,label in [(initial,'Fresh odometry'),(fresh,'Fresh all-edge GN20')]:
        e=residual_jacobians(x,data)[0]
        axes[0,0].plot(np.linalg.norm(e[:,:2],axis=1),label=label,alpha=.8,lw=.8)
        axes[0,1].plot(abs(e[:,2]),label=label,alpha=.8,lw=.8)
    for ax,title,unit in zip(axes[0],['All 1483 translation residuals','All 1483 heading residuals'],['m','rad']):
        ax.set(xlabel='Acquisition edge index',ylabel=unit,title=title,yscale='symlog');ax.legend()
    e=residual_jacobians(fresh,data)[0]
    axes[1,0].plot(np.linalg.norm(e[:,:2],axis=1),lw=.8,color='tab:orange')
    axes[1,1].plot(abs(e[:,2]),lw=.8,color='tab:orange')
    for ax,title,unit in zip(axes[1],['Fresh fitted translation: full physical scale','Fresh fitted heading: full physical scale'],['m','rad']):
        ax.set(xlabel='Acquisition edge index',ylabel=unit,title=title)
    return fig

def convergence_figure(history,fresh,original):
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    axes[0].semilogy(history,'o-');axes[0].set(xlabel='GN iteration',ylabel='Full weighted error (dimensionless)',title='Undamped original schedule; first step increases cost')
    axes[1].plot(np.linalg.norm(fresh[:,:2]-original[:,:2],axis=1));axes[1].set(xlabel='Pose index',ylabel='Position difference (m)',title='Method agreement, no alignment')
    axes[2].plot(wrap(fresh[:,2]-original[:,2]));axes[2].set(xlabel='Pose index',ylabel='Circular heading difference (rad)',title='Heading agreement, fixed identity anchor')
    return fig
