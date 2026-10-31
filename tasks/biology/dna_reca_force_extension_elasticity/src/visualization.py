"""Measured polymer mechanics, with the rejected approximation clearly labeled."""
import numpy as np
import matplotlib.pyplot as plt
from .solvers import submission
def rmse(prediction,observation):
    """RMS error in force units, without normalization or alignment."""
    return float(np.sqrt(np.mean((prediction-observation)**2)))
def local_covariance(jacobian,cost,nobs):
    """Unweighted local linearized covariance, residual variance estimated."""
    return np.linalg.inv(jacobian.T@jacobian)*(2*cost/(nobs-jacobian.shape[1]))
def make_figures(data,fit,original,rejected):
    fresh=submission(fit,data);fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    fig2,res=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,ax,rx in zip(['control','reca'],axes,res):
        mask=data['mask_'+name];x=data['distance_'+name][mask];y=data['force_'+name][mask];q=fresh['predicted_force_'+name][0]
        ax.scatter(data['distance_'+name],data['force_'+name],s=4,color='.8',label='All native observations')
        ax.scatter(x,y,s=5,color='black',label='Fixed force≤30pN fit set')
        ax.plot(x,q,lw=2,label='Fresh Marko–Siggia');ax.plot(x,original['predicted_force_'+name][0],'--',label='Original Marko–Siggia')
        ax.plot(x,rejected['output_force_'+name],':',label='Rejected Odijk approximation')
        ax.set(title='DNA '+('without RecA' if name=='control' else 'with RecA'),xlabel='Extension (µm)',ylabel='Force (pN)');ax.legend(fontsize=7)
        rx.scatter(y,q-y,s=6,label='Fresh Marko–Siggia');rx.scatter(y,rejected['output_force_'+name]-y,s=5,alpha=.4,label='Rejected Odijk')
        rx.axhline(0,color='black',lw=.7);rx.axvspan(0,10,color='orange',alpha=.1,label='Outside Odijk stated 10–30pN range')
        rx.set(title=name,xlabel='Observed force (pN)',ylabel='Prediction residual (pN)');rx.legend(fontsize=7)
    fig3,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for j,name in enumerate(['control','reca']):
        y=data['force_'+name][data['mask_'+name]]
        errors=[rmse(original['predicted_force_'+name][0],y),rmse(fresh['predicted_force_'+name][0],y),rmse(rejected['output_force_'+name],y)]
        axes[0].bar(np.arange(3)+j*4,errors,label=name)
    axes[0].set(xticks=[1,5],xticklabels=['Control: original/fresh/rejected','RecA: original/fresh/rejected'],ylabel='Observation RMSE (pN)',title='Same observations and fixed fit support');axes[0].tick_params(axis='x',labelsize=7)
    cov=local_covariance(fit.jac,fit.cost,len(fit.fun));sd=np.sqrt(np.diag(cov));p=fit.x
    axes[1].errorbar([0,1],p[[1,3]],yerr=1.96*sd[[1,3]],fmt='o',capsize=5)
    axes[1].set(xticks=[0,1],xticklabels=['Control','RecA'],ylabel='Contour length (µm)',title='Local covariance interval, model conditional')
    return [fig,fig2,fig3]
