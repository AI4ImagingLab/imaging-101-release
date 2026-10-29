"""Measured spin contrast, depth profiles and conditional sensitivity plots."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import predictions,depth_profiles
from .solvers import LOWER,UPPER

def spin_asymmetry(channels):
    return (channels[0]-channels[1])/(channels[0]+channels[1])

def asymmetry_uncertainty(channels,errors):
    total=channels.sum(axis=0)
    return 2*np.sqrt((channels[1]*errors[0])**2+(channels[0]*errors[1])**2)/total**2

def weighted_residuals(data,fit):return (fit['prediction']-data['reflectivity'])/data['uncertainty']

def observation_figure(data,dead,uniform,original):
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);q=data['q'][0]
    for j,label in enumerate(['++','--']):
        color=f'C{j}';axes[0,0].errorbar(q,data['reflectivity'][j],data['uncertainty'][j],fmt='.',ms=3,color=color,alpha=.65,label='Measured '+label)
        axes[0,0].semilogy(q,dead['prediction'][j],color=color,label='Fresh dead-layer '+label)
        axes[0,0].semilogy(q,original['dead_prediction'][0,j],'x',color=color,ms=3,alpha=.5)
    axes[0,0].set(title='All 168 measured intensities; original crosses',xlabel='Nominal Q [Angstrom^-1]',ylabel='Reflectivity');axes[0,0].legend(fontsize=7)
    sa=spin_asymmetry(data['reflectivity']);se=asymmetry_uncertainty(data['reflectivity'],data['uncertainty'])
    axes[0,1].errorbar(q,sa,se,fmt='.',color='black',ms=3,label='Measured spin asymmetry')
    axes[0,1].plot(q,spin_asymmetry(uniform['prediction']),'--',label='Fresh uniform magnetization')
    axes[0,1].plot(q,spin_asymmetry(dead['prediction']),label='Fresh suppressed interface')
    axes[0,1].set(title='Magnetic contrast and phase of oscillations',xlabel='Nominal Q [Angstrom^-1]',ylabel='(R++ - R--) / (R++ + R--)');axes[0,1].legend(fontsize=7)
    for j,axis in enumerate(axes[1]):
        axis.plot(q,weighted_residuals(data,uniform)[j],'o--',ms=3,label='Uniform')
        axis.plot(q,weighted_residuals(data,dead)[j],'.-',ms=3,label='Suppressed interface')
        axis.axhline(0,color='grey',ls=':');axis.set(title=['++ residuals','-- residuals'][j],xlabel='Nominal Q [Angstrom^-1]',ylabel='(Prediction - measurement) / sigma');axis.legend(fontsize=8)
    return fig

def profile_figure(data,dead,uniform,original):
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);z=data['depth']
    for j,axis in enumerate(axes[0]):
        axis.plot(z,uniform['profiles'][:,j],'--',label='Fresh uniform')
        axis.plot(z,dead['profiles'][:,j],label='Fresh suppressed interface')
        axis.plot(z[::15],original['dead_profiles'][0,::15,j],'x',ms=4,label='Independent original')
        axis.axvline(0,color='grey',ls=':');axis.axvline(dead['parameters'][-1],color='grey',ls=':')
        axis.set(title=['Nuclear composition profile','Magnetization-proportional SLD profile'][j],xlabel='Depth from substrate/film boundary [Angstrom]',ylabel='SLD [10^-6 Angstrom^-2]');axis.legend(fontsize=8)
    names=['I','broadening','offset','rough bottom','rough top','dead depth','mag SLD','nuclear SLD','thickness']
    diff=(dead['parameters']-original['dead_parameters'][0])/(UPPER-LOWER)
    axes[1,0].barh(names,diff,color='C2');axes[1,0].set(title='Fresh versus independently fitted original parameters',xlabel='Difference / declared physical bound range')
    axes[1,1].bar(['Uniform','Suppressed interface'],[uniform['reduced_chi_square'],dead['reduced_chi_square']],color=['C0','C1'])
    for i,r in enumerate([uniform,dead]):axes[1,1].text(i,r['reduced_chi_square']+.015,f"{r['reduced_chi_square']:.4f}",ha='center')
    axes[1,1].axhline(1,color='grey',ls=':',label='Ideal independent noise scale');axes[1,1].set(title='Both models actually fitted, including broadening',ylabel='Chi-square / degrees of freedom',ylim=(0,1.95));axes[1,1].legend(fontsize=7)
    return fig

def sensitivity(data,fit):
    p=fit['parameters'];steps=(UPPER-LOWER)*1e-5;jac=[]
    for i,h in enumerate(steps):
        plus=p.copy();minus=p.copy();plus[i]+=h;minus[i]-=h
        jac.append(((predictions(plus,data)-predictions(minus,data))/(2*h)/data['uncertainty']).ravel())
    jac=np.array(jac).T;cov=np.linalg.pinv(jac.T@jac);std=np.sqrt(np.diag(cov));correlation=cov/np.outer(std,std)
    depths=np.linspace(0,20,81);broadening=np.linspace(.018,.04,81);costs=[]
    for index,grid in [(5,depths),(1,broadening)]:
        row=[]
        for value in grid:
            trial=p.copy();trial[index]=value;r=(predictions(trial,data)-data['reflectivity'])/data['uncertainty'];row.append(np.sum(r*r))
        costs.append(np.array(row))
    return dict(jacobian=jac,conditional_covariance=cov,standard_deviation=std,correlation=correlation,depths=depths,broadening=broadening,costs=costs)

def sensitivity_figure(data,fit,diagnostic):
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);base=2*fit['objective']
    for axis,key,values,index in [(axes[0,0],'dead-layer thickness',diagnostic['depths'],5),(axes[0,1],'additional broadening',diagnostic['broadening'],1)]:
        cost=diagnostic['costs'][0 if index==5 else 1];axis.plot(values,cost-base);axis.axvline(fit['parameters'][index],color='black',ls=':')
        axis.set(title='Conditional '+key+' scan\nOther parameters fixed',xlabel=key+(' [Angstrom]' if index==5 else ' [degree]'),ylabel='Chi-square minus fitted chi-square')
    indices=[1,3,4,5,6,7,8];labels=['broad','rough B','rough T','dead','mag','rho','h'];corr=diagnostic['correlation'][np.ix_(indices,indices)]
    im=axes[1,0].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');fig.colorbar(im,ax=axes[1,0],label='Local linearized correlation')
    axes[1,0].set(xticks=range(7),xticklabels=labels,yticks=range(7),yticklabels=labels,title='Conditional curvature; diagonal-noise assumption')
    hist=fit['history'];axes[1,1].semilogy(hist[:,0],np.maximum(hist[:,1]-fit['objective'],1e-14),'.-');axes[1,1].set(title='Actual explicit Nelder-Mead optimization history',xlabel='Iteration',ylabel='Objective above final (floor 1e-14)')
    return fig
