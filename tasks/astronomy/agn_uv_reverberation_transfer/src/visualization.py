"""Figures from actual arrays; conditional uncertainty and local modes labeled."""
import numpy as np
import matplotlib.pyplot as plt
from .solvers import likelihood,predict
from .preprocessing import prepare
NAMES=['UV continuum','Lyman alpha','C IV'];UNITS=['Continuum flux [native unit]','Line flux [native unit]','Line flux [native unit]']
def observations(obs):
 fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
 for i,a in enumerate(ax):a.errorbar(obs[i,:,0],obs[i,:,1],yerr=obs[i,:,2],fmt='.',capsize=2,label=NAMES[i]);a.set_ylabel(UNITS[i]);a.legend()
 ax[-1].set_xlabel('Observer epoch [JD - 2440000, day]');fig.suptitle('All180 actual IUE measurements; no synthetic observations');return fig

def reconstruction(obs,out,info,original):
 fig,ax=plt.subplots(3,2,figsize=(15,9),layout='constrained');t=info['time']
 for i in range(3):
  a,b=ax[i];m=out['reconstruction'][0,i];sd=info['conditional_sd'][i]
  a.fill_between(t,m-sd,m+sd,alpha=.22,label='Conditional 1 SD');a.plot(t,original['reconstruction'][i],ls='--',label='Original MAP');a.plot(t,m,lw=1,label='Fresh local');a.errorbar(obs[i,:,0],obs[i,:,1],yerr=obs[i,:,2],fmt='k.',alpha=.5,label='Measured')
  a.axvspan(t[0],obs[i,0,0],color='grey',alpha=.15);a.axvspan(obs[i,-1,0],t[-1],color='grey',alpha=.15);a.set_title(NAMES[i]+'; grey = extrapolation');a.set_ylabel(UNITS[i]);a.set_xlabel('Epoch [JD - 2440000]')
  res=(np.interp(obs[i,:,0],t,m)-obs[i,:,1])/obs[i,:,2];b.axhline(0,color='grey');b.plot(obs[i,:,0],res,'.-');b.set_title('Observed residual; WRMS %.3f'%np.sqrt(np.mean(res**2)));b.set_ylabel('(fit - measured) / error');b.set_xlabel('Epoch [JD - 2440000]')
 ax[0,0].legend(fontsize=8,ncol=2);return fig

def transfer_modes(info):
 fig,ax=plt.subplots(1,3,figsize=(16,4.4),layout='constrained');colors=['C0','C1','C2']
 for n,p in enumerate(info['all_parameters']):
  for j in range(2):
   lag,w,a=p[2+j*3:5+j*3];x=[lag-w/2,lag-w/2,lag+w/2,lag+w/2];ax[j].plot(x,[0,a/w,a/w,0],color=colors[n],label='Start%d'%n)
 ax[0].set_title('Lyman alpha transfer');ax[1].set_title('C IV transfer')
 for a in ax[:2]:a.set_xlabel('Response delay [day]');a.set_ylabel('Response scale per day');a.legend()
 ax[2].bar(np.arange(3),info['all_logp'].max()-info['all_logp']);ax[2].set_xticks(range(3),['Start0','Start1','Start2']);ax[2].set_ylabel('Best minus mode log objective');ax[2].set_title('Converged local modes, not posterior weights');return fig

def lag_slice(obs,info):
 d=prepare(obs);p=info['parameters'];lags=np.linspace(0,25,31);z=np.empty((31,31))
 for i,x in enumerate(lags):
  for j,y in enumerate(lags):
   q=p.copy();q[2]=x;q[5]=y;z[j,i]=likelihood(q,d)
 fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');im=ax[0].imshow(z.max()-z,origin='lower',extent=[0,25,0,25],aspect='auto',vmax=50,cmap='magma_r');fig.colorbar(im,ax=ax[0],label='Slice max minus log objective');ax[0].plot(p[2],p[5],'rx',ms=10);ax[0].set(xlabel='Lyman alpha lag [day]',ylabel='C IV lag [day]',title='Other six parameters fixed: not profile/posterior')
 for n,q in enumerate(info['all_parameters']):ax[1].scatter(np.exp(q[1]),np.exp(q[0]),s=80,label='Start%d'%n)
 ax[1].set(xlabel='Continuum damping time [day]',ylabel='Variability amplitude [continuum unit]',title='Mode coupling with stochastic continuum');ax[1].legend();return fig,z

def mode_profiles(obs,info):
 d=prepare(obs);fig,ax=plt.subplots(3,1,figsize=(12,8),layout='constrained',sharex=True)
 for k,p in enumerate(info['all_parameters']):
  m,sd=predict(p,d,info['time'])
  for i,a in enumerate(ax):a.plot(info['time'],m[i],label='Converged start%d'%k)
 for i,a in enumerate(ax):a.plot(obs[i,:,0],obs[i,:,1],'k.',alpha=.5);a.set_ylabel(NAMES[i]);a.legend(ncol=3,fontsize=8)
 ax[-1].set_xlabel('Epoch [JD - 2440000, day]');fig.suptitle('Different transfer modes can reconstruct similar measured histories');return fig

def parity_uncertainty(out,info,original):
 fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 for i in range(3):
  ax[0].plot(info['time'],out['reconstruction'][0,i]-original['reconstruction'][i],label=NAMES[i]);ax[1].plot(info['time'],info['conditional_sd'][i],label=NAMES[i])
 ax[0].set(title='Fresh minus original: numerical agreement',xlabel='Epoch [JD - 2440000]',ylabel='Flux difference [channel native unit]');ax[1].set(title='Conditional SD; excludes parameter uncertainty',xlabel='Epoch [JD - 2440000]',ylabel='Flux SD [channel native unit]')
 for a in ax:a.legend();return_value=None
 return fig
