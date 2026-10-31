"""Review plots; leave backend selection to caller."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import SPECIES,RATE_NAMES,concentrations,charge

def plot_observations(c,q,initial,result,upstream,raw):
    time=np.linspace(0,64,150)
    fig,axes=plt.subplots(3,2,figsize=(13,12),layout='constrained')
    for i,ax in enumerate(axes.flat[:5]):
        for run in raw['concentration_repeats']:
            ax.scatter(run[:,0],run[:,i+1],s=12,alpha=.3,color='grey')
        ax.plot(time,upstream['concentrations'][:,i],ls='--',label='Corrected original execution',color='black')
        ax.plot(time,result['reconstruction'][0,:,i],label='Fresh local joint fit')
        ax.set(xlabel='Time [h]',ylabel='Concentration [micromol/L]',title=SPECIES[i]);ax.legend(fontsize=8)
    ax=axes.flat[5]
    for run in raw['charge_repeats']:ax.plot(run[:,0],run[:,1],alpha=.25,color='grey')
    ax.plot(time,upstream['charge'],'k--',label='Corrected original execution')
    ax.plot(time,result['charge'][0],label='Fresh local joint fit')
    ax.set(xlabel='Time [h]',ylabel='Charge [C]',title='Joint electrical observation');ax.legend(fontsize=8)
    return fig

def plot_hidden_and_residuals(c,q,initial,result):
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');t=np.linspace(0,64,150)
    for j in range(5):
        axes[0,0].plot(t,result['reconstruction'][0,:,5+j],label=SPECIES[5+j])
        axes[0,1].plot(t,result['reconstruction'][0,:,10+j],label=SPECIES[10+j])
    for ax in axes[0]:ax.set(xlabel='Time [h]',ylabel='Inferred concentration [micromol/L]');ax.legend(fontsize=8)
    axes[0,0].set_title('Unobserved first degradation pools (method inference)')
    axes[0,1].set_title('Unobserved further degradation pools (method inference)')
    rates=result['rates'][0]
    cm=concentrations(c[:,0],initial,rates)
    for j in range(5):axes[1,0].plot(c[:,0],c[:,j+1]-cm[:,j],'.-',label=SPECIES[j])
    axes[1,0].set(xlabel='Time [h]',ylabel='Observed minus fitted [micromol/L]',title='Chemical residuals');axes[1,0].legend(fontsize=8)
    axes[1,1].plot(q[:,0],q[:,1]-charge(concentrations(q[:,0],initial,rates)))
    axes[1,1].set(xlabel='Time [h]',ylabel='Observed minus fitted [C]',title='Electrical residuals')
    return fig

def plot_ablation(result,chemical_only):
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained');x=np.arange(11);t=np.linspace(0,64,150)
    axes[0].bar(x-.18,result['rates'][0],.36,label='Fresh joint fit')
    axes[0].bar(x+.18,chemical_only['rates'][0],.36,label='Fresh concentration only')
    axes[0].set(xticks=x,xticklabels=RATE_NAMES,ylabel='Rate [1/h]',title='Effect of electrical constraint');axes[0].tick_params(axis='x',rotation=45);axes[0].legend(fontsize=8)
    axes[0].set_yscale('symlog',linthresh=1e-5)
    axes[0].annotate('kDx is unobservable from chemical assays alone',xy=(10,chemical_only['rates'][0,10]),xytext=(2,.3),arrowprops={'arrowstyle':'->'},fontsize=8)
    axes[1].plot(t,result['charge'][0],label='Fresh joint fit')
    axes[1].plot(t,chemical_only['charge'][0],label='Concentration-only prediction')
    axes[1].set(xlabel='Time [h]',ylabel='Predicted charge [C]',title='Electrical consequence of ablation');axes[1].legend(fontsize=8)
    return fig
