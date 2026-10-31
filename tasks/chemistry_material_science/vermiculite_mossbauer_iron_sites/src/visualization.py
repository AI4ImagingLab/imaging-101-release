import numpy as np
import matplotlib.pyplot as plt
from .physics_model import transmission

SAMPLES=['Louisa, Virginia','Palabora, South Africa','Enoree, South Carolina','Libby, Montana']

def review_figures(velocity,observed,fresh,upstream):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,ax in enumerate(axes.flat):
        ax.plot(velocity[i],observed[i],'.',ms=2,color='.6',label='Real measured')
        ax.plot(velocity[i],upstream['prediction'][i],lw=2,label='Original-method run')
        ax.plot(velocity[i],fresh['transmission'][0,i],'--',label='Fresh cleaned')
        ax.set(title=SAMPLES[i],xlabel='Calibrated velocity [mm/s]',ylabel='Normalized transmitted intensity');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout()
    second,axes=plt.subplots(2,2,figsize=(12,7))
    for i,ax in enumerate(axes.flat):
        ax.plot(velocity[i],1000*(observed[i]-fresh['transmission'][0,i]),lw=.8)
        ax.axhline(0,color='k',lw=.7);ax.set(title=SAMPLES[i],xlabel='Calibrated velocity [mm/s]',ylabel='Measured minus fresh [1e-3]');ax.grid(alpha=.2)
    second.tight_layout()
    third,axes=plt.subplots(1,3,figsize=(13,4))
    quantities=fresh['reconstruction'][0];colors=plt.cm.tab10(np.arange(4))
    for i,name in enumerate(SAMPLES):
        for j,title in enumerate(['Isomer shift','Full quadrupole splitting','Absorber FWHM']):
            axes[j].scatter(np.arange(1,5)+(i-1.5)*.1,quantities[i,:,j],color=colors[i],label=name)
            axes[j].set(xlabel='Canonical component index',ylabel=title+' [mm/s]');axes[j].grid(alpha=.2)
    axes[0].legend(fontsize=7);third.tight_layout()
    return [fig,second,third]
