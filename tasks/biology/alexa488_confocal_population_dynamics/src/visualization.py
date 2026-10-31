import numpy as np
import matplotlib.pyplot as plt
from .physics_model import correlation_model

def comparison_figure(t,g,parameters,model,upstream,without_triplet):
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    ax=axs[0,0];ax.semilogx(t,g,'.',label='Author observation');ax.semilogx(t,model,label='Fresh local fit');ax.semilogx(t,upstream,'--',label='Fresh upstream run');ax.semilogx(t,without_triplet,':',label='Fresh no-triplet comparison');ax.legend(fontsize=8);ax.set(xlabel='Lag (ms)',ylabel='Excess correlation',title='Alexa488 confocal correlation')
    axs[0,1].semilogx(t,g-model,'.',label='Full model');axs[0,1].semilogx(t,g-without_triplet,'.',alpha=.6,label='No triplet');axs[0,1].axhline(0,color='k',lw=.6);axs[0,1].legend();axs[0,1].set(xlabel='Lag (ms)',ylabel='Observation - prediction',title='Residuals on every acquired lag')
    q=parameters.copy();q[1]=0
    axs[1,0].semilogx(t,model-correlation_model(q,t));axs[1,0].set(xlabel='Lag (ms)',ylabel='Triplet excess correlation',title='Fitted short-lag contribution')
    eps=np.linspace(-.2,.2,61)
    for j,label in enumerate(['Effective molecule number','Triplet fraction','Residence time']):
        loss=[]
        for e in eps:
            p=parameters.copy();p[j]*=1+e;loss.append(np.sum((correlation_model(p,t)-g)**2))
        axs[1,1].plot(100*eps,loss,label=label)
    axs[1,1].set(xlabel='Parameter perturbation (%)',ylabel='Unweighted SSE',title='One-at-a-time physical sensitivity');axs[1,1].legend(fontsize=8)
    return fig

