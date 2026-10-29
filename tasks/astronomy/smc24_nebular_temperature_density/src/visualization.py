"""Active seven-diagnostic comparison from local statistical equilibrium."""
import numpy as np
import matplotlib.pyplot as plt
from .solvers import DIAGNOSTICS,line_ratio

def diagnostic_figure(ions,ratios,conditions):
    temperature=np.linspace(7500,16000,75);density=np.geomspace(500,25000,85)
    fig,ax=plt.subplots(figsize=(11,7),layout='constrained')
    colors=plt.cm.tab10(np.arange(7))
    handles=[]
    for (name,_),color in zip(DIAGNOSTICS.items(),colors):
        values=line_ratio(ions,name,temperature,density)
        level=ratios[name]
        # Assume independent5percent errors in component fluxes; ratios with
        # summed lines have conservative sqrt(2)*5percent bands for context.
        err=np.sqrt(2)*.05
        ax.contourf(temperature,density,values.T,levels=[level*(1-err),level*(1+err)],colors=[color],alpha=.13)
        ax.contour(temperature,density,values.T,levels=[level],colors=[color],linewidths=1.6)
        handles.append(plt.Line2D([],[],color=color,label=name))
    for i,(t,n) in enumerate(conditions):ax.plot(t,n,'k*',ms=10);ax.annotate(str(i+1),(t,n),xytext=(5,5),textcoords='offset points')
    ax.set(xlabel='Electron temperature (K)',ylabel='Electron density (cm$^{-3}$)',yscale='log',title='Seven observed diagnostics; four fresh crossed solutions')
    ax.legend(handles=handles,loc='upper right',title='Ratios; shaded assumed errors')
    return fig
