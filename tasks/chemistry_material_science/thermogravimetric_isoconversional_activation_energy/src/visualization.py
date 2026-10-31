"""Notebook figures from measured thermograms; no backend mutation."""
import matplotlib.pyplot as plt
def thermograms(traces,beta):
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for b,x in zip(beta,traces):
        axs[0].plot(x[:,1],x[:,2],label=f'{b:.3f} K/min');axs[1].plot(x[:,0],x[:,1])
    axs[0].set(xlabel='Temperature (C)',ylabel='Mass (mg)');axs[0].legend()
    axs[1].set(xlabel='Time (min)',ylabel='Temperature (C)')
    return fig
