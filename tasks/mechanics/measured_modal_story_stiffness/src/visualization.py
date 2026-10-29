"""Fresh parameter comparisons and diagnostics."""
import numpy as np
import matplotlib.pyplot as plt
def relative_error(a,b):return float(np.max(abs(np.asarray(a)/b-1)))
def stiffness_figure(fresh,original):
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');x=np.arange(4)
    ax.bar(x-.18,original.ravel(),.36,label='Reproduced original');ax.bar(x+.18,fresh.ravel(),.36,label='Fresh local');ax.axhline(10,color='gray',ls='--',label='Nominal model');ax.set(xticks=x,xticklabels=['1','2','3','4'],xlabel='Story',ylabel='Stiffness (lbf/in)');ax.legend();return fig
