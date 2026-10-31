"""Scientific profile plotting; callers own backend and IO."""
import numpy as np
import matplotlib.pyplot as plt
def relative_error(estimate,reference):
    """Maximum componentwise relative error; zero reference rejected."""
    if np.any(np.asarray(reference)==0):raise ValueError('Zero reference')
    return float(np.max(np.abs(np.asarray(estimate)/reference-1)))
def profile_figure(rows):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,r in enumerate(rows):
        lo,hi=r['fit_indices'];d=r['distance'];colors=['#147d92','#d47c32']
        for side,c in enumerate(colors):
            for mode in range(2):
                axes[i,mode].plot(d,r['profiles'][side,:,mode],'.',ms=2,color=c,alpha=.45,label=['Increasing row','Decreasing row'][side])
                axes[i,mode].plot(d[lo:hi],np.polyval(r['coefficients'][side,mode],d[lo:hi]),color=c,lw=2)
        for mode in range(2):axes[i,mode].axvspan(d[lo],d[hi],color='gray',alpha=.08);axes[i,mode].set(xlabel='Distance from excitation (mm)',ylabel=['log(amplitude / mK)','Phase (rad)'][mode],title=f'Acquisition {i+1}');axes[i,mode].legend();axes[i,mode].grid(alpha=.2)
    return fig
