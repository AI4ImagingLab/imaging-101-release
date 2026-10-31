"""Numerical summary and scientific panels; no backend side effect."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import correlation_forward
def metrics(estimate,reference):
    """Uncentered NCC and reference-range NRMSE; reject degenerate reference."""
    a,b=np.ravel(estimate),np.ravel(reference)
    scale=np.ptp(b);norm=np.linalg.norm(b)
    if scale<=0 or norm<=0:raise ValueError('Degenerate reference')
    ncc=float(a@b/(np.linalg.norm(a)*norm)) if np.linalg.norm(a)>0 else 0.
    return ncc,float(np.sqrt(np.mean((a-b)**2))/scale)
def diagnostic_figure(rows):
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    colors=['#167d9a','#c66737','#7971ac']
    for r,c in zip(rows,colors):
        label=f"Repeat {len(axes[0,0].lines)+1}"
        axes[0,0].semilogx(r['time'],r['correlation'],color=c,label=label)
        axes[0,1].loglog(r['time'],r['raw_msd'],'.',color=c,alpha=.4)
        axes[0,1].loglog(r['time'],r['msd'],color=c)
        axes[0,2].semilogx(r['time'],r['exponent'],color=c)
        axes[1,0].loglog(r['omega'],r['storage'],color=c)
        axes[1,0].loglog(r['omega'],r['loss'],'--',color=c)
        axes[1,1].semilogx(r['time'],correlation_forward(r['msd'],r['intercept'])-r['correlation'],color=c)
        axes[1,2].plot(np.arange(40,130,10),r['cv_scores'],'o-',color=c)
    titles=['Measured correlation','Raw dots / smoothed MSD','Local exponent','Storage solid / loss dashed','Forward correlation residual','Leave-one-out intercept CV']
    xl=['Lag (us)','Lag (us)','Lag (us)','Angular frequency (s^-1)','Lag (us)','Window upper limit (us)']
    yl=['g2 - 1','MSD (nm²)','alpha','Modulus (Pa)','Predicted - measured','MSE']
    for ax,title,x,y in zip(axes.flat,titles,xl,yl):ax.set(title=title,xlabel=x,ylabel=y);ax.grid(alpha=.2)
    axes[0,0].legend();return fig
