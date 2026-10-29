import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import EFFECTIVE_PIXELS_PER_METRE,prepare,disk_mask
from .physics_model import stress_basis,forward

def agreement(x,r):
    a=x.ravel();b=r.ravel();n=np.linalg.norm(a)
    return float(np.dot(a,b)/(n*np.linalg.norm(b))) if n else 0.,float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))

def panel(ax,a,title,cmap='magma',vmin=None,vmax=None):
    extent=np.array([-.5,180.5,180.5,-.5])/EFFECTIVE_PIXELS_PER_METRE*1e3
    h=ax.imshow(a,extent=extent,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set(xlabel='Column coordinate (mm)',ylabel='Row coordinate (mm)',title=title);return h

def acquisition_figure(data,result):
    y=prepare(data['rgb'])*disk_mask();b=stress_basis(data['contact_angles'],alpha=result['contact_directions'][0]);pred=forward(result['forces'][0],b)
    fig,axs=plt.subplots(1,4,figsize=(12,3.4),layout='constrained')
    panel(axs[0],data['rgb'],'a  Measured RGB')
    h=panel(axs[1],y,'b  Optical channel',vmin=0,vmax=1);panel(axs[2],pred,'c  Fresh fitted response',vmin=0,vmax=1);fig.colorbar(h,ax=list(axs[1:3]),label='Normalized intensity (1)')
    h=panel(axs[3],pred-y,'d  Model minus measured','RdBu_r',-.5,.5);fig.colorbar(h,ax=axs[3],label='Intensity residual (1)');return fig

def stress_figure(fresh,upstream):
    fig,axs=plt.subplots(1,3,figsize=(10,3.8),layout='constrained')
    vmax=float(np.quantile(upstream[upstream>0],.99))
    h=panel(axs[0],upstream,'a  Original inferred stress',vmin=0,vmax=vmax);panel(axs[1],fresh,'b  Fresh inferred stress',vmin=0,vmax=vmax);fig.colorbar(h,ax=list(axs[:2]),label='Integrated stress difference (N/m)\ncolour clipped at source99th percentile',extend='max')
    d=fresh-upstream;v=max(float(np.max(abs(d))),1e-12);h=panel(axs[2],d,'c  Numerical difference','RdBu_r',-v,v);fig.colorbar(h,ax=axs[2],label='Difference (N/m)');return fig

def sensitivity_figure(rows,fields):
    fig,axs=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    labels=['Bottom','Right','Top','Left'];x=np.arange(4);width=.18
    for i,(name,r) in enumerate(rows.items()):axs[0].bar(x+(i-1.5)*width,r['forces'][0],width,label=name)
    axs[0].set(xticks=x,xticklabels=labels,ylabel='Force magnitude (N)',title='a  Optical-calibration sensitivity');axs[0].tick_params(axis='x',rotation=35);axs[0].legend(fontsize=7)
    for ax,(name,a) in zip(axs[1:],fields.items()):h=panel(ax,a,name,vmin=0,vmax=1)
    fig.colorbar(h,ax=list(axs[1:]),label='Predicted normalized intensity (1)');return fig
