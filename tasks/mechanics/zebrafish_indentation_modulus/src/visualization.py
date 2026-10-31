"""Review figures and metrics. Importing this module never changes a backend."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import preprocess
from .physics_model import contact_force

def agreement(estimate,reference):
    """Cosine NCC and reference-dynamic-range NRMSE, without alignment."""
    x=np.asarray(estimate).ravel();y=np.asarray(reference).ravel()
    norm=np.linalg.norm(x)*np.linalg.norm(y)
    if np.ptp(y)==0 or np.linalg.norm(y)==0:raise ValueError('Degenerate reference')
    return {'ncc':float(x@y/norm) if norm else 0.,'nrmse':float(np.sqrt(np.mean((x-y)**2))/np.ptp(y))}

def curve(data,i):
    a,b=data['offsets'][i:i+2]
    x,y,o=preprocess(data['force'][a:b],data['height'][a:b],data['spring_constant'][i])
    return x,y,o,data['segment'][a:b]==0

def show_fits(data,result,indices,title):
    """Draw actual observed and fresh fitted approach traces, plus residuals."""
    fig,axes=plt.subplots(2,len(indices),figsize=(3.1*len(indices),5),squeeze=False)
    for j,i in enumerate(indices):
        x,y,o,m=curve(data,int(i));pred=contact_force(x[m],result['reconstruction'][0,i],result['contact_position'][0,i]-o)
        axes[0,j].plot(x[m]*1e6,y[m]*1e9,lw=.65,label='measured')
        axes[0,j].plot(x[m]*1e6,pred*1e9,lw=1.3,label='fresh fit')
        axes[0,j].set_title(f'Curve {i}: {result["reconstruction"][0,i]:.1f} Pa')
        axes[1,j].plot(x[m]*1e6,(y[m]-pred)*1e9,lw=.65)
        axes[1,j].axhline(0,color='black',lw=.5)
        axes[1,j].set_xlabel('Relative tip coordinate (µm)')
    axes[0,0].set_ylabel('Force (nN)');axes[1,0].set_ylabel('Unweighted residual (nN)');axes[0,0].legend()
    fig.suptitle(title);fig.tight_layout();return fig
