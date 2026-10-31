import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import profiles,detrend
from .physics_model import heat_matrix

NAMES=['Ljut','LK1','MB1','MT2','SI1','SOM1','ZT1']

def metrics(x,reference):
    norm=np.linalg.norm(x)*np.linalg.norm(reference)
    return {'ncc':float(np.sum(x*reference)/norm) if norm else 0.,'nrmse':float(np.sqrt(np.mean((x-reference)**2))/np.ptp(reference))}

def data_plot(data):
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for name,(z,y) in zip(NAMES,profiles(data)):
        axes[0].plot(y,z,'.-',label=name)
    axes[0].invert_yaxis();axes[0].set(xlabel='Temperature (C)',ylabel='Depth (m)',title='All 237 measured records');axes[0].legend()
    loc=data['site_coordinates'];points=axes[1].scatter(loc[:,1],loc[:,0],c=data['logging_year'],s=80)
    fig.colorbar(points,ax=axes[1],label='Logging year (CE)');axes[1].margins(.12)
    for name,xy in zip(NAMES,loc):axes[1].annotate(name,xy[::-1],xytext=(4,4),textcoords='offset points')
    axes[1].set(xlabel='Longitude (degrees east)',ylabel='Latitude (degrees north)',title='All seven Slovenian archive sites')
    return fig

def results_plot(data,fresh,reference):
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for i,ax in enumerate(axes.flat):
        if i==7:ax.axis('off');continue
        years=data['logging_year'][i]-30*(np.arange(21)+.5)
        ax.plot(years,reference[i],label='Original SVD',lw=3,alpha=.55)
        ax.plot(years,fresh[i],'--',label='Fresh local SVD')
        ax.set(title=NAMES[i],xlabel='Year CE',ylabel='Surface anomaly (K)')
    axes[0,0].legend();return fig

def residual_plot(data,fresh):
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');rms=[]
    for i,(z,temp) in enumerate(profiles(data)):
        anomaly,_=detrend(z,temp);fit=heat_matrix(z)@fresh[i];res=fit-anomaly[:,1]
        color=plt.get_cmap('tab10')(i)
        axes[0].plot(anomaly[:,1],z,'.',color=color);axes[0].plot(fit,z,'-',color=color,label=NAMES[i])
        axes[1].plot(res,z,'.-',color=color,label=NAMES[i]);rms.append(np.sqrt(np.mean(res**2)))
    for ax in axes[:2]:ax.invert_yaxis();ax.set_ylabel('Depth (m)')
    axes[0].set(xlabel='Temperature anomaly (K)',title='Measured anomaly and heat prediction');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Prediction - measurement (K)',title='Full-profile residuals');axes[1].axvline(0,color='k',lw=.5)
    axes[2].bar(NAMES,rms);axes[2].set(ylabel='RMS (K)',title='Observation consistency by site');return fig
