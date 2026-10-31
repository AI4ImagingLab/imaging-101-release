import matplotlib.pyplot as plt
import numpy as np

def field_figure(parameters,upstream,predicted,observed,frequency):
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,label in enumerate(['Effective absorption (1/mm)','Effective reduced scattering (1/mm)']):
        lo=min(parameters[:,:,i].min(),upstream[:,:,i].min());hi=max(parameters[:,:,i].max(),upstream[:,:,i].max());im=axes[i,0].imshow(parameters[:,:,i],vmin=lo,vmax=hi);axes[i,0].set_title('Fresh local\n'+label);fig.colorbar(im,ax=axes[i,0]);im=axes[i,1].imshow(upstream[:,:,i],vmin=lo,vmax=hi);axes[i,1].set_title('Independent source\n'+label);fig.colorbar(im,ax=axes[i,1])
    residual=predicted-observed;im=axes[0,2].imshow(np.sqrt(np.mean(residual**2,axis=2)));axes[0,2].set_title('Every-cell reflectance residual RMS');fig.colorbar(im,ax=axes[0,2]);axes[1,2].bar(np.arange(5),np.sqrt(np.mean(residual**2,axis=(0,1))));axes[1,2].set(xticks=np.arange(5),xticklabels=[f'{f:g}' for f in frequency],xlabel='Spatial frequency (1/mm)',ylabel='Reflectance residual RMS',title='All five acquisition conditions')
    for ax in axes[:,:2].flat:ax.set(xlabel='Column block',ylabel='Row block')
    return fig

def residual_validity_figure(parameters,predicted,observed,frequency):
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained');res=predicted-observed;bound=max(np.max(abs(res)),1e-12)
    for i in range(5):
        ax=axes.flat[i];im=ax.imshow(res[:,:,i],cmap='coolwarm',vmin=-bound,vmax=bound);ax.set_title(f'All-pixel residual at {frequency[i]:g}/mm');fig.colorbar(im,ax=ax)
    mua=parameters[:,:,0];mus=parameters[:,:,1];transport=mua+mus;indicator=np.sqrt(3*mua*transport+(2*np.pi*frequency[-1])**2)/transport;im=axes[1,2].imshow(indicator);axes[1,2].set_title('High-frequency effective attenuation / transport');fig.colorbar(im,ax=axes[1,2]);return fig

def index_sensitivity_figure(base,low,high):
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for i,label in enumerate(['Absorption','Reduced scattering']):
        im=axes[i].imshow(100*(high[:,:,i]-low[:,:,i])/base[:,:,i],cmap='coolwarm');axes[i].set_title(label+': n1.45 minus n1.35 / n1.4 (%)');fig.colorbar(im,ax=axes[i])
    fig.suptitle('Fresh hypothetical model-index perturbation; no measured calibration error claimed');return fig
