"""Review figures; all scientific metrics use supplied unclipped arrays."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import forward


def metrics(estimate,reference):
    """Return cosine NCC and reference-dynamic-range NRMSE."""
    a,b=np.asarray(estimate).ravel(),np.asarray(reference).ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))


def acquisition_figure(observation,background,result):
    prediction=forward(background,result['kernel'][0],result['reconstruction'][0],result['mask'][0])
    residual=observation-prediction
    fig,ax=plt.subplots(1,4,figsize=(15,4),layout='constrained')
    for a,image,title in zip(ax[:3],[observation,background,prediction],['Real blurred observation','Independent background','Fresh model prediction']):
        a.imshow(np.clip(image[...,::-1],0,1));a.set_title(title);a.axis('off')
    limit=np.max(np.abs(residual))
    image=ax[3].imshow(residual.mean(axis=2),cmap='RdBu_r',vmin=-limit,vmax=limit)
    ax[3].set_title('Mean-channel residual');ax[3].axis('off')
    fig.colorbar(image,ax=ax[3],shrink=.65,label='Normalized intensity')
    return fig


def comparison_figure(results):
    """Rows are explicitly identified measurements, columns are all unknowns."""
    fig,ax=plt.subplots(len(results),3,figsize=(10,3*len(results)),layout='constrained')
    maximum=max(float(z['kernel'].max()) for z in results.values())
    for row,(name,z) in enumerate(results.items()):
        ax[row,0].imshow(np.clip(z['reconstruction'][0,...,::-1],0,1))
        ax[row,1].imshow(z['mask'][0],vmin=0,vmax=1,cmap='gray')
        ax[row,2].imshow(z['kernel'][0],vmin=0,vmax=maximum,cmap='magma')
        for col,title in enumerate(['Premultiplied appearance (RGB display)','Opacity','Exposure kernel']):
            ax[row,col].set_title(name+'\n'+title,fontsize=10)
            ax[row,col].axis('off')
    fig.suptitle('Display range only: appearance and opacity 0 to 1; common kernel scale')
    return fig


def metric_figure(results,reference):
    keys=('reconstruction','mask','kernel')
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    width=.8/len(results);centers=np.arange(3)
    for index,(name,z) in enumerate(results.items()):
        values=[metrics(z[k],reference[k]) for k in keys]
        position=centers-.4+width*(index+.5)
        axes[0].bar(position,[v[0] for v in values],width,label=name)
        axes[1].bar(position,[max(v[1],1e-16) for v in values],width,label=name)
    axes[0].set_ylabel('Cosine NCC');axes[0].set_ylim(0,1.005)
    axes[1].set_ylabel('Dynamic-range NRMSE');axes[1].set_yscale('log')
    for a in axes:
        a.set_xticks(centers,['Appearance','Opacity','Kernel'])
        a.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    axes[1].set_title('Log display floor 1e-16; numerical values in table')
    return fig
