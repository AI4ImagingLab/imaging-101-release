"""Evidence-bearing spectral and retrospective-acquisition comparisons."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import spectral_bins,PATTERN_PERIOD_US
from .solvers import reconstruct
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
EDGES_NM=np.linspace(530.,730.,9)
COUNTS=np.array([256,512,1024,2048,4096])
ORDERS=['acquisition','sequency','random','random_dc']
ORDER_LABELS=['Recorded acquisition order','Low Walsh index','Random actual pairs','Random with measured DC']

def metric_values(estimate,reference):
    """Same declared uncentred NCC/dynamic-range NRMSE, for review plots."""
    span=np.ptp(reference);norm=np.linalg.norm(estimate)*np.linalg.norm(reference)
    if span<=0 or np.linalg.norm(reference)==0:raise ValueError('Degenerate reference')
    return dict(ncc=0. if norm==0 else float(np.vdot(estimate,reference).real/norm),nrmse=float(np.sqrt(np.mean((estimate-reference)**2))/span))

def compute_experiments(data,fresh):
    """Actively reconstruct20 sampling cases using actual spectrally binned data."""
    wave=data['wavelength_nm'];binned=spectral_bins(data['measurements'],wave,EDGES_NM)
    reduced=dict(data,measurements=binned)
    cases=np.array([[reconstruct(reduced,pair_count=int(n),ordering=o)['reconstruction'][0] for n in COUNTS] for o in ORDERS])
    full_bands=spectral_bins(fresh,wave,EDGES_NM)
    errors=np.array([[metric_values(x,full_bands)['nrmse'] for x in row] for row in cases])
    mean_wave=np.array([wave[(wave>=a)&(wave<b)].mean() for a,b in zip(EDGES_NM[:-1],EDGES_NM[1:])])
    return dict(images=cases,nrmse=errors,counts=COUNTS.copy(),band_sum=full_bands,edges_nm=EDGES_NM.copy(),centre_nm=mean_wave,binned_measurements=binned)

def spectral_plate(experiments):
    """Eight native spatial fields on a shared detector-count scale."""
    images=experiments['band_sum'];limit=np.percentile(images,99.5)
    fig,axes=plt.subplots(2,4,figsize=(12,6.2),layout='constrained')
    for k,ax in enumerate(axes.flat):
        im=ax.imshow(images[...,k],origin='upper',cmap='magma',vmin=0,vmax=limit,interpolation='nearest')
        ax.set_title(f"({chr(97+k)}) {EDGES_NM[k]:.0f}–{EDGES_NM[k+1]:.0f} nm")
        ax.set_xlabel('Column [native pixel]');ax.set_ylabel('Row [native pixel]')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Band-summed intensity [detector counts/pixel]')
    fig.suptitle('Fresh reconstruction · complete real tomato field',fontsize=15)
    return fig

def parity_plate(fresh_bands,upstream_bands):
    """Original result and fresh reconstruction, with an honest zero-difference view."""
    a=upstream_bands.sum(axis=-1);b=fresh_bands.sum(axis=-1);delta=b-a
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    lo,hi=np.percentile(a,[.5,99.5])
    for ax,x,title in zip(axes[:2],[a,b],['(a) Independently rerun original','(b) Fresh local inverse']):
        im=ax.imshow(x,cmap='magma',vmin=lo,vmax=hi,interpolation='nearest');ax.set_title(title)
    fig.colorbar(im,ax=axes[:2],shrink=.8,label='530–730 nm summed counts/pixel')
    bound=float(np.max(abs(delta)));display=bound if bound else 1.
    im=axes[2].imshow(delta,cmap='RdBu_r',vmin=-display,vmax=display,interpolation='nearest');axes[2].set_title('(c) Fresh minus original')
    fig.colorbar(im,ax=axes[2],shrink=.8,label='Difference [counts/pixel]')
    if bound==0:axes[2].text(.5,.5,'Bitwise equal\nmaximum difference = 0',transform=axes[2].transAxes,ha='center',va='center',bbox=dict(facecolor='white',edgecolor='#666666'))
    for ax in axes:ax.set_xlabel('Column [native pixel]');ax.set_ylabel('Row [native pixel]')
    return fig

def sampling_plate(experiments):
    """Spatial-information tradeoffs; each image uses an actual measured subset."""
    selection=[0,2,3,4];full=experiments['band_sum'].sum(axis=-1)
    lo,hi=np.percentile(full,[.5,99.5])
    fig,axes=plt.subplots(4,4,figsize=(13,11.5),layout='constrained')
    for row in range(4):
        for col,k in enumerate(selection):
            ax=axes[row,col];n=int(COUNTS[k]);x=experiments['images'][row,k].sum(axis=-1)
            im=ax.imshow(x,cmap='magma',vmin=lo,vmax=hi,interpolation='nearest')
            ax.set_title(f'{100*n/4096:g}% · {n} pairs\nNRMSE {experiments["nrmse"][row,k]:.4f}')
            ax.set_xticks([0,32,63]);ax.set_yticks([0,32,63])
            if col==0:ax.set_ylabel(ORDER_LABELS[row]+'\nRow [pixel]')
            if row==3:ax.set_xlabel('Column [pixel]')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='530–730 nm summed counts/pixel')
    fig.suptitle('Retrospective sampling of real observations · full-data reference is not truth',fontsize=14)
    return fig

def diagnostics_plate(data,fresh,experiments):
    """Method-reference error, temporal stability, pixel spectra and signed noise."""
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for r,(label,color) in enumerate(zip(ORDER_LABELS,COLORS)):
        axes[0,0].plot(COUNTS/4096*100,experiments['nrmse'][r],marker=['o','s','^','D'][r],color=color,label=label)
    axes[0,0].set(xlabel='Retained complementary pairs [%]',ylabel='NRMSE vs full measured inverse',title='(a) Spatial information with fewer measurements');axes[0,0].legend(fontsize=9)
    pairs=experiments['binned_measurements'][::2]+experiments['binned_measurements'][1::2]
    for k in [0,3,7]:
        axes[0,1].plot(pairs[:,k]/np.median(pairs[:,k]),lw=.65,label=f'{EDGES_NM[k]:.0f}–{EDGES_NM[k+1]:.0f} nm')
    axes[0,1].set(xlabel='Acquired pair index',ylabel='Pair sum / its median [dimensionless]',title='(b) Raw complement-sum stability');axes[0,1].legend(fontsize=9)
    for (y,x),color in zip([(16,32),(32,32),(48,32)],COLORS):
        axes[1,0].plot(data['wavelength_nm'],fresh[y,x],lw=.65,color=color,label=f'row {y}, col {x}')
    axes[1,0].set(xlabel='Wavelength [nm]',ylabel='Intensity [counts/pixel]',title='(c) Native 2048-channel pixel spectra');axes[1,0].legend(fontsize=9)
    negative=np.mean(fresh<0,axis=(0,1))*100
    axes[1,1].plot(data['wavelength_nm'],negative,color='#0072B2',lw=1)
    axes[1,1].set(xlabel='Wavelength [nm]',ylabel='Negative reconstructed pixels [%]',title='(d) Noise-sensitive bands remain signed')
    for ax in axes.flat:ax.grid(alpha=.2)
    return fig
