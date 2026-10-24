"""Computed microwave-map comparisons and source-data return values."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import mixing_matrix,forward
from .solvers import reconstruct,separate
NAMES=('CMB','Synchrotron (22.8 GHz)','Dust (93.5 GHz)')
def style():
    plt.rcParams.update({'figure.dpi':125,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','axes.labelcolor':'#24354b','text.color':'#24354b','figure.facecolor':'white','axes.facecolor':'#f8fafc','savefig.bbox':'tight'})
def compare(a,b):
    return float(np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))
def extent(data):return [data['galactic_longitude'].min(),data['galactic_longitude'].max(),data['galactic_latitude'].min(),data['galactic_latitude'].max()]
def image(ax,data,array,title,cmap='RdBu_r',limit=None):
    limit=limit or max(abs(array.min()),abs(array.max()),1e-12)
    im=ax.imshow(array,origin='lower',extent=extent(data),aspect='auto',cmap=cmap,vmin=-limit,vmax=limit)
    ax.set(xlabel='Galactic longitude (deg)',ylabel='Galactic latitude (deg)',title=title);return im
def acquisition(data):
    style();fig,axs=plt.subplots(1,5,figsize=(16,3.1),layout='constrained');maps=data['measurements']
    for i,ax in enumerate(axs):
        im=image(ax,data,maps[i],f'{data["frequencies"][i]:g} GHz');fig.colorbar(im,ax=ax,label='Thermodynamic microK',shrink=.75)
    fig.suptitle('Real WMAP nine-year observations | Common 1 degree angular resolution',fontweight='bold')
    return fig,{'measurements':maps}
def spectra(data):
    style();fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained');nu=np.geomspace(20,110,301);curve=mixing_matrix(nu);A=mixing_matrix(data['frequencies'])
    for k,color in enumerate(['#344c6e','#138b95','#b56a40']):axs[0].loglog(nu,curve[:,k],label=NAMES[k],color=color);axs[0].scatter(data['frequencies'],A[:,k],s=18,color=color)
    axs[0].set(xlabel='Frequency (GHz)',ylabel='Amplitude relative to component reference',title='A  |  Fixed spectral laws');axs[0].legend(fontsize=8)
    from matplotlib.ticker import FixedLocator,FixedFormatter,NullFormatter
    axs[0].xaxis.set_major_locator(FixedLocator(data['frequencies']))
    axs[0].xaxis.set_major_formatter(FixedFormatter([f'{x:g}' for x in data['frequencies']]))
    axs[0].xaxis.set_minor_formatter(NullFormatter())
    im=axs[1].imshow(A,aspect='auto',cmap='viridis');fig.colorbar(im,ax=axs[1],label='Thermodynamic response');axs[1].set(xticks=range(3),xticklabels=['CMB','Synch','Dust'],yticks=range(5),yticklabels=data['frequencies'],xlabel='Component',ylabel='Frequency (GHz)',title='B  |  Acquisition mixing matrix')
    singular=np.linalg.svd(A,compute_uv=False);axs[2].semilogy(np.arange(1,4),singular,'o-',color='#138b95');axs[2].set(xlabel='Mode index',ylabel='Singular value',title=f'C  |  Condition number {singular[0]/singular[-1]:.2f}')
    return fig,{'frequency_grid':nu,'spectra':curve,'mixing':A,'singular_values':singular}
def components(data,maps,author):
    style();fig,axs=plt.subplots(3,3,figsize=(12,9),layout='constrained')
    for k in range(3):
        lim=max(np.abs(maps[k]).max(),np.abs(author[k]).max())
        for col,(value,label) in enumerate([(author[k],'Reproduced author'),(maps[k],'Fresh local'),(maps[k]-author[k],'Numerical difference')]):
            im=image(axs[k,col],data,value,f'{label}\n{NAMES[k]}',limit=lim if col<2 else None);fig.colorbar(im,ax=axs[k,col],label='Thermodynamic microK',shrink=.8)
    return fig,{'fresh':maps,'reproduced_author':author,'difference':maps-author}
def residuals(data,pred):
    style();res=pred-data['measurements'];fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained');lim=np.abs(res).max()
    for i,ax in enumerate(axs.flat[:5]):
        im=image(ax,data,res[i],f'{data["frequencies"][i]:g} GHz residual',limit=lim);fig.colorbar(im,ax=ax,label='Predicted − observed (microK)',shrink=.8)
    rms=np.sqrt(np.mean(res*res,axis=(1,2)));axs[1,2].bar(data['frequencies'],rms,width=5,color='#344c6e');axs[1,2].set(xlabel='Frequency (GHz)',ylabel='Residual RMS (microK)',title='All 3072 sky directions included')
    return fig,{'prediction':pred,'residual':res,'band_rms':rms}
def ablation(data,maps):
    style();A=mixing_matrix(data['frequencies']);two,_,_=separate(data['measurements'],A[:,:2]);pred2=forward(two,A[:,:2]);pred3=forward(maps,A)
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,arr,title in zip(axs[:2],[two[0],two[0]-maps[0]],['A  |  CMB from two-component inverse','B  |  CMB bias from omitting dust']):im=image(ax,data,arr,title);fig.colorbar(im,ax=ax,label='Thermodynamic microK')
    rms=[np.sqrt(np.mean((pred2-data['measurements'])**2)),np.sqrt(np.mean((pred3-data['measurements'])**2))];axs[2].bar(['CMB+synch','CMB+synch+dust'],rms,color=['#b56a40','#138b95']);axs[2].set(ylabel='Five-band residual RMS (microK)',title='C  |  Computed model comparison');axs[2].tick_params(axis='x',labelrotation=12)
    return fig,{'two_component_maps':two,'omitted_dust_cmb_difference':two[0]-maps[0],'residual_rms':np.array(rms)}
def sensitivity(data,maps):
    style();betas=np.linspace(-3.25,-2.75,9);sweep=[];rms=[]
    for beta in betas:
        m,p,_=reconstruct(data,beta_sync=float(beta));sweep.append(m);rms.append(np.sqrt(np.mean((p-data['measurements'])**2)))
    loo=[];A=mixing_matrix(data['frequencies'])
    for i in range(5):keep=np.arange(5)!=i;loo.append(separate(data['measurements'][keep],A[keep])[0])
    loo=np.array(loo);fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained');axs[0].plot(betas,rms,'o-',color='#138b95');axs[0].axvline(-3.,ls=':',color='#b56a40');axs[0].set(xlabel='Assumed synchrotron index',ylabel='Residual RMS (microK)',title='A  |  Spectral-model sensitivity')
    change=np.sqrt(np.mean((np.array(sweep)[:,0]-maps[0])**2,axis=(1,2)));axs[1].plot(betas,change,'o-',color='#344c6e');axs[1].set(xlabel='Assumed synchrotron index',ylabel='CMB map RMS change (microK)',title='B  |  Foreground leakage')
    leave=np.sqrt(np.mean((loo[:,0]-maps[0])**2,axis=(1,2)));axs[2].bar(np.arange(5),leave,color='#b56a40');axs[2].set(xticks=np.arange(5),xticklabels=[f'{x:g}' for x in data['frequencies']],xlabel='Omitted band (GHz)',ylabel='CMB map RMS change (microK)',title='C  |  Leave-one-band-out stability')
    return fig,{'sync_indices':betas,'sweep_maps':np.array(sweep),'residual_rms':np.array(rms),'cmb_change':change,'leave_one_out_maps':loo,'leave_one_out_cmb_change':leave}
