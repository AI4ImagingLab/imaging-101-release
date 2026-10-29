"""Fresh reconstruction comparisons and explicitly labeled diagnostic fits."""
import numpy as np
import matplotlib.pyplot as plt
from .solvers import MODELS,solve
LABELS=('metal','chloride','hydroxide','thiocyanate','sulfide','thiosulfate')
def comparisons(energy,observation,standards,result,upstream):
    f1,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for j,s in enumerate(standards):ax[0].plot(energy,s,label=LABELS[j])
    ax[0].plot(energy,observation,'k.',ms=2,label='measured sample');ax[0].legend(fontsize=8)
    ax[0].set(xlabel='Photon energy (eV)',ylabel='Normalized absorption',title='Six independently measured standards')
    im=ax[1].pcolormesh(energy,np.arange(6),standards,shading='nearest');ax[1].invert_yaxis();ax[1].set_yticks(range(6),LABELS)
    ax[1].set(xlabel='Photon energy (eV)',title='Library spectral features');f1.colorbar(im,ax=ax[1],label='Normalized absorption')
    f2,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained');best=result['best']
    ax[0,0].plot(energy,observation,'k.',ms=3,label='measured')
    for i in [0,3,best]:ax[0,0].plot(energy,result['fits'][i],label=f'fresh model {i+1}')
    ax[0,0].plot(energy,upstream['fits'][best],'--',lw=1,label='native selected reference')
    ax[0,0].legend(fontsize=8);ax[0,0].set(title='Computed fits: 2, 3 and 4 standards',ylabel='Normalized absorption')
    for i in [0,3,best]:ax[1,0].plot(energy,observation-result['fits'][i],label=f'model {i+1}')
    ax[1,0].axhline(0,color='k',lw=.5);ax[1,0].set(xlabel='Photon energy (eV)',ylabel='Observed minus fitted',title='Measured residual structure')
    for j,s in enumerate(standards):
        if result['weights'][best,j]>0:ax[0,1].plot(energy,result['weights'][best,j]*s,label=LABELS[j])
    ax[0,1].legend(fontsize=8);ax[0,1].set(title='Fresh selected weighted components',ylabel='Normalized absorption')
    ax[1,1].plot(energy,result['fits'][best]-upstream['fits'][best]);ax[1,1].set(xlabel='Photon energy (eV)',ylabel='Fresh minus native',title='Numerical parity; native is not truth')
    f3,ax=plt.subplots(2,1,figsize=(12,7),layout='constrained');x=np.arange(11)
    ax[0].bar(x-.17,upstream['statistics'][:,1],.34,label='native reproduced')
    ax[0].bar(x+.17,result['statistics'][:,1],.34,label='fresh local');ax[0].legend()
    ax[0].set(ylabel='Reduced chi-square (assumed scale)',title='All eleven original comparisons retained')
    ax[0].set_xticks(x,x+1);ax[0].set_xlabel('Model number')
    bottom=np.zeros(11)
    for j,name in enumerate(LABELS):ax[1].bar(x,result['weights'][:,j],bottom=bottom,label=name);bottom+=result['weights'][:,j]
    ax[1].set_xticks(x,[','.join(str(k+1) for k in m) for m in MODELS],rotation=30)
    ax[1].set(xlabel='Included standard indices',ylabel='Fitted spectral weight');ax[1].legend(ncol=3,fontsize=8)
    return f1,f2,f3
def ambiguity(energy,observation,standards,result):
    f1,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    corr=np.corrcoef(standards);im=ax[0].imshow(corr,vmin=.8,vmax=1,cmap='viridis')
    ax[0].set_xticks(range(6),LABELS,rotation=50,ha='right');ax[0].set_yticks(range(6),LABELS)
    ax[0].set_title('Pearson correlation of measured library');f1.colorbar(im,ax=ax[0],label='Correlation')
    for ix in [0,3,7]:
        m=MODELS[ix];sv=np.linalg.svd((standards[list(m[:-1])]-standards[m[-1]]).T,compute_uv=False)
        ax[1].semilogy(np.arange(1,len(sv)+1),sv,'o-',label=f'model {ix+1}, cond {sv[0]/sv[-1]:.1f}')
    ax[1].legend();ax[1].set(xlabel='Singular-value index',ylabel='Singular value',title='Affine mixture identifiability')
    shifts=[-.2,0,.2];weights=[];rms=[];which=[]
    for shift in shifts:
        moved=np.array([np.interp(energy-shift,energy,s) for s in standards])
        r=solve(observation,moved);i=r['best'];weights.append(r['weights'][i]);rms.append(np.sqrt(np.mean((observation-r['fits'][i])**2)));which.append(i+1)
    f2,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained');weights=np.array(weights)
    for j,name in enumerate(LABELS):ax[0].plot(shifts,weights[:,j],'o-',label=name)
    ax[0].legend(fontsize=8);ax[0].set(xlabel='Supplementary library shift (eV)',ylabel='Re-fitted spectral weight',title='Computed calibration sensitivity')
    ax[1].plot(shifts,rms,'o-');ax[1].margins(x=.16,y=.2)
    for x,y,m in zip(shifts,rms,which):ax[1].annotate(f'model {m}',(x,y),xytext=((-4 if x>=0 else 4),7),ha=('right' if x>=0 else 'left'),textcoords='offset points')
    ax[1].set(xlabel='Supplementary library shift (eV)',ylabel='Measured-fit RMS',title='Shifted standards are diagnostics, not data')
    return f1,f2
