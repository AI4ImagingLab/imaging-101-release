from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct
from .physics_model import forward,PIXEL_SIZE,WAVELENGTH,NA
def error(x,y):return float(np.linalg.norm(x-y)/np.linalg.norm(y))
def run_experiments(data,upstream):
    nominal,stage=reconstruct(data);tv,tvs=reconstruct(data,method='TV');rows=[]
    for factor in [.1,.3,1,3,10]:
        x,_=reconstruct(data,reg_u=.1*factor,reg_p=.005*factor)
        prediction=forward(x['absorption'][0],x['reconstruction'][0],stage['Hu'],stage['Hp'])
        rows.append([factor,error(prediction,stage['normalized']),np.std(x['reconstruction'])])
    metrics={k+'_upstream_relative_l2':error(nominal[k],upstream[k]) for k in nominal}
    metrics['tikhonov_observation_residual']=error(forward(nominal['absorption'][0],nominal['reconstruction'][0],stage['Hu'],stage['Hp']),stage['normalized'])
    return {'nominal':nominal,'tv':tv,**stage,'tv_history':tvs['history'],'sweep':np.asarray(rows),'metrics':metrics}
def make_figures(data,e):
    figs=[];extent=[0,512*PIXEL_SIZE,512*PIXEL_SIZE,0]
    fig,ax=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for i,a in enumerate(ax.ravel()):
        im=a.imshow(data['intensities'][i],cmap='gray',extent=extent);a.set(title=f'Real measured DPC exposure {i+1}',xlabel='x (µm)',ylabel='y (µm)');fig.colorbar(im,ax=a,label='ADU')
    figs.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for i,a in enumerate(ax.ravel()):
        im=a.imshow(np.fft.fftshift(e['source'][i]),cmap='gray');a.set(title=f'Illumination source {i+1}, rotation '+str([0,180,90,270][i])+'°',xlabel='Spatial-frequency column bin',ylabel='Spatial-frequency row bin');fig.colorbar(im,ax=a,label='Relative source weight')
    figs.append(fig)
    fig,ax=plt.subplots(2,4,figsize=(14,6),constrained_layout=True)
    for i in range(4):
        for row,key in [(0,'Hu'),(1,'Hp')]:
            values=e[key][i].real if row==0 else e[key][i].imag
            im=ax[row,i].imshow(np.fft.fftshift(values),cmap='RdBu_r');ax[row,i].set(title=('Log-amplitude' if row==0 else 'Phase')+f' transfer {i+1}',xlabel='Fourier column bin',ylabel='Fourier row bin');fig.colorbar(im,ax=ax[row,i],label='Dimensionless response')
    figs.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for row,method in enumerate(['nominal','tv']):
        for col,key in enumerate(['absorption','reconstruction']):
            im=ax[row,col].imshow(e[method][key][0],cmap='gray',extent=extent,vmin=-.15 if col==0 else -1,vmax=.02 if col==0 else 3)
            ax[row,col].set(title=('Fresh Tikhonov' if row==0 else 'Fresh third-order TV')+(' log-amplitude' if col==0 else ' phase delay'),xlabel='x (µm)',ylabel='y (µm)');fig.colorbar(im,ax=ax[row,col],label='ln amplitude (dimensionless)' if col==0 else 'Phase delay (rad)')
    figs.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    ax[0,0].plot(np.arange(1,len(e['tv_history'])+1),e['tv_history'],'o-');ax[0,0].set(title='Actual ten TV inverse updates',xlabel='Iteration',ylabel='Normalized-intensity residual norm')
    ax[0,1].semilogx(e['sweep'][:,0],e['sweep'][:,1],'o-');ax[0,1].set(title='Fresh quadratic regularization sweep',xlabel='Multiplier on both regularizers',ylabel='Relative observation residual')
    for method in ['nominal','tv']:ax[1,0].plot(np.arange(512)*PIXEL_SIZE,e[method]['reconstruction'][0,256],label=method)
    ax[1,0].set(title='Computed central-row phase comparison',xlabel='x (µm)',ylabel='Phase delay (rad)');ax[1,0].legend()
    residual=forward(e['nominal']['absorption'][0],e['nominal']['reconstruction'][0],e['Hu'],e['Hp'])-e['normalized'];im=ax[1,1].imshow(residual[0],cmap='RdBu_r',extent=extent);ax[1,1].set(title='Fresh forward residual, exposure 1',xlabel='x (µm)',ylabel='y (µm)');fig.colorbar(im,ax=ax[1,1],label='Normalized intensity');figs.append(fig)
    return figs
def export(figures,e,directory):
    p=Path(directory);p.mkdir(parents=True,exist_ok=True)
    for i,f in enumerate(figures,1):
        for ext in ['png','svg','pdf']:f.savefig(p/f'figure_{i}.{ext}',dpi=140)
    np.savez_compressed(p/'numeric_sources.npz',sweep=e['sweep'],tv_history=e['tv_history'],tikhonov_phase=e['nominal']['reconstruction'],tikhonov_absorption=e['nominal']['absorption'],tv_phase=e['tv']['reconstruction'],tv_absorption=e['tv']['absorption'])
    (p/'metrics.json').write_text(json.dumps(e['metrics'],indent=2),encoding='utf8')
