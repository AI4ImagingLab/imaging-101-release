from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct
def relative_error(x,y):return float(np.linalg.norm(x-y)/np.linalg.norm(y))
def run_experiments(data,upstream):
    result,stage=reconstruct(data);alternatives=[];rows=[]
    for fraction in [.25,.375,.5,.625,.75]:
        out,_=reconstruct(data,size_fraction=fraction);x=out['reconstruction'][0];alternatives.append(x);rows.append([fraction,relative_error(x,result['reconstruction'][0]),np.std(abs(x)),np.std(np.angle(x))])
    return {**stage,'nominal':result['reconstruction'][0],'alternatives':np.asarray(alternatives),'sweep':np.asarray(rows),'metrics':{'upstream_relative_l2':relative_error(result['reconstruction'],upstream['reconstruction']),'minimum_empty_field_amplitude':float(np.min(abs(stage['background_field']))),'phase_range_rad':[float(stage['unwrapped'].min()),float(stage['unwrapped'].max())]}}
def make_figures(data,e):
    figures=[];fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for a,key,title in zip(ax[0],['hologram','empty_hologram'],['Real HL60 object hologram','Independent empty-field hologram']):
        im=a.imshow(data[key],cmap='gray');a.set(title=title,xlabel='Detector column (pixel)',ylabel='Detector row (pixel)');fig.colorbar(im,ax=a,label='ADU')
    for a,key,title in zip(ax[1],['fft_origin','fft_filtered'],['Measured centered spectrum','Actually isolated object sideband']):
        im=a.imshow(np.log1p(abs(e[key])),cmap='viridis');a.set(title=title,xlabel='Fourier column bin',ylabel='Fourier row bin');fig.colorbar(im,ax=a,label='log(1+ spectral magnitude)')
    figures.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    values=[abs(e['nominal']),np.angle(e['nominal']),e['phase_difference'],e['unwrapped']]
    for a,x,title,unit in zip(ax.ravel(),values,['Fresh full relative amplitude','Fresh principal relative phase','Original-example phase difference convention','Fresh continuous phase for comparison'],['Relative amplitude','rad','rad','rad']):
        im=a.imshow(x,cmap='viridis' if unit=='Relative amplitude' else 'coolwarm');a.set(title=title,xlabel='Column (pixel)',ylabel='Row (pixel)');fig.colorbar(im,ax=a,label=unit)
    figures.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for a,x,title in zip(ax[0],[e['alternatives'][0],e['alternatives'][-1]],['Fresh narrow aperture','Fresh wide aperture']):
        im=a.imshow(np.angle(x),cmap='coolwarm');a.set(title=title,xlabel='Column (pixel)',ylabel='Row (pixel)');fig.colorbar(im,ax=a,label='Principal phase (rad)')
    for i,label in [(0,'Narrow'),(2,'Nominal'),(4,'Wide')]:ax[1,0].plot(np.angle(e['alternatives'][i,100]),label=label)
    ax[1,0].set(title='Actually reconstructed central-row phase',xlabel='Column (pixel)',ylabel='Principal phase (rad)');ax[1,0].legend()
    ax[1,1].plot(e['sweep'][:,0],e['sweep'][:,1],'o-');ax[1,1].set(title='Measured finite-aperture sensitivity',xlabel='Fraction of sideband distance',ylabel='Complex change relative to nominal');figures.append(fig)
    return figures
def export(figures,e,directory):
    p=Path(directory);p.mkdir(parents=True,exist_ok=True)
    for i,f in enumerate(figures,1):
        for ext in ['png','svg','pdf']:f.savefig(p/f'figure_{i}.{ext}',dpi=140)
    np.savez_compressed(p/'numeric_sources.npz',sweep=e['sweep'],amplitude=abs(e['nominal']),phase=np.angle(e['nominal']),unwrapped=e['unwrapped'],carrier=e['carrier'])
    (p/'metrics.json').write_text(json.dumps(e['metrics'],indent=2),encoding='utf8')
