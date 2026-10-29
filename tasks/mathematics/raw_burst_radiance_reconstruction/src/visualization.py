from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from .solvers import reconstruct,patchesRMS,NOISE
def relative_l2(x,y):return float(np.linalg.norm(x.astype(float)-y)/np.linalg.norm(y))
def preview(x):
    x=x.astype(float)
    return (x[0::2,0::2]+x[1::2,0::2]+x[0::2,1::2]+x[1::2,1::2])/4-64
def alternate_dc_weights(tiles,index):
    maps=[]
    for di,dj in [(0,0),(1,0),(0,1),(1,1)]:
        ref=tiles[0,...,di::2,dj::2];alt=tiles[index,...,di::2,dj::2]
        # DC DFT equals the exact tile sum; no Fourier amplitude fit is used.
        difference=ref.astype(float).sum(axis=(-2,-1))-alt.astype(float).sum(axis=(-2,-1))
        variance=NOISE[0]*patchesRMS(ref)+NOISE[1];scaled=16**2/16*2*75*variance
        maps.append(scaled/(difference*difference+scaled))
    return np.mean(maps,axis=0)
def run_experiments(data,references):
    torch.set_num_threads(1);solutions={};nominal,stages=reconstruct(data);solutions['nominal']=nominal['reconstruction'][0]
    for label,kwargs in [('temporal8',{'temporal':8}),('reference_only',{'temporal':0}),('no_spatial',{'spatial':0}),('no_alignment',{'align':False})]:solutions[label]=reconstruct(data,**kwargs)[0]['reconstruction'][0]
    counts=[]
    for n in [1,2,5,9,10]:
        image=solutions['nominal'] if n==10 else reconstruct(data,frame_count=n)[0]['reconstruction'][0]
        if n==9:solutions['first9']=image
        counts.append([n,relative_l2(image,solutions['nominal']),np.std(preview(image)-preview(solutions['nominal']))])
    upstream=references['corrected'];legacy=references['unmodified'];nom=solutions['nominal'];ref=data['frames'][0]
    dc1=alternate_dc_weights(stages['aligned_tiles'],1);dc9=alternate_dc_weights(stages['aligned_tiles'],9)
    return {'solutions':solutions,'motion':stages['motion'],'counts':np.asarray(counts),'dc_frame1':dc1,'dc_frame9':dc9,'legacy':legacy,'corrected':upstream,'reference':ref,'metrics':{'corrected_upstream_relative_l2':relative_l2(nom,upstream),'unmodified_source75_relative_change':relative_l2(nom,legacy),'single_reference_relative_error':relative_l2(ref,upstream),'no_alignment_relative_change':relative_l2(solutions['no_alignment'],nom),'spatial_filter_relative_change':relative_l2(solutions['no_spatial'],nom),'last_frame_relative_contribution':relative_l2(nom,solutions['first9']),'last_frame_mean_brightness_change_ADU':float(np.mean(nom.astype(float)-solutions['first9'])),'frame1_mean_alternate_DC_weight':float(dc1.mean()),'frame9_mean_alternate_DC_weight':float(dc9.mean())}}
def make_figures(data,e):
    figures=[];limits=(0,float(np.percentile(preview(e['reference']),99.7)))
    fig,ax=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    for a,t in zip(ax.flat,[0,1,5,9]):
        im=a.imshow(preview(data['frames'][t]),cmap='gray',vmin=limits[0],vmax=limits[1]);a.set(title=f'Real measured RAW frame {t}',xlabel='Half-resolution column',ylabel='Half-resolution row');fig.colorbar(im,ax=a,label='2×2 mean signal above black (ADU)')
    figures.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    panels=[('Noisy reference',e['reference']),('Fresh10-frame reconstruction',e['solutions']['nominal']),('Unmodified source75',e['legacy']),('No alignment',e['solutions']['no_alignment']),('Temporal strength8',e['solutions']['temporal8']),('No spatial Wiener filter',e['solutions']['no_spatial'])]
    for a,(title,x) in zip(ax.flat,panels):
        image=preview(x);im=a.imshow(image[150:350,150:350],cmap='gray',vmin=limits[0],vmax=limits[1]);a.set(title=title+'\nSame native display region',xlabel='Half-resolution column offset150',ylabel='Half-resolution row offset150');fig.colorbar(im,ax=a,label='Mean signal above black (ADU)')
    figures.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,9),constrained_layout=True);motion=e['motion'];yy,xx=np.indices(motion.shape[1:3]);s=4
    ax[0,0].quiver(xx[::s,::s]*16,yy[::s,::s]*16,motion[0,::s,::s,1],motion[0,::s,::s,0]);ax[0,0].set(title='Equal-exposure frame1 pullback motion',xlabel='Padded sensor column',ylabel='Padded sensor row');ax[0,0].invert_yaxis()
    delta=e['solutions']['nominal'].astype(float)-e['legacy'];lim=max(1,np.percentile(abs(delta),99.5));im=ax[0,1].imshow(delta,cmap='RdBu_r',vmin=-lim,vmax=lim);ax[0,1].set(title='Corrected minus unmodified source75',xlabel='Sensor column',ylabel='Sensor row');fig.colorbar(im,ax=ax[0,1],label='ADU')
    for k in ['reference_only','no_spatial','no_alignment']:
        d=preview(e['solutions'][k])-preview(e['solutions']['nominal']);ax[1,0].hist(d.ravel(),bins=100,density=True,histtype='step',label=k)
    ax[1,0].set(xlim=(-15,15),title='Computed differences to nominal',xlabel='2×2 mean difference (ADU)',ylabel='Probability density');ax[1,0].legend()
    for k in ['reference_only','nominal','temporal8']:ax[1,1].plot(preview(e['solutions'][k])[256],label=k,lw=.8)
    ax[1,1].set(title='Fresh image central-row profile',xlabel='Half-resolution column',ylabel='Signal above black (ADU)');ax[1,1].legend();figures.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,9),constrained_layout=True);counts=e['counts']
    ax[0,0].plot(counts[:,0],counts[:,1],'o-');ax[0,0].set(title='Actually reconstructed frame-count sweep',xlabel='Included measured frames',ylabel='Relative change to10-frame result')
    ax[0,1].plot(counts[:,0],counts[:,2],'o-');ax[0,1].set(title='Observed spatial difference variation',xlabel='Included measured frames',ylabel='Difference std (ADU); no truth assumed')
    for i,key in enumerate(['reference_only','nominal']):
        x=preview(e['solutions'][key]);power=abs(np.fft.fftshift(np.fft.fft2(x-x.mean())))**2;im=ax[1,i].imshow(np.log10(1+power),cmap='magma');ax[1,i].set(title=key+' spatial log power',xlabel='Fourier column bin',ylabel='Fourier row bin');fig.colorbar(im,ax=ax[1,i],label='log10(1+ADU² spectral power)')
    figures.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for a,key,title in [(ax[0,0],'dc_frame1','Equal-exposure frame1'),(ax[0,1],'dc_frame9','Different-exposure frame9')]:
        im=a.imshow(e[key],vmin=0,vmax=1,cmap='viridis');a.set(title=title+' alternate DC weight',xlabel='Overlapped tile column',ylabel='Overlapped tile row');fig.colorbar(im,ax=a,label='Alternate coefficient weight before1/10 average')
    delta=e['solutions']['nominal'].astype(float)-e['solutions']['first9'];lim=max(1,float(np.percentile(abs(delta),99.5)))
    im=ax[1,0].imshow(delta,vmin=-lim,vmax=lim,cmap='RdBu_r');ax[1,0].set(title='All10 minus first9 reconstructed frames',xlabel='Native sensor column',ylabel='Native sensor row');fig.colorbar(im,ax=ax[1,0],label='ADU')
    means=[float(np.mean(delta[di::2,dj::2])) for di,dj in [(0,0),(0,1),(1,0),(1,1)]];ax[1,1].bar(['B','G1','G2','R'],means);ax[1,1].axhline(0,color='black',lw=.5);ax[1,1].set(title='Measured last-frame brightness contribution',ylabel='Mean10-minus9 change (ADU)',xlabel='Bayer site; no radiometric truth assumed')
    figures.append(fig);return figures
def export(figures,e,directory):
    p=Path(directory);p.mkdir(exist_ok=True,parents=True)
    for i,fig in enumerate(figures,1):
        for ext in ['png','svg','pdf']:fig.savefig(p/f'figure_{i}.{ext}',dpi=140)
    np.savez_compressed(p/'numeric_sources.npz',**e['solutions'],motion=e['motion'],counts=e['counts'],dc_frame1=e['dc_frame1'],dc_frame9=e['dc_frame9'],unmodified_source75=e['legacy'])
    (p/'metrics.json').write_text(json.dumps(e['metrics'],indent=2),encoding='utf8')
