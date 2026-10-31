"""Measured chords, independent contacts and correlated geometry ensembles."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import ellipse_contour,contact_time_residuals,negative_margin

def make_figures(data,output,stages,upstream,comparison,export_dir=None):
    figures=[];p=output['parameters'][0];curve=output['reconstruction'][0];points=data['limb_points'];negative=data['negative_path'];theta=data['contour_angles']
    names=['Outeniqua','Onduruquea','Tivoli','Windhoek C14','Windhoek D16'];colors=plt.get_cmap('tab10').colors
    def finish(fig,title):fig.suptitle(title,fontsize=14);fig.tight_layout(rect=(0,0,1,.95));figures.append(fig)
    def chords(ax):
        for i,name in enumerate(names):
            line=points[2*i:2*i+2];ax.plot(line[:,0],line[:,1],'o-',color=colors[i],label=name,ms=4)
        cut=np.linalg.norm(negative,axis=1)<240
        ax.plot(negative[cut,0],negative[cut,1],'k--',label='Hakos negative path')
        ax.set(xlabel='East f [km]',ylabel='North g [km]',aspect='equal');ax.legend(fontsize=7)
    fig,axs=plt.subplots(1,2,figsize=(12,5));chords(axs[0]);axs[0].set_title('Five instruments, four positive sites')
    ax=axs[1];t=(data['times_jd']-data['times_jd'].min())*86400
    ax.errorbar(np.arange(10),t,data['timing_sigma'],fmt='o');ax.set(xticks=np.arange(10),xticklabels=[f'{i+1}{s}' for i in range(5) for s in ['I','E']],xlabel='Instrument/contact',ylabel='Time after first contact [s]',title='Real contacts; author-example timing weights')
    finish(fig,'1. Sparse real2017occultation geometry and supplied uncertainty weights')
    fig,axs=plt.subplots(1,2,figsize=(12,5));chords(axs[0]);axs[0].plot(curve[:,0],curve[:,1],'r',lw=2,label='Fresh best ellipse');axs[0].plot(p[0],p[1],'r+')
    source=ellipse_contour(comparison['source_best'],theta);axs[1].plot(theta,(curve-source)[:,0],label='f difference');axs[1].plot(theta,(curve-source)[:,1],label='g difference');axs[1].set(xlabel='Polar angle [rad]',ylabel='Fresh minus reproduced original [km]');axs[1].legend()
    finish(fig,'2. Fresh projected boundary and independent source agreement')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));dt=contact_time_residuals(p,points,data['velocity'])
    axs[0].bar(np.arange(10),dt/data['timing_sigma'],color=[colors[i//2] for i in range(10)]);axs[0].axhline(0,color='k',lw=.7);axs[0].set(xlabel='Contact index',ylabel='Predicted time shift / timing sigma',title='Independent ray/ellipse quadratic')
    cut=np.linalg.norm(negative-p[:2],axis=1)<220;path=negative[cut];delta=path-p[:2];angle=np.arctan2(delta[:,1],delta[:,0]);limb=ellipse_contour(p,angle)-p[:2]
    clearance=np.linalg.norm(delta,axis=1)-np.linalg.norm(limb,axis=1);axs[1].plot(path[:,0],clearance);axs[1].axhline(0,color='r');axs[1].set(xlabel='Hakos f [km]',ylabel='Radial clearance [km]',title='All negative samples remain outside')
    finish(fig,'3. Every positive contact and the negative observation independently checked')
    fig,axs=plt.subplots(1,3,figsize=(14,4));trials=stages['parameters'];loss=stages['chi2'];selected=loss<loss.min()+3
    for ax,(i,j),labels in zip(axs,[(2,3),(2,4),(0,1)],[('a [km]','oblateness'),('a [km]','Pole angle [degree]'),('Center f [km]','Center g [km]')]):
        im=ax.scatter(trials[selected,i],trials[selected,j],c=loss[selected],s=8,cmap='viridis');ax.plot(p[i],p[j],'r+');ax.set(xlabel=labels[0],ylabel=labels[1]);fig.colorbar(im,ax=ax,label='Radial chi-square')
    finish(fig,'4. Actual accepted search exposes correlated shape and position')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));cut=loss<loss.min()+10;clear=stages['negative_clear']
    axs[0].hist(trials[cut,4],bins=35,histtype='step',label='Positive contacts only');axs[0].hist(trials[cut&clear,4],bins=35,histtype='step',label='Full negative path');axs[0].hist(comparison['exposure_parameters'][:,4],bins=35,histtype='step',label='Exposure-aware negative');axs[0].set(xlabel='Pole angle [degree]',ylabel='Accepted candidate count');axs[0].legend(fontsize=8)
    labels=['Positive only','Continuous negative','Exposure negative'];values=[len(trials),int(clear.sum()),len(comparison['exposure_parameters'])];axs[1].bar(labels,values);axs[1].set(ylabel='Retained candidate count')
    for i,v in enumerate(values):axs[1].text(i,v,str(v),ha='center',va='bottom')
    finish(fig,'5. Both original negative-filter comparisons reproduced with local geometry')
    fig,axs=plt.subplots(1,2,figsize=(12,5));chords(axs[0]);axs[0].plot(curve[:,0],curve[:,1],'r-',label='Standard fresh');other=ellipse_contour(comparison['seed_best'],theta);axs[0].plot(other[:,0],other[:,1],'b--',label='Independent MC seed');axs[0].legend(fontsize=7)
    labels=['f0 [km]','g0 [km]','a [km]','oblateness','PA [degree]'];difference=comparison['seed_best']-p;difference[4]=(difference[4]+90)%180-90;scale=comparison['parameter_bounds'];axs[1].bar(labels,difference/scale);axs[1].axhline(0,color='k',lw=.7);axs[1].set(ylabel='Seed change / declared parameter tolerance')
    finish(fig,'6. Independent full-search seed sensitivity; five geometric unknowns')
    if export_dir is not None:
        dest=Path(export_dir);dest.mkdir(parents=True,exist_ok=True)
        numeric={**{'input_'+k:v for k,v in data.items()},**{'fresh_'+k:v for k,v in output.items()},**{'search_'+k:np.asarray(v) for k,v in stages.items()},**{'comparison_'+k:np.asarray(v) for k,v in comparison.items()},**{'source_'+k:v for k,v in upstream.items()},'contact_time_residual_s':dt,'negative_clearance_km':clearance}
        np.savez_compressed(dest/'numeric_sources.npz',**numeric)
        for i,fig in enumerate(figures,1):
            for ext in ['png','svg','pdf']:fig.savefig(dest/f'figure_{i:02}.{ext}',dpi=150,bbox_inches='tight')
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.suffix in ('.png','.svg','.pdf','.npz')};(dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return figures
