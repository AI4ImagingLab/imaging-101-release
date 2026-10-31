"""Scientific plots from actual supplied/computed arrays; no backend selection."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COLORS=['#246a93','#d37035','#3d876b','#8f5d9f']

def save_figure(fig,name,directory,numeric):
    """Write actual PNG/SVG/PDF and their named numeric consumer source NPZ."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:fig.savefig(directory/(name+'.'+suffix),dpi=170,bbox_inches='tight')
    np.savez_compressed(directory/(name+'.npz'),**numeric)
    return fig

def _surface(ax,mesh,faces,title,view=(22,-55)):
    radius=np.linalg.norm(mesh,axis=1);facecolor=plt.get_cmap('viridis')((radius[faces].mean(axis=1)-25)/25)
    collection=Poly3DCollection(mesh[faces],facecolors=facecolor,edgecolors=(.1,.2,.25,.2),linewidths=.18)
    ax.add_collection3d(collection);ax.set(xlim=(-50,50),ylim=(-50,50),zlim=(-50,50),xlabel='X / km',ylabel='Y / km',zlabel='Z / km',title=title)
    ax.set_box_aspect([1,1,1]);ax.view_init(*view);ax.tick_params(labelsize=7)
    ax.set(xticks=[-40,0,40],yticks=[-40,0,40],zticks=[-40,0,40])

def data_figure(raw):
    d={k:v[0] for k,v in raw.items()};fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
    for j in range(2):
        image=np.maximum(d['ao_images'][j],0);psf=np.maximum(d['ao_psf'][j],0)
        im=axes[0,j].imshow(image,cmap='magma',origin='lower');fig.colorbar(im,ax=axes[0,j],label='Relative instrumental intensity',shrink=.75)
        axes[0,j].set(title=f'Observed AO exposure{j+1}: native150×150',xlabel='Pixel x',ylabel='Pixel y')
        im=axes[1,j].imshow(psf/psf.max(),cmap='magma',origin='lower');fig.colorbar(im,ax=axes[1,j],label='PSF / its peak',shrink=.75)
        axes[1,j].set(title=f'Matched observed PSF{j+1}',xlabel='Pixel x',ylabel='Pixel y')
    axes[0,2].scatter(d['lc_time'].ravel()/365.25,d['lc_brightness'].ravel(),c=d['lc_curve_index'],cmap='turbo',s=12,alpha=.8)
    axes[0,2].set(title='All30curves /923observations',xlabel='Years from supplied epoch',ylabel='Relative brightness')
    for i,chord in enumerate(d['oc_chords']):axes[1,2].plot(chord[[0,2]],chord[[1,3]],'-o',color=COLORS[0],markersize=2,lw=.8)
    axes[1,2].set(title='All18measured occultation chords',xlabel='Sky-plane x / km',ylabel='Sky-plane y / km');axes[1,2].set_aspect('equal')
    fig.suptitle('Complementary real observations: brightness, blur and geometry',fontsize=16)
    return fig

def shape_figure(initial,fresh,faces):
    fig=plt.figure(figsize=(12,8.2),layout='constrained')
    for row,mesh in enumerate([initial,fresh]):
        for col,view in enumerate([(15,-90),(15,0),(65,-65)]):
            ax=fig.add_subplot(2,3,row*3+col+1,projection='3d');_surface(ax,mesh,faces,('Independent ellipsoid seed' if row==0 else 'Fresh joint reconstruction')+f' · view{col+1}',view)
    fig.suptitle('The entire ordered three-dimensional surface is reconstructed',fontsize=16)
    return fig

def comparison_figure(fresh,original,other,subdivision,subdivision_faces,faces,boundaries,metrics):
    fig=plt.figure(figsize=(13,8.2),layout='constrained')
    reference=original['reconstruction'][0];theta=np.arccos(np.clip(reference[:,2]/np.linalg.norm(reference,axis=1),-1,1));longitude=np.arctan2(reference[:,1],reference[:,0])
    for col,(name,mesh) in enumerate([('Fresh − original O1',fresh['reconstruction'][0]),('Original O0 − original O1',other['reconstruction'][0])]):
        ax=fig.add_subplot(2,3,col+1);difference=np.linalg.norm(mesh,axis=1)-np.linalg.norm(reference,axis=1);limit=max(float(abs(difference).max()),1e-9)
        plot=ax.scatter(np.rad2deg(longitude),np.rad2deg(theta),c=difference,cmap='coolwarm',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),s=20)
        fig.colorbar(plot,ax=ax,label='Radius difference / km',shrink=.8);ax.set(title=name,xlabel='Body longitude / deg',ylabel='Body colatitude / deg')
    ax=fig.add_subplot(2,3,3,projection='3d');_surface(ax,subdivision,subdivision_faces,'Original subdivision snapshot\nDifferent author config;578vertices')
    ax=fig.add_subplot(2,3,4);x=np.arange(3);width=.27
    for i,(name,output) in enumerate([('Fresh',fresh),('Independent original O0',other)]):
        error=[np.sqrt(np.mean((output['reconstruction'][0,:,j]-reference[:,j])**2))/np.ptp(reference[:,j]) for j in range(3)]
        ax.bar(x+(i-.5)*width,error,width,label=name,color=COLORS[i])
    ax.scatter(x,[boundaries['mesh_'+a+'_range_nrmse'] for a in 'xyz'],marker='_',s=360,color='black',label='Pre-frozen limit')
    ax.set(xticks=x,xticklabels=list('XYZ'),ylabel='Coordinate-specific range NRMSE',title='Every coordinate has its own scale');ax.legend(fontsize=8)
    ax=fig.add_subplot(2,3,(5,6));keys=[key for key in metrics if key!='complete_weighted_objective'];ratio=[metrics[key]/boundaries[key] for key in keys]
    ax.barh(np.arange(len(keys)),ratio,color=COLORS[0]);ax.axvline(1,color='black',ls='--',lw=1)
    ax.set(yticks=np.arange(len(keys)),yticklabels=['Mesh X','Mesh Y','Mesh Z','Latitude','Longitude','Period','AO shifts','AO log gains','OC shift'],xlabel='Measured error / independently frozen limit',title='All surface, spin and nuisance quantities');ax.invert_yaxis()
    fig.suptitle('Method agreement and independent numerical uncertainty · reference is not truth',fontsize=15)
    return fig

def lightcurve_figure(data,details,source_details,initial_model):
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.8),layout='constrained');d=data;observed=d['lc_brightness'];fresh=details['lightcurve_model'];original=source_details['lightcurve_model'];labels=d['lc_curve_index']
    curves=np.arange(30);width=.27
    for i,(name,model) in enumerate([('Seed',initial_model),('Original O1',original),('Fresh',fresh)]):
        error=[np.sqrt(np.mean((observed[labels==curve]-model[labels==curve])**2)) for curve in curves]
        axes[0,0].bar(curves+(i-1)*width,error,width,label=name,color=COLORS[i])
    axes[0,0].set(title='Every photometric curve',xlabel='Native curve index',ylabel='Unweighted relative-brightness RMS');axes[0,0].legend(fontsize=8)
    for ax,curve in zip([axes[0,1],axes[0,2],axes[1,0],axes[1,1]],[0,9,19,29]):
        rows=np.flatnonzero(labels==curve);time=(d['lc_time'][rows]-d['lc_time'][rows].min())*24;order=np.argsort(time)
        ax.scatter(time,observed[rows],color='black',s=13,label='Measured')
        ax.plot(time[order],original[rows][order],color=COLORS[1],label='Original O1',lw=1.1);ax.plot(time[order],fresh[rows][order],color=COLORS[0],label='Fresh',lw=1.1,ls='--')
        ax.set(title=f'Curve{curve} ·{len(rows)}samples',xlabel='Hours from curve start',ylabel='Relative brightness')
    axes[0,1].legend(fontsize=8)
    axes[1,2].scatter(np.arange(923),observed-fresh,c=labels,cmap='turbo',s=9);axes[1,2].axhline(0,color='black',lw=.8)
    axes[1,2].set(title='All923 fresh residuals',xlabel='Native sample index',ylabel='Observed − fresh brightness')
    fig.suptitle('Photometry constrains rotation across the full time span',fontsize=16)
    return fig

def image_occultation_figure(raw,fresh,details,source_details,projection):
    d={k:v[0] for k,v in raw.items()};fig,axes=plt.subplots(2,4,figsize=(15,7.7),layout='constrained')
    residuals=[];models=[];observed=[]
    for j in range(2):
        image=np.maximum(d['ao_images'][j],0);image=image/image.sum();spectrum=np.empty((150,76),complex);spectrum.ravel()[0]=np.exp(fresh['ao_log_gains'][0,j]);spectrum.ravel()[1:]=details['ao_models'][j]
        yy,xx=np.indices(spectrum.shape);model=np.fft.irfft2(spectrum*((-1.)**(xx+yy)),s=(150,150));models.append(model);observed.append(image);residuals.append(image-model)
    common=max(float(image.max()) for image in observed+models);limit=max(float(abs(image).max()) for image in residuals)
    for j in range(2):
        for col,(title,image) in enumerate([('Observed',observed[j]),('Fresh model',models[j]),('Observed − model',residuals[j])]):
            norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit) if col==2 else None
            options=dict(cmap='coolwarm',norm=norm) if col==2 else dict(cmap='magma',vmin=0,vmax=common)
            artist=axes[j,col].imshow(image,origin='lower',**options);fig.colorbar(artist,ax=axes[j,col],shrink=.73,label='Fraction of measured image flux')
            axes[j,col].set(title=f'Exposure{j+1} · {title}',xlabel='Pixel x',ylabel='Pixel y')
    for i,chord in enumerate(d['oc_chords']):axes[0,3].plot(chord[[0,2]],chord[[1,3]],'-',color='black',lw=.8)
    axes[0,3].triplot(projection[:,0],projection[:,1],d['facets'],color='grey',lw=.2,alpha=.25)
    for endpoint,marker in [(details['occultation_endpoints'],'o'),(source_details['occultation_endpoints'],'x')]:
        axes[0,3].scatter(endpoint[:,[0,2]],endpoint[:,[1,3]],s=14,marker=marker,label='Fresh' if marker=='o' else 'Original O1',color=COLORS[0] if marker=='o' else COLORS[1])
    axes[0,3].set(title='All18chord lines and outer intersections',xlabel='Sky x / km',ylabel='Sky y / km');axes[0,3].set_aspect('equal');axes[0,3].legend(fontsize=8)
    observed_endpoint=d['oc_chords'];error=details['occultation_endpoints']-observed_endpoint
    for column,label in enumerate(['Entry x','Entry y','Exit x','Exit y']):axes[1,3].plot(np.arange(18),error[:,column],marker='.',ms=3,lw=.8,label=label)
    axes[1,3].axhline(0,color='black',lw=.7);axes[1,3].set(title='All72 endpoint residuals',xlabel='Chord index',ylabel='Predicted − measured / km');axes[1,3].legend(ncol=2,fontsize=7)
    fig.suptitle('Resolved-image and occultation checks retain their physical scales',fontsize=16)
    return fig,dict(display_observed=np.array(observed),display_model=np.array(models),display_residual=np.array(residuals),endpoint_error=error)

def diagnostic_figure(fresh_history,source_history,other_history,sensitivity,arithmetic_failure_ratio):
    fig,axes=plt.subplots(2,3,figsize=(13.5,8),layout='constrained')
    for name,history,color in [('Original O1',source_history,COLORS[1]),('Original O0',other_history,COLORS[2]),('Fresh',fresh_history,COLORS[0])]:
        axes[0,0].semilogy(history[:,0],history[:,2],color=color,label=name,lw=1);accepted=history[:,4].astype(bool)
        axes[0,0].scatter(history[accepted,0],history[accepted,2],color=color,s=10)
    axes[0,0].set(title='Full accepted/rejected native LM trials',xlabel='Trial index',ylabel='Complete weighted objective');axes[0,0].legend(fontsize=8)
    axes[0,1].semilogy(fresh_history[:,0],fresh_history[:,3],color=COLORS[0],marker='.',ms=4)
    axes[0,1].set(title='Actual fresh damping decisions',xlabel='Trial index',ylabel='Diagonal damping lambda')
    axes[0,2].bar(['Latitude','Longitude'],arithmetic_failure_ratio,color=COLORS[3]);axes[0,2].axhline(1,color='black',ls='--',lw=1)
    axes[0,2].set(title='Recorded recursion-path full-run failure\nNot the accepted implementation',ylabel='Measured error / pre-frozen limit')
    for ax,name,xlabel in zip(axes[1],['period','latitude','scale'],['Period perturbation / microhour','Latitude perturbation / degree','Uniform shape scale']):
        x,components,missed=sensitivity[name]
        if name=='period':x=x*1e6
        for column,label in enumerate(['Photometry','AO','Occultation','Regularizers']):ax.plot(x,components[:,column],'-o',ms=3,lw=1,label=label)
        for value,count,contribution in zip(x,missed,components[:,2]):
            if count:
                ax.scatter([value],[contribution],s=45,facecolors='none',edgecolors='#a83d38',zorder=5)
                ax.annotate(f'{count} missed',xy=(value,contribution),xytext=(0,8),textcoords='offset points',fontsize=7,ha='center')
        ax.set(title='Actual forward sensitivity · '+name,xlabel=xlabel,ylabel='Weighted objective contribution')
    axes[1,0].legend(fontsize=7)
    fig.suptitle('Numerical reproducibility and model sensitivity are not physical truth',fontsize=16)
    return fig
