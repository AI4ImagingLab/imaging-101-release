"""Computed review panels; no backend selection or source-library imports."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm
def point_map(ax,xy,values,title,units,valid=None,cmap='viridis',limits=None,symmetric=False):
    values=np.asarray(values);good=np.isfinite(values)
    if valid is not None:good&=np.asarray(valid,dtype=bool)
    ax.scatter(xy[:,0],xy[:,1],c='.85',s=3,rasterized=True)
    norm=None
    if symmetric:
        scale=float(np.max(abs(values[good]),initial=1e-12));norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale)
    elif limits is not None:norm=Normalize(*limits)
    image=ax.scatter(xy[good,0],xy[good,1],c=values[good],s=4,cmap=cmap,norm=norm,rasterized=True)
    ax.set(title=title,xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal')
    ax.figure.colorbar(image,ax=ax,label=units)
    return image
def observations_figure(data):
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,label in enumerate(['oblique','planview']):
        xy=data['mode_xy_'+label];values=data['signal_'+label];time=data['time_'+label]
        point_map(axs[row,0],xy,values[:,0],label+': first measured green channel','ADU',cmap='gray',limits=(0,255))
        ids=np.linspace(0,len(xy)-1,3,dtype=int)
        for i in ids:axs[row,1].plot(time,values[i]/255,label=f'Row{i}')
        axs[row,1].set(title=label+': observed temporal traces',xlabel='Native time (s)',ylabel='Intensity /255');axs[row,1].legend(fontsize=8)
        axs[row,2].scatter(xy[:,0],xy[:,1],s=1,label='M-grid');w=data['wave_xy_'+label];axs[row,2].scatter(w[:,0],w[:,1],s=1,label='K-grid');q=data['reconstruction_xy'];axs[row,2].scatter(q[:,0],q[:,1],s=.2,c='.7',label='B-grid');axs[row,2].set_aspect('equal');axs[row,2].set(title=label+': all calibrated sampling grids',xlabel='x (m)',ylabel='y (m)');axs[row,2].legend(markerscale=4,fontsize=8)
    return fig
def mode_figure(records):
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,record in enumerate(records):
        mode=sorted(record['modes'],key=lambda m:(m['start_seconds'],m['window_seconds'],m['period']))[0]
        point_map(axs[row,0],record['mode_xy'],mode['phases'],record['label']+': fresh mode phase','rad',cmap='twilight',limits=(-np.pi,np.pi))
        point_map(axs[row,1],record['mode_xy'],mode['amplitudes'],f"Fresh amplitude, T={mode['period']:.3f}s",'relative optical amplitude')
        modes=record['modes']
        for window,color in [(60,'tab:blue'),(90,'tab:orange')]:
            selected=[m for m in modes if m['window_seconds']==window];axs[row,2].scatter([m['start_seconds'] for m in selected],[m['period'] for m in selected],color=color,label=f'{window}s')
        axs[row,2].set(title='Retained periods by window',xlabel='Window start (s)',ylabel='Period (s)');axs[row,2].legend(fontsize=8)
    return fig
def wave_figure(fields):
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,label in enumerate(['oblique','planview']):
        f=next(v for v in fields if v['record']==label);j=1
        point_map(axs[row,0],f['wave_xy'],f['ks'][:,j],f"{label}: fresh k, radius={f['radii'][j]:.1f}m",'rad/m')
        gamma=f['omega']**2/(9.80665*f['ks'][:,j])
        point_map(axs[row,1],f['wave_xy'],gamma,'Fresh dimensionless dispersion parameter','gamma',limits=(0,1.2))
        point_map(axs[row,2],f['wave_xy'],f['std_gamma'][:,j],'Local gamma variation (not posterior SD)','dimensionless')
    return fig
def bed_figure(xy,submission,survey,native,required):
    bed=submission['reconstruction'][0];valid=submission['validity'][0];nv=native['validity'];nb=native['bed']
    fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained');limits=(float(survey.min()),float(survey.max()))
    point_map(axs[0,0],xy,survey,'Independent previous-day survey','Bed elevation (m)',limits=limits)
    point_map(axs[0,1],xy,nb,'Original sequential same-config run','Bed elevation (m)',valid=nv,limits=limits)
    point_map(axs[0,2],xy,bed,'Fresh local full-path reconstruction','Bed elevation (m)',valid=valid,limits=limits)
    point_map(axs[1,0],xy,bed-survey,'Fresh minus independent survey','Signed error (m)',valid=valid,cmap='RdBu_r',symmetric=True)
    point_map(axs[1,1],xy,valid.astype(float)+required.astype(float),'Fresh and fixed-interior coverage','0neither,1one,2both',limits=(0,2))
    axs[1,2].scatter(survey[valid],bed[valid],s=3,alpha=.25);axs[1,2].plot(limits,limits,'k--');axs[1,2].set(title=f'Every submitted valid site (n={valid.sum()})',xlabel='Survey elevation (m)',ylabel='Fresh elevation (m)');axs[1,2].set_aspect('equal')
    return fig
