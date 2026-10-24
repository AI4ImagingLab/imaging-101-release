"""Scientific plots from actual RF/reconstruction arrays; no backend mutation."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,Normalize
from matplotlib.patches import Circle
from .physics_model import SAMPLING_FREQUENCY,START_TIME,phase_consistency,predicted_radial

COLORS=['#237eb3','#d66e38','#8c5694','#2a947c','#a64f67']
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
        'figure.facecolor':'white','axes.facecolor':'#f7f9fc','axes.spines.top':False,'axes.spines.right':False,
        'savefig.facecolor':'white','svg.fonttype':'none'})

def field(ax,result,values,title,label,norm=None,cmap='RdBu_r',extend='neither'):
    x=result['x'];z=result['z'];mask=result['roi'];dx=np.median(np.diff(x[0]));dz=np.median(np.diff(z[:,0]))
    image=ax.imshow(np.ma.array(values,mask=~mask),extent=[(x.min()-dx/2)*100,(x.max()+dx/2)*100,(z.max()+dz/2)*100,(z.min()-dz/2)*100],norm=norm,cmap=cmap,interpolation='nearest',aspect='equal')
    ax.add_patch(Circle((0,2.3),1,fill=False,color='#475569',linewidth=.65))
    ax.set(title=title,xlabel='Lateral x [cm]',ylabel='Depth z [cm]')
    cb=ax.figure.colorbar(image,ax=ax,shrink=.78,pad=.025,extend=extend);cb.set_label(label)
    return image

def arrows(ax,result,velocity,stride=15,key=False):
    sl=(slice(None,None,stride),slice(None,None,stride));m=result['roi'][sl]
    q=ax.quiver(result['x'][sl][m]*100,result['z'][sl][m]*100,velocity[0][sl][m],velocity[1][sl][m],angles='xy',scale_units='xy',scale=1.4,width=.005,color='#14263d')
    if key:ax.quiverkey(q,.20,.07,.5,'0.5 m/s',labelpos='E',coordinates='axes')

def plot_observations(observations):
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');rf=observations['rf']
    ax=axes[0,0];xe=(np.arange(128)-63.5)*.000298
    ax.plot(xe*100,np.zeros(128),'.',color=COLORS[0],ms=3,label='All 128 receive elements')
    ax.add_patch(Circle((0,2.3),1,facecolor='#d9e8f5',edgecolor='#46637e',label='Known disk support'))
    for angle,color,label in [(np.pi/12,COLORS[1],'+15° receive'),(-np.pi/12,COLORS[2],'−15° receive')]:
        origin=-.023*np.tan(angle)*100
        ax.annotate('',xy=(0,2.3),xytext=(origin,0),arrowprops={'arrowstyle':'->','color':color,'lw':2})
        ax.text(-1.68 if angle>0 else .86,.62,label,color=color,fontsize=10)
    ax.plot([0,0],[0,2.3],'--',color='#64748b',lw=1,label='Normal plane-wave transmit')
    ax.set(xlim=(-2,2),ylim=(3.6,-.2),aspect='equal',xlabel='Lateral x [cm]',ylabel='Depth z [cm]',title='A  Measured solid motion; no true speed supplied');ax.legend(loc='lower left',fontsize=8)
    im=axes[0,1].imshow(rf[:,:,0],extent=[-.5,127.5,(START_TIME+(334-.5)/SAMPLING_FREQUENCY)*1e6,(START_TIME-.5/SAMPLING_FREQUENCY)*1e6],aspect='auto',cmap='RdBu_r',vmin=-np.max(abs(rf[:,:,0])),vmax=np.max(abs(rf[:,:,0])))
    axes[0,1].set(title='B  Actual first RF pulse',xlabel='Receive element index',ylabel='Fast time [µs]');fig.colorbar(im,ax=axes[0,1],label='Signed ADC counts',shrink=.85)
    spectrum=np.mean(abs(np.fft.rfft(rf,n=2048,axis=0))**2,axis=(1,2));frequency=np.fft.rfftfreq(2048,1/SAMPLING_FREQUENCY)/1e6
    db=10*np.log10(np.maximum(spectrum/spectrum.max(),1e-12))
    axes[1,0].plot(frequency,db,color=COLORS[0])
    alias=abs(5-SAMPLING_FREQUENCY/1e6)
    axes[1,0].axvspan(alias-.375,alias+.375,color=COLORS[1],alpha=.15,label='Expected alias band')
    axes[1,0].axvline(alias,color=COLORS[1],ls='--',label='5 MHz carrier → sampled alias')
    axes[1,0].set(title='C  Mean spectrum of every RF trace',xlabel='Sampled positive frequency [MHz]',ylabel='Power relative to peak [dB]',xlim=(0,SAMPLING_FREQUENCY/2e6),ylim=(-80,3));axes[1,0].legend(fontsize=9)
    time=(START_TIME+np.arange(334)/SAMPLING_FREQUENCY)*1e6
    for index,color in [(31,COLORS[0]),(95,COLORS[1])]:axes[1,1].plot(time,rf[:,index,0],color=color,lw=1,label=f'Element {index}, pulse 0')
    axes[1,1].set(title='D  Actual uncalibrated echo amplitudes',xlabel='Fast time [µs]',ylabel='ADC counts');axes[1,1].legend(fontsize=9)
    fig.suptitle('The observations are undersampled RF, not pre-estimated velocities',fontsize=17,fontweight='bold')
    return fig,{'rf_first_pulse':rf[:,:,0],'frequency_mhz':frequency,'mean_spectrum_db':db,'trace_time_us':time,'selected_traces':rf[:,[31,95],0],'element_x_m':xe}

def plot_atlas(result):
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained');mask=result['roi'];norm=TwoSlopeNorm(vcenter=0,vmin=-.65,vmax=.65)
    for row,(name,v) in enumerate([('Raw two-angle',result['raw_velocity']),('Selected robust',result['velocity'])]):
        for j,label in enumerate(['vₓ','v_z']):
            count=int(np.count_nonzero(abs(v[j,mask])>.65))
            field(axes[row,j],result,v[j],f'{chr(65+row*3+j)}  {name} {label}',f'{label} [m/s]',norm=norm,extend='both' if count else 'neither')
            if count:axes[row,j].text(.02,.97,f'{count} values beyond ±0.65',transform=axes[row,j].transAxes,va='top',fontsize=8,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
        speed=np.linalg.norm(v,axis=0);count=np.count_nonzero(speed[mask]>.8)
        field(axes[row,2],result,speed,f'{chr(67+row*3)}  {name} speed and direction','Speed [m/s]',norm=Normalize(0,.8),cmap='viridis',extend='max' if count else 'neither');arrows(axes[row,2],result,v,key=row==0)
    fig.suptitle('Every disk pixel has two recovered velocity components',fontsize=17,fontweight='bold')
    return fig,{'raw_vector_mps':result['raw_velocity'],'selected_vector_mps':result['velocity'],'roi':mask,'signed_display_limit_mps':np.array(.65),'speed_display_limit_mps':np.array(.8)}

def plot_parity(result,original,thresholds):
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained');mask=result['roi'];fresh=result['velocity'];reference=original['smoothed_vector'].copy();reference[:,~mask]=0;errors=[]
    for j,label in enumerate(['vx','vz']):
        a=reference[j,mask];b=fresh[j,mask];axes[0,j].scatter(a,b,s=2,alpha=.18,rasterized=True,color=COLORS[j])
        lo=min(a.min(),b.min());hi=max(a.max(),b.max());axes[0,j].plot([lo,hi],[lo,hi],'--',color='#64748b',lw=1)
        axes[0,j].set(title=f'{chr(65+j)}  All 31,404 {label} values',xlabel='Independent official-author run [m/s]',ylabel='Fresh local field [m/s]')
        difference=fresh[j]-reference[j];limit=max(np.max(abs(difference[mask])),1e-16)
        field(axes[1,j],result,difference,f'{chr(68+j)}  Signed fresh − author {label}','Difference [m/s]',norm=TwoSlopeNorm(vcenter=0,vmin=-limit,vmax=limit))
        normalized=abs(b-a)/np.ptp(a);errors.append(float(np.sqrt(np.mean(normalized**2))))
        axes[0,2].hist(np.maximum(normalized,1e-18),bins=np.geomspace(1e-18,1e-7,48),histtype='step',lw=1.7,color=COLORS[j],label=label)
        axes[0,2].axvline(thresholds[label]['nrmse_boundary'],ls='--',color=COLORS[j])
    axes[0,2].set(xscale='log',title='C  Pixel differences / reference range',xlabel='Absolute normalized difference',ylabel='Count');axes[0,2].legend()
    values=np.array([thresholds[k]['nrmse_boundary'] for k in ['vx','vz']]);positions=np.arange(2)
    for y,a,b in zip(positions,errors,values):axes[1,2].plot([a,b],[y,y],color='#aab5c3',ls=':',lw=1.5)
    axes[1,2].scatter(errors,positions,s=55,color=COLORS[0],label='Actual fresh NRMSE',zorder=3)
    axes[1,2].scatter(values,positions,s=65,marker='|',linewidths=2.5,color=COLORS[1],label='Pre-extraction frozen boundary',zorder=3)
    axes[1,2].set(xscale='log',yticks=positions,yticklabels=['vx','vz'],title='F  Numerical reproduction; no speed truth',xlabel='Range-normalized RMS error');axes[1,2].legend(fontsize=8)
    axes[1,2].set_xticks([1e-15,1e-12,1e-9]);axes[1,2].minorticks_off();axes[1,2].set_ylim(-.5,1.5)
    fig.suptitle('Original and fresh are independent method measurements',fontsize=17,fontweight='bold')
    return fig,{'original_vector_mps':reference,'fresh_vector_mps':fresh,'differences_mps':fresh-reference,'nrmse':np.array(errors),'nrmse_boundaries':values,'roi':mask,'histogram_zero_display_floor':np.array(1e-18)}

def plot_phase(result,phase_limit):
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained');mask=result['roi'];vn=.74
    for j,name in enumerate(['+15°','−15°']):field(axes[0,j],result,result['radial'][j],f'{chr(65+j)}  Observed signed Doppler {name}','d [m/s]',norm=TwoSlopeNorm(vcenter=0,vmin=-vn,vmax=vn))
    coherence=np.min(1-result['variance']/(2*(vn/np.pi)**2),axis=0)
    field(axes[0,2],result,coherence,'C  Smaller two-view |AC| / packet power','Finite-packet correlation ratio',norm=Normalize(0,1),cmap='viridis')
    prediction=np.exp(-1j*np.pi*predicted_radial(result['velocity'])/vn)
    difference=np.angle(prediction*np.conj(result['autocorrelation']))
    phase_range=max(float(np.max(abs(difference[:,mask]))),1e-6)
    for j,name in enumerate(['+15°','−15°']):field(axes[1,j],result,difference[j],f'{chr(68+j)}  Predicted − observed phase {name}','Wrapped phase residual [rad]',norm=TwoSlopeNorm(vcenter=0,vmin=-phase_range,vmax=phase_range),cmap='coolwarm')
    errors=difference[:,mask].ravel();weights=abs(result['autocorrelation'][:,mask]).ravel()
    axes[1,2].hist(errors,bins=np.linspace(-phase_range,phase_range,65),density=True,histtype='step',color=COLORS[0],label='Equal pixel weight')
    axes[1,2].hist(errors,bins=np.linspace(-phase_range,phase_range,65),weights=weights,density=True,histtype='step',color=COLORS[1],label='Echo magnitude weight')
    metric=phase_consistency(result['velocity'],result['autocorrelation'],mask)
    axes[1,2].set(title='F  Actual residuals expose smoothing bias',xlabel='Wrapped phase residual [rad]',ylabel='Density');axes[1,2].legend(fontsize=8,loc='upper right')
    axes[1,2].text(.04,.95,f'Phasor RMS {metric:.5f}\nLimit {phase_limit:.5f}',transform=axes[1,2].transAxes,va='top',fontsize=8,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    fig.suptitle('Vector quality also has to explain the measured complex phase',fontsize=17,fontweight='bold')
    return fig,{'radial_mps':result['radial'],'autocorrelation':result['autocorrelation'],'wrapped_phase_residual_rad':difference,'packet_correlation_ratio':coherence,'weighted_phasor_rms':np.array(metric),'phase_boundary':np.array(phase_limit),'signed_phase_display_limit_rad':np.array(phase_range),'roi':mask}

def plot_comparisons(result,comparisons):
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained');mask=result['roi'];names=list(comparisons)
    rawspeed=np.linalg.norm(result['raw_velocity'],axis=0);speed=np.linalg.norm(result['velocity'],axis=0)
    axes[0,0].scatter(rawspeed[mask],speed[mask],s=2,alpha=.15,rasterized=True,color=COLORS[0]);hi=max(rawspeed[mask].max(),speed[mask].max());axes[0,0].plot([0,hi],[0,hi],'--',color='#64748b')
    axes[0,0].set(title='A  The prior changes measured speed estimates',xlabel='Raw two-angle speed [m/s]',ylabel='Selected robust speed [m/s]')
    data_errors=[];phase_errors=[]
    for name,item in comparisons.items():
        data_errors.append(np.sqrt(np.mean((item['velocity'][:,mask]-result['raw_velocity'][:,mask])**2)))
        phase_errors.append(phase_consistency(item['velocity'],result['autocorrelation'],mask))
    bars=axes[0,1].bar(np.arange(len(names)),data_errors,color=[COLORS[j%5] for j in range(len(names))]);axes[0,1].bar_label(bars,fmt='%.3g',padding=3,fontsize=8);axes[0,1].margins(y=.12);axes[0,1].set(title='B  Actual departure from raw vectors',ylabel='Two-component RMS [m/s]',xticks=np.arange(len(names)),xticklabels=names);axes[0,1].tick_params(axis='x',rotation=32,labelsize=8)
    bars=axes[0,2].bar(np.arange(len(names)),phase_errors,color=[COLORS[j%5] for j in range(len(names))]);axes[0,2].bar_label(bars,fmt='%.3g',padding=3,fontsize=8);axes[0,2].margins(y=.12);axes[0,2].axhline(.057214058693848345,ls='--',color='#64748b',label='Selected task boundary');axes[0,2].set(title='C  Same observed-phase check for every field',ylabel='Weighted phasor RMS',xticks=np.arange(len(names)),xticklabels=names);axes[0,2].tick_params(axis='x',rotation=32,labelsize=8);axes[0,2].legend(fontsize=8)
    weights=result['smoothing']['weights_used'];field(axes[1,0],result,weights,'D  Actual weights used in final robust solve','Bisquare weight',norm=Normalize(0,1),cmap='viridis')
    residual=np.linalg.norm(result['raw_velocity']-result['smoothed_full'],axis=0)
    axes[1,1].scatter(residual[mask],weights[mask],s=2,alpha=.12,rasterized=True,color=COLORS[2]);axes[1,1].set(title='E  Downweighting is not a ground-truth label',xlabel='Raw − selected vector residual [m/s]',ylabel='Final-solve weight')
    plain=comparisons['Nonrobust 10⁴']['velocity'];difference=plain[0]-result['velocity'][0];limit=max(np.max(abs(difference[mask])),1e-6)
    field(axes[1,2],result,difference,'F  Nonrobust − selected lateral velocity','Δvx [m/s]',norm=TwoSlopeNorm(vcenter=0,vmin=-limit,vmax=limit))
    fig.suptitle('Raw, quadratic and robust estimates make different trade-offs',fontsize=17,fontweight='bold')
    numeric={'method_names':np.array(names),'vector_rms_from_raw_mps':np.array(data_errors),'weighted_phasor_rms':np.array(phase_errors),'final_solve_weights':weights,'vector_residual_mps':residual,'nonrobust_minus_selected_vx':difference,'roi':mask}
    for i,item in enumerate(comparisons.values()):numeric[f'method_{i}_velocity_mps']=item['velocity']
    return fig,numeric

def rotation_diagnostic(result,velocity):
    mask=result['roi'];x=result['x'][mask];z=result['z'][mask]-.023
    omega=float(np.sum(-z*velocity[0,mask]+x*velocity[1,mask])/np.sum(x*x+z*z))
    model=np.stack([-omega*(result['z']-.023),omega*result['x']]);rms=float(np.sqrt(np.mean((velocity[:,mask]-model[:,mask])**2)))
    return omega,rms

def plot_sensitivity(result,lags,windows):
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained');mask=result['roi'];row=int(np.argmin(abs(result['z'][:,0]-.023)));inside=mask[row]
    for j,(lag,item) in enumerate(lags.items()):axes[0,0].plot(result['x'][row,inside]*100,item['velocity'][1,row,inside],color=COLORS[j],label=f'Lag {lag}, VN={.74/lag:.3f}m/s')
    axes[0,0].set(title='A  Actual depth-velocity profiles at disk centre',xlabel='Lateral x [cm]',ylabel='vz [m/s]');axes[0,0].legend(fontsize=8)
    errors=[]
    for lag,item in lags.items():errors.append([np.sqrt(np.mean((item['velocity'][j,mask]-result['velocity'][j,mask])**2))/np.ptp(result['velocity'][j,mask]) for j in range(2)])
    errors=np.array(errors);positions=np.arange(len(lags))
    axes[0,1].bar(positions-.17,errors[:,0],width=.32,color=COLORS[0],label='vx');axes[0,1].bar(positions+.17,errors[:,1],width=.32,color=COLORS[1],label='vz')
    axes[0,1].set(title='B  Differences from selected lag1 field',xlabel='Ensemble lag',ylabel='Range-normalized RMS difference',xticks=positions,xticklabels=list(lags));axes[0,1].legend()
    delta=lags[4]['velocity'][1]-result['velocity'][1];limit=max(np.max(abs(delta[mask])),1e-6)
    field(axes[0,2],result,delta,'C  Lag4 − selected depth velocity','Δvz [m/s]',norm=TwoSlopeNorm(vcenter=0,vmin=-limit,vmax=limit))
    for j,(window,item) in enumerate(windows.items()):axes[1,0].plot(result['x'][row,inside]*100,item['velocity'][0,row,inside],color=COLORS[j%5],label=f'{window} × {window}')
    axes[1,0].set(title='D  Spatial averaging changes lateral estimates',xlabel='Lateral x [cm]',ylabel='vx [m/s]');axes[1,0].legend(fontsize=8)
    diagnostic=np.array([rotation_diagnostic(result,item['velocity']) for item in windows.values()]);p=np.arange(len(windows))
    axes[1,1].bar(p,diagnostic[:,1],color=COLORS[3]);axes[1,1].set(title='E  Fitted rigid-motion diagnostic; no speed truth',xlabel='Hamming window side',ylabel='RMS from fitted rotation [m/s]',xticks=p,xticklabels=list(windows))
    history=result['smoothing']['history'];counts=np.array([h['iterations'] for h in history]);change=np.array([h['relative_change'] for h in history])
    ax=axes[1,2];ax.bar(np.arange(1,4),counts,color=COLORS[0],alpha=.7,label='Inner updates');ax.axhline(100,color='#64748b',ls='--',label='Per-pass budget')
    ax.set(title='F  Finite protocol retains its first-pass cap',xlabel='Robust pass',ylabel='Number of inner updates',xticks=[1,2,3],ylim=(0,115));right=ax.twinx();right.plot([1,2,3],change,'o-',color=COLORS[1],label='Final relative change');right.axhline(1e-3,color=COLORS[1],ls=':');right.set_ylabel('Final relative field change',color=COLORS[1]);right.set_ylim(0,.003)
    ax.legend(loc='upper right',fontsize=7);right.legend(loc='center right',fontsize=7)
    fig.suptitle('Temporal aliasing, spatial averaging and finite stopping all matter',fontsize=17,fontweight='bold')
    numeric={'lags':np.array(list(lags)),'lag_component_nrmse':errors,'window_sides':np.array(list(windows)),'fitted_omega_radps':diagnostic[:,0],'rms_from_fitted_rotation_mps':diagnostic[:,1],'inner_iterations':counts,'inner_final_relative_change':change,'roi':mask}
    for k,item in lags.items():numeric[f'lag{k}_velocity_mps']=item['velocity']
    for k,item in windows.items():numeric[f'window{k}_velocity_mps']=item['velocity']
    return fig,numeric

def export_figure(fig,figure_id,directory,numeric):
    if directory:
        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
        for extension in ['png','svg','pdf']:fig.savefig(path/f'{figure_id}.{extension}',dpi=150,bbox_inches='tight')
        np.savez_compressed(path/f'{figure_id}_numeric.npz',**numeric)
    return fig
