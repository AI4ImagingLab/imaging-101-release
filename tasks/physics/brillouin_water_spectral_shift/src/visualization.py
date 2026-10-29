"""Fresh measured-water comparisons and physically labeled sensitivity studies."""
import contextlib
import io
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct,REGIONS
from .physics_model import spectral_model,VIPA,setup

def relative_error(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def experiments(raw,result,details):
    """Actively recompute strip widths, calibration drift handling and L2 fitting."""
    variants={}
    with contextlib.redirect_stdout(io.StringIO()):
        for label,kwargs in [('width1',{'arc_width':1}),('width3',{'arc_width':3}),('first_calibration',{'calibration_mode':'first'}),('L2_peaks',{'power':2})]:
            variants[label],_=reconstruct(raw,**kwargs)
    instrument=setup();peaks=details['calibration_peak_positions']
    frequencies=np.empty_like(peaks)
    expected=np.empty_like(peaks)
    for i in range(2):
        for j in range(10):
            p=result['calibration_parameters'][0,i,j]
            frequencies[i,j]=VIPA(peaks[i,j],p)-instrument.f0
            expected[i,j]=instrument.calibration.shifts+p[3]*instrument.calibration.orders
    calibration_residual=frequencies-expected
    source={'calibration_residual_Hz':calibration_residual,'calibration_expected_Hz':expected,'calibration_peaks_pixel':peaks}
    for label,v in variants.items():source[label+'_shift_Hz']=v['reconstruction']
    measurements={'upstream_objective':'quartic spectral residual, nonlinear VIPA sum-of-squares','calibration_rms_MHz':float(np.sqrt(np.mean(calibration_residual**2))/1e6),'calibration_max_abs_MHz':float(abs(calibration_residual).max()/1e6),'mean_shift_GHz':float(result['reconstruction'].mean()/1e9),'frame_and_region_std_MHz':float(result['reconstruction'].std()/1e6),'variant_rms_change_MHz':{k:float(np.sqrt(np.mean((v['reconstruction']-result['reconstruction'])**2))/1e6) for k,v in variants.items()}}
    return variants,source,measurements

def make_figures(raw,result,details,reference,variants,source):
    """Return exactly three figures, with full comparison arrays in source."""
    figs={};plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ax=axs[0,0];image=raw['calibration_images'][0,0].mean(axis=0)
    source['calibration_camera_mean_ADU']=image
    im=ax.imshow(image,origin='upper',extent=(-.5,399.5,399.5,-.5),cmap='magma')
    arc=result['calibration_arcs'][0,0];pts=details['points'][0]
    ax.plot(arc[:,2,1],arc[:,2,0],color='cyan',lw=1,label='Fitted circular centerline');ax.scatter(pts[:,1],pts[:,0],facecolors='none',edgecolors='lime',s=45,label='Detected calibration peaks')
    ax.set(title='A  Independent calibration camera mean',xlabel='Column (pixel)',ylabel='Row (pixel)');ax.legend(fontsize=8);fig.colorbar(im,ax=ax,label='Camera intensity (ADU)')
    ax=axs[0,1]
    for i in range(2):ax.plot(np.arange(500),details['calibration_spectra'][i].mean(axis=0),label=f'Calibration {i+1}')
    ax.set(title='B  All-frame extracted calibration spectra',xlabel='Arc sample index (1)',ylabel='Intensity (ADU)');ax.legend()
    ax=axs[1,0]
    for i in range(2):
        f=result['calibration_frequency'][0,i]
        ax.plot(f.mean(axis=0)/1e9,label=f'Calibration {i+1} mean');ax.fill_between(np.arange(500),f.min(axis=0)/1e9,f.max(axis=0)/1e9,alpha=.15)
    ax.set(title='C  Nonlinear VIPA frequency calibration',xlabel='Arc sample index (1)',ylabel='Frequency relative to laser (GHz)');ax.legend(fontsize=9)
    ax=axs[1,1];x=np.arange(10)
    for r in range(2):
        fresh=result['reconstruction'][0,:,:,r]/1e9;ref=reference['reconstruction'][0,:,:,r]/1e9
        ax.plot(x,ref.mean(axis=1),color=['#0072B2','#D55E00'][r],lw=3,alpha=.3,label=f'Original region {r+1} mean')
        ax.plot(x,fresh.mean(axis=1),'o-',color=['#0072B2','#D55E00'][r],label=f'Fresh region {r+1} mean')
        for j in range(2):ax.scatter(x,fresh[:,j],marker=['_','x'][j],color=['#0072B2','#D55E00'][r],s=23)
    ax.set(title='D  Water: ten acquisition positions, two frames',xlabel='Acquisition position index (1)',ylabel='Brillouin shift (GHz)');ax.legend(fontsize=8,ncol=2)
    figs['reconstruction']=fig

    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    f=result['frequency_axis'][0,0,0];y=result['extracted_spectra'][0,0,0];p=result['peak_parameters'][0,0,0]
    ax=axs[0,0];ax.plot(f/1e9,y,'.',ms=3,color='.55',label='Measured first frame')
    residual=np.full((4,500),np.nan)
    for j,r in enumerate(REGIONS):
        a=np.argmin(abs(f-r[0]));b=np.argmin(abs(f-r[1]));model=spectral_model(f[a:b],p[j]);ax.plot(f[a:b]/1e9,model,lw=1.8,label=f'Region {j+1} fit');residual[j,a:b]=y[a:b]-model
    ax.set(title='A  Full extracted spectrum and regional fits',xlabel='Frequency relative to laser (GHz)',ylabel='Intensity (ADU)');ax.legend(fontsize=8)
    ax=axs[0,1]
    for j in range(4):ax.plot(f/1e9,residual[j],label=f'Region {j+1}')
    ax.axhline(0,color='.5',lw=.8);ax.set(title='B  Signed regional forward residuals',xlabel='Frequency relative to laser (GHz)',ylabel='Measured minus fitted (ADU)');ax.legend(fontsize=8)
    source['representative_residual_ADU']=residual;source['representative_frequency_Hz']=f
    ax=axs[1,0]
    for k,label in [('width1','3-pixel strip'),('width3','7-pixel strip'),('L2_peaks','L2 peak fit')]:
        delta=(variants[k]['reconstruction']-result['reconstruction'])[0]/1e6
        ax.plot(np.arange(10),delta.mean(axis=(1,2)),'o-',label=label)
        ax.fill_between(np.arange(10),delta.min(axis=(1,2)),delta.max(axis=(1,2)),alpha=.12)
    ax.axhline(0,color='.5',lw=.8);ax.set(title='C  Extraction and objective sensitivity',xlabel='Acquisition position index (1)',ylabel='Shift change from standard (MHz)');ax.legend(fontsize=8)
    ax=axs[1,1];delta=(variants['first_calibration']['reconstruction']-result['reconstruction'])[0]/1e6
    for r in range(2):ax.plot(np.arange(10),delta[:,:,r].mean(axis=1),'o-',label=f'Region {r+1}')
    ax.axhline(0,color='.5',lw=.8);ax.set(title='D  Omit second calibration: drift sensitivity',xlabel='Acquisition position index (1)',ylabel='Shift change from interpolated axis (MHz)');ax.legend(fontsize=8)
    figs['robustness']=fig

    fig,axs=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    ax=axs[0];cr=source['calibration_residual_Hz']/1e6
    for i in range(2):
        for j in range(10):ax.plot(np.arange(6),cr[i,j],'-o',alpha=.3,color=['#0072B2','#D55E00'][i],label=f'Calibration {i+1}' if j==0 else None,ms=3)
    ax.axhline(0,color='.5',lw=.8);ax.set(title='A  Independent standard residuals',xlabel='Ordered calibration peak index (1)',ylabel='Calibrated minus known frequency (MHz)');ax.legend(fontsize=8)
    ax=axs[1];shift=result['reconstruction'][0]/1e6
    for j in range(2):ax.plot(np.arange(10),shift[:,j,0]-shift[:,j,1],'o-',label=f'Frame {j+1}')
    ax.axhline(0,color='.5',lw=.8);ax.legend(fontsize=8)
    ax.set(title='B  Stokes/anti-Stokes agreement',xlabel='Acquisition position index (1)',ylabel='Region 1 minus region 2 shift (MHz)')
    ax=axs[2]
    for i in range(2):
        for j in range(2):ax.plot(np.arange(10),result['peak_parameters'][0,:,j,i,1]/1e6,marker=['o','x'][j],linestyle=['-','--'][j],color=['#0072B2','#D55E00'][i],alpha=.7,label=f'Region {i+1}, frame {j+1}')
    ax.legend(fontsize=7)
    ax.set(title='C  Observed Brillouin linewidths',xlabel='Acquisition position index (1)',ylabel='Fitted instrument-broadened FWHM (MHz)')
    figs['physical_diagnostics']=fig
    return figs
