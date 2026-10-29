"""Scientific comparisons using only measured inputs and computed fit dictionaries."""
import numpy as np
import matplotlib.pyplot as plt
from .solvers import design_system,solve_scattering

def log_ratio_metrics(output,reference,mask):
    values={}
    for key in output:
        a=output[key];b=reference[key]
        if key=='site_amplification':a=a[0][mask];b=b[0][mask]
        values[key]=float(np.max(abs(np.log(a/b))))
    return values

def acquisition_figure(data,details):
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    time=data['start_time'][0,0]+np.arange(4401)/20
    for j,label in enumerate(['component 1','component 2','component 3']):axes[0,0].plot(time,data['velocity'][0,0,j]*1e6,label=label,lw=.7)
    axes[0,0].set(xlabel='Time after origin [s]',ylabel='Velocity [micrometres/s]',title='Actual GR record: event 1, station slot 1');axes[0,0].legend(fontsize=8)
    r=details[0][2];p=r['pairs'][0]
    axes[0,1].semilogy(time,p['unsmoothed'],alpha=.5,label='Noise-subtracted observed energy',lw=.6)
    axes[0,1].semilogy(time,p['energy'],label='Bartlett-smoothed observation',lw=1)
    axes[0,1].axhline(p['noise'],color='grey',ls=':',label='Measured pre-event noise')
    axes[0,1].axvspan(*p['bulk_interval'],alpha=.15,color='orange',label='Bulk window')
    axes[0,1].axvspan(p['tcoda'][0],p['tcoda'][-1],alpha=.08,color='green',label='Coda window')
    axes[0,1].set(xlabel='Time after origin [s]',ylabel='Energy [J m^-3 Hz^-1]',title='1.5 Hz observation and selected windows');axes[0,1].legend(fontsize=7)
    for j,p in enumerate(r['pairs']):
        color=f'C{j}';amplitude=r['source']*r['sites'][j]
        axes[1,0].semilogy(p['tcoda'],p['Ecoda'],color=color,lw=.6,alpha=.55)
        axes[1,0].semilogy(p['tcoda'],amplitude*r['Gcoda'][j]*np.exp(-r['absorption']*p['tcoda']),color=color,label=f"slot {p['station_slot']+1} fit")
        axes[1,0].scatter([p['tbulk']],[p['Ebulk']],color=color,s=20,marker='s')
    axes[1,0].set(xlabel='Time after origin [s]',ylabel='Energy [J m^-3 Hz^-1]',title='Fresh full model; faint observations, square bulk means');axes[1,0].legend(fontsize=8)
    available=data['fit_valid'].sum(axis=2)
    im=axes[1,1].imshow(available,vmin=0,vmax=5,cmap='Blues');fig.colorbar(im,ax=axes[1,1],label='Usable station records')
    for e in range(5):
        for f in range(5):axes[1,1].text(f,e,str(available[e,f]),ha='center',va='center',color='white')
    axes[1,1].set(xticks=range(5),xticklabels=['.375','.75','1.5','3','6'],yticks=range(5),yticklabels=range(1,6),xlabel='Band centre [Hz]',ylabel='Event',title='All 25 fits: observed coda availability')
    return fig

def spectra_figure(data,output,reference):
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);freq=data['frequency_bands'].mean(axis=1)
    for e in range(5):
        for k,axis in enumerate(axes[0]):
            axis.loglog(freq,output['reconstruction'][0,e,:,k],'-o',label=f'event {e+1}',ms=3)
            axis.loglog(freq,reference['reconstruction'][0,e,:,k],'x',color=f'C{e}',ms=6)
        axes[1,0].loglog(freq,output['source_energy'][0,e],'-o',label=f'event {e+1}',ms=3)
        axes[1,0].loglog(freq,reference['source_energy'][0,e],'x',color=f'C{e}',ms=6)
    axes[0,0].set(title='Scattering: fresh lines / original crosses',xlabel='Frequency [Hz]',ylabel='Scattering [m^-1]');axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='Intrinsic absorption: event-dependent spectra',xlabel='Frequency [Hz]',ylabel='Absorption [s^-1]')
    axes[1,0].set(title='Source spectra in the geometric-site gauge',xlabel='Frequency [Hz]',ylabel='Source energy [J Hz^-1]')
    names=['BUG','TNS','CLZ','BFO','FUR']
    for station in range(5):
        for e in range(5):
            slots=np.flatnonzero(data['station_index'][e]==station)
            if not len(slots):continue
            slot=slots[0];mask=data['fit_valid'][e,:,slot]
            axes[1,1].loglog(freq[mask],output['site_amplification'][0,e,mask,slot],color=f'C{station}',alpha=.45,marker='.',label=names[station] if e==0 else None)
    axes[1,1].axhline(1,color='grey',ls=':');axes[1,1].set(title='Site responses grouped by actual station identity',xlabel='Frequency [Hz]',ylabel='Relative energy amplification');axes[1,1].legend(fontsize=8)
    from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
    for axis in axes.flat:
        axis.xaxis.set_major_locator(FixedLocator(freq));axis.xaxis.set_major_formatter(FixedFormatter(['.375','.75','1.5','3','6']));axis.xaxis.set_minor_locator(NullLocator())
    return fig

def diagnostics_figure(data,output,reference,details):
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    ratios=np.max(abs(np.log(output['reconstruction'][0]/reference['reconstruction'][0])),axis=-1)
    im=axes[0,0].imshow(ratios,cmap='magma',vmin=0,vmax=max(float(ratios.max()),1e-15));fig.colorbar(im,ax=axes[0,0],label='Max |log(fresh/original)|')
    axes[0,0].set(title='Transport agreement across all 25 fits',xlabel='Frequency index',ylabel='Event index')
    errors=np.array([[r['error'] for r in row] for row in details]);im=axes[0,1].imshow(errors,cmap='viridis');fig.colorbar(im,ax=axes[0,1],label='Weighted log-energy residual')
    axes[0,1].set(title='Nonzero model mismatch, not truth error',xlabel='Frequency index',ylabel='Event index')
    r=details[0][2];grid=np.geomspace(r['scattering']/8,r['scattering']*8,81);profiles=[solve_scattering(r['pairs'],g) for g in grid]
    axes[1,0].semilogx(grid,[p['error'] for p in profiles],label='Profiled residual');axes[1,0].axvline(r['scattering'],color='black',ls=':',label='Selected g')
    twin=axes[1,0].twinx();twin.semilogx(grid,[p['absorption'] for p in profiles],color='C1',ls='--');twin.set_ylabel('Profiled absorption [s^-1]',color='C1')
    axes[1,0].set(title='Scattering/absorption coupling: event 1 at 1.5 Hz',xlabel='Scattering [m^-1]',ylabel='Weighted log residual');axes[1,0].legend(fontsize=8)
    all_residual=[]
    for row in details:
        for r in row:
            _,energy,_=design_system(r['pairs']);all_residual.extend(np.log(energy/r['model']))
    axes[1,1].hist(all_residual,bins=70,density=True,color='C2',alpha=.8);axes[1,1].axvline(0,color='black',ls=':')
    axes[1,1].set(title='All measured coda/bulk residuals (correlated)',xlabel='log(observed / fitted energy)',ylabel='Empirical density')
    return fig

def fixed_scattering_comparison(details):
    common=float(np.exp(np.mean(np.log([[r['scattering'] for r in row] for row in details]))))
    fixed=[[solve_scattering(r['pairs'],common) for r in row] for row in details]
    return common,fixed

def ablation_figure(details,common,fixed):
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    full=np.array([[r['error'] for r in row] for row in details]);reduced=np.array([[r['error'] for r in row] for row in fixed]);x=np.arange(25)
    axes[0,0].plot(x,full.ravel(),'o-',label='25 independently fitted scattering values');axes[0,0].plot(x,reduced.ravel(),'s--',label='One common scattering value; amplitudes/b refitted')
    axes[0,0].set(title='Fresh fixed-scattering comparison',xlabel='Event-major fit index',ylabel='Weighted log residual');axes[0,0].legend(fontsize=7)
    b=np.array([[r['absorption'] for r in row] for row in details]);bf=np.array([[r['absorption'] for r in row] for row in fixed])
    axes[0,1].scatter(b,bf);lo=min(b.min(),bf.min());hi=max(b.max(),bf.max());axes[0,1].plot([lo,hi],[lo,hi],'k:');axes[0,1].set(title='Absorption compensates for fixed scattering',xlabel='Full fit b [s^-1]',ylabel='Fixed-scattering b [s^-1]')
    w=np.array([[r['source'] for r in row] for row in details]);wf=np.array([[r['source'] for r in row] for row in fixed]);im=axes[1,0].imshow(np.log(wf/w),cmap='RdBu_r');fig.colorbar(im,ax=axes[1,0],label='log(fixed / full source)');axes[1,0].set(title='Source-spectrum sensitivity to transport model',xlabel='Frequency index',ylabel='Event index')
    r=details[0][2];q=fixed[0][2];p=r['pairs'][0]
    for fit,label,style in [(r,'Full inversion','-'),(q,'Common scattering','--')]:
        model=fit['source']*fit['sites'][0]*fit['Gcoda'][0]*np.exp(-fit['absorption']*p['tcoda']);axes[1,1].plot(p['tcoda'],np.log(p['Ecoda']/model),style,label=label)
    axes[1,1].axhline(0,color='grey',ls=':');axes[1,1].set(title=f'Conditional comparison at common g={common:.3g} m^-1',xlabel='Time after origin [s]',ylabel='Log coda residual');axes[1,1].legend(fontsize=8)
    return fig
