"""Scientific figures from computed fields; no backend selection or reconstruction."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,TwoSlopeNorm
from .physics_model import abs2,scaled_residual

COLORS=['#147d92','#d67832','#7456a5','#4b8e4d','#c24760']

def aligned_field(field,reference,alignment,grid):
    a=np.fft.fftshift(np.fft.ifft(field,norm='ortho'))
    if alignment['conjugate_time_reversal']:a=a.conj()
    a*=np.exp(1j*grid.w*alignment['time_shift_seconds']-1j*alignment['global_phase_radians'])
    return np.fft.fft(np.fft.ifftshift(a),norm='ortho')

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':110,'savefig.dpi':180,'axes.labelcolor':'#263745','text.color':'#263745','axes.edgecolor':'#91a1ab','grid.alpha':.2})

def save(fig,directory,name):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg','pdf']:fig.savefig(directory/(name+'.'+ext),bbox_inches='tight')
    return fig

def acquisition(measured,model,directory):
    style();tau=model.delays*1e12;f=model.grid.w/(2*np.pi*1e12);fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    im=ax[0,0].pcolormesh(tau,f,measured.T,norm=LogNorm(1e-6,1),cmap='magma',shading='auto',rasterized=True);fig.colorbar(im,ax=ax[0,0],label='Relative intensity');ax[0,0].set_title('A  Complete processed experimental trace')
    im=ax[0,1].pcolormesh(tau,f,measured.T,cmap='magma',vmin=0,vmax=1,shading='auto',rasterized=True);ax[0,1].set(xlim=(-2.5,2.5),ylim=(-3,3),title='B  Pulse-bearing region (display zoom)');fig.colorbar(im,ax=ax[0,1],label='Relative intensity')
    ax[1,0].plot(tau,measured.sum(axis=1),color=COLORS[0]);ax[1,0].set(title='C  Delay marginal',ylabel='Summed intensity (relative)',xlabel='Actual delay (ps)')
    ax[1,1].plot(f,measured.sum(axis=0),color=COLORS[1]);ax[1,1].set(title='D  Frequency marginal',ylabel='Summed intensity (relative)',xlabel='Signal detuning (THz)')
    for a in ax[0]:a.set(xlabel='Actual delay (ps)',ylabel='Signal detuning (THz)')
    fig.suptitle('1 | A nonlinear optical gate records intensity, not phase',fontsize=17,fontweight='bold');return save(fig,directory,'figure1_acquisition')

def pulse(field,reference,model,alignment,directory):
    style();g=model.grid;e=aligned_field(field,reference,alignment,g);center=g.t[np.argmax(abs(reference))];t=(g.t-center)*1e12;f=g.w/(2*np.pi*1e12);fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for q,label,c,ls in [(reference,'Reproduced original COPRA','#333333','--'),(e,'Fresh cleaned COPRA',COLORS[0],'-')]:
        I=abs(q)**2*1e-12;sp=g.forward(q);J=abs(sp)**2;J/=J.max();phase=np.unwrap(np.angle(q));phase-=phase[np.argmax(I)];phasew=np.unwrap(np.angle(sp));phasew-=phasew[np.argmax(J)]
        ax[0,0].plot(t,I,label=label,color=c,ls=ls,lw=2);ax[1,0].plot(f,J,color=c,ls=ls,lw=2)
        # All phases are plotted faintly; solid phase highlights appreciable intensity.
        ax[0,1].plot(t,phase,color=c,ls=ls,alpha=.22);ax[0,1].plot(t,np.where(I>.01*I.max(),phase,np.nan),color=c,ls=ls,lw=2)
        ax[1,1].plot(f,phasew,color=c,ls=ls,alpha=.22);ax[1,1].plot(f,np.where(J>.01,phasew,np.nan),color=c,ls=ls,lw=2)
    ax[0,0].set(xlim=(-2,2),xlabel='Time relative to reference peak (ps)',ylabel='Unit-energy intensity (ps⁻¹)',title='A  Temporal envelope');ax[0,0].legend(fontsize=8)
    ax[0,1].set(xlim=(-2,2),ylim=(-12,12),xlabel='Time relative to reference peak (ps)',ylabel='Envelope phase (rad)',title='B  Temporal phase; faint = weak field')
    ax[1,0].set(xlim=(-3,3),xlabel='Fundamental detuning (THz)',ylabel='Peak-normalized spectral intensity',title='C  Spectral envelope')
    ax[1,1].set(xlim=(-3,3),ylim=(-12,12),xlabel='Fundamental detuning (THz)',ylabel='Spectral phase (rad)',title='D  Phase is poorly constrained in weak tails')
    fig.suptitle('2 | Recovering both amplitude and phase of a temporal field',fontsize=17,fontweight='bold');return save(fig,directory,'figure2_complex_field')

def residuals(measured,model,spectrum,directory):
    style();T=model.trace(spectrum);R,mu,_=scaled_residual(measured,T);pred=mu*T;res=measured-pred;f=model.grid.w/(2*np.pi*1e12);tau=model.delays*1e12
    fig,ax=plt.subplots(1,3,figsize=(15,4.9),layout='constrained');norm=TwoSlopeNorm(vmin=-np.max(abs(res)),vcenter=0,vmax=np.max(abs(res)))
    for i,(a,v,title) in enumerate(zip(ax,[measured,pred,res],['A  Measured','B  Fresh predicted','C  Measured − predicted'])):
        im=a.pcolormesh(tau,f,v.T,cmap='RdBu_r' if i==2 else 'magma',norm=norm if i==2 else None,vmin=None if i==2 else 0,vmax=None if i==2 else 1,shading='auto',rasterized=True);a.set(xlim=(-2.5,2.5),ylim=(-3,3),xlabel='Actual delay (ps)',ylabel='Signal detuning (THz)',title=title);fig.colorbar(im,ax=a,label='Relative intensity')
    fig.suptitle(f'3 | Complete-data physical consistency: R = {100*R:.4f}%',fontsize=17,fontweight='bold');fig.text(.5,-.01,'Display zoom only; residual metric uses all512×512 samples. Remaining structure includes model/preprocessing mismatch.',ha='center',fontsize=10);return save(fig,directory,'figure3_trace_residuals')

def convergence(fresh,upstream,directory):
    style();fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for i,(name,result) in enumerate(fresh.items()):
        errors=result['trace_error'];x=np.arange(1,len(errors)+1);ax[0].plot(x,errors,color=COLORS[i],alpha=.32,lw=.6);ax[0].plot(x,np.minimum.accumulate(errors),color=COLORS[i],label=name,lw=2)
        ax[1].plot(x,np.minimum.accumulate(upstream[name]['trace_error']),color=COLORS[i],ls='--',lw=2,label=name+' original');ax[1].plot(x,np.minimum.accumulate(errors),color=COLORS[i],lw=1,label=name+' cleaned')
    for a in ax:a.set(xlabel='Iteration',ylabel='Trace error R',yscale='log');a.grid(True);a.legend(fontsize=8)
    ax[0].set_title('A  Fresh histories and running minima');ax[1].set_title('B  Independent original / cleaned histories')
    fig.suptitle('4 | COPRA, PIE and a PCGPA regrid diagnostic',fontsize=17,fontweight='bold');fig.text(.5,-.01,'PCGPA uses linearly regridded delays; COPRA/PIE preserve actual scan delays. Source global logs can lag one iterate.',ha='center',fontsize=9);return save(fig,directory,'figure4_convergence')

def comparisons(rows,profiles,model,directory):
    style();names=list(rows);x=np.arange(len(names));fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for a,key,title,mult in [(ax[0,0],'trace_R','A  Fit on the original actual-delay observations',100),(ax[0,1],'complex_relative_l2','B  Complete complex field vs method reference',1)]:
        values=[rows[n][key]*mult for n in names];bars=a.bar(x,values,color=COLORS[:len(names)]);a.set_xticks(x,names,rotation=15);a.set_title(title);a.set_ylabel('Trace R (%)' if mult==100 else 'Aligned relative complex L2 error')
        for b,v in zip(bars,values):a.text(b.get_x()+b.get_width()/2,b.get_height(),f'{v:.4f}',ha='center',va='bottom',fontsize=9)
        a.margins(y=.2)
    for i,(name,e) in enumerate(profiles.items()):
        ax[1,0].plot(model.grid.t*1e12,abs(e)**2*1e-12,color=COLORS[i],label=name);ax[1,1].semilogy(model.grid.t*1e12,np.maximum(abs(e)**2*1e-12,1e-9),color=COLORS[i],label=name)
    ax[1,0].set(xlim=(-2,2),xlabel='Reference-gauge time (ps)',ylabel='Unit-energy intensity (ps⁻¹)',title='C  Aligned pulse profiles');ax[1,0].legend(fontsize=8)
    ax[1,1].set(xlabel='Reference-gauge time (ps)',ylabel='Unit-energy intensity (ps⁻¹), log',title='D  Full-window weak structure is retained');ax[1,1].legend(fontsize=8)
    fig.suptitle('5 | A small residual does not make the method result physical truth',fontsize=17,fontweight='bold');return save(fig,directory,'figure5_method_comparison')

def uncertainty(stability,controls,sensitivity,directory,field_boundary,trace_boundary):
    style();fig,ax=plt.subplots(2,2,figsize=(13,8.5),layout='constrained');names=list(stability);x=np.arange(len(names));errs=[stability[n]['complex_relative_l2'] for n in names];rs=[stability[n]['trace_R'] for n in names]
    ax[0,0].bar(x,errs,color=COLORS[0]);ax[0,0].axhline(field_boundary,color=COLORS[4],ls='--',label='Frozen field boundary');ax[0,0].set_xticks(x,names,rotation=35,ha='right');ax[0,0].set(ylabel='Aligned full-complex L2 error',title='A  Independent original seeds, widths and long runs');ax[0,0].legend(fontsize=8)
    ax[0,1].scatter(errs,np.array(rs)*100,c=COLORS[0],s=60);ax[0,1].axvline(field_boundary,color=COLORS[4],ls='--');ax[0,1].axhline(trace_boundary*100,color=COLORS[4],ls='--');ax[0,1].set(xlabel='Aligned complete-field error',ylabel='Trace R (%)',title='B  Coherent alternatives still differ in weak phase')
    cn=list(controls);xx=np.arange(len(cn));ax[1,0].bar(xx,[controls[n]['trace_R']*100 for n in cn],color=COLORS[1]);ax[1,0].axhline(trace_boundary*100,color=COLORS[4],ls='--',label='Frozen trace boundary');ax[1,0].set_xticks(xx,cn,rotation=28,ha='right');ax[1,0].set(ylabel='Trace R (%)',title='C  Independently computed poor controls');ax[1,0].legend(fontsize=8)
    for i,(name,result) in enumerate(sensitivity.items()):ax[1,1].semilogy(np.arange(1,len(result['trace_error'])+1),np.minimum.accumulate(result['trace_error']),color=COLORS[i],label=name)
    ax[1,1].set(xlabel='Iteration',ylabel='Running minimum trace R',title='D  Fresh global-step sensitivity (same200-step budget)');ax[1,1].legend(fontsize=8);ax[1,1].grid(True)
    fig.suptitle('6 | Moderate phase uncertainty and nonvacuous quality controls',fontsize=17,fontweight='bold');return save(fig,directory,'figure6_uncertainty')
