"""Computed scientific comparisons; no reconstruction caches enter experiments."""
import numpy as np
from scipy.optimize import lsq_linear
from .solvers import reconstruct
from .physics_model import second_difference

def metrics(x,reference):
    a=x.ravel();b=reference.ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)>0 else 0.,float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))

def experiments(data,truth):
    h=data['system_matrix'];obs=data['observations'];out={}
    for p in [.25,1.,4.]:out[f'prior_{p}']=reconstruct(h,obs,prior_factor=p)[0]
    for k in [1,10,199]:out[f'repeats_{k}']=reconstruct(h,obs[:k])[0]
    for budget in [100,1000,10000,100000]:out[f'budget_{budget}']=reconstruct(h,obs,iterations=budget)[0]
    d=second_difference(h.shape[1]);y=obs.mean(0);v=.01**2/len(obs);q=h.shape[1]/h.shape[0]*.01**2
    a=np.vstack((h/np.sqrt(v),d/np.sqrt(q)));b=np.r_[y/np.sqrt(v),np.zeros(d.shape[0])]
    # Independent generic bound-constrained linear least squares; review only.
    # BVLS solves the explicit whitened quadratic, never the selected CP pipeline.
    opt=lsq_linear(a,b,bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=2000)
    out['independent_qp']=opt.x;out['constant_air']=np.ones(h.shape[1]);out['zero']=np.zeros(h.shape[1])
    out['qp_record']=np.array([opt.success,opt.nit,2*opt.cost,opt.optimality])
    out['quality_table']=np.array([metrics(out[k],truth) for k in out if np.asarray(out[k]).shape==truth.shape])
    return out

def figures(data,truth,fresh,diagnostics,comparisons,upstream):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    depth=data['depth_nm'];h=data['system_matrix'];obs=data['observations'];eal=data['attenuation_length_nm'];y=obs.mean(0);figs=[]
    def title(fig,text):fig.suptitle(text,x=.02,ha='left',weight='bold',fontsize=15)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');title(fig,'01  |  Ten overlapping escape-depth kernels')
    im=ax[0].imshow(h,aspect='auto',extent=[depth[0],depth[-1],9.5,-.5],cmap='viridis',norm=LogNorm(vmin=h.min(),vmax=h.max()));fig.colorbar(im,ax=ax[0],label='Weight (µm³)');ax[0].set(xlabel='Depth (nm)',ylabel='Attenuation channel',title='Provided cylindrical calibration')
    for i in [0,3,6,9]:ax[1].plot(depth,h[i]/h[i].sum(),label=f'{eal[i]:.2f} nm')
    ax[1].set(xlabel='Depth (nm)',ylabel='Normalized spatial sensitivity',title='Broad, correlated depth probes');ax[1].legend(frameon=False)
    ax[2].errorbar(eal,y,yerr=obs.std(0,ddof=1),fmt='o',color='#176b87',capsize=3,label='Repeat mean ± repeat SD');ax[2].set(xlabel='Attenuation length (nm)',ylabel='Peak area (µm³)',title='All 199 noisy acquisitions');ax[2].legend(fontsize=8,frameon=False);figs.append(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');title(fig,'02  |  Fresh reconstruction and independently known concentration')
    for a in ax[:2]:
        a.plot(depth,truth,color='black',lw=2,label='Author simulated truth');a.plot(depth,upstream,'--',color='#e39d29',lw=2,label='Reproduced original CP');a.plot(depth,fresh,color='#176b87',lw=1.4,label='Fresh local CP');a.set(xlabel='Depth (nm)',ylabel='Relative chemical density')
    ax[0].set(title='All 201 samples are estimated');ax[0].legend(frameon=False,fontsize=8);ax[1].set(xlim=(-2,2),title='Interface detail; no fixed endpoints')
    ax[2].plot(depth,fresh-truth,color='#b44c66',label='Fresh − truth');ax[2].axhline(0,color='.5',lw=.7);ax[2].set(xlabel='Depth (nm)',ylabel='Signed density error',title=f'NCC {metrics(fresh,truth)[0]:.6f}  |  NRMSE {metrics(fresh,truth)[1]:.5f}');figs.append(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');title(fig,'03  |  Observation fit, spatial bias and source parity')
    ax[0].errorbar(eal,y,yerr=.01/np.sqrt(len(obs)),fmt='o',capsize=4,color='black',label='Mean ± known SEM');ax[0].plot(eal,h@fresh,'-',color='#176b87',label='Fresh prediction');ax[0].set(xlabel='Attenuation length (nm)',ylabel='Peak area (µm³)',title='Forward prediction');ax[0].legend(frameon=False)
    ax[1].axhline(0,color='.5');ax[1].bar(np.arange(10),(h@fresh-y)/(.01/np.sqrt(len(obs))),color='#b44c66');ax[1].set(xlabel='Attenuation channel',ylabel='Residual / known SEM',title='Whitened data residual')
    ax[2].plot(depth,fresh-upstream,color='#176b87');ax[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax[2].set(xlabel='Depth (nm)',ylabel='Fresh − reproduced original',title=f'Max absolute parity error {np.max(abs(fresh-upstream)):.2e}');figs.append(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');title(fig,'04  |  Iteration budget and independent quadratic optimum')
    history=diagnostics['history'];ax[0].loglog(history[:,0],history[:,1],color='#176b87');ax[0].set(xlabel='CP updates',ylabel='Whitened objective',title='Full selected 100000-update run')
    budgets=[100,1000,10000,100000];err=[metrics(comparisons[f'budget_{k}'],truth)[1] for k in budgets];ax[1].semilogx(budgets,err,'o-',color='#b44c66');ax[1].set(xlabel='CP updates',ylabel='Range NRMSE against truth',title='Early stopping can preserve bias')
    ax[2].plot(depth,truth,'k',label='Truth');ax[2].plot(depth,fresh,color='#176b87',label='Selected original CP');ax[2].plot(depth,comparisons['independent_qp'],'--',color='#e39d29',label='Independent same-objective QP');ax[2].set(xlabel='Depth (nm)',ylabel='Relative density',title='Optimizer comparison, same input');ax[2].legend(frameon=False,fontsize=8);figs.append(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');title(fig,'05  |  Sensitivity to prior strength and number of repeats')
    colors=['#e39d29','#176b87','#b44c66']
    for p,c in zip([.25,1.,4.],colors):ax[0].plot(depth,comparisons[f'prior_{p}'],color=c,label=f'Prior covariance × {p:g}')
    ax[0].plot(depth,truth,'k--',lw=1,label='Truth');ax[0].set(xlabel='Depth (nm)',ylabel='Relative density',title='All runs use 100000 CP updates');ax[0].legend(frameon=False,fontsize=8)
    for k,c in zip([1,10,199],colors):ax[1].plot(depth,comparisons[f'repeats_{k}'],color=c,label=f'{k} repeats')
    ax[1].plot(depth,truth,'k--',lw=1);ax[1].set(xlabel='Depth (nm)',ylabel='Relative density',title='Leading acquisitions; no selection');ax[1].legend(frameon=False)
    labels=['1','10','199'];ax[2].bar(labels,[metrics(comparisons[f'repeats_{k}'],truth)[1] for k in [1,10,199]],color=colors);ax[2].set(xlabel='Number of noisy repeats',ylabel='Range NRMSE',title='Noise reduction versus prior bias');figs.append(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');title(fig,'06  |  Why spatial regularity is needed')
    u,s,v=np.linalg.svd(h,full_matrices=False);ax[0].semilogy(np.arange(1,11),s/s[0],'o-',color='#176b87');ax[0].set(xlabel='Singular mode',ylabel='Relative singular value',title=f'10 observations for 201 unknowns; κ={s[0]/s[-1]:.1e}')
    for i,c in zip([0,3,9],colors):ax[1].plot(depth,v[i],color=c,label=f'Mode {i+1}')
    ax[1].set(xlabel='Depth (nm)',ylabel='Right singular-vector amplitude',title='Weak modes oscillate in depth');ax[1].legend(frameon=False)
    names=['Fresh CP','QP','Constant 1','Zero'];values=[metrics(a,truth)[1] for a in [fresh,comparisons['independent_qp'],comparisons['constant_air'],comparisons['zero']]];bars=ax[2].bar(names,values,color=['#176b87','#e39d29','.6','.8']);ax[2].tick_params(axis='x',rotation=20);ax[2].set(ylabel='Range NRMSE',title='Nontrivial reconstruction versus controls')
    for b,val in zip(bars,values):ax[2].text(b.get_x()+b.get_width()/2,val+.01,f'{val:.3f}',ha='center',fontsize=9)
    figs.append(fig);return figs
