"""Actually computed physical-field comparisons for the active review notebook."""
import numpy as np
from .solvers import reconstruct

def field_metrics(result,truth):
    out={}
    for name,x,r in [('xx',result['strain'][0,0],truth['strain'][0,0]),('xy',result['strain'][0,1],truth['strain'][0,1]),('yy',result['strain'][0,2],truth['strain'][0,2]),('d0',result['lattice_spacing'][0],truth['lattice_spacing'][0])]:
        a=x.ravel();b=r.ravel();na=np.linalg.norm(a);out[name]=(float(a@b/(na*np.linalg.norm(b))) if na else 0.,float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
    return out

def sensitivity(data):
    cases={};records={}
    for exponent in [10,14]:
        result,info=reconstruct(data,svd_cutoff=10.**(-exponent));cases[str(exponent)]=result;records[str(exponent)]=info['history']
    return cases,records

def figures(data,truth,fresh,diagnostics,original,qualified,cases,histories):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    figs=[];extent=[0,20,-5,5];labels=['εxx (µstrain)','εxy (µstrain)','εyy (µstrain)','d0 (Å)']
    def title(fig,text):fig.suptitle(text,x=.02,ha='left',weight='bold',fontsize=15)
    def arrays(result):return [a*1e6 for a in result['strain'][0]]+[result['lattice_spacing'][0]]
    ref=arrays(truth)
    linear={'strain':original['linear_strain'],'lattice_spacing':np.full_like(truth['lattice_spacing'],4.056)}
    fig,ax=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');title(fig,'01  |  Nonlinear neutron transmission acquisition')
    for i in np.linspace(0,len(data['entry'])-1,90,dtype=int):ax[0].plot([data['entry'][i,0]*1000,data['exit'][i,0]*1000],[data['entry'][i,1]*1000,data['exit'][i,1]*1000],color='#176b87',alpha=.14,lw=.6)
    ax[0].plot([0,20,20,0,0],[-5,-5,5,5,-5],'k',lw=1);ax[0].scatter(data['traction_points'][:,0]*1000,data['traction_points'][:,1]*1000,s=5,color='#e39d29');ax[0].set(xlabel='x (mm)',ylabel='y (mm)',title='90 evenly sampled rays shown; all 1798 fitted\nKnown zero-traction boundaries',aspect='equal')
    angles=np.arctan2(data['direction'][:,1],data['direction'][:,0])*180/np.pi;sc=ax[1].scatter(angles,data['path_length']*1000,c=data['observations'],s=5,cmap='viridis');fig.colorbar(sc,ax=ax[1],label='Measured mean lattice spacing (Å)');ax[1].set(xlabel='Ray angle (degree)',ylabel='Path length (mm)',title='Absolute Bragg-edge measurements')
    residual=(diagnostics['prediction']-data['observations'])/4.056e-4;ax[2].hist(residual,bins=45,color='#176b87',alpha=.85);ax[2].set(xlabel='Prediction residual / known noise SD',ylabel='Number of rays',title=f'Fresh residual RMS {np.sqrt(np.mean(residual**2)):.3f}');figs.append(fig)
    rows=[('Author simulated truth',truth),('Original heuristic',original),('Constant-d0 linear GP',linear),('Qualified original',qualified),('Fresh local Newton',fresh)]
    fig,ax=plt.subplots(5,4,figsize=(14,8.8),layout='constrained');title(fig,'02  |  Joint fields and the cost of assuming constant lattice spacing\nTriangles indicate display saturation; all metrics use full values')
    for c in range(4):
        vmax=max(abs(ref[c].min()),abs(ref[c].max())) if c<3 else ref[c].max();vmin=-vmax if c<3 else ref[c].min()
        for r,(name,result) in enumerate(rows):
            image=ax[r,c].imshow(arrays(result)[c],origin='lower',extent=extent,cmap='RdBu_r' if c<3 else 'viridis',vmin=vmin,vmax=vmax,aspect='equal');ax[r,c].set_xticks([0,10,20]);ax[r,c].set_yticks([-5,0,5])
            if r==0:ax[r,c].set_title(labels[c])
            if c==0:ax[r,c].set_ylabel(name+'\ny (mm)',fontsize=8)
            if r==4:ax[r,c].set_xlabel('x (mm)')
        low=any(np.min(arrays(result)[c])<vmin for _,result in rows);high=any(np.max(arrays(result)[c])>vmax for _,result in rows)
        extend='both' if low and high else 'min' if low else 'max' if high else 'neither'
        fig.colorbar(image,ax=ax[:,c],shrink=.7,label=labels[c],extend=extend)
    figs.append(fig)
    fig,ax=plt.subplots(3,4,figsize=(14,5.9),layout='constrained');title(fig,'03  |  Signed physical-field error under one scale per quantity')
    error_rows=[('Original heuristic',original),('Constant-d0 linear GP',linear),('Fresh qualified Newton',fresh)]
    for c in range(4):
        errors=[arrays(result)[c]-ref[c] for _,result in error_rows];lim=max(np.max(abs(a)) for a in errors)
        for r,((name,_),error) in enumerate(zip(error_rows,errors)):
            im=ax[r,c].imshow(error,origin='lower',extent=extent,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='equal')
            ax[r,c].text(.025,.055,f'RMS {np.sqrt(np.mean(error**2)):.2g}',transform=ax[r,c].transAxes,fontsize=7,bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1})
            if r==0:ax[r,c].set_title(labels[c])
            if c==0:ax[r,c].set_ylabel(name+'\ny (mm)',fontsize=8)
            if r==2:ax[r,c].set_xlabel('x (mm)')
        fig.colorbar(im,ax=ax[:,c],shrink=.7,label='Signed error: '+labels[c])
    figs.append(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');title(fig,'04  |  Physical accuracy and original-to-local field parity')
    names=['xx','xy','yy','d0'];positions=np.arange(4);width=.24
    for offset,(name,result,color) in enumerate([('Original heuristic',original,'#e39d29'),('Constant-d0 GP',linear,'.65'),('Fresh qualified',fresh,'#176b87')]):
        m=field_metrics(result,truth);ax[0].bar(positions+(offset-1)*width,[m[k][1] for k in names],width,label=name,color=color);ax[1].bar(positions+(offset-1)*width,[m[k][0] for k in names],width,color=color)
    ax[0].set(xticks=positions,xticklabels=names,ylabel='Range NRMSE',title='All four unknown fields are evaluated');ax[0].legend(fontsize=7,frameon=False)
    ax[1].set(xticks=positions,xticklabels=names,ylabel='Cosine NCC',title='No shift, scale or component alignment')
    for c,(x,z) in enumerate(zip(arrays(fresh),arrays(qualified))):
        error=np.sqrt(np.mean((x-z)**2))/np.ptp(z);ax[2].bar(names[c],error,color='#176b87')
    ax[2].set(yscale='log',ylabel='Range NRMSE versus qualified original',title='Full physical-field parity');ax[2].axhline(2e-5,color='#b44c66',ls='--',label='Frozen parity boundary');ax[2].legend(fontsize=8,frameon=False);figs.append(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');title(fig,'05  |  Objective descent, numerical rank and geometry checks')
    ax[0].semilogy(np.arange(len(qualified['history'])), -qualified['history'],'o--',color='#e39d29',label='Qualified original');ax[0].semilogy(diagnostics['history'][:,0],diagnostics['history'][:,1],'.-',color='#176b87',label='Fresh local');ax[0].set(xlabel='Newton update',ylabel='Negative log objective',title='Same Eq19 objective and line search');ax[0].legend(frameon=False)
    s=diagnostics['first']['singular_values'];ax[1].semilogy(np.arange(1,len(s)+1),s/s[0],color='#176b87');ax[1].axhline(1e-12,color='#b44c66',ls='--');ax[1].set(xlabel='Dimensionless singular mode',ylabel='Relative singular value',title=f"Fixed initial coordinate scale; rank {diagnostics['first']['rank']}")
    e=data['entry'];x=data['exit'];n=data['direction'];length=data['path_length'];ax[2].hist(np.linalg.norm(x-e-n*length[:,None],axis=1)*1e12,bins=30,color='#176b87');ax[2].set(xlabel='Endpoint/direction closure error (pm)',ylabel='Number of rays',title='Independent geometric consistency');figs.append(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');title(fig,'06  |  Rank sensitivity: include the failed numerical convention')
    xs=data['grid_x'][0]*1000;row=20
    for key,color,label in [('10','#b44c66','Cutoff 1e-10 (failed quality)'),('12','#176b87','Selected cutoff 1e-12'),('14','#e39d29','Cutoff 1e-14')]:
        result=fresh if key=='12' else cases[key];ax[0].plot(xs,result['lattice_spacing'][0,row],color=color,label=label);ax[1].plot(xs,result['strain'][0,1,row]*1e6,color=color);m=field_metrics(result,truth);ax[2].plot(positions,[m[k][1] for k in names],'o-',color=color,label=label)
    ax[0].plot(xs,truth['lattice_spacing'][0,row],'k--',label='Truth');ax[1].plot(xs,truth['strain'][0,1,row]*1e6,'k--')
    ax[0].set(xlabel='x (mm)',ylabel='Stress-free lattice spacing (Å)',title='Central-row concentration-related field');ax[0].legend(fontsize=7,frameon=False)
    ax[1].set(xlabel='x (mm)',ylabel='Shear strain (µstrain)',title='Same native basis and observations')
    ax[2].set(xticks=positions,xticklabels=names,ylabel='Range NRMSE',title='Changing numerical rank changes bias');figs.append(fig)
    return figs
