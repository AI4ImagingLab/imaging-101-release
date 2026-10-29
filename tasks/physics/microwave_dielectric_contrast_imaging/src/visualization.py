"""Actual BIM comparisons; historical author runs are labeled separately."""
from pathlib import Path
import json,time
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct
def ncc(x,y):
    reference_norm=np.linalg.norm(y);estimate_norm=np.linalg.norm(x)
    if reference_norm==0:raise ValueError('Undefined zero reference norm')
    return 0. if estimate_norm==0 else float(np.vdot(x.ravel(),y.ravel()).real/(estimate_norm*reference_norm))
def nrmse(x,y):
    span=float(y.max()-y.min())
    if span<=0:raise ValueError('Undefined constant reference dynamic range')
    return float(np.sqrt(np.mean((x-y)**2))/span)
def run_experiments(data,truth,upstream,historical):
    results=[];stages=[];table=[];runtimes=[]
    for alpha in [.001,.01,.1]:
        start=time.perf_counter();output,stage=reconstruct(data,alpha=alpha);elapsed=time.perf_counter()-start
        image=output['reconstruction'][0];results.append(image);stages.append(stage);runtimes.append(elapsed)
        table.append([alpha,ncc(image,truth),nrmse(image,truth),stage['physical_residual'][-1],elapsed])
        print('Fresh alpha',alpha,'completed in',round(elapsed,3),'s; physical residual',stage['physical_residual'][-1],flush=True)
    nominal=results[0];stage=stages[0];chi=nominal.ravel();predicted=stage['GS']@(chi[:,None]*stage['final_total_field'])
    historical_errors=historical['relative_permittivity_percent'];representatives=[]
    for k in range(3):
        order=np.argsort(historical_errors[k]);index=order[len(order)//2];representatives.append(historical['contrast_images'][k,index])
    return {'results':np.asarray(results),'stages':stages,'table':np.asarray(table),'truth':truth,'first_born':stage['complex_contrasts'][0].real.reshape(30,30),'predicted':predicted,'historical_errors':historical_errors,'historical_representatives':np.asarray(representatives),'metrics':{'nominal_ncc':ncc(nominal,truth),'nominal_nrmse':nrmse(nominal,truth),'physical_observation_residual':float(stage['physical_residual'][-1]),'final_state_residual':float(stage['state_residual'][-1]),'max_difference_from_original':float(np.max(abs(nominal-upstream))),'first_born_ncc':ncc(stage['complex_contrasts'][0].real.reshape(30,30),truth),'first_born_nrmse':nrmse(stage['complex_contrasts'][0].real.reshape(30,30),truth)}}
def make_figures(data,e):
    figs=[];extent=[-.4,.4,-.4,.4];angles=data['receiver_angles'];illum=data['illumination_angles']
    fig,ax=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    ax[0,0].plot(np.cos(angles),np.sin(angles),'o',label='9 receiver positions');ax[0,0].plot([-.4,.4,.4,-.4,-.4],[-.4,-.4,.4,.4,-.4],label='Image support');ax[0,0].set(title='Calibrated acquisition geometry',xlabel='x (m)',ylabel='y (m)',aspect='equal');ax[0,0].legend()
    ax[0,1].quiver(np.zeros(10),np.zeros(10),np.cos(illum),np.sin(illum),angles='xy',scale_units='xy',scale=1);ax[0,1].set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),title='10 plane-wave propagation directions',xlabel='Normalized x direction',ylabel='Normalized y direction',aspect='equal')
    im=ax[1,0].imshow(abs(data['measurements']),aspect='auto',origin='lower');ax[1,0].set(title='Existing author simulated observations',xlabel='Illumination index',ylabel='Receiver index');fig.colorbar(im,ax=ax[1,0],label='Scattered field amplitude / E0')
    im=ax[1,1].imshow(np.angle(data['measurements']),aspect='auto',origin='lower',cmap='twilight',vmin=-np.pi,vmax=np.pi);ax[1,1].set(title='Retained complex observation phase',xlabel='Illumination index',ylabel='Receiver index');fig.colorbar(im,ax=ax[1,1],label='Phase (rad)');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    panels=[('Actual30-cell mean truth',e['truth']),('Fresh first-Born linear update',e['first_born']),('Fresh15-step BIM, alpha0.001',e['results'][0]),('Fresh BIM minus actual truth',e['results'][0]-e['truth']),('Fresh15-step BIM, alpha0.01',e['results'][1]),('Fresh15-step BIM, alpha0.1',e['results'][2])]
    lower=min(-.15,float(e['results'].min()),float(e['first_born'].min()));upper=max(1.,float(e['results'].max()),float(e['first_born'].max()))
    for index,(a,(title,image)) in enumerate(zip(ax.flat,panels)):
        if index==3:
            bound=max(.1,float(np.max(abs(image))));im=a.imshow(image,origin='lower',extent=extent,cmap='RdBu_r',vmin=-bound,vmax=bound)
        else:im=a.imshow(image,origin='lower',extent=extent,cmap='viridis',vmin=lower,vmax=upper)
        a.set(title=title,xlabel='x (m)',ylabel='y (m)');fig.colorbar(im,ax=a,label='Relative dielectric contrast, dimensionless')
    figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for alpha,stage in zip([.001,.01,.1],e['stages']):
        step=np.arange(1,16);label=f'alpha {alpha}'
        ax[0,0].semilogy(step,stage['legacy_objective'],'o-',ms=3,label=label)
        ax[0,1].semilogy(step,stage['physical_residual'],'o-',ms=3,label=label)
        images=stage['complex_contrasts'].real.reshape(15,30,30)
        ax[1,0].plot(step,[ncc(x,e['truth']) for x in images],'o-',ms=3,label=label)
        ax[1,1].plot(step,[nrmse(x,e['truth']) for x in images],'o-',ms=3,label=label)
        ax[0,2].plot(step,[len(x) for x in stage['inner_history']],'o-',ms=3,label=label)
        ax[1,2].semilogy(step,stage['state_residual'],'o-',ms=3,label=label)
    settings=[('Legacy complex-intermediate diagnostic','Squared unnormalized field mismatch'),('Physical REAL contrast consistency','Relative observation residual'),('Actual inner solver work','Inner iterations'),('Actual-truth image cosine','Uncentered NCC'),('Actual-truth spatial error','Dynamic-range NRMSE'),('Independent dense state-equation check','Relative state residual')]
    for a,(title,ylabel) in zip(ax.flat,settings):a.set(title=title,xlabel='Outer BIM update',ylabel=ylabel);a.legend(fontsize=8)
    ax[0,1].axhline(.0255218106788596,color='black',ls='--',lw=.8,label='Fixed criterion');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(15,9),constrained_layout=True);stage=e['stages'][0]
    field=stage['final_total_field'][:,0].reshape(30,30);incident=stage['incident'][:,0].reshape(30,30)
    limit=float(max(np.max(abs(field.real)),np.max(abs(incident.real))))
    for a,title,value in [(ax[0,0],'Incident real field, illumination0',incident.real),(ax[0,1],'Fresh internal real total field',field.real),(ax[0,2],'Fresh internal imaginary total field',field.imag)]:
        im=a.imshow(value,origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit);a.set(title=title,xlabel='x (m)',ylabel='y (m)');fig.colorbar(im,ax=a,label='Electric field / E0')
    measured=data['measurements'];prediction=e['predicted'];limit=float(max(np.max(abs(measured.real)),np.max(abs(prediction.real))))
    for a,title,value in [(ax[1,0],'Observed scattered real field',measured.real),(ax[1,1],'Fresh REAL-contrast forward prediction',prediction.real)]:
        im=a.imshow(value,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit);a.set(title=title,xlabel='Illumination index',ylabel='Receiver index');fig.colorbar(im,ax=a,label='Scattered field / E0')
    residual=prediction-measured;limit=float(max(np.max(abs(residual.real)),np.max(abs(residual.imag))));im=ax[1,2].imshow(residual.real,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit);ax[1,2].set(title='Physical forward minus observed, real',xlabel='Illumination index',ylabel='Receiver index');fig.colorbar(im,ax=ax[1,2],label='Scattered field / E0');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for k,title in enumerate(['DE','PSO','GA']):
        im=ax[0,k].imshow(e['historical_representatives'][k],origin='lower',extent=extent,cmap='viridis',vmin=0,vmax=1);ax[0,k].set(title=title+' cached author median-ranked run\nHistorical native60 display, NOT reproduced',xlabel='x (m)',ylabel='y (m)');fig.colorbar(im,ax=ax[0,k],label='Author cached contrast, dimensionless')
    ax[1,0].boxplot(e['historical_errors'].T,tick_labels=['DE','PSO','GA']);ax[1,0].set(title='Cached author30-run groups only\nDifferent metric/grid; not fresh BIM comparison',ylabel='Author relative-permittivity percent error')
    ax[1,1].bar(['0.001','0.01','0.1'],e['table'][:,1]);ax[1,1].set(title='Fresh BIM actual-truth image cosine',xlabel='Tikhonov alpha',ylabel='Uncentered NCC')
    ax[1,2].bar(['0.001','0.01','0.1'],e['table'][:,2]);ax[1,2].set(title='Fresh BIM actual-truth spatial error',xlabel='Tikhonov alpha',ylabel='Dynamic-range NRMSE');figs.append(fig)
    return figs
def export(figures,experiments,directory):
    path=Path(directory);path.mkdir(exist_ok=True,parents=True)
    for i,figure in enumerate(figures,1):
        for ext in ['png','svg','pdf']:figure.savefig(path/f'figure_{i}.{ext}',dpi=140)
    values={'fresh_results':experiments['results'],'truth':experiments['truth'],'first_born':experiments['first_born'],'parameter_table':experiments['table'],'physical_prediction':experiments['predicted'],'historical_errors':experiments['historical_errors'],'historical_representatives':experiments['historical_representatives']}
    for i,stage in enumerate(experiments['stages']):
        for key in ['legacy_objective','physical_residual','state_residual','complex_contrasts','final_total_field','incident']:values[f'alpha{i}_{key}']=stage[key]
        values[f'alpha{i}_inner_counts']=np.asarray([len(x) for x in stage['inner_history']])
    np.savez_compressed(path/'numeric_sources.npz',**values)
    (path/'metrics.json').write_text(json.dumps(experiments['metrics'],indent=2),encoding='utf8')
