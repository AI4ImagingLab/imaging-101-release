"""Six scientific figure groups from actually computed arrays; no backend selection."""
from pathlib import Path
import json,hashlib,textwrap
import numpy as np
import matplotlib.pyplot as plt

BLUE='#2563eb'
ORANGE='#d97706'
GRAY='#64748b'
SPECIES=('TAM','TEMPO')
EXTENT=(-1.515,1.485,-1.515,1.485)

def _finish(fig,title,caption):
 fig.suptitle(title,fontsize=16,fontweight='bold')
 fig.supxlabel(textwrap.fill(caption,width=int(fig.get_figwidth()*13)),fontsize=9,color='#334155')
 return fig

def _field(ax,image,title,cap=None):
 cap=float(np.max(abs(image))) if cap is None else cap
 image_object=ax.imshow(image,origin='lower',extent=EXTENT,cmap='RdBu_r',vmin=-cap,vmax=cap,interpolation='nearest')
 ax.set(title=title,xlabel='x (cm)',ylabel='y (cm)')
 ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1])
 return image_object

def acquisition_figure(data,reference):
 fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained')
 field=data['magnetic_field_G'];observed=data['projections'];cap=float(np.percentile(abs(observed),99.5))
 im=ax[0,0].imshow(observed,aspect='auto',extent=[field[0],field[-1],39.5,-.5],cmap='RdBu_r',vmin=-cap,vmax=cap)
 ax[0,0].set(title='A  All 40 measured synthetic spectra',xlabel='Homogeneous field (G)',ylabel='Projection index')
 fig.colorbar(im,ax=ax[0,0],label='Receiver signal (a.u.)',extend='both')
 for j in range(2):ax[0,1].plot(field,data['reference_spectra'][j]/1e6,color=(BLUE,ORANGE)[j],label=SPECIES[j],lw=1.4)
 ax[0,1].set(title='B  Real spectral response templates',xlabel='Homogeneous field (G)',ylabel='Receiver signal / relative amount (10⁶ a.u.)');ax[0,1].legend()
 g=data['gradients_G_cm'];ax[1,0].scatter(g[0],g[1],c=np.arange(40),cmap='viridis',s=24);ax[1,0].axhline(0,color='#cbd5e1',lw=.7);ax[1,0].axvline(0,color='#cbd5e1',lw=.7)
 ax[1,0].set(title='C  All calibrated gradient directions',xlabel='gx (G/cm)',ylabel='gy (G/cm)',aspect='equal')
 noise=reference['original_noise'].astype(float)/5000.;grid=np.linspace(-4.5,4.5,401)
 ax[1,1].hist(noise.ravel(),bins=70,density=True,histtype='step',lw=1.8,color=BLUE,label='Actual generated receiver noise')
 ax[1,1].plot(grid,np.exp(-grid**2/2)/np.sqrt(2*np.pi),color='black',ls='--',label='Known N(0,1)')
 ax[1,1].set(title='D  Known noise, independently identified',xlabel='Noise / 5000',ylabel='Probability density');ax[1,1].legend(fontsize=8)
 return _finish(fig,'01  Chemical signatures encode two spatial fields','Author-defined simulated observations, using real response templates; not real spatial measurements. Panel A display saturates only the outer 0.5% amplitudes; all numeric observations remain in every reconstruction and metric.')

def field_comparison_figure(truth,source,fresh):
 fig,ax=plt.subplots(2,4,figsize=(15,8),layout='constrained');cap=float(max(np.max(abs(truth)),np.max(abs(source)),np.max(abs(fresh))));errcap=float(np.max(abs(fresh-truth)))
 for j in range(2):
  for k,(image,label) in enumerate(((truth[j],'Analytical truth'),(source[j],'Original source: float64'),(fresh[j],'Fresh local signed TV'))):
   image_object=_field(ax[j,k],image,f'{SPECIES[j]} · {label}',cap=cap)
  error_object=_field(ax[j,3],fresh[j]-truth[j],f'{SPECIES[j]} · fresh − truth',cap=errcap)
 fig.colorbar(image_object,ax=ax[:,:3].ravel().tolist(),label='Relative amount / cm²',shrink=.86)
 fig.colorbar(error_object,ax=ax[:,3].tolist(),label='Signed error / cm²',shrink=.86)
 numerical=np.linalg.norm(fresh-source)/np.linalg.norm(source)
 return _finish(fig,'02  Complete fields: truth, original run and fresh reconstruction',f'All 20,000 density values are shown and evaluated. Shared symmetric density scale for both species/methods; error panels share their own full symmetric range. Fresh/original relative L2 difference = {numerical:.2e}.')

def method_maps_figure(review,names):
 fig,ax=plt.subplots(2,6,figsize=(18,7.5),layout='constrained');methods=review['method_reconstructions']
 short=['Signed TV','Nonnegative TV','TV weight ÷4','TV weight ×4','Independent fits','Backprojection']
 for j in range(2):
  for k in range(6):
   image_object=_field(ax[j,k],methods[k,j],f'{short[k]}\n{SPECIES[j]}',cap=1.25)
   minimum,maximum=float(methods[k,j].min()),float(methods[k,j].max())
   ax[j,k].text(.02,.02,f'full range\n[{minimum:.2g}, {maximum:.2g}]',transform=ax[j,k].transAxes,fontsize=7,va='bottom',bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
 fig.colorbar(image_object,ax=ax.ravel().tolist(),label='Relative amount / cm²',extend='both',shrink=.75)
 return _finish(fig,'03  Source coupling, prior strength and positivity matter','Every panel is freshly computed on the same complete primary observations. Common display limits ±1.25 saturate outliers (triangles); full ranges and untruncated metrics are retained. Independent fits omit chemical coupling; backprojection is initialization, not a converged inverse.')

def bias_and_metrics_figure(review,truth):
 fig,ax=plt.subplots(2,2,figsize=(14,9),layout='constrained');m=review['method_metrics'];x=np.arange(6);labels=['Signed','Nonneg','TV÷4','TV×4','Separate','Adjoint']
 for j in range(2):
  ax[0,0].bar(x+(j-.5)*.34,m[:,j,1],width=.34,color=(BLUE,ORANGE)[j],label=SPECIES[j])
  ax[0,1].bar(x+(j-.5)*.34,m[:,j,2]*100,width=.34,color=(BLUE,ORANGE)[j],label=SPECIES[j])
 ax[0,0].set_yscale('log');ax[0,0].set(title='A  Whole-field normalized error',ylabel='Dynamic-range NRMSE');ax[0,0].set_xticks(x,labels);ax[0,0].legend()
 ax[0,1].axhline(15,color='#b91c1c',ls='--',label='Fixed 15% amount gate')
 ax[0,1].set(title='B  Integrated-amount error',ylabel='Relative error (%)');ax[0,1].set_xticks(x,labels);ax[0,1].legend(fontsize=8)
 fields=review['method_reconstructions']
 for j in range(2):
  for index,style,word in ((0,'-','signed'),(1,'--','nonnegative')):
   background=np.sort(fields[index,j][truth[j]==0]);cdf=np.arange(1,len(background)+1)/len(background)
   ax[1,0].plot(background,cdf,color=(BLUE,ORANGE)[j],ls=style,lw=1.4,label=f'{SPECIES[j]} {word}')
 ax[1,0].axvline(0,color=GRAY,lw=.8);ax[1,0].set(xlim=(-.18,.25),title='C  Known zero-density background',xlabel='Estimated density (relative amount/cm²)',ylabel='Empirical cumulative fraction');ax[1,0].legend(fontsize=8,ncol=2)
 grid=(np.arange(100)-50)*.03;row=58
 for j in range(2):
  color=(BLUE,ORANGE)[j];ax[1,1].plot(grid,truth[j,row],color=color,ls='--',lw=1.1,label=f'{SPECIES[j]} truth')
  ax[1,1].plot(grid,fields[0,j,row],color=color,lw=1.4,label=f'{SPECIES[j]} signed')
  ax[1,1].plot(grid,fields[1,j,row],color=color,ls=':',lw=1.8,label=f'{SPECIES[j]} nonneg')
 ax[1,1].set(title='D  Same physical line, y = +0.24 cm',xlabel='x (cm)',ylabel='Density (relative amount/cm²)');ax[1,1].legend(fontsize=7,ncol=2)
 return _finish(fig,'04  Better-looking positivity can bias chemical amount','The nonnegative variant improves local RMS error but truncates background fluctuations and fails the unchanged TAM amount gate. Its failure is preserved. Stronger TV performs differently on this realization; the primary author configuration is not retuned using truth. Background CDF zoom is diagnostic only.')

def sensitivity_figure(review):
 fig,ax=plt.subplots(3,3,figsize=(13.5,11.5),layout='constrained')
 for j in range(2):
  for k,s in enumerate(review['noise_scales']):image_object=_field(ax[j,k],review['noise_reconstructions'][k,j],f'{SPECIES[j]} · noise {s:g}σ',cap=1.25)
 fig.colorbar(image_object,ax=ax[:2].ravel().tolist(),label='Relative amount / cm²',extend='both',shrink=.8)
 for j in range(2):
  color=(BLUE,ORANGE)[j]
  ax[2,0].plot(review['noise_scales'],review['noise_metrics'][:,j,1],marker='o',color=color,label=SPECIES[j])
  ax[2,1].plot(review['angular_counts'],review['angular_metrics'][:,j,1],marker='o',color=color,label=SPECIES[j])
  ax[2,2].plot(review['noise_scales'],review['noise_metrics'][:,j,2]*100,marker='o',color=color,label=SPECIES[j])
 ax[2,0].set(title='A  Noise sensitivity',xlabel='Known noise amplitude / σ',ylabel='Dynamic-range NRMSE');ax[2,0].legend()
 ax[2,1].set(title='B  Angular coverage sensitivity',xlabel='Number of directions',ylabel='Dynamic-range NRMSE');ax[2,1].set_xticks([10,20,40]);ax[2,1].legend()
 ax[2,2].set(title='C  Amount recovery sensitivity',xlabel='Known noise amplitude / σ',ylabel='Relative amount error (%)');ax[2,2].axhline(15,ls='--',color=GRAY);ax[2,2].legend()
 return _finish(fig,'05  Controlled noise and sampling experiments','All plotted fields were reconstructed in this review. Noise cases scale one fixed realization and do not provide bootstrap coverage. Angular cases retain the complete spatial field and spectral channels; the bound task uses all40 directions. Display caps are identified by colorbar triangles.')

def physics_and_residuals_figure(data,fresh,diagnostics,physics,physics_names,metrics):
 fig,ax=plt.subplots(2,3,figsize=(15,9),layout='constrained');field=data['magnetic_field_G'];projection=13
 ax[0,0].plot(field,physics['point_expected'][projection],color='black',ls='--',lw=2,label='Independent shifted-spectrum formula')
 ax[0,0].plot(field,physics['point_predicted'][projection],color=BLUE,lw=1,label='Fresh direct Fourier operator')
 ax[0,0].set(title='A  Known shifted positive point source',xlabel='Homogeneous field (G)',ylabel='Receiver signal (a.u.)');ax[0,0].legend(fontsize=7)
 errors=physics['errors'];thresholds=np.array([1e-10,1e-10,1e-10,1e-12,1e-5]);position=np.arange(5)
 ax[0,1].scatter(np.maximum(errors,1e-17),position,color=BLUE,label='Actual error',zorder=3);ax[0,1].scatter(thresholds,position,color=GRAY,marker='|',s=110,label='Frozen limit')
 ax[0,1].set(xscale='log',xlim=(1e-17,1e-4),title='B  Independent physics checks',xlabel='Relative discrepancy');ax[0,1].set_yticks(position,physics_names);ax[0,1].legend(fontsize=8)
 history=diagnostics['history'];ax[0,2].semilogy(history[:,0],history[:,1],color=BLUE);ax[0,2].axhline(1e-5,color=GRAY,ls='--')
 ax[0,2].set(title=f'C  Original stopping rule: {diagnostics["iterations"]} steps',xlabel='Iteration',ylabel='Relative primal update')
 energy=diagnostics['energy_history'];ax[1,0].plot(energy[:,0],energy[:,1]/energy[0,1],color=BLUE)
 ax[1,0].set(title='D  Actual objective trajectory',xlabel='Iteration',ylabel='Objective / first logged value')
 from .preprocessing import make_model
 residual=make_model(data).project(fresh)-data['projections'];normalized=residual.ravel()/5000.;grid=np.linspace(-4.5,4.5,301)
 ax[1,1].hist(normalized,bins=70,density=True,histtype='step',color=BLUE,lw=1.7,label='Fresh full observation residual')
 ax[1,1].plot(grid,np.exp(-grid**2/2)/np.sqrt(2*np.pi),color='black',ls='--',label='Known unit Gaussian')
 ratio=np.sqrt(np.mean(normalized**2));corr=np.corrcoef(residual[:,:-1].ravel(),residual[:,1:].ravel())[0,1]
 ax[1,1].set(title=f'E  RMS/σ={ratio:.4f}; lag-one r={corr:.4f}',xlabel='Full residual / 5000',ylabel='Probability density');ax[1,1].legend(fontsize=7)
 ax[1,2].axis('off');table=ax[1,2].table(cellText=[[f'{v:.5f}' for v in row] for row in metrics],rowLabels=SPECIES,colLabels=['NCC','NRMSE','Amount error'],cellLoc='center',loc='center');table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.5)
 ax[1,2].set_title('F  Fresh whole-field quality');ax[1,2].text(.05,.12,'No alignment, crop, positivity clipping\nor hidden-target calibration.\nBoth chemical fields are required.',transform=ax[1,2].transAxes,fontsize=9,color='#334155')
 return _finish(fig,'06  Mathematical consistency and known-noise observation fit','All physics checks and convergence traces were computed through local modules. High raw relative residual is expected for this known-noise acquisition; its actual RMS and correlation are tested independently. Source parity alone is not the scientific accuracy criterion.')

def save_group(fig,directory,name,arrays,metadata):
 """Write the paired publication figures and actual numeric sources."""
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 for extension in ('png','svg','pdf'):fig.savefig(directory/f'{name}.{extension}',dpi=220,bbox_inches='tight',facecolor='white')
 numeric=directory/f'{name}.npz';np.savez_compressed(numeric,**{key:np.asarray(value) for key,value in arrays.items()})
 index=dict(metadata);index['numeric_arrays']={k:dict(shape=list(np.asarray(v).shape),dtype=str(np.asarray(v).dtype)) for k,v in arrays.items()};index['numeric_sha256']=hashlib.sha256(numeric.read_bytes()).hexdigest();index['figure_files']=[f'{name}.{ext}' for ext in ('png','svg','pdf')]
 (directory/f'{name}.json').write_text(json.dumps(index,indent=2,allow_nan=False),encoding='utf-8')
