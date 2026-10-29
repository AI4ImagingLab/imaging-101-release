import numpy as np
import matplotlib.pyplot as plt
def metrics(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    return dict(ncc=float(a@b/np.linalg.norm(a)/np.linalg.norm(b)),nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
def image_comparison(data,observed,original_observed):
    fig,axes=plt.subplots(3,4,figsize=(13,9),layout='constrained');extent=[15.72,-15.72,-15.72,15.72]
    for i,band in enumerate('grz'):
        residual=(data['images'][i]-observed[i])*np.sqrt(data['inverse_variance'][i]);limit=np.percentile(data['images'][i],99.5)
        for j,(x,title)in enumerate(zip([data['images'][i],original_observed[i],observed[i],residual],[band+' measured',band+' original convolved',band+' fresh convolved',band+' normalized residual'])):
            a=axes[i,j];masked=np.ma.array(x,mask=data['mask'][0]);im=a.imshow(masked,origin='lower',extent=extent,cmap='RdBu_r'if j==3 else'magma',vmin=-5 if j==3 else 0,vmax=5 if j==3 else limit)
            a.set(title=title,xlabel='East offset(arcsec)',ylabel='North offset(arcsec)');fig.colorbar(im,ax=a,label='Residual/RMS'if j==3 else'Flux(nanomaggies/pixel)')
    fig.suptitle('Legacy Surveys / D. Lang (Perimeter Institute)\nMasked observed neighbors; images99.5% display range; residual display ±5RMS')
    return fig
