"""Scientific review plots; importing never selects a plotting backend."""
from pathlib import Path
import numpy as np

def metric_pair(estimate,reference):
    """Return uncentered cosine NCC and reference-range NRMSE."""
    u=np.asarray(estimate,dtype=np.float64).ravel();v=np.asarray(reference,dtype=np.float64).ravel()
    dynamic=np.max(v)-np.min(v);denominator=np.linalg.norm(u)*np.linalg.norm(v)
    if dynamic<=0 or np.linalg.norm(v)==0:raise ValueError('Degenerate reference')
    return float(np.dot(u,v)/denominator) if denominator>0 else 0.,float(np.sqrt(np.mean((u-v)**2))/dynamic)

def summarize(maps,reference):
    """Return separately named metrics for both required physical fields."""
    result={}
    for key in ['reconstruction','arterial_transit_time']:
        ncc,nrmse=metric_pair(maps[key],reference[key]);result[key+'_ncc']=ncc;result[key+'_nrmse']=nrmse
    return result

def save_figure(fig,directory,stem):
    """Write PNG (180 dpi), SVG and PDF to explicit directory, or do nothing."""
    if directory is None:return
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for extension in ['png','svg','pdf']:fig.savefig(directory/f'{stem}.{extension}',dpi=180,bbox_inches='tight')

def map_panel(array,title,unit,vmin=None,vmax=None,cmap='viridis'):
    """Create (fig,axes) for a batch-first five-slice physical field.

    Bounds and palette affect display only; the numerical map is untouched.
    """
    import matplotlib.pyplot as plt
    a=np.asarray(array)[0];fig,axes=plt.subplots(1,a.shape[0],figsize=(13,3.1),layout='constrained')
    for z,ax in enumerate(np.atleast_1d(axes)):
        im=ax.imshow(a[z],vmin=vmin,vmax=vmax,cmap=cmap,origin='lower');ax.set_title(f'z = {z} mm');ax.set_xlabel('x [mm]')
    axes[0].set_ylabel('y [mm]');fig.colorbar(im,ax=np.atleast_1d(axes).tolist(),label=unit,shrink=.8);fig.suptitle(title)
    return fig,axes
