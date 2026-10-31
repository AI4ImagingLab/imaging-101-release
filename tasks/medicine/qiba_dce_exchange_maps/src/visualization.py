"""Actual scientific figure export, without selecting a plotting backend."""
from pathlib import Path
import hashlib,json
import numpy as np

def export_figure(fig,directory,name,numeric):
    """Save PNG/SVG/PDF and paired NPZ, with SHA256 provenance JSON."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    files=[]
    for suffix in ('png','svg','pdf'):
        path=directory/(name+'.'+suffix)
        fig.savefig(path,dpi=150,bbox_inches='tight')
        files.append(path)
    path=directory/(name+'.npz');np.savez_compressed(path,**numeric);files.append(path)
    records={f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files}
    (directory/(name+'.json')).write_text(json.dumps({'figure':name,'numeric_consumer':'paired figure panels and plotted experiments','files':records},indent=2),encoding='utf8')
    return records

def field_image(ax,values,title,units,limits=None,signed=False):
    """Display a native raster and correctly scaled colorbar."""
    import matplotlib.pyplot as plt
    values=np.asarray(values)
    kwargs={}
    if signed:
        bound=float(np.max(abs(values))) or 1.;kwargs=dict(vmin=-bound,vmax=bound,cmap='RdBu_r')
    else:
        kwargs=dict(cmap='viridis')
        if limits is not None:kwargs.update(vmin=limits[0],vmax=limits[1])
    im=ax.imshow(values,origin='upper',**kwargs)
    ax.set_title(title,fontsize=10);ax.set_xlabel('Native column');ax.set_ylabel('Native row')
    plt.colorbar(im,ax=ax,shrink=.75,label=units)
    return im
