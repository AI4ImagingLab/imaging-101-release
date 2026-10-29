"""Scientific figure export helpers; importing this module never selects a backend."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt

def export_figure(fig,directory,name,numeric,caption):
    """Export the computed figure in PNG/SVG/PDF and its named numeric sources."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    fig.suptitle(caption,fontsize=13);fig.tight_layout(rect=(0,0,1,.95))
    for suffix in ['png','svg','pdf']:fig.savefig(directory/(name+'.'+suffix),dpi=150,bbox_inches='tight')
    np.savez_compressed(directory/(name+'_numeric.npz'),**numeric)
    record=dict(caption=caption,consumer='active reconstruction notebook; '+name,files={})
    for path in directory.glob(name+'*'):
        if path.is_file():record['files'][path.name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    (directory/(name+'_source.json')).write_text(json.dumps(record,indent=2),encoding='utf8')
    return directory/(name+'.png')

def image_panel(ax,image,title,extent=None,signed=False,limit=None):
    """Display every supplied sample, with symmetric limits for signed values."""
    if signed:
        bound=float(np.max(np.abs(image))) if limit is None else float(limit)
        bound=max(bound,1e-16);kwargs=dict(cmap='RdBu_r',vmin=-bound,vmax=bound)
    else:kwargs=dict(cmap='magma',vmin=0,vmax=limit)
    im=ax.imshow(image,origin='lower',aspect='auto',extent=extent,**kwargs);ax.set_title(title,fontsize=10)
    return im
