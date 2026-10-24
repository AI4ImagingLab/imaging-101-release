"""Scientific plotting helpers; importing this module never selects a backend."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt

def depth_image(ax,depth,mask,title,vmin=None,vmax=None):
    """Plot the full sensor region; exclude only observation-unsupported pixels."""
    value=np.ma.array(depth,mask=~np.asarray(mask,dtype=bool))
    im=ax.imshow(value,cmap='viridis',vmin=vmin,vmax=vmax,interpolation='nearest')
    ax.set(title=title,xlabel='ROI column [pixel]',ylabel='ROI row [pixel]')
    return im

def export_figure(fig,directory,name,arrays,description):
    """Write actual figure and its numeric inputs/provenance to private review storage.

    Numeric inputs must be finite except explicitly identified internal losses
    outside observation support. No missing experiment is fabricated here.
    """
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    paths={}
    for ext in ['png','svg','pdf']:
        path=directory/(name+'.'+ext);fig.savefig(path,dpi=135,bbox_inches='tight')
        paths[ext]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    numeric=directory/(name+'.npz');np.savez_compressed(numeric,**arrays)
    record={'name':name,'description':description,'files':paths,'numeric_file':numeric.name,
            'numeric_sha256':hashlib.sha256(numeric.read_bytes()).hexdigest(),
            'array_shapes':{k:list(np.asarray(v).shape) for k,v in arrays.items()},
            'consumer':'canonical active reconstruction notebook; embedded figure and maintainer report'}
    (directory/(name+'.json')).write_text(json.dumps(record,indent=2),encoding='utf8')
    return record
