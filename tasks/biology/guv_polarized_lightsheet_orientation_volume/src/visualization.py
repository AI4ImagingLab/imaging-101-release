"""Maintainer scientific diagnostics and traceable figure export, without backend changes."""
from pathlib import Path
import hashlib,json
import numpy as np
from .physics_model import real_harmonics

def coefficient_metrics(estimate,reference):
    """Return independent per-SH cosine NCC, dynamic-range NRMSE and maxabs."""
    rows=[]
    for j in range(15):
        x=np.asarray(estimate[...,j],float).ravel();r=np.asarray(reference[...,j],float).ravel()
        if np.ptp(r)==0 or np.linalg.norm(r)==0:raise ValueError('Degenerate reference')
        error=x-r;norm=np.linalg.norm(x)
        rows.append({'ncc':float(x@r/norm/np.linalg.norm(r)) if norm else 0.,'nrmse':float(np.sqrt(np.mean(error*error))/np.ptp(r)),'max_abs':float(np.max(abs(error)))})
    return rows

def angular_samples(number=256):
    """Return deterministic Fibonacci sphere directions and15-column SH values."""
    if number<16:raise ValueError('At least16 quadrature-display directions required')
    index=np.arange(number);z=1-2*(index+.5)/number;phi=index*np.pi*(3-np.sqrt(5));theta=np.arccos(z)
    directions=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),z],axis=-1)
    return directions,real_harmonics(theta,phi)

def orientation_diagnostics(field,number=256):
    """Calculate GFA, sampled ODF peaks/negative-lobe diagnostics without clipping."""
    directions,Y=angular_samples(number);angular=np.asarray(field)@Y.T
    power=np.sum(np.asarray(field)**2,axis=-1)
    fraction=np.divide(field[...,0]**2,power,out=np.ones_like(power),where=power!=0)
    gfa=np.sqrt(np.maximum(0,1-fraction))
    return {'gfa':gfa,'peak_direction':directions[np.argmax(angular,axis=-1)],'negative_fraction':np.mean(angular<0,axis=-1),'angular_minimum':np.min(angular,axis=-1),'directions':directions,'harmonics':Y}

def save_figure(fig,directory,name,numerics,description):
    """Export actual computed figure PNG/SVG/PDF plus paired numeric NPZ+JSON."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);files={}
    for ext in ['png','svg','pdf']:
        dest=directory/(name+'.'+ext);fig.savefig(dest,dpi=160,bbox_inches='tight')
        files[ext]={'file':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    numeric=directory/(name+'.npz');np.savez_compressed(numeric,**numerics)
    files['numeric']={'file':numeric.name,'sha256':hashlib.sha256(numeric.read_bytes()).hexdigest()}
    record={'name':name,'description':description,'source':'Fresh active local notebook computation unless upstream arrays are explicitly named','artifacts':files}
    (directory/(name+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf8')
    return record
