"""Author spectral support, integrated without discarding any band."""
from pathlib import Path
import numpy as np

def prepare(path=Path('data/raw_data.npz')):
    with np.load(path,allow_pickle=False) as data:
        spectra=data['spectra'][0];support=data['support_indices'][0]
        b=data['b_values'][0];ppm=data['chemical_shift'][0]
    groups=np.split(support,np.flatnonzero(np.diff(support)>1)+1)
    observations=np.array([spectra[g].sum(axis=0) for g in groups])
    noise=float(np.median(np.std(spectra[:3000],axis=0,ddof=1)))
    eta=np.array([np.hypot(1.05*np.sqrt(30)*noise*np.sqrt(len(g)),.001*np.linalg.norm(y)) for g,y in zip(groups,observations)])
    return {'spectra':spectra,'support':support,'groups':groups,'observations':observations,'b_values':b,'ppm':ppm,'eta':eta,'noise':noise}
