"""Public measured band data and independent opacity/chemistry calibration grids."""
from pathlib import Path
import numpy as np

PARAMETERS=('Rs','Mp','Rp','T','log_scatt_factor','logZ','log_cloudtop_P','error_multiple')
SPECIES=('H','He','C','N','O','Na','K','H2','N2','O2','OH','HF','MgH','SiH','HCl','CO','NO','SiO','TiO','VO','HCN','H2CO','CH4','OCS','CO2','C2H2','C2H4','H2O','H2S','NH3','PH3','NO2','SO2','O3')

def load_observations(path):
    """Return independent arrays with the single benchmark batch axis removed."""
    with np.load(Path(path),allow_pickle=False) as z:
        result={k:z[k][0].copy() for k in z.files}
    return result

def bracket_grid(target,grid,max_cutoff=np.inf):
    """Original inclusive enclosing calibration nodes, including cutoff boundary."""
    condition=np.zeros(len(grid),dtype=bool);start=None;end=None
    for i in range(len(grid)):
        if start is None:
            if grid[i]>np.min(target):start=i-1
            if grid[i]==np.min(target):start=i
        if end is None:
            if grid[i]>=np.max(target) or grid[i]>=max_cutoff:end=i+1
    condition[start:end]=True
    return condition
