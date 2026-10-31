"""Load only public experimental observations and average author repeats."""
from pathlib import Path
import numpy as np

def load_data(path=Path('data/raw_data.npz')):
    with np.load(path, allow_pickle=False) as data:
        return {key:data[key][0] for key in data.files}

def preprocess(data):
    """Return mean concentrations/charge at independently averaged times."""
    c = compensated_mean(data['concentration_repeats'])
    q = compensated_mean(data['charge_repeats'])
    initial = np.r_[c[0,1:],np.zeros(10)]
    return c,q,initial

def compensated_mean(values):
    """Pandas groupby mean uses Kahan accumulation in original row order."""
    total=np.zeros(values.shape[1:],dtype=np.float64)
    compensation=np.zeros_like(total)
    for row in values:
        y=row-compensation
        updated=total+y
        compensation=(updated-total)-y
        total=updated
    return total/len(values)
