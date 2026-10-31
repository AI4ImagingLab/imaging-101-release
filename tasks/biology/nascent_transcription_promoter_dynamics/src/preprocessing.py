"""Real measured trace packing, preserving every author-selected frame."""
from pathlib import Path
import numpy as np
def load_data(path=None):
    path=Path(path) if path else Path(__file__).resolve().parents[1]/'data/raw_data.npz'
    with np.load(path,allow_pickle=False) as z:values=z['fluorescence'][0].copy();lengths=z['lengths'][0].copy()
    traces=[values[i,:int(n)].reshape(1,-1) for i,n in enumerate(lengths)]
    return values,lengths,{'Processed Signals':traces,'Matrix Mean':np.mean([x.mean() for x in traces]),'Matrix Max':max(x.max() for x in traces),'Signal Lengths':np.unique(lengths)}
