"""Full native measured THz time traces in calibrated photocurrent units."""
from pathlib import Path
import numpy as np
DT_PS=.05003462406015038
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as f:waveforms=f['waveforms']
    if waveforms.ndim!=3 or waveforms.shape[0]!=1 or not np.isfinite(waveforms).all():raise ValueError('Expected finite batch-first waveforms')
    return waveforms[0]
