#!/usr/bin/env python
"""Print public reconstruction diagnostics without requiring hidden labels."""
from pathlib import Path
import sys
import numpy as np

out = Path('outputs/case_000.npz')
raw_path = Path('public/visible_data/cases/case_000/input_data/raw_data.npz')
if not out.exists():
    sys.exit('Missing outputs/case_000.npz')
recon = np.load(out)['reconstruction']
print('reconstruction shape:', recon.shape, 'dtype:', recon.dtype)
print('finite:', bool(np.isfinite(recon).all()))
print('min/max/sum:', float(np.nanmin(recon)), float(np.nanmax(recon)), float(np.nansum(recon)))
if recon.ndim == 3:
    print('plane sums:', [float(v) for v in recon.sum(axis=(1, 2))])
    print('plane maxima:', [float(v) for v in recon.max(axis=(1, 2))])
if raw_path.exists():
    y = np.load(raw_path)['measurements'][0]
    print('measurement channel-sum stats:', float(y.sum(axis=-1).min()), float(y.sum(axis=-1).max()), float(y.sum()))
