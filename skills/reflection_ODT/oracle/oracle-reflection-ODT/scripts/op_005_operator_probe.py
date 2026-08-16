import json
import numpy as np
from pathlib import Path

root = Path.cwd()
with open(root / 'public' / 'visible_data' / 'cases' / 'case_000' / 'params_data.json', 'r') as f:
    p = json.load(f)
nz, ny, nx = p['volume_shape']
n0 = p['n0']
res = [v * n0 / p['wavelength_um'] for v in p['res_um']]
fx = np.fft.fftfreq(nx) / res[0]
fy = np.fft.fftfreq(ny) / res[1]
FY, FX = np.meshgrid(fy, fx, indexing='ij')
cg = np.sqrt(np.maximum(1.0 - FX**2 - FY**2, 1e-8))
eva = np.exp(np.minimum((cg - 0.2) * 5.0, 0.0))
cutoff = np.sqrt(1.0 - (p['NA_obj'] / n0)**2)
t = np.exp(np.minimum(cg - cutoff, 0.01) * 10000.0)
pupil = t / (1.0 + t)
print('res', res)
print('c_gamma range', float(cg.min()), float(cg.max()))
print('eva range', float(eva.min()), float(eva.max()))
print('pupil range', float(pupil.min()), float(pupil.max()), 'mean', float(pupil.mean()))
