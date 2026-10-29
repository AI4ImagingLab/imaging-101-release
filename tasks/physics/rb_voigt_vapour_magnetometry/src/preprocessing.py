import numpy as np

def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as a:
        result={angle:(a[f'detuning_{angle}'][0],a[f'transmission_{angle}'][0]) for angle in [0,45,90]}
    for x,y in result.values():
        if x.dtype!=np.float64 or y.dtype!=np.float64 or x.shape!=y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):raise ValueError('Invalid native spectrum')
    return result
