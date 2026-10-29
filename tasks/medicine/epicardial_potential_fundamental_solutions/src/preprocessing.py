"""Read the complete public geometry and measured jacket potentials."""
import numpy as np
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as z:return {k:z[k][0] for k in z.files}
