from pathlib import Path
import numpy as np
SQUARE_SIZE=.0022
LAYERS=((.057,1.0003),(.012,1.48899),(.043,1.34),(.8,1.0003))
SHAPE=(641,641)
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as z:r=z['reference'];d=z['measurements']
    for a in [r,d]:
        if a.shape!=(1,*SHAPE) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('reference and measurements must be finite float32 (1,641,641)')
    return r[0].copy(),d[0].copy()
