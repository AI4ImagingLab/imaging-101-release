"""Strict observation loading; no target-derived calibration."""
from pathlib import Path
import numpy as np
KEYS={'observations','camera_matrix','rotation','inverse_rotation','translation','camera_centres','distortion'}
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as z:
        if set(z.files)!=KEYS:raise ValueError('Unexpected observation keys')
        data={k:z[k] for k in z.files}
    if data['observations'].shape!=(1,100,4,1024,1024) or data['observations'].dtype!=np.uint32:raise ValueError('Observation shape/dtype')
    if np.max(data['observations'])>255:raise ValueError('Counts exceed native sensor convention')
    for key,shape in [('camera_matrix',(1,4,3,3)),('rotation',(1,4,3,3)),('inverse_rotation',(1,4,3,3)),('translation',(1,4,3)),('camera_centres',(1,4,3)),('distortion',(1,4,5))]:
        a=data[key]
        if a.shape!=shape or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Invalid '+key)
    return data
def camera_parameters(data):
    return np.ascontiguousarray(np.concatenate([data[k][0].reshape(4,-1) for k in ['camera_matrix','rotation','inverse_rotation','translation','camera_centres','distortion']],axis=1))
