"""Native real MRE harmonic-wave observations; explicit public contract."""
from pathlib import Path
import numpy as np
DENSITY=1000.0
def load_data(path="data/raw_data.npz"):
    """Return wave(F,x,y,z,C)complex128, support(x,y,z)bool,frequencies(F),coordinates(x,y,z)."""
    with np.load(Path(path),allow_pickle=False) as z:data={k:z[k] for k in z.files}
    shapes={"wave":(1,2,3,25,80,128),"frequency":(1,2),"support":(1,25,80,128),"x":(1,128),"y":(1,80),"z":(1,25)}
    if set(data)!=set(shapes):raise ValueError("Invalid input keys")
    for k,s in shapes.items():
        dtype=np.complex128 if k=="wave" else np.uint8 if k=="support" else np.float64
        if data[k].shape!=s or data[k].dtype!=dtype or not np.isfinite(data[k]).all():raise ValueError("Invalid "+k)
    if not np.isin(data["support"],[0,1]).all():raise ValueError("Support must be binary")
    for k in ['x','y','z']:
        if not np.all(np.diff(data[k][0])>0):raise ValueError("Coordinates must increase")
    if not np.all(data['frequency']>0):raise ValueError("Frequencies must be positive")
    return data['wave'][0].transpose(0,4,3,2,1),data['support'][0].transpose(2,1,0).astype(bool),data['frequency'][0],tuple(data[k][0] for k in ['x','y','z'])
