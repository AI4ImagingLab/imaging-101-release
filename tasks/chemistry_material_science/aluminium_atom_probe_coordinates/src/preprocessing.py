"""Public detector positions and standing voltages; no atom-coordinate input."""
import numpy as np
def load_data(path):
    """Return float64(200000,2) detector cm and float64(200000,) standing voltage V."""
    with np.load(path,allow_pickle=False) as z:xy=z['detector_positions'];v=z['standing_voltage']
    if xy.shape!=(1,200000,2) or xy.dtype!=np.float64 or not np.isfinite(xy).all():raise ValueError('Invalid detector_positions')
    if v.shape!=(1,200000) or v.dtype!=np.float64 or not np.isfinite(v).all() or (v<=0).any():raise ValueError('Invalid standing_voltage')
    return xy[0],v[0]
def validate_events(xy,voltage):
    """Validate finite unbatched event arrays, preserving the native float64 precision."""
    xy=np.asarray(xy,dtype=np.float64);v=np.asarray(voltage,dtype=np.float64)
    if xy.ndim!=2 or xy.shape[1]!=2 or v.shape!=(len(xy),) or not len(xy):raise ValueError('Event shapes disagree')
    if not np.isfinite(xy).all() or not np.isfinite(v).all() or (v<=0).any():raise ValueError('Invalid finite event values')
    return xy,v
