"""Visible measurements; stored thermal fields are Kelvin, not Celsius."""
import numpy as np

def load_data(path='data/raw_data.npz'):
    """Return temperature:(25,66,50) K; time:(25,) s; x:(50,),y:(66,) m."""
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k][0].copy() for k in ('temperature','time','x','y')}
    shapes={'temperature':(25,66,50),'time':(25,),'x':(50,),'y':(66,)}
    for key,value in data.items():
        if value.shape!=shapes[key] or value.dtype!=np.float64 or not np.isfinite(value).all():
            raise ValueError('Invalid shape, dtype or nonfinite values: '+key)
    if np.any(data['temperature']<=0):raise ValueError('Absolute temperature must be positive')
    for key,step in [('time',1.),('x',.0127),('y',.0127)]:
        if not np.allclose(np.diff(data[key]),step,rtol=1e-12,atol=1e-14):raise ValueError('Unexpected acquisition grid: '+key)
    return data
