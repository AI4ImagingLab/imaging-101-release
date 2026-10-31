"""Complete 100-frame reconstruction, using only owned explicit mathematics."""
import numpy as np,time
from .preprocessing import camera_parameters
from .physics_model import library
def reconstruct(data):
    """Return required batched arrays and identified runtime diagnostics.

    All400 images are consumed; fresh observation-only OTF calibration uses
    five images per camera. Capacity is allocation only, never truncation.
    """
    cameras=camera_parameters(data);raw=np.ascontiguousarray(data['observations'][0]);capacity=2000000
    rows=np.empty((capacity,5));count=np.zeros(1,np.int32);otf=np.empty((4,3));statistics=np.empty((100,4));start=time.perf_counter()
    code=library().numeric_tracking(cameras,raw,100,capacity,rows,count,otf,statistics)
    if code:raise RuntimeError('Local full tracking failed, code '+str(code))
    rows=rows[:count[0]]
    result={'positions':rows[:,2:][None].copy(),'frame_index':rows[:,1].astype(np.int16)[None],'track_id':rows[:,0].astype(np.int32)[None]}
    diagnostics={'otf':otf,'statistics':statistics,'runtime_seconds':time.perf_counter()-start}
    return result,diagnostics
