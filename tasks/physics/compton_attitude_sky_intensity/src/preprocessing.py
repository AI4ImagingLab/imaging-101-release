"""Read allowlisted NPZ observations and target-independent calibration.

Response integers are streamed from standard NPZ into exact uint16 storage;
calibration is applied to float32 computation tiles in the acquisition operator.
No upstream package, external response file, or reconstruction is accessed.
"""
from pathlib import Path
from dataclasses import dataclass
import zipfile
import numpy as np
from .physics_model import ComptonOperator

@dataclass
class ObservationData:
    events: np.ndarray
    background: np.ndarray
    operator: ComptonOperator

def _read_response(path):
    with zipfile.ZipFile(path) as archive,archive.open('response_counts.npy') as handle:
        version=np.lib.format.read_magic(handle)
        if version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(handle)
        elif version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(handle)
        else:raise ValueError('Unsupported NPY header version')
        if shape!=(1,3072,1,1,60,3072) or fortran or dtype!=np.dtype('uint16'):
            raise ValueError('Invalid response count contract')
        response=np.empty((3072,60*3072),dtype=np.uint16)
        target=response.reshape(-1);offset=0
        while offset<target.size:
            count=min(2**20,target.size-offset)
            block=handle.read(count*dtype.itemsize)
            if len(block)!=count*dtype.itemsize:raise ValueError('Truncated response counts')
            values=np.frombuffer(block,dtype=dtype)
            target[offset:offset+count]=values
            offset+=count
        if handle.read(1):raise ValueError('Unexpected trailing response values')
    return response

def load_observations(path):
    """Return whole physical-bin observations and the calibrated operator."""
    path=Path(path)
    with np.load(path,allow_pickle=False) as data:
        events=data['events'];background=data['background']
        for name,array in [('events',events),('background',background)]:
            if array.shape!=(1,278,1,60,3072) or array.dtype!=np.uint8:
                raise ValueError(name+' has invalid shape or dtype')
        coordinates=data['attitude_coordinates']
        weights=data['attitude_weights_s']
        exposure=data['exposure_cm2_s_sr']
        area=data['response_area_cm2']
    if coordinates.shape!=(1,3,3416064) or coordinates.dtype!=np.int32:raise ValueError('Invalid attitude coordinates')
    if weights.shape!=(1,3416064) or weights.dtype!=np.float32:raise ValueError('Invalid attitude weights')
    if area.shape!=(1,1) or not np.isfinite(area).all() or area.item()<=0:raise ValueError('Invalid instrument area')
    if exposure.shape!=(1,3072) or exposure.dtype!=np.float32 or not np.isfinite(exposure).all() or np.any(exposure<=0):raise ValueError('Invalid exposure')
    if not np.isfinite(weights).all() or np.any(weights<0):raise ValueError('Invalid attitude weights')
    # Original HDF5 includes one underflow slot on every axis. Its nonzero
    # coordinate records are 1-based physical indices; all are physical bins.
    if np.any(coordinates<1) or np.any(coordinates[0,0]>278) or np.any(coordinates[0,1:]>3072):raise ValueError('Invalid padded attitude indices')
    response=_read_response(path)
    operator=ComptonOperator(response,coordinates[0]-1,weights[0],exposure,response_area=area.item())
    return ObservationData(events[0].reshape(278,-1),background[0].reshape(278,-1),operator)
